"""Strict, dependency-free analysis for paper-oriented evaluation artifacts.

This module implements the procedures disclosed in Appendices A.3, A.5, and B.1:
label-blind patient-preserving task caps, whole-cohort binary ranking metrics with
median format imputation, equal-task macro averages, joint patient-cluster bootstrap,
matched-seed t intervals, and the two N=8 inference-scaling controls.  It does not
claim to reconstruct unreleased author prompts, cohorts, or scoring manifests.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from vapa.artifacts import (
    ArtifactContentKind,
    artifact_fingerprint,
    atomic_write_text,
    canonical_json_dumps,
    fingerprint_file,
    guard_artifact_write_path,
    validate_output_paths,
)
from vapa.data.io import DataValidationError, load_json, load_jsonl
from vapa.evaluation.metrics import MetricInputError, auprc, auroc
from vapa.evaluation.statistics import paired_t_interval
from vapa.provenance import package_code_fingerprint

ANALYSIS_SCHEMA_VERSION = 1
DEFAULT_TASK_CAP = 2_500
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
ANALYSIS_WARNING = (
    "This analysis implements disclosed procedures but is not an author-exact "
    "reconstruction of prompts, data, or private scoring artifacts."
)


class AnalysisInputError(DataValidationError):
    """Raised when an analysis artifact cannot support the declared estimand."""


def _exact_keys(
    value: Mapping[str, object],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    location: str,
) -> None:
    missing = required - set(value)
    unexpected = set(value) - required - optional
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if unexpected:
            details.append(f"unexpected={sorted(unexpected)}")
        raise AnalysisInputError(f"{location} has incompatible fields: {', '.join(details)}")


def _identifier(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise AnalysisInputError(f"{location} must be a non-empty, trimmed string")
    return value


def _nonnegative_integer(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AnalysisInputError(f"{location} must be a nonnegative integer")
    return value


def _finite_number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise AnalysisInputError(f"{location} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise AnalysisInputError(f"{location} must be a finite number")
    return converted


def _sha256(value: object, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AnalysisInputError(f"{location} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class MethodContrast:
    name: str
    left_system: str
    right_system: str

    def __post_init__(self) -> None:
        _identifier(self.name, "contrast.name")
        _identifier(self.left_system, "contrast.left_system")
        _identifier(self.right_system, "contrast.right_system")
        if self.left_system == self.right_system:
            raise AnalysisInputError("a contrast requires two different systems")

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "left_system": self.left_system,
            "right_system": self.right_system,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> MethodContrast:
        _exact_keys(
            value,
            required={"name", "left_system", "right_system"},
            location="contrast",
        )
        return cls(
            name=value["name"],  # type: ignore[arg-type]
            left_system=value["left_system"],  # type: ignore[arg-type]
            right_system=value["right_system"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class AnalysisManifest:
    analysis_id: str
    records_sha256: str
    seed: int = 0
    task_cap: int = DEFAULT_TASK_CAP
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES
    contrasts: tuple[MethodContrast, ...] = ()
    paper_exact: bool = False
    schema_version: int = ANALYSIS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.analysis_id, "analysis_id")
        _sha256(self.records_sha256, "records_sha256")
        _nonnegative_integer(self.seed, "seed")
        if (
            isinstance(self.task_cap, bool)
            or not isinstance(self.task_cap, int)
            or not 1 <= self.task_cap <= DEFAULT_TASK_CAP
        ):
            raise AnalysisInputError("task_cap must be an integer in [1, 2500]")
        if (
            isinstance(self.bootstrap_replicates, bool)
            or not isinstance(self.bootstrap_replicates, int)
            or not 2 <= self.bootstrap_replicates <= 100_000
        ):
            raise AnalysisInputError("bootstrap_replicates must be in [2, 100000]")
        if self.paper_exact is not False:
            raise AnalysisInputError("public analysis manifests must set paper_exact=false")
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != ANALYSIS_SCHEMA_VERSION
        ):
            raise AnalysisInputError("unsupported analysis manifest schema_version")
        if any(not isinstance(contrast, MethodContrast) for contrast in self.contrasts):
            raise AnalysisInputError("contrasts must contain MethodContrast values")
        names = [contrast.name for contrast in self.contrasts]
        if len(names) != len(set(names)):
            raise AnalysisInputError("contrast names must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "analysis_id": self.analysis_id,
            "paper_exact": self.paper_exact,
            "records_sha256": self.records_sha256,
            "seed": self.seed,
            "task_cap": self.task_cap,
            "bootstrap_replicates": self.bootstrap_replicates,
            "contrasts": [contrast.to_dict() for contrast in self.contrasts],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> AnalysisManifest:
        required = {"schema_version", "analysis_id", "paper_exact", "records_sha256"}
        optional = {"seed", "task_cap", "bootstrap_replicates", "contrasts"}
        _exact_keys(value, required=required, optional=optional, location="analysis manifest")
        raw_contrasts = value.get("contrasts", [])
        if not isinstance(raw_contrasts, list) or any(
            not isinstance(item, Mapping) for item in raw_contrasts
        ):
            raise AnalysisInputError("analysis manifest contrasts must be an array of objects")
        return cls(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            analysis_id=value["analysis_id"],  # type: ignore[arg-type]
            paper_exact=value["paper_exact"],  # type: ignore[arg-type]
            records_sha256=value["records_sha256"],  # type: ignore[arg-type]
            seed=value.get("seed", 0),  # type: ignore[arg-type]
            task_cap=value.get("task_cap", DEFAULT_TASK_CAP),  # type: ignore[arg-type]
            bootstrap_replicates=value.get("bootstrap_replicates", DEFAULT_BOOTSTRAP_REPLICATES),  # type: ignore[arg-type]
            contrasts=tuple(MethodContrast.from_dict(item) for item in raw_contrasts),
        )


@dataclass(frozen=True, slots=True)
class CohortInstance:
    task_id: str
    patient_id: str
    instance_id: str

    def __post_init__(self) -> None:
        _identifier(self.task_id, "task_id")
        _identifier(self.patient_id, "patient_id")
        _identifier(self.instance_id, "instance_id")

    def to_dict(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "patient_id": self.patient_id,
            "instance_id": self.instance_id,
        }


@dataclass(frozen=True, slots=True)
class BinaryPrediction:
    system_id: str
    seed: int | None
    task_id: str
    patient_id: str
    instance_id: str
    label: int
    score: float | None
    format_valid: bool
    schema_version: int = ANALYSIS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("system_id", "task_id", "patient_id", "instance_id"):
            _identifier(getattr(self, name), name)
        if self.seed is not None:
            _nonnegative_integer(self.seed, "seed")
        if (
            isinstance(self.label, bool)
            or not isinstance(self.label, int)
            or self.label not in {0, 1}
        ):
            raise AnalysisInputError("label must be integer 0 or 1")
        if not isinstance(self.format_valid, bool):
            raise AnalysisInputError("format_valid must be boolean")
        if self.format_valid:
            score = _finite_number(self.score, "score")
            if not 0.0 <= score <= 1.0:
                raise AnalysisInputError("valid binary scores must be in [0, 1]")
            object.__setattr__(self, "score", score)
        elif self.score is not None:
            raise AnalysisInputError("format failures must carry score=null")
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != ANALYSIS_SCHEMA_VERSION
        ):
            raise AnalysisInputError("unsupported binary prediction schema_version")

    @property
    def cohort_key(self) -> tuple[str, str]:
        return self.task_id, self.instance_id

    @property
    def block_key(self) -> tuple[str, int | None]:
        return self.system_id, self.seed

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "system_id": self.system_id,
            "seed": self.seed,
            "task_id": self.task_id,
            "patient_id": self.patient_id,
            "instance_id": self.instance_id,
            "label": self.label,
            "score": self.score,
            "format_valid": self.format_valid,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> BinaryPrediction:
        required = {
            "schema_version",
            "system_id",
            "seed",
            "task_id",
            "patient_id",
            "instance_id",
            "label",
            "score",
            "format_valid",
        }
        _exact_keys(value, required=required, location="binary prediction")
        return cls(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            system_id=value["system_id"],  # type: ignore[arg-type]
            seed=value["seed"],  # type: ignore[arg-type]
            task_id=value["task_id"],  # type: ignore[arg-type]
            patient_id=value["patient_id"],  # type: ignore[arg-type]
            instance_id=value["instance_id"],  # type: ignore[arg-type]
            label=value["label"],  # type: ignore[arg-type]
            score=value["score"],  # type: ignore[arg-type]
            format_valid=value["format_valid"],  # type: ignore[arg-type]
        )


def _keyed_digest(*, seed: int, namespace: str, values: Sequence[str]) -> str:
    return artifact_fingerprint(
        {
            "namespace": namespace,
            "seed": seed,
            "values": list(values),
        }
    )


def frozen_task_cap(
    instances: Iterable[CohortInstance],
    *,
    cap: int = DEFAULT_TASK_CAP,
    seed: int = 0,
) -> tuple[CohortInstance, ...]:
    """Apply the frozen, label-blind round-r patient-preserving task cap."""

    if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= DEFAULT_TASK_CAP:
        raise AnalysisInputError("cap must be an integer in [1, 2500]")
    _nonnegative_integer(seed, "seed")
    by_task: dict[str, dict[str, list[CohortInstance]]] = defaultdict(lambda: defaultdict(list))
    seen: set[tuple[str, str]] = set()
    for instance in instances:
        if not isinstance(instance, CohortInstance):
            raise TypeError("frozen_task_cap requires CohortInstance values")
        key = (instance.task_id, instance.instance_id)
        if key in seen:
            raise AnalysisInputError(f"duplicate cohort instance: {key!r}")
        seen.add(key)
        by_task[instance.task_id][instance.patient_id].append(instance)
    if not seen:
        raise AnalysisInputError("the frozen task cap requires at least one instance")

    selected: list[CohortInstance] = []
    for task_id in sorted(by_task):
        patients = by_task[task_id]
        if len(patients) > cap:
            raise AnalysisInputError(
                f"task {task_id!r} has {len(patients)} patients, exceeding cap {cap}; "
                "the disclosed rule cannot retain every patient"
            )
        patient_order = sorted(
            patients,
            key=lambda patient_id: (
                _keyed_digest(
                    seed=seed,
                    namespace="vapa-analysis-patient-order-v1",
                    values=(task_id, patient_id),
                ),
                patient_id,
            ),
        )
        ordered_instances: dict[str, list[CohortInstance]] = {}
        for patient_id in patient_order:
            ordered_instances[patient_id] = sorted(
                patients[patient_id],
                key=lambda instance: (
                    _keyed_digest(
                        seed=seed,
                        namespace="vapa-analysis-instance-order-v1",
                        values=(task_id, patient_id, instance.instance_id),
                    ),
                    instance.instance_id,
                ),
            )
        round_index = 0
        task_selected: list[CohortInstance] = []
        while len(task_selected) < cap:
            admitted = False
            for patient_id in patient_order:
                candidates = ordered_instances[patient_id]
                if round_index < len(candidates):
                    task_selected.append(candidates[round_index])
                    admitted = True
                    if len(task_selected) == cap:
                        break
            if not admitted:
                break
            round_index += 1
        selected.extend(task_selected)
    return tuple(selected)


def _normalize_predictions(
    values: Iterable[BinaryPrediction | Mapping[str, object]],
) -> tuple[BinaryPrediction, ...]:
    predictions = tuple(
        value if isinstance(value, BinaryPrediction) else BinaryPrediction.from_dict(value)
        for value in values
    )
    if not predictions:
        raise AnalysisInputError("binary analysis requires at least one prediction")
    return predictions


def _validate_complete_panel(
    predictions: Sequence[BinaryPrediction],
) -> tuple[tuple[CohortInstance, ...], dict[tuple[str, int | None], set[tuple[str, str]]]]:
    cohort: dict[tuple[str, str], tuple[str, int]] = {}
    blocks: dict[tuple[str, int | None], set[tuple[str, str]]] = defaultdict(set)
    seen: set[tuple[str, int | None, str, str]] = set()
    for prediction in predictions:
        identity = (
            prediction.system_id,
            prediction.seed,
            prediction.task_id,
            prediction.instance_id,
        )
        if identity in seen:
            raise AnalysisInputError(f"duplicate system/seed/task/instance record: {identity!r}")
        seen.add(identity)
        cohort_value = (prediction.patient_id, prediction.label)
        previous = cohort.setdefault(prediction.cohort_key, cohort_value)
        if previous != cohort_value:
            raise AnalysisInputError(
                f"patient or label disagreement for cohort instance {prediction.cohort_key!r}"
            )
        blocks[prediction.block_key].add(prediction.cohort_key)
    expected = set(cohort)
    for block, actual in blocks.items():
        if actual != expected:
            missing = len(expected - actual)
            extra = len(actual - expected)
            raise AnalysisInputError(
                f"prediction panel block {block!r} is incomplete: missing={missing}, extra={extra}"
            )
    seeds_by_system: dict[str, set[int | None]] = defaultdict(set)
    for system_id, seed in blocks:
        seeds_by_system[system_id].add(seed)
    mixed = [
        system_id
        for system_id, seeds in seeds_by_system.items()
        if None in seeds and len(seeds) > 1
    ]
    if mixed:
        raise AnalysisInputError(f"systems cannot mix fixed and trained-seed records: {mixed}")
    roster = tuple(
        CohortInstance(task_id, patient_id, instance_id)
        for (task_id, instance_id), (patient_id, _) in sorted(cohort.items())
    )
    return roster, blocks


@dataclass(frozen=True, slots=True)
class _ScoredObservation:
    patient_id: str
    label: int
    score: float


def _scored_groups(
    predictions: Sequence[BinaryPrediction],
) -> tuple[
    dict[tuple[str, int | None, str], tuple[_ScoredObservation, ...]],
    dict[tuple[str, int | None, str], dict[str, float | int]],
]:
    grouped: dict[tuple[str, int | None, str], list[BinaryPrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[(prediction.system_id, prediction.seed, prediction.task_id)].append(prediction)
    scored: dict[tuple[str, int | None, str], tuple[_ScoredObservation, ...]] = {}
    details: dict[tuple[str, int | None, str], dict[str, float | int]] = {}
    for key, members in grouped.items():
        valid_scores = [member.score for member in members if member.format_valid]
        if not valid_scores:
            raise AnalysisInputError(
                f"system/seed/task block {key!r} has no valid score for median imputation"
            )
        median_score = statistics.median(valid_scores)  # type: ignore[arg-type]
        observations = tuple(
            _ScoredObservation(
                member.patient_id,
                member.label,
                median_score if member.score is None else member.score,
            )
            for member in members
        )
        failures = sum(not member.format_valid for member in members)
        scored[key] = observations
        details[key] = {
            "instances": len(members),
            "patients": len({member.patient_id for member in members}),
            "positives": sum(member.label for member in members),
            "format_failures": failures,
            "format_failure_rate": failures / len(members),
            "imputation_score": median_score,
        }
    return scored, details


def _ranking_metrics(
    observations: Sequence[_ScoredObservation],
    multiplicities: Mapping[str, int] | None = None,
) -> tuple[float, float]:
    labels: list[int] = []
    scores: list[float] = []
    for observation in observations:
        copies = 1 if multiplicities is None else multiplicities.get(observation.patient_id, 0)
        for _ in range(copies):
            labels.append(observation.label)
            scores.append(observation.score)
    try:
        return 100.0 * auprc(labels, scores), 100.0 * auroc(labels, scores)
    except MetricInputError as error:
        raise AnalysisInputError(
            "a task lacks both classes under the requested patient resample"
        ) from error


def binary_task_metrics(
    values: Iterable[BinaryPrediction | Mapping[str, object]],
) -> dict[str, object]:
    """Compute median-imputed per-task metrics and equal-task macro averages."""

    predictions = _normalize_predictions(values)
    _validate_complete_panel(predictions)
    scored, details = _scored_groups(predictions)
    systems: dict[str, dict[str, object]] = {}
    system_ids = sorted({prediction.system_id for prediction in predictions})
    for system_id in system_ids:
        run_keys = sorted(
            {seed for candidate_system, seed, _ in scored if candidate_system == system_id},
            key=lambda seed: -1 if seed is None else seed,
        )
        runs: dict[str, object] = {}
        for seed in run_keys:
            tasks: dict[str, object] = {}
            auprc_values: list[float] = []
            auroc_values: list[float] = []
            task_ids = sorted(
                task_id
                for candidate_system, candidate_seed, task_id in scored
                if candidate_system == system_id and candidate_seed == seed
            )
            for task_id in task_ids:
                key = (system_id, seed, task_id)
                task_auprc, task_auroc = _ranking_metrics(scored[key])
                auprc_values.append(task_auprc)
                auroc_values.append(task_auroc)
                tasks[task_id] = {
                    **details[key],
                    "auprc": task_auprc,
                    "auroc": task_auroc,
                }
            run_name = "fixed" if seed is None else str(seed)
            runs[run_name] = {
                "seed": seed,
                "tasks": tasks,
                "macro_auprc": statistics.fmean(auprc_values),
                "macro_auroc": statistics.fmean(auroc_values),
            }
        system_result: dict[str, object] = {"runs": runs}
        if "fixed" not in runs and len(runs) >= 2:
            run_values = tuple(runs.values())
            auprc_values = [run["macro_auprc"] for run in run_values]  # type: ignore[index]
            auroc_values = [run["macro_auroc"] for run in run_values]  # type: ignore[index]
            system_result["seed_aggregate"] = {
                "seeds": len(runs),
                "macro_auprc_mean": statistics.fmean(auprc_values),  # type: ignore[arg-type]
                "macro_auprc_standard_deviation": statistics.stdev(auprc_values),  # type: ignore[arg-type]
                "macro_auroc_mean": statistics.fmean(auroc_values),  # type: ignore[arg-type]
                "macro_auroc_standard_deviation": statistics.stdev(auroc_values),  # type: ignore[arg-type]
            }
        systems[system_id] = system_result
    return {"scale": "percent", "systems": systems}


def paired_seed_t_summary(
    left: Mapping[int, float],
    right: Mapping[int, float],
) -> dict[str, object]:
    """Aggregate one matched-seed contrast with the Appendix B.1 t interval."""

    if set(left) != set(right) or len(left) < 2:
        raise AnalysisInputError("paired seed inputs require the same two or more seed IDs")
    seed_ids = sorted(left)
    differences: list[float] = []
    for seed in seed_ids:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise AnalysisInputError("paired seed IDs must be nonnegative integers")
        left_value = _finite_number(left[seed], f"left[{seed}]")
        right_value = _finite_number(right[seed], f"right[{seed}]")
        differences.append(left_value - right_value)
    interval = paired_t_interval(differences)
    return {
        "estimate": interval.estimate,
        "lower": interval.lower,
        "upper": interval.upper,
        "n": interval.n,
        "seed_ids": seed_ids,
        "within_seed_differences": differences,
    }


def _system_seed_values(
    metric_results: Mapping[str, object],
    system_id: str,
    metric: str,
) -> tuple[dict[int, float], float | None]:
    systems = metric_results.get("systems")
    if not isinstance(systems, Mapping) or system_id not in systems:
        raise AnalysisInputError(f"unknown contrast system {system_id!r}")
    system = systems[system_id]
    if not isinstance(system, Mapping) or not isinstance(system.get("runs"), Mapping):
        raise AnalysisInputError("internal metric result schema is malformed")
    runs = system["runs"]
    assert isinstance(runs, Mapping)
    if "fixed" in runs:
        if len(runs) != 1:
            raise AnalysisInputError("a system cannot mix fixed and trained-seed records")
        fixed = runs["fixed"]
        if not isinstance(fixed, Mapping):
            raise AnalysisInputError("internal fixed-system run schema is malformed")
        return {}, _finite_number(fixed.get(metric), f"{system_id}.fixed.{metric}")
    values: dict[int, float] = {}
    for raw_run in runs.values():
        if not isinstance(raw_run, Mapping):
            raise AnalysisInputError("internal metric run schema is malformed")
        seed = raw_run.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise AnalysisInputError("trained-system runs require integer seeds")
        values[seed] = _finite_number(raw_run.get(metric), f"{system_id}.{seed}.{metric}")
    return values, None


def paired_seed_contrasts(
    metric_results: Mapping[str, object],
    contrasts: Iterable[MethodContrast],
) -> dict[str, object]:
    output: dict[str, object] = {}
    for contrast in contrasts:
        metrics: dict[str, object] = {}
        for metric in ("macro_auprc", "macro_auroc"):
            left, left_fixed = _system_seed_values(metric_results, contrast.left_system, metric)
            right, right_fixed = _system_seed_values(metric_results, contrast.right_system, metric)
            if left_fixed is not None and right_fixed is not None:
                break
            if left_fixed is not None:
                left = {seed: left_fixed for seed in right}
            if right_fixed is not None:
                right = {seed: right_fixed for seed in left}
            metrics[metric] = paired_seed_t_summary(left, right)
        if metrics:
            output[contrast.name] = {
                "left_system": contrast.left_system,
                "right_system": contrast.right_system,
                "interval": "paired_seed_t_95",
                "metrics": metrics,
            }
    return output


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_summary(estimate: float, replicates: Sequence[float]) -> dict[str, object]:
    return {
        "estimate": estimate,
        "lower": _percentile(replicates, 0.025),
        "upper": _percentile(replicates, 0.975),
        "bootstrap_standard_error": statistics.stdev(replicates),
        "replicates": len(replicates),
    }


def joint_patient_cluster_bootstrap(
    values: Iterable[BinaryPrediction | Mapping[str, object]],
    *,
    systems: Iterable[str] | None = None,
    contrasts: Iterable[MethodContrast] = (),
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = 0,
) -> dict[str, object]:
    """Resample patient IDs once over the task union and apply draws to all methods."""

    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or not 2 <= replicates <= 100_000
    ):
        raise AnalysisInputError("bootstrap replicates must be in [2, 100000]")
    _nonnegative_integer(seed, "seed")
    predictions = _normalize_predictions(values)
    roster, blocks = _validate_complete_panel(predictions)
    all_systems = sorted({prediction.system_id for prediction in predictions})
    selected_systems = all_systems if systems is None else sorted(set(systems))
    if not selected_systems or set(selected_systems) - set(all_systems):
        raise AnalysisInputError("bootstrap systems must be a non-empty known subset")
    for system_id in selected_systems:
        seeds = {block_seed for block_system, block_seed in blocks if block_system == system_id}
        if seeds != {None}:
            raise AnalysisInputError(
                "patient bootstrap is for fixed systems; trained systems use paired seed intervals"
            )
    contrast_list = tuple(contrasts)
    for contrast in contrast_list:
        if (
            contrast.left_system not in selected_systems
            or contrast.right_system not in selected_systems
        ):
            raise AnalysisInputError(
                f"bootstrap contrast {contrast.name!r} references an unselected system"
            )

    filtered = tuple(
        prediction for prediction in predictions if prediction.system_id in selected_systems
    )
    scored, _ = _scored_groups(filtered)
    tasks = sorted({instance.task_id for instance in roster})
    patients = sorted({instance.patient_id for instance in roster})
    if not patients:
        raise AnalysisInputError("patient bootstrap requires at least one patient")

    def macro(system_id: str, multiplicities: Mapping[str, int] | None) -> dict[str, float]:
        task_auprc: list[float] = []
        task_auroc: list[float] = []
        for task_id in tasks:
            key = (system_id, None, task_id)
            if key not in scored:
                raise AnalysisInputError(f"system {system_id!r} lacks task {task_id!r}")
            one_auprc, one_auroc = _ranking_metrics(scored[key], multiplicities)
            task_auprc.append(one_auprc)
            task_auroc.append(one_auroc)
        return {
            "macro_auprc": statistics.fmean(task_auprc),
            "macro_auroc": statistics.fmean(task_auroc),
        }

    estimates = {system_id: macro(system_id, None) for system_id in selected_systems}
    samples: dict[str, dict[str, list[float]]] = {
        system_id: {"macro_auprc": [], "macro_auroc": []} for system_id in selected_systems
    }
    contrast_samples: dict[str, dict[str, list[float]]] = {
        contrast.name: {"macro_auprc": [], "macro_auroc": []} for contrast in contrast_list
    }
    generator = random.Random(seed)
    for _ in range(replicates):
        multiplicities = Counter(
            patients[generator.randrange(len(patients))] for _ in range(len(patients))
        )
        replicate_metrics: dict[str, dict[str, float]] = {}
        for system_id in selected_systems:
            result = macro(system_id, multiplicities)
            replicate_metrics[system_id] = result
            for metric, metric_value in result.items():
                samples[system_id][metric].append(metric_value)
        for contrast in contrast_list:
            for metric in ("macro_auprc", "macro_auroc"):
                contrast_samples[contrast.name][metric].append(
                    replicate_metrics[contrast.left_system][metric]
                    - replicate_metrics[contrast.right_system][metric]
                )

    endpoints = {
        system_id: {
            metric: _bootstrap_summary(estimates[system_id][metric], metric_samples)
            for metric, metric_samples in samples[system_id].items()
        }
        for system_id in selected_systems
    }
    paired: dict[str, object] = {}
    for contrast in contrast_list:
        paired[contrast.name] = {
            "left_system": contrast.left_system,
            "right_system": contrast.right_system,
            "metrics": {
                metric: _bootstrap_summary(
                    estimates[contrast.left_system][metric]
                    - estimates[contrast.right_system][metric],
                    metric_samples,
                )
                for metric, metric_samples in contrast_samples[contrast.name].items()
            },
        }
    return {
        "method": "joint_patient_cluster_percentile_95",
        "seed": seed,
        "replicates": replicates,
        "patient_union_size": len(patients),
        "task_count": len(tasks),
        "endpoints": endpoints,
        "paired_contrasts": paired,
    }


@dataclass(frozen=True, slots=True)
class TrajectoryCandidate:
    trajectory_id: str
    answer: object
    answer_valid: bool
    policy_logprob: float
    generated_tokens: int
    evidence_chain: tuple[str, ...] = ()
    binary_score: float | None = None

    def __post_init__(self) -> None:
        _identifier(self.trajectory_id, "trajectory_id")
        if not isinstance(self.answer_valid, bool):
            raise AnalysisInputError("answer_valid must be boolean")
        if not self.answer_valid and self.answer is not None:
            raise AnalysisInputError("invalid answers must use answer=null")
        if self.answer_valid:
            try:
                canonical_json_dumps(self.answer)
            except (TypeError, ValueError) as error:
                raise AnalysisInputError("answer must be finite JSON-compatible data") from error
        logprob = _finite_number(self.policy_logprob, "policy_logprob")
        if logprob > 0.0:
            raise AnalysisInputError("policy_logprob cannot be positive")
        object.__setattr__(self, "policy_logprob", logprob)
        if (
            isinstance(self.generated_tokens, bool)
            or not isinstance(self.generated_tokens, int)
            or self.generated_tokens < 1
        ):
            raise AnalysisInputError("generated_tokens must be a positive integer")
        if any(
            not isinstance(pointer, str) or not pointer.strip() for pointer in self.evidence_chain
        ):
            raise AnalysisInputError("evidence_chain must contain non-empty strings")
        object.__setattr__(self, "evidence_chain", tuple(self.evidence_chain))
        if self.binary_score is not None:
            binary_score = _finite_number(self.binary_score, "binary_score")
            if not 0.0 <= binary_score <= 1.0:
                raise AnalysisInputError("binary_score must be in [0, 1]")
            object.__setattr__(self, "binary_score", binary_score)

    @property
    def length_normalized_logprob(self) -> float:
        return self.policy_logprob / self.generated_tokens


def self_consistency(
    candidates: Sequence[TrajectoryCandidate],
    *,
    n: int = 8,
    binary: bool = False,
    canonicalizer: Callable[[object], str] | None = None,
) -> dict[str, object]:
    """Select the deployable N-sample plurality or average binary class scores."""

    if isinstance(n, bool) or not isinstance(n, int) or len(candidates) != n or n < 1:
        raise AnalysisInputError(f"self-consistency requires exactly N={n} trajectories")
    if any(not isinstance(candidate, TrajectoryCandidate) for candidate in candidates):
        raise TypeError("self_consistency requires TrajectoryCandidate values")
    if len({candidate.trajectory_id for candidate in candidates}) != len(candidates):
        raise AnalysisInputError("trajectory IDs must be unique")
    if binary:
        scores = [candidate.binary_score for candidate in candidates]
        if any(score is None for score in scores):
            raise AnalysisInputError("binary self-consistency requires every class score")
        return {
            "method": "self_consistency_binary_average",
            "deployable": True,
            "n": n,
            "binary_score": statistics.fmean(scores),  # type: ignore[arg-type]
        }

    normalize = canonicalizer or canonical_json_dumps
    groups: dict[str, list[TrajectoryCandidate]] = defaultdict(list)
    for candidate in candidates:
        if not candidate.answer_valid:
            continue
        canonical = normalize(candidate.answer)
        if not isinstance(canonical, str) or not canonical:
            raise AnalysisInputError("answer canonicalizer must return a non-empty string")
        groups[canonical].append(candidate)
    if not groups:
        raise AnalysisInputError("self-consistency has no valid terminal answer")
    ranked_groups = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            -statistics.fmean(member.length_normalized_logprob for member in item[1]),
            item[0],
        ),
    )
    canonical_answer, supporters = ranked_groups[0]
    supporting = sorted(
        supporters,
        key=lambda candidate: (-candidate.policy_logprob, candidate.trajectory_id),
    )[0]
    return {
        "method": "self_consistency_plurality",
        "deployable": True,
        "n": n,
        "valid_trajectories": sum(len(group) for group in groups.values()),
        "plurality_count": len(supporters),
        "canonical_answer": canonical_answer,
        "answer": supporting.answer,
        "mean_length_normalized_logprob": statistics.fmean(
            member.length_normalized_logprob for member in supporters
        ),
        "supporting_trajectory_id": supporting.trajectory_id,
        "evidence_chain": list(supporting.evidence_chain),
    }


def oracle_best_of_n(
    candidates: Sequence[TrajectoryCandidate],
    metric_outcomes: Mapping[str, Mapping[str, float | bool]],
    *,
    n: int = 8,
) -> dict[str, object]:
    """Return a hidden-outcome sampling ceiling, explicitly marked nondeployable."""

    if isinstance(n, bool) or not isinstance(n, int) or len(candidates) != n or n < 1:
        raise AnalysisInputError(f"oracle best-of-N requires exactly N={n} trajectories")
    if any(not isinstance(candidate, TrajectoryCandidate) for candidate in candidates):
        raise TypeError("oracle_best_of_n requires TrajectoryCandidate values")
    candidate_ids = {candidate.trajectory_id for candidate in candidates}
    if len(candidate_ids) != len(candidates):
        raise AnalysisInputError("trajectory IDs must be unique")
    if not metric_outcomes:
        raise AnalysisInputError("oracle best-of-N requires at least one metric")
    selected: dict[str, object] = {}
    for metric, outcomes in sorted(metric_outcomes.items()):
        _identifier(metric, "oracle metric")
        if set(outcomes) != candidate_ids:
            raise AnalysisInputError(
                f"oracle metric {metric!r} must score every candidate exactly once"
            )
        ranked = sorted(
            (
                (
                    float(value)
                    if isinstance(value, bool)
                    else _finite_number(value, f"oracle.{metric}.{trajectory_id}"),
                    trajectory_id,
                )
                for trajectory_id, value in outcomes.items()
            ),
            key=lambda item: (-item[0], item[1]),
        )
        value, trajectory_id = ranked[0]
        selected[metric] = {"value": value, "trajectory_id": trajectory_id}
    return {
        "method": "oracle_best_of_n",
        "deployable": False,
        "n": n,
        "warning": "Uses hidden metric outcomes and may select a different trajectory per metric.",
        "metrics": selected,
    }


def analyze_binary_results(
    manifest_path: str | Path,
    records_path: str | Path,
    output_path: str | Path,
    *,
    content_kind: ArtifactContentKind | str = ArtifactContentKind.DERIVED,
    repository_root: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Validate, cap, analyze, and atomically emit one canonical JSON artifact."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    records_file = Path(records_path).expanduser().resolve()
    destination = guard_artifact_write_path(
        output_path, content_kind=content_kind, repository_root=repository_root
    )
    validate_output_paths([destination], inputs=[manifest_file, records_file], overwrite=overwrite)
    if manifest_file.suffix.casefold() != ".json":
        raise AnalysisInputError("analysis manifest must be a .json file")
    if records_file.suffix.casefold() not in {".jsonl", ".ndjson"}:
        raise AnalysisInputError("binary prediction records must be JSONL")
    manifest_fingerprint = fingerprint_file(manifest_file)
    records_fingerprint = fingerprint_file(records_file)
    implementation_sha256 = package_code_fingerprint()
    raw_manifest = load_json(manifest_file)
    if not isinstance(raw_manifest, Mapping):
        raise AnalysisInputError("analysis manifest must be a JSON object")
    manifest = AnalysisManifest.from_dict(raw_manifest)
    if records_fingerprint.sha256 != manifest.records_sha256:
        raise AnalysisInputError("prediction JSONL does not match manifest records_sha256")
    raw_records = load_jsonl(records_file)
    if any(not isinstance(value, Mapping) for value in raw_records):
        raise AnalysisInputError("every binary prediction record must be an object")
    predictions = tuple(BinaryPrediction.from_dict(value) for value in raw_records)  # type: ignore[arg-type]
    if fingerprint_file(manifest_file) != manifest_fingerprint:
        raise AnalysisInputError("analysis manifest changed while it was being read")
    if fingerprint_file(records_file) != records_fingerprint:
        raise AnalysisInputError("binary prediction records changed while they were being read")
    roster, _ = _validate_complete_panel(predictions)
    selected_roster = frozen_task_cap(roster, cap=manifest.task_cap, seed=manifest.seed)
    selected_keys = {(item.task_id, item.instance_id) for item in selected_roster}
    selected_predictions = tuple(
        prediction for prediction in predictions if prediction.cohort_key in selected_keys
    )
    _validate_complete_panel(selected_predictions)
    metrics = binary_task_metrics(selected_predictions)

    systems = sorted({prediction.system_id for prediction in selected_predictions})
    system_modes: dict[str, str] = {}
    for system_id in systems:
        seeds = {
            prediction.seed
            for prediction in selected_predictions
            if prediction.system_id == system_id
        }
        if seeds == {None}:
            system_modes[system_id] = "fixed"
        elif None not in seeds and len(seeds) >= 2:
            system_modes[system_id] = "trained"
        else:
            raise AnalysisInputError(
                f"system {system_id!r} must be fixed or have at least two integer seeds"
            )
    for contrast in manifest.contrasts:
        if contrast.left_system not in system_modes or contrast.right_system not in system_modes:
            raise AnalysisInputError(f"contrast {contrast.name!r} references an unknown system")
    fixed_systems = [system for system in systems if system_modes[system] == "fixed"]
    fixed_contrasts = tuple(
        contrast
        for contrast in manifest.contrasts
        if system_modes[contrast.left_system] == system_modes[contrast.right_system] == "fixed"
    )
    bootstrap: dict[str, object] | None = None
    if fixed_systems:
        bootstrap = joint_patient_cluster_bootstrap(
            selected_predictions,
            systems=fixed_systems,
            contrasts=fixed_contrasts,
            replicates=manifest.bootstrap_replicates,
            seed=manifest.seed,
        )
    seed_contrasts = paired_seed_contrasts(metrics, manifest.contrasts)

    task_before = Counter(instance.task_id for instance in roster)
    task_after = Counter(instance.task_id for instance in selected_roster)
    patients_after: dict[str, set[str]] = defaultdict(set)
    for instance in selected_roster:
        patients_after[instance.task_id].add(instance.patient_id)
    output: dict[str, object] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_kind": "binary_task_transfer",
        "analysis_id": manifest.analysis_id,
        "paper_exact": False,
        "warning": ANALYSIS_WARNING,
        "provenance": {
            "manifest": manifest_fingerprint.to_dict(),
            "records": records_fingerprint.to_dict(),
            "configuration_sha256": artifact_fingerprint(manifest.to_dict()),
            "implementation": "vapa.evaluation.analysis:binary-v1",
            "implementation_sha256": implementation_sha256,
        },
        "configuration": manifest.to_dict(),
        "cohort": {
            "selection_rule": "frozen_label_blind_patient_round_robin",
            "task_cap": manifest.task_cap,
            "selection_sha256": artifact_fingerprint(
                [instance.to_dict() for instance in selected_roster]
            ),
            "tasks": {
                task_id: {
                    "instances_before": task_before[task_id],
                    "instances_after": task_after[task_id],
                    "patients_after": len(patients_after[task_id]),
                }
                for task_id in sorted(task_before)
            },
        },
        "metrics": metrics,
        "uncertainty": {
            "fixed_system_patient_bootstrap": bootstrap,
            "trained_system_paired_seed_contrasts": seed_contrasts,
        },
    }
    if package_code_fingerprint() != implementation_sha256:
        raise AnalysisInputError("the VAPA implementation changed during analysis")
    atomic_write_text(destination, canonical_json_dumps(output) + "\n", overwrite=overwrite)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze strict binary evaluation JSONL under a frozen manifest."
    )
    parser.add_argument("manifest", type=Path, help="analysis JSON manifest")
    parser.add_argument("records", type=Path, help="complete binary prediction JSONL panel")
    parser.add_argument("output", type=Path, help="canonical aggregate JSON output")
    parser.add_argument(
        "--content-kind",
        choices=tuple(item.value for item in ArtifactContentKind),
        default=ArtifactContentKind.DERIVED.value,
    )
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = analyze_binary_results(
        arguments.manifest,
        arguments.records,
        arguments.output,
        content_kind=arguments.content_kind,
        repository_root=arguments.repository_root,
        overwrite=arguments.overwrite,
    )
    print(canonical_json_dumps(result))
    return 0


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "ANALYSIS_WARNING",
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_TASK_CAP",
    "AnalysisInputError",
    "AnalysisManifest",
    "BinaryPrediction",
    "CohortInstance",
    "MethodContrast",
    "TrajectoryCandidate",
    "analyze_binary_results",
    "binary_task_metrics",
    "build_parser",
    "frozen_task_cap",
    "joint_patient_cluster_bootstrap",
    "main",
    "oracle_best_of_n",
    "paired_seed_contrasts",
    "paired_seed_t_summary",
    "self_consistency",
]
