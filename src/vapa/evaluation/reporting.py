"""Manifest-driven factorial, horizon, and named-profile experiment reports.

This complements the binary-transfer analysis with the remaining public statistical
control paths described by the manuscript.  It consumes patient-level metric rows,
uses equal task weights, keeps matched seeds aligned, and applies one declared Holm
family to every confirmatory t test in the report.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vapa.artifacts import (
    ArtifactContentKind,
    artifact_fingerprint,
    atomic_write_text,
    canonical_json_dumps,
    fingerprint_file,
    guard_artifact_write_path,
    strict_json_loads,
    strict_jsonl_loads,
    validate_output_paths,
)
from vapa.evaluation.statistics import (
    PairedTest,
    holm_adjust,
    ordinary_least_squares_slope,
    paired_t_test,
)
from vapa.provenance import package_code_fingerprint
from vapa.training.factorial import Arm, factorial_contrasts

EXPERIMENT_RECORD_SCHEMA_VERSION = "vapa-experiment-metric-v1"
EXPERIMENT_REPORT_SCHEMA_VERSION = "vapa-experiment-report-v1"
EXPERIMENT_REPORT_WARNING = (
    "This public analysis engine implements the declared statistical protocol; "
    "it does not contain author data, runs, or results."
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ExperimentReportError(ValueError):
    """Raised when a report input is incomplete, ambiguous, or unauthenticated."""


@dataclass(frozen=True, slots=True)
class ExperimentMetricRecord:
    system_id: str
    seed: int
    task_id: str
    patient_id: str
    metric: str
    value: float
    horizon_days: int | None
    profile: str | None
    decision_depth: int | None = None


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentReportError(f"{name} must be non-empty text")
    return value


def _string_list(value: object, name: str, *, minimum: int = 1) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ExperimentReportError(f"{name} must contain at least {minimum} names")
    result = tuple(_nonempty(item, name) for item in value)
    if len(set(result)) != len(result):
        raise ExperimentReportError(f"{name} contains duplicates")
    return result


def _strict_fields(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    if set(value) != expected:
        raise ExperimentReportError(
            f"{location} schema mismatch; "
            f"missing={sorted(expected - set(value))}, unknown={sorted(set(value) - expected)}"
        )


def load_experiment_records(path: str | Path) -> tuple[ExperimentMetricRecord, ...]:
    source = Path(path)
    try:
        content = source.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise ExperimentReportError("experiment records must be UTF-8") from error
    try:
        records = strict_jsonl_loads(content, source=str(source))
    except ValueError as error:
        raise ExperimentReportError(str(error)) from error
    rows: list[ExperimentMetricRecord] = []
    seen: set[tuple[object, ...]] = set()
    expected = {
        "schema_version",
        "system_id",
        "seed",
        "task_id",
        "patient_id",
        "metric",
        "value",
        "profile",
    }
    for line_number, raw in enumerate(records, start=1):
        if not isinstance(raw, Mapping):
            raise ExperimentReportError(f"{source}:{line_number}: row must be an object")
        axes = set(raw) & {"horizon_days", "decision_depth"}
        if not axes:
            raise ExperimentReportError(
                "metric row requires horizon_days or decision_depth (null if unused)"
            )
        _strict_fields(raw, expected | axes, f"{source}:{line_number}")
        if raw["schema_version"] != EXPERIMENT_RECORD_SCHEMA_VERSION:
            raise ExperimentReportError(f"{source}:{line_number}: unsupported schema_version")
        seed = raw["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ExperimentReportError(f"{source}:{line_number}: seed must be nonnegative")
        value = raw["value"]
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ExperimentReportError(f"{source}:{line_number}: value must be finite")
        for axis in axes:
            coordinate = raw[axis]
            if coordinate is not None and (type(coordinate) is not int or coordinate <= 0):
                raise ExperimentReportError(
                    f"{source}:{line_number}: {axis} must be positive or null"
                )
        if raw.get("horizon_days") is not None and raw.get("decision_depth") is not None:
            raise ExperimentReportError("a metric row cannot mix days and decision depth")
        profile = raw["profile"]
        if profile is not None:
            profile = _nonempty(profile, "profile")
        row = ExperimentMetricRecord(
            system_id=_nonempty(raw["system_id"], "system_id"),
            seed=seed,
            task_id=_nonempty(raw["task_id"], "task_id"),
            patient_id=_nonempty(raw["patient_id"], "patient_id"),
            metric=_nonempty(raw["metric"], "metric"),
            value=float(value),
            horizon_days=raw.get("horizon_days"),
            profile=profile,
            decision_depth=raw.get("decision_depth"),
        )
        identity = (
            row.system_id,
            row.seed,
            row.task_id,
            row.patient_id,
            row.metric,
            row.horizon_days,
            row.profile,
            row.decision_depth,
        )
        if identity in seen:
            raise ExperimentReportError(f"{source}:{line_number}: duplicate metric row")
        seen.add(identity)
        rows.append(row)
    return tuple(rows)


def _load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        raw = strict_json_loads(path.read_bytes())
    except (TypeError, ValueError) as error:
        raise ExperimentReportError(f"invalid report manifest: {error}") from error
    if not isinstance(raw, Mapping):
        raise ExperimentReportError("report manifest must be an object")
    expected = {
        "schema_version",
        "report_id",
        "records_sha256",
        "factorial",
        "horizon",
        "profile",
    }
    _strict_fields(
        raw,
        expected | ({"confirmatory_tests"} if "confirmatory_tests" in raw else set()),
        "report manifest",
    )
    if raw["schema_version"] != EXPERIMENT_REPORT_SCHEMA_VERSION:
        raise ExperimentReportError("unsupported report manifest schema_version")
    _nonempty(raw["report_id"], "report_id")
    if not isinstance(raw["records_sha256"], str) or not _SHA256.fullmatch(raw["records_sha256"]):
        raise ExperimentReportError("records_sha256 must be a lowercase SHA-256 digest")
    if all(raw[name] is None for name in ("factorial", "horizon", "profile")):
        raise ExperimentReportError("report manifest enables no analysis section")
    return raw


def _task_macro(rows: Iterable[ExperimentMetricRecord]) -> tuple[float, tuple[str, ...]]:
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_task[row.task_id].append(row.value)
    if not by_task:
        raise ExperimentReportError("an analysis cell contains no observations")
    tasks = tuple(sorted(by_task))
    return statistics.fmean(statistics.fmean(by_task[task]) for task in tasks), tasks


def _patient_panel(rows: Iterable[ExperimentMetricRecord]) -> frozenset[tuple[str, str]]:
    return frozenset((row.task_id, row.patient_id) for row in rows)


def _test_payload(test: PairedTest) -> dict[str, object]:
    return {
        "estimate": test.estimate,
        "lower": test.lower,
        "upper": test.upper,
        "n": test.n,
        "t_statistic": test.t_statistic if math.isfinite(test.t_statistic) else None,
        "p_value": test.p_value,
    }


def _factorial_report(
    records: Sequence[ExperimentMetricRecord], section: object
) -> tuple[dict[str, object], dict[str, float]]:
    if not isinstance(section, Mapping):
        raise ExperimentReportError("factorial must be an object or null")
    _strict_fields(section, {"arms", "metrics"}, "factorial")
    arms = section["arms"]
    if not isinstance(arms, Mapping) or set(arms) != {arm.value for arm in Arm}:
        raise ExperimentReportError("factorial.arms must map exactly a1, a2, a3, and a4")
    systems = {
        Arm(name): _nonempty(value, f"factorial.arms.{name}") for name, value in arms.items()
    }
    if len(set(systems.values())) != 4:
        raise ExperimentReportError("factorial arms must reference four distinct systems")
    metrics = _string_list(section["metrics"], "factorial.metrics")
    report: dict[str, object] = {}
    p_values: dict[str, float] = {}
    for metric in metrics:
        seed_values: dict[Arm, dict[int, float]] = {}
        task_panels: dict[Arm, dict[int, tuple[str, ...]]] = {}
        patient_panels: dict[int, frozenset[tuple[str, str]]] = {}
        for arm, system in systems.items():
            selected = [
                row
                for row in records
                if row.system_id == system
                and row.metric == metric
                and row.horizon_days is None
                and row.decision_depth is None
                and row.profile is None
            ]
            seeds = sorted({row.seed for row in selected})
            if len(seeds) < 2:
                raise ExperimentReportError(
                    f"factorial metric {metric!r} requires at least two seeds for {arm.value}"
                )
            seed_values[arm] = {}
            task_panels[arm] = {}
            for seed in seeds:
                cell = [row for row in selected if row.seed == seed]
                panel = _patient_panel(cell)
                if patient_panels.setdefault(seed, panel) != panel:
                    raise ExperimentReportError("factorial arms have unmatched patient panels")
                value, tasks = _task_macro(cell)
                seed_values[arm][seed] = value
                task_panels[arm][seed] = tasks
        seed_sets = {tuple(sorted(values)) for values in seed_values.values()}
        if len(seed_sets) != 1:
            raise ExperimentReportError(f"factorial metric {metric!r} has unmatched seeds")
        seeds = next(iter(seed_sets))
        panels = {tasks for arm in Arm for tasks in task_panels[arm].values()}
        if len(panels) != 1:
            raise ExperimentReportError(f"factorial metric {metric!r} has unmatched task panels")
        tasks = next(iter(panels))
        contrasts: dict[str, list[float]] = defaultdict(list)
        arm_rows: dict[str, list[float]] = {arm.value: [] for arm in Arm}
        for seed in seeds:
            values = {arm: seed_values[arm][seed] for arm in Arm}
            calculated = factorial_contrasts(values)
            for name, value in asdict(calculated).items():
                contrasts[name].append(value)
            for arm in Arm:
                arm_rows[arm.value].append(values[arm])
        contrast_report: dict[str, object] = {}
        for name, values in sorted(contrasts.items()):
            test = paired_t_test(values)
            identity = f"factorial:{metric}:{name}"
            p_values[identity] = test.p_value
            contrast_report[name] = {**_test_payload(test), "seed_values": values}
        report[metric] = {
            "seeds": list(seeds),
            "tasks": list(tasks),
            "arm_seed_values": arm_rows,
            "contrasts": contrast_report,
        }
    return report, p_values


def _parse_contrasts(value: object, location: str) -> tuple[tuple[str, str, str], ...]:
    if not isinstance(value, list):
        raise ExperimentReportError(f"{location} must be an array")
    result: list[tuple[str, str, str]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ExperimentReportError(f"{location}[{index}] must be an object")
        _strict_fields(raw, {"name", "left", "right"}, f"{location}[{index}]")
        result.append(
            (
                _nonempty(raw["name"], "contrast name"),
                _nonempty(raw["left"], "contrast left"),
                _nonempty(raw["right"], "contrast right"),
            )
        )
        if result[-1][1] == result[-1][2]:
            raise ExperimentReportError(f"{location} cannot compare a system to itself")
    if len({item[0] for item in result}) != len(result):
        raise ExperimentReportError(f"{location} has duplicate names")
    return tuple(result)


def _horizon_report(
    records: Sequence[ExperimentMetricRecord], section: object
) -> tuple[dict[str, object], dict[str, float]]:
    if not isinstance(section, Mapping):
        raise ExperimentReportError("horizon must be an object or null")
    coordinates_key = "decision_depths" if "decision_depths" in section else "horizons_days"
    axis = "decision_depth" if coordinates_key == "decision_depths" else "horizon_days"
    slope_key = "slope_per_step" if axis == "decision_depth" else "slope_per_day"
    _strict_fields(section, {"systems", "metrics", coordinates_key, "contrasts"}, "horizon")
    systems = _string_list(section["systems"], "horizon.systems")
    metrics = _string_list(section["metrics"], "horizon.metrics")
    horizons_raw = section[coordinates_key]
    if (
        not isinstance(horizons_raw, list)
        or len(horizons_raw) < 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in horizons_raw
        )
    ):
        raise ExperimentReportError(
            f"horizon.{coordinates_key} needs at least two positive integers"
        )
    horizons = tuple(horizons_raw)
    if len(set(horizons)) != len(horizons):
        raise ExperimentReportError(f"horizon.{coordinates_key} contains duplicates")
    contrasts = _parse_contrasts(section["contrasts"], "horizon.contrasts")
    if any(left not in systems or right not in systems for _, left, right in contrasts):
        raise ExperimentReportError("horizon contrast references an undeclared system")
    report: dict[str, object] = {}
    p_values: dict[str, float] = {}
    for metric in metrics:
        slopes: dict[str, dict[int, float]] = {}
        system_report: dict[str, object] = {}
        system_tasks: dict[str, tuple[str, ...]] = {}
        patient_panels: dict[tuple[int, int], frozenset[tuple[str, str]]] = {}
        for system in systems:
            selected = [
                row
                for row in records
                if row.system_id == system and row.metric == metric and row.profile is None
            ]
            seeds = sorted({row.seed for row in selected if getattr(row, axis) in horizons})
            if len(seeds) < 2:
                raise ExperimentReportError(
                    f"horizon metric {metric!r} requires at least two seeds for {system!r}"
                )
            slopes[system] = {}
            curves: dict[str, list[float]] = {}
            for seed in seeds:
                values: list[float] = []
                panels: list[tuple[str, ...]] = []
                for horizon in horizons:
                    cell = [
                        row
                        for row in selected
                        if row.seed == seed and getattr(row, axis) == horizon
                    ]
                    panel = _patient_panel(cell)
                    if patient_panels.setdefault((seed, horizon), panel) != panel:
                        raise ExperimentReportError("horizon curves have unmatched patient panels")
                    value, tasks = _task_macro(cell)
                    values.append(value)
                    panels.append(tasks)
                if len(set(panels)) != 1:
                    raise ExperimentReportError(
                        f"horizon metric {metric!r} has unmatched tasks across horizons"
                    )
                if system not in system_tasks:
                    system_tasks[system] = panels[0]
                elif system_tasks[system] != panels[0]:
                    raise ExperimentReportError(
                        f"horizon metric {metric!r} has unmatched tasks across seeds"
                    )
                slopes[system][seed] = ordinary_least_squares_slope(horizons, values)
                curves[str(seed)] = values
            test = paired_t_test(slopes[system].values())
            identity = f"horizon:{metric}:{system}:slope"
            p_values[identity] = test.p_value
            system_report[system] = {
                coordinates_key: list(horizons),
                "seed_curves": curves,
                slope_key: {
                    **_test_payload(test),
                    "seed_values": [slopes[system][seed] for seed in seeds],
                },
            }
        contrast_report: dict[str, object] = {}
        for name, left, right in contrasts:
            if system_tasks[left] != system_tasks[right]:
                raise ExperimentReportError(f"horizon contrast {name!r} has unmatched tasks")
            if set(slopes[left]) != set(slopes[right]):
                raise ExperimentReportError(f"horizon contrast {name!r} has unmatched seeds")
            values = [slopes[left][seed] - slopes[right][seed] for seed in sorted(slopes[left])]
            test = paired_t_test(values)
            identity = f"horizon:{metric}:contrast:{name}"
            p_values[identity] = test.p_value
            contrast_report[name] = {
                **_test_payload(test),
                "left": left,
                "right": right,
                "seed_values": values,
            }
        report[metric] = {"axis": axis, "systems": system_report, "contrasts": contrast_report}
    return report, p_values


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _resampled_task_macro(
    tasks: Mapping[str, Mapping[str, float]], multiplicities: Mapping[str, int]
) -> float | None:
    means: list[float] = []
    for patients in tasks.values():
        count = sum(multiplicities.get(patient, 0) for patient in patients)
        if count == 0:
            return None  # The frozen equal-task endpoint is undefined for this draw.
        means.append(
            math.fsum(value * multiplicities.get(patient, 0) for patient, value in patients.items())
            / count
        )
    return statistics.fmean(means)


def _profile_report(
    records: Sequence[ExperimentMetricRecord], section: object
) -> dict[str, object]:
    if not isinstance(section, Mapping):
        raise ExperimentReportError("profile must be an object or null")
    _strict_fields(
        section,
        {"systems", "metrics", "profiles", "bootstrap_draws", "bootstrap_seed", "contrasts"},
        "profile",
    )
    systems = _string_list(section["systems"], "profile.systems")
    metrics = _string_list(section["metrics"], "profile.metrics")
    profiles = _string_list(section["profiles"], "profile.profiles")
    contrasts = _parse_contrasts(section["contrasts"], "profile.contrasts")
    if any(left not in systems or right not in systems for _, left, right in contrasts):
        raise ExperimentReportError("profile contrast references an undeclared system")
    draws = section["bootstrap_draws"]
    bootstrap_seed = section["bootstrap_seed"]
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 100:
        raise ExperimentReportError("profile.bootstrap_draws must be at least 100")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ExperimentReportError("profile.bootstrap_seed must be nonnegative")
    report: dict[str, object] = {}
    for profile in profiles:
        selected_profile = [
            row
            for row in records
            if row.profile == profile and row.system_id in systems and row.metric in metrics
        ]
        if any(
            row.horizon_days is not None or row.decision_depth is not None
            for row in selected_profile
        ):
            raise ExperimentReportError("profile rows cannot mix horizon-specific observations")
        rng = random.Random(
            int(artifact_fingerprint({"seed": bootstrap_seed, "profile": profile}), 16)
        )
        population = sorted({row.patient_id for row in selected_profile})
        if len(population) < 2:
            raise ExperimentReportError(f"profile {profile!r} requires at least two patients")
        cells: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
        estimates: dict[tuple[str, str], float] = {}
        reference_panel: frozenset[tuple[int, str, str]] | None = None
        for system in systems:
            for metric in metrics:
                cell = [
                    row
                    for row in selected_profile
                    if row.system_id == system and row.metric == metric
                ]
                if not cell:
                    raise ExperimentReportError(
                        f"profile {profile!r} has no {metric!r} rows for {system!r}"
                    )
                panel = frozenset((row.seed, row.task_id, row.patient_id) for row in cell)
                if reference_panel is None:
                    reference_panel = panel
                if panel != reference_panel:
                    raise ExperimentReportError(
                        f"profile {profile!r} requires matched seed/task/patient panels"
                    )
                seed_panels = {
                    _patient_panel(row for row in cell if row.seed == seed)
                    for seed in {row.seed for row in cell}
                }
                if len(seed_panels) != 1:
                    raise ExperimentReportError("profile seeds have unmatched patient panels")
                # Every seed has the same roster: average seeds before patient resampling.
                values: dict[tuple[str, str], list[float]] = defaultdict(list)
                for row in cell:
                    values[(row.task_id, row.patient_id)].append(row.value)
                tasks: dict[str, dict[str, float]] = defaultdict(dict)
                for (task, patient), observations in sorted(values.items()):
                    tasks[task][patient] = statistics.fmean(observations)
                cells[(system, metric)] = tasks
                estimates[(system, metric)] = _resampled_task_macro(
                    tasks, dict.fromkeys(population, 1)
                )
        replicates: dict[tuple[str, str], list[float]] = {key: [] for key in cells}
        contrast_replicates: dict[tuple[str, str], list[float]] = {
            (name, metric): [] for name, _, _ in contrasts for metric in metrics
        }
        for _ in range(draws):
            sampled = Counter(population[rng.randrange(len(population))] for _ in population)
            current = {key: _resampled_task_macro(cell, sampled) for key, cell in cells.items()}
            if any(value is None for value in current.values()):
                continue
            for key, value in current.items():
                replicates[key].append(value)
            for name, left, right in contrasts:
                for metric in metrics:
                    contrast_replicates[(name, metric)].append(
                        current[(left, metric)] - current[(right, metric)]
                    )
        cells_report: dict[str, object] = {}
        valid_draws = len(next(iter(replicates.values())))
        if valid_draws < 2:
            raise ExperimentReportError("fewer than two profile bootstrap draws retain every task")
        for (system, metric), values in sorted(replicates.items()):
            cells_report[f"{system}:{metric}"] = {
                "estimate": estimates[(system, metric)],
                "lower": _percentile(values, 0.025),
                "upper": _percentile(values, 0.975),
                "patients": len(population),
            }
        contrasts_report: dict[str, object] = {}
        for (name, metric), values in sorted(contrast_replicates.items()):
            left, right = next((left, right) for item, left, right in contrasts if item == name)
            contrasts_report[f"{name}:{metric}"] = {
                "estimate": estimates[(left, metric)] - estimates[(right, metric)],
                "lower": _percentile(values, 0.025),
                "upper": _percentile(values, 0.975),
                "left": left,
                "right": right,
            }
        report[profile] = {
            "bootstrap_draws": draws,
            "valid_bootstrap_draws": valid_draws,
            "undefined_bootstrap_draws": draws - valid_draws,
            "seed_aggregation": "equal_seed_mean_before_patient_resampling",
            "joint_patient_population": len(population),
            "cells": cells_report,
            "contrasts": contrasts_report,
        }
    return report


def analyze_experiments(
    manifest_path: str | Path,
    records_path: str | Path,
    output_path: str | Path,
    *,
    content_kind: ArtifactContentKind | str = ArtifactContentKind.CREDENTIALED,
    repository_root: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    manifest_source = Path(manifest_path).expanduser().resolve()
    records_source = Path(records_path).expanduser().resolve()
    destination = guard_artifact_write_path(
        output_path, content_kind=content_kind, repository_root=repository_root
    )
    validate_output_paths(
        [destination], inputs=[manifest_source, records_source], overwrite=overwrite
    )
    implementation = package_code_fingerprint()
    manifest_before = fingerprint_file(manifest_source)
    records_before = fingerprint_file(records_source)
    manifest = _load_manifest(manifest_source)
    if records_before.sha256 != manifest["records_sha256"]:
        raise ExperimentReportError("experiment records do not match records_sha256")
    records = load_experiment_records(records_source)
    if (
        fingerprint_file(manifest_source) != manifest_before
        or fingerprint_file(records_source) != records_before
    ):
        raise ExperimentReportError("analysis inputs changed while they were being loaded")

    p_values: dict[str, float] = {}
    factorial: dict[str, object] | None = None
    horizon: dict[str, object] | None = None
    profile: dict[str, object] | None = None
    if manifest["factorial"] is not None:
        factorial, values = _factorial_report(records, manifest["factorial"])
        p_values.update(values)
    if manifest["horizon"] is not None:
        horizon, values = _horizon_report(records, manifest["horizon"])
        p_values.update(values)
    if manifest["profile"] is not None:
        profile = _profile_report(records, manifest["profile"])
    selected_tests = _string_list(
        manifest.get("confirmatory_tests", sorted(p_values)), "confirmatory_tests", minimum=0
    )
    if unknown := set(selected_tests) - set(p_values):
        raise ExperimentReportError(
            f"confirmatory_tests references unknown tests: {sorted(unknown)}"
        )
    adjusted = holm_adjust({name: p_values[name] for name in selected_tests})
    for metric_name, metric_payload in (factorial or {}).items():
        if isinstance(metric_payload, dict):
            for name, payload in metric_payload.get("contrasts", {}).items():
                payload["holm_p_value"] = adjusted.get(f"factorial:{metric_name}:{name}")
    if horizon is not None:
        for metric_name, metric_payload in horizon.items():
            for system, payload in metric_payload["systems"].items():
                slope_key = (
                    "slope_per_step"
                    if metric_payload["axis"] == "decision_depth"
                    else "slope_per_day"
                )
                payload[slope_key]["holm_p_value"] = adjusted.get(
                    f"horizon:{metric_name}:{system}:slope"
                )
            for name, payload in metric_payload["contrasts"].items():
                payload["holm_p_value"] = adjusted.get(f"horizon:{metric_name}:contrast:{name}")
    result: dict[str, object] = {
        "schema_version": EXPERIMENT_REPORT_SCHEMA_VERSION,
        "report_id": manifest["report_id"],
        "paper_exact": False,
        "warning": EXPERIMENT_REPORT_WARNING,
        "implementation_sha256": implementation,
        "inputs": {
            "manifest": manifest_before.to_dict(),
            "records": records_before.to_dict(),
            "records_count": len(records),
        },
        "factorial": factorial,
        "horizon": horizon,
        "profile": profile,
        "holm_family": {name: adjusted[name] for name in sorted(adjusted)},
    }
    if (
        fingerprint_file(manifest_source) != manifest_before
        or fingerprint_file(records_source) != records_before
        or package_code_fingerprint() != implementation
    ):
        raise ExperimentReportError("analysis inputs or implementation changed during reporting")
    atomic_write_text(destination, canonical_json_dumps(result) + "\n", overwrite=overwrite)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze factorial, horizon, and named-profile experiment metrics."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("records", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--content-kind",
        choices=[item.value for item in ArtifactContentKind],
        default=ArtifactContentKind.CREDENTIALED.value,
    )
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = analyze_experiments(
        arguments.manifest,
        arguments.records,
        arguments.output,
        content_kind=arguments.content_kind,
        repository_root=arguments.repository_root,
        overwrite=arguments.overwrite,
    )
    print(canonical_json_dumps({"report_id": result["report_id"], "output": str(arguments.output)}))
    return 0


__all__ = [
    "EXPERIMENT_RECORD_SCHEMA_VERSION",
    "EXPERIMENT_REPORT_SCHEMA_VERSION",
    "EXPERIMENT_REPORT_WARNING",
    "ExperimentMetricRecord",
    "ExperimentReportError",
    "analyze_experiments",
    "load_experiment_records",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
