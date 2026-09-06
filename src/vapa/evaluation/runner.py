"""Resume-safe evaluation runner with per-instance and aggregate artifacts."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vapa.artifacts import (
    ArtifactContentKind,
    atomic_write_text,
    canonical_json_dumps,
    guard_artifact_write_path,
)
from vapa.data.episodes import episode_to_record
from vapa.data.io import DataValidationError, load_json, load_jsonl, sha256_json
from vapa.evaluation.analysis import (
    DEFAULT_TASK_CAP,
    BinaryPrediction,
    CohortInstance,
    frozen_task_cap,
)
from vapa.inference import (
    CheckpointFactory,
    CheckpointManagerFactory,
    InferenceResult,
    PolicyFactory,
    binary_label,
    build_inference_engine,
    checkpoint_identity,
    deterministic_instance_seed,
    episode_fingerprint,
    evaluation_task_id,
    factory_identity,
    heuristic_policy_factory,
    path_checkpoint_factory,
    select_episodes,
)
from vapa.provenance import package_code_fingerprint
from vapa.rollouts import ManagerFactory, OutcomeScorer, exact_outcome_scorer
from vapa.schemas import Episode

EVALUATION_SCHEMA_VERSION = 2
ResultCallback = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    output_directory: Path
    predictions_path: Path
    metrics_path: Path
    run_manifest_path: Path
    binary_predictions_path: Path | None
    metrics: Mapping[str, Any]
    newly_completed: int
    resumed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_directory": str(self.output_directory),
            "predictions_path": str(self.predictions_path),
            "metrics_path": str(self.metrics_path),
            "run_manifest_path": str(self.run_manifest_path),
            "binary_predictions_path": (
                None if self.binary_predictions_path is None else str(self.binary_predictions_path)
            ),
            "newly_completed": self.newly_completed,
            "resumed": self.resumed,
            "metrics": dict(self.metrics),
        }


def _run_manifest(
    episodes: tuple[Episode, ...],
    *,
    selection: Mapping[str, Any],
    seed: int,
    max_instances: int | None,
    greedy: bool,
    checkpoint_path: str | Path | None,
    checkpoint_factory: CheckpointFactory,
    policy_factory: PolicyFactory,
    manager_factory: ManagerFactory | None,
    checkpoint_manager_factory: CheckpointManagerFactory | None,
    outcome_scorer: OutcomeScorer,
    policy_id: str | None,
    content_kind: ArtifactContentKind,
    binary_readout: bool,
    analysis_seed: int | None,
) -> dict[str, Any]:
    episode_records = [episode_to_record(episode) for episode in episodes]
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "implementation_sha256": package_code_fingerprint(),
        "data_sha256": sha256_json(episode_records),
        "instance_count": len(episodes),
        "instance_ids_sha256": sha256_json([episode.task.instance_id for episode in episodes]),
        "seed": seed,
        "max_instances": max_instances,
        "selection": dict(selection),
        "greedy": greedy,
        "content_kind": content_kind.value,
        "checkpoint": checkpoint_identity(checkpoint_path),
        "checkpoint_factory": factory_identity(checkpoint_factory),
        "policy_factory": factory_identity(policy_factory),
        "policy_id": policy_id,
        "binary_readout": binary_readout,
        "analysis_seed": analysis_seed,
        "manager_factory": (
            factory_identity(checkpoint_manager_factory)
            if checkpoint_manager_factory is not None
            else (
                "vapa:default-state-manager"
                if manager_factory is None
                else factory_identity(manager_factory)
            )
        ),
        "outcome_scorer": factory_identity(outcome_scorer),
    }


def _load_existing_manifest(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    raw = load_json(path)
    if not isinstance(raw, Mapping):
        raise DataValidationError("existing evaluation run manifest must be an object")
    return raw


def _load_existing_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.stat().st_size == 0:
        return []
    raw = load_jsonl(path)
    output: list[dict[str, Any]] = []
    for index, record in enumerate(raw):
        if not isinstance(record, Mapping):
            raise DataValidationError(f"existing prediction record {index} must be an object")
        output.append(dict(record))
    return output


_PREDICTION_FIELDS = frozenset(
    {
        "schema_version",
        "instance_id",
        "episode_sha256",
        "patient_id",
        "task_id",
        "family",
        "seed",
        "status",
        "error",
        "prediction",
        "reference",
        "answer_evidence",
        "reference_evidence",
        "reference_evidence_covered",
        "outcome_reward",
        "success",
        "terminated",
        "turn_count",
        "action_cost",
        "max_action_cost",
        "sampled_tokens",
        "binary_score",
    }
)


def _validate_record_payload(raw: Mapping[str, Any], episode: Episode, *, seed: int) -> None:
    missing = _PREDICTION_FIELDS - set(raw)
    unknown = set(raw) - _PREDICTION_FIELDS
    if missing or unknown:
        raise DataValidationError(
            "prediction record fields mismatch: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    instance_id = episode.task.instance_id
    if raw["patient_id"] != episode.task.patient_id:
        raise DataValidationError(
            f"existing prediction for {instance_id!r} has the wrong patient_id"
        )
    if raw["task_id"] != evaluation_task_id(episode):
        raise DataValidationError(
            f"existing prediction for {instance_id!r} has the wrong evaluation task_id"
        )
    if raw["family"] != episode.task.family:
        raise DataValidationError(
            f"existing prediction for {instance_id!r} has the wrong task family"
        )
    if raw["seed"] != deterministic_instance_seed(seed, instance_id):
        raise DataValidationError(
            f"existing prediction for {instance_id!r} has the wrong instance seed"
        )
    if raw["reference"] != episode.gold_answer:
        raise DataValidationError(
            f"existing prediction for {instance_id!r} has the wrong reference answer"
        )
    answer_evidence = raw["answer_evidence"]
    reference_evidence = raw["reference_evidence"]
    if not isinstance(answer_evidence, list) or any(
        not isinstance(pointer, str) or not pointer for pointer in answer_evidence
    ):
        raise DataValidationError("prediction answer_evidence must be an array of strings")
    if reference_evidence != list(episode.reference_evidence):
        raise DataValidationError(
            f"existing prediction for {instance_id!r} has the wrong reference evidence"
        )
    expected_coverage = (
        None if not reference_evidence else set(reference_evidence).issubset(answer_evidence)
    )
    if raw["reference_evidence_covered"] != expected_coverage:
        raise DataValidationError(
            f"existing prediction for {instance_id!r} has inconsistent evidence coverage"
        )
    status = raw["status"]
    error = raw["error"]
    if status == "ok" and error is not None:
        raise DataValidationError("successful prediction records cannot contain an error")
    if status == "error" and (not isinstance(error, str) or not error):
        raise DataValidationError("error prediction records require an error message")
    outcome = raw["outcome_reward"]
    if (
        isinstance(outcome, bool)
        or not isinstance(outcome, int | float)
        or not math.isfinite(outcome)
        or not -1.0 <= outcome <= 1.0
    ):
        raise DataValidationError("prediction outcome_reward must be finite and in [-1, 1]")
    if raw["success"] is not (status == "ok" and outcome > 0.0):
        raise DataValidationError("prediction success is inconsistent with its outcome")
    for field, minimum in (
        ("turn_count", 0),
        ("action_cost", 0),
        ("max_action_cost", 1),
        ("sampled_tokens", 0),
    ):
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise DataValidationError(
                f"prediction field {field!r} must be an integer of at least {minimum}"
            )
    if raw["action_cost"] > raw["max_action_cost"]:
        raise DataValidationError("prediction action_cost exceeds max_action_cost")
    if not isinstance(raw["terminated"], bool):
        raise DataValidationError("prediction terminated must be boolean")
    binary_score = raw["binary_score"]
    if binary_score is not None and (
        isinstance(binary_score, bool)
        or not isinstance(binary_score, int | float)
        or not math.isfinite(binary_score)
        or not 0.0 <= binary_score <= 1.0
    ):
        raise DataValidationError("prediction binary_score must be null or finite in [0, 1]")
    if binary_score is not None and status != "ok":
        raise DataValidationError("error prediction records cannot contain a binary_score")


def _validate_existing_records(
    records: Iterable[Mapping[str, Any]], episodes: tuple[Episode, ...], *, seed: int
) -> dict[str, dict[str, Any]]:
    expected = {episode.task.instance_id: episode for episode in episodes}
    output: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(records):
        instance_id = raw.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise DataValidationError(f"existing prediction record {index} has invalid instance_id")
        if instance_id in output:
            raise DataValidationError(f"duplicate existing prediction for {instance_id!r}")
        if instance_id not in expected:
            raise DataValidationError(
                f"existing prediction has unknown instance_id {instance_id!r}"
            )
        if raw.get("schema_version") != EVALUATION_SCHEMA_VERSION:
            raise DataValidationError(
                f"existing prediction for {instance_id!r} has unsupported schema_version"
            )
        if raw.get("episode_sha256") != episode_fingerprint(expected[instance_id]):
            raise DataValidationError(
                f"existing prediction for {instance_id!r} does not match the episode"
            )
        if raw.get("status") not in {"ok", "error"}:
            raise DataValidationError(f"existing prediction for {instance_id!r} has invalid status")
        _validate_record_payload(raw, expected[instance_id], seed=seed)
        output[instance_id] = dict(raw)
    return output


def _write_predictions(
    path: Path,
    records: Mapping[str, Mapping[str, Any]],
    episodes: tuple[Episode, ...],
) -> None:
    ordered = [
        records[episode.task.instance_id]
        for episode in episodes
        if episode.task.instance_id in records
    ]
    payload = "".join(canonical_json_dumps(record) + "\n" for record in ordered)
    atomic_write_text(path, payload)


def _select_for_evaluation(
    episodes: Iterable[Episode],
    *,
    max_instances: int | None,
    task_cap: int | None,
    selection_seed: int,
) -> tuple[tuple[Episode, ...], dict[str, Any]]:
    source = select_episodes(episodes)
    source_ids = [episode.task.instance_id for episode in source]
    source_roster = [
        {
            "task_id": evaluation_task_id(episode),
            "patient_id": episode.task.patient_id,
            "instance_id": episode.task.instance_id,
        }
        for episode in source
    ]
    if task_cap is not None:
        if max_instances is not None:
            raise ValueError("max_instances and task_cap are mutually exclusive")
        roster = tuple(
            CohortInstance(
                task_id=evaluation_task_id(episode),
                patient_id=episode.task.patient_id,
                instance_id=episode.task.instance_id,
            )
            for episode in source
        )
        selected_roster = frozen_task_cap(roster, cap=task_cap, seed=selection_seed)
        selected_ids = {item.instance_id for item in selected_roster}
        selected = tuple(episode for episode in source if episode.task.instance_id in selected_ids)
        method = "frozen-label-blind-patient-round-robin-v1"
    else:
        selected = select_episodes(source, max_instances=max_instances)
        method = "lexicographic-prefix-v1" if max_instances is not None else "all-v1"
    selected_ids_ordered = [episode.task.instance_id for episode in selected]
    return selected, {
        "method": method,
        "task_cap": task_cap,
        "selection_seed": selection_seed if task_cap is not None else None,
        "source_instance_count": len(source),
        "source_instance_ids_sha256": sha256_json(source_ids),
        "source_roster_sha256": sha256_json(source_roster),
        "selected_instance_count": len(selected),
        "selected_instance_ids_sha256": sha256_json(selected_ids_ordered),
    }


def _write_binary_predictions(
    path: Path,
    records: Mapping[str, Mapping[str, Any]],
    episodes: tuple[Episode, ...],
    *,
    system_id: str,
    analysis_seed: int | None,
) -> None:
    output: list[dict[str, object]] = []
    for episode in episodes:
        record = records[episode.task.instance_id]
        score = record["binary_score"]
        output.append(
            BinaryPrediction(
                system_id=system_id,
                seed=analysis_seed,
                task_id=evaluation_task_id(episode),
                patient_id=episode.task.patient_id,
                instance_id=episode.task.instance_id,
                label=binary_label(episode),
                score=(
                    score
                    if isinstance(score, int | float) and not isinstance(score, bool)
                    else None
                ),
                format_valid=score is not None,
            ).to_dict()
        )
    atomic_write_text(
        path,
        "".join(canonical_json_dumps(record) + "\n" for record in output),
    )


def _error_record(episode: Episode, *, seed: int, error: Exception) -> dict[str, Any]:
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "instance_id": episode.task.instance_id,
        "episode_sha256": episode_fingerprint(episode),
        "patient_id": episode.task.patient_id,
        "task_id": evaluation_task_id(episode),
        "family": episode.task.family,
        "seed": deterministic_instance_seed(seed, episode.task.instance_id),
        "status": "error",
        "error": f"{type(error).__name__}: {error}",
        "prediction": None,
        "reference": episode.gold_answer,
        "answer_evidence": [],
        "reference_evidence": list(episode.reference_evidence),
        "reference_evidence_covered": False if episode.reference_evidence else None,
        "outcome_reward": -1.0,
        "success": False,
        "terminated": False,
        "turn_count": 0,
        "action_cost": 0,
        "max_action_cost": 1,
        "sampled_tokens": 0,
        "binary_score": None,
    }


def _numeric(record: Mapping[str, Any], field: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise DataValidationError(f"prediction field {field!r} must be a finite number")
    return float(value)


def _record_metrics(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "instances": 0,
            "successful": 0,
            "errors": 0,
            "task_success": None,
            "macro_task_success": None,
            "cost_normalized_success": None,
            "mean_action_cost": None,
            "mean_sampled_tokens": None,
            "termination_rate": None,
            "reference_evidence_coverage": None,
            "by_family": {},
            "by_task": {},
        }

    successes: list[float] = []
    costs: list[float] = []
    maxima: list[float] = []
    tokens: list[float] = []
    terminated: list[float] = []
    grounded: list[float] = []
    families: list[str] = []
    task_ids: list[str] = []
    errors = 0
    for record in records:
        success = record.get("success")
        if not isinstance(success, bool):
            raise DataValidationError("prediction field 'success' must be boolean")
        family = record.get("family")
        if not isinstance(family, str) or not family:
            raise DataValidationError("prediction field 'family' must be a non-empty string")
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise DataValidationError("prediction field 'task_id' must be a non-empty string")
        status = record.get("status")
        if status not in {"ok", "error"}:
            raise DataValidationError("prediction field 'status' is invalid")
        errors += int(status == "error")
        cost = _numeric(record, "action_cost")
        maximum = _numeric(record, "max_action_cost")
        if cost < 0 or maximum <= 0 or cost > maximum:
            raise DataValidationError("prediction action costs violate their bounds")
        token_count = _numeric(record, "sampled_tokens")
        if token_count < 0:
            raise DataValidationError("prediction sampled_tokens cannot be negative")
        is_terminated = record.get("terminated")
        if not isinstance(is_terminated, bool):
            raise DataValidationError("prediction field 'terminated' must be boolean")
        coverage = record.get("reference_evidence_covered")
        if coverage is not None and not isinstance(coverage, bool):
            raise DataValidationError("reference_evidence_covered must be boolean or null")
        successes.append(float(success))
        costs.append(cost)
        maxima.append(maximum)
        tokens.append(token_count)
        terminated.append(float(is_terminated))
        families.append(family)
        task_ids.append(task_id)
        if coverage is not None:
            grounded.append(float(coverage))

    count = len(records)
    family_values: dict[str, list[int]] = defaultdict(list)
    for index, family in enumerate(families):
        family_values[family].append(index)
    by_family = {
        family: {
            "instances": len(indices),
            "task_success": 100.0 * math.fsum(successes[index] for index in indices) / len(indices),
            "mean_action_cost": math.fsum(costs[index] for index in indices) / len(indices),
        }
        for family, indices in sorted(family_values.items())
    }
    task_values: dict[str, list[int]] = defaultdict(list)
    for index, task_id in enumerate(task_ids):
        task_values[task_id].append(index)
    cost_contributions = [
        success * (1.0 - cost / maximum)
        for success, cost, maximum in zip(successes, costs, maxima, strict=True)
    ]
    by_task = {
        task_id: {
            "instances": len(indices),
            "task_success": 100.0 * math.fsum(successes[index] for index in indices) / len(indices),
            "cost_normalized_success": 100.0
            * math.fsum(cost_contributions[index] for index in indices)
            / len(indices),
        }
        for task_id, indices in sorted(task_values.items())
    }
    task_success_values = [
        math.fsum(successes[index] for index in indices) / len(indices)
        for indices in task_values.values()
    ]
    task_cost_values = [
        math.fsum(cost_contributions[index] for index in indices) / len(indices)
        for indices in task_values.values()
    ]
    return {
        "instances": count,
        "successful": int(math.fsum(successes)),
        "errors": errors,
        "task_success": 100.0 * math.fsum(successes) / count,
        "macro_task_success": 100.0 * math.fsum(task_success_values) / len(task_success_values),
        "cost_normalized_success": 100.0 * math.fsum(task_cost_values) / len(task_cost_values),
        "mean_action_cost": math.fsum(costs) / count,
        "mean_sampled_tokens": math.fsum(tokens) / count,
        "termination_rate": 100.0 * math.fsum(terminated) / count,
        "reference_evidence_coverage": (
            None if not grounded else 100.0 * math.fsum(grounded) / len(grounded)
        ),
        "by_family": by_family,
        "by_task": by_task,
    }


def evaluate_episodes(
    episodes: Iterable[Episode],
    output_directory: str | Path,
    *,
    seed: int = 0,
    max_instances: int | None = None,
    task_cap: int | None = None,
    selection_seed: int | None = None,
    greedy: bool = True,
    checkpoint_path: str | Path | None = None,
    checkpoint_factory: CheckpointFactory = path_checkpoint_factory,
    policy_factory: PolicyFactory = heuristic_policy_factory,
    manager_factory: ManagerFactory | None = None,
    checkpoint_manager_factory: CheckpointManagerFactory | None = None,
    outcome_scorer: OutcomeScorer = exact_outcome_scorer,
    policy_id: str | None = None,
    binary_readout: bool = False,
    analysis_seed: int | None = None,
    continue_on_error: bool = False,
    retry_errors: bool = False,
    on_result: ResultCallback | None = None,
    content_kind: ArtifactContentKind | str = ArtifactContentKind.CREDENTIALED,
    repository_root: str | Path | None = None,
    allow_synthetic_outcome_scorer: bool = False,
) -> EvaluationResult:
    """Evaluate selected episodes and atomically checkpoint per-instance JSONL.

    Existing results are resumed only when the complete immutable run manifest
    agrees.  ``on_result`` runs after the result is durable and is primarily
    useful for orchestration and interruption tests.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if selection_seed is None:
        selection_seed = seed
    if (
        isinstance(selection_seed, bool)
        or not isinstance(selection_seed, int)
        or selection_seed < 0
    ):
        raise ValueError("selection_seed must be a non-negative integer or None")
    if analysis_seed is not None and (
        isinstance(analysis_seed, bool) or not isinstance(analysis_seed, int) or analysis_seed < 0
    ):
        raise ValueError("analysis_seed must be a non-negative integer or None")
    if not isinstance(binary_readout, bool):
        raise TypeError("binary_readout must be boolean")
    if not binary_readout and analysis_seed is not None:
        raise ValueError("analysis_seed is valid only with binary_readout")
    if policy_id is not None and (not isinstance(policy_id, str) or not policy_id.strip()):
        raise ValueError("policy_id must be a non-empty string or None")
    if binary_readout and policy_id is None:
        raise ValueError("binary_readout requires policy_id for analysis provenance")
    if binary_readout and task_cap is None:
        task_cap = DEFAULT_TASK_CAP
    if not isinstance(allow_synthetic_outcome_scorer, bool):
        raise TypeError("allow_synthetic_outcome_scorer must be boolean")
    if (
        checkpoint_path is not None
        and outcome_scorer is exact_outcome_scorer
        and not allow_synthetic_outcome_scorer
    ):
        raise ValueError(
            "checkpoint-backed evaluation requires an explicit task-specific outcome_scorer; "
            "the exact-string scorer is synthetic-only"
        )
    try:
        disclosure = ArtifactContentKind(content_kind)
    except (TypeError, ValueError) as error:
        choices = ", ".join(item.value for item in ArtifactContentKind)
        raise ValueError(f"content_kind must be one of: {choices}") from error
    if manager_factory is not None and checkpoint_manager_factory is not None:
        raise ValueError("manager_factory and checkpoint_manager_factory are mutually exclusive")
    if (
        checkpoint_path is not None
        and manager_factory is None
        and checkpoint_manager_factory is None
    ):
        raise ValueError(
            "checkpoint-backed evaluation requires an explicit manager_factory or "
            "checkpoint_manager_factory"
        )
    selected, selection = _select_for_evaluation(
        episodes,
        max_instances=max_instances,
        task_cap=task_cap,
        selection_seed=selection_seed,
    )
    if binary_readout:
        for episode in selected:
            binary_label(episode)
    unresolved_destination = Path(output_directory).expanduser().resolve(strict=False)
    destination = guard_artifact_write_path(
        unresolved_destination,
        content_kind=disclosure,
        repository_root=repository_root,
    )
    predictions_path = destination / "predictions.jsonl"
    metrics_path = destination / "metrics.json"
    manifest_path = destination / "run_manifest.json"
    binary_predictions_path = destination / "binary_predictions.jsonl" if binary_readout else None
    destination.mkdir(parents=True, exist_ok=True)

    expected_manifest = _run_manifest(
        selected,
        selection=selection,
        seed=seed,
        max_instances=max_instances,
        greedy=greedy,
        checkpoint_path=checkpoint_path,
        checkpoint_factory=checkpoint_factory,
        policy_factory=policy_factory,
        manager_factory=manager_factory,
        checkpoint_manager_factory=checkpoint_manager_factory,
        outcome_scorer=outcome_scorer,
        policy_id=policy_id,
        content_kind=disclosure,
        binary_readout=binary_readout,
        analysis_seed=analysis_seed,
    )
    existing_manifest = _load_existing_manifest(manifest_path)
    if existing_manifest is None:
        if predictions_path.exists() or metrics_path.exists():
            raise DataValidationError(
                "evaluation outputs exist without a run manifest; choose a new output directory"
            )
        atomic_write_text(manifest_path, canonical_json_dumps(expected_manifest) + "\n")
    elif dict(existing_manifest) != expected_manifest:
        raise DataValidationError(
            "evaluation run manifest mismatch; data, seed, checkpoint, or factories changed"
        )

    records = _validate_existing_records(
        _load_existing_records(predictions_path), selected, seed=seed
    )
    if retry_errors:
        records = {
            instance_id: record
            for instance_id, record in records.items()
            if record.get("status") != "error"
        }
        _write_predictions(predictions_path, records, selected)
    resumed = len(records)
    pending = [episode for episode in selected if episode.task.instance_id not in records]
    newly_completed = 0
    if pending:
        engine = build_inference_engine(
            checkpoint_path=checkpoint_path,
            checkpoint_factory=checkpoint_factory,
            policy_factory=policy_factory,
            manager_factory=manager_factory,
            checkpoint_manager_factory=checkpoint_manager_factory,
            outcome_scorer=outcome_scorer,
            binary_readout=binary_readout,
        )
        if checkpoint_identity(checkpoint_path) != expected_manifest["checkpoint"]:
            raise DataValidationError(
                "checkpoint changed while the evaluation policy was being constructed"
            )
        for episode in pending:
            try:
                result: InferenceResult = engine.run(episode, seed=seed, greedy=greedy)
                record = result.to_record()
            except Exception as error:
                if not continue_on_error:
                    raise
                record = _error_record(episode, seed=seed, error=error)
            _validate_record_payload(record, episode, seed=seed)
            records[episode.task.instance_id] = record
            newly_completed += 1
            _write_predictions(predictions_path, records, selected)
            if on_result is not None:
                on_result(record)
        if checkpoint_identity(checkpoint_path) != expected_manifest["checkpoint"]:
            raise DataValidationError("checkpoint changed while evaluation was running")
    elif not predictions_path.exists():
        _write_predictions(predictions_path, records, selected)

    ordered = [records[episode.task.instance_id] for episode in selected]
    metrics = _record_metrics(ordered)
    aggregate = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "data_sha256": expected_manifest["data_sha256"],
        "metrics": metrics,
    }
    atomic_write_text(metrics_path, canonical_json_dumps(aggregate) + "\n")
    if binary_predictions_path is not None:
        assert policy_id is not None
        _write_binary_predictions(
            binary_predictions_path,
            records,
            selected,
            system_id=policy_id,
            analysis_seed=analysis_seed,
        )
    return EvaluationResult(
        output_directory=destination,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        run_manifest_path=manifest_path,
        binary_predictions_path=binary_predictions_path,
        metrics=metrics,
        newly_completed=newly_completed,
        resumed=resumed,
    )
