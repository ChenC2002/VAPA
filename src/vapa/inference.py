"""Deterministic, backend-neutral inference over prepared episodes."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from vapa.data.episodes import episode_to_record
from vapa.data.io import sha256_file, sha256_json
from vapa.environment.state_manager import StateManager
from vapa.policies.base import Policy
from vapa.policies.heuristic import HeuristicPolicy
from vapa.rollouts import (
    ManagerFactory,
    OutcomeScorer,
    Rollout,
    RolloutRunner,
    exact_outcome_scorer,
)
from vapa.schemas import ActionKind, Episode


@runtime_checkable
class CheckpointFactory(Protocol):
    """Load a backend-specific checkpoint object once per evaluation run."""

    def __call__(self, checkpoint_path: Path | None) -> object | None: ...


@runtime_checkable
class PolicyFactory(Protocol):
    """Construct a policy from the object returned by a checkpoint factory."""

    def __call__(self, checkpoint: object | None) -> Policy: ...


@runtime_checkable
class CheckpointManagerFactory(Protocol):
    """Construct the episode manager factory bound into a loaded checkpoint."""

    def __call__(self, checkpoint: object | None) -> ManagerFactory: ...


def path_checkpoint_factory(checkpoint_path: Path | None) -> object | None:
    """Pass an existing checkpoint path to a policy factory without interpreting it."""

    return checkpoint_path


def heuristic_policy_factory(checkpoint: object | None) -> Policy:
    """Create the dependency-free baseline; it intentionally accepts no checkpoint."""

    if checkpoint is not None:
        raise ValueError("the heuristic policy does not consume a checkpoint")
    return HeuristicPolicy()


def import_factory(specification: str) -> Any:
    """Import a ``module:attribute`` callable for a backend integration."""

    if not isinstance(specification, str) or specification.count(":") != 1:
        raise ValueError("factory specification must use module:attribute syntax")
    module_name, attribute_path = specification.split(":", 1)
    if not module_name or not attribute_path or any(not part for part in attribute_path.split(".")):
        raise ValueError("factory specification must use module:attribute syntax")
    value: Any = importlib.import_module(module_name)
    for component in attribute_path.split("."):
        value = getattr(value, component)
    if not callable(value):
        raise TypeError(f"imported factory {specification!r} is not callable")
    return value


def _identity_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("callable identity cannot contain non-finite values")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _identity_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return _identity_value(asdict(value))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("callable identity mappings require string keys")
        return {key: _identity_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_identity_value(item) for item in value]
    if callable(value):
        return {"callable": factory_identity(value)}
    raise TypeError(f"cannot content-address bound callable value of type {type(value).__name__}")


def factory_identity(factory: Any) -> str:
    """Return a source- and binding-addressed callable identity."""

    if isinstance(factory, partial):
        payload = {
            "function": factory_identity(factory.func),
            "args": _identity_value(factory.args),
            "keywords": _identity_value(factory.keywords or {}),
        }
        return f"functools:partial@sha256:{sha256_json(payload)}"
    module = getattr(factory, "__module__", type(factory).__module__)
    qualified = getattr(factory, "__qualname__", type(factory).__qualname__)
    source_target = factory if inspect.isroutine(factory) else type(factory)
    try:
        source_file = inspect.getsourcefile(source_target)
    except TypeError:
        source_file = None
    file_digest = None
    if source_file is not None and Path(source_file).is_file():
        file_digest = sha256_file(source_file)
    try:
        source = inspect.getsource(source_target)
    except (OSError, TypeError):
        source = None
    if source is None and file_digest is None:
        raise ValueError(
            f"cannot content-address callable {module}:{qualified}; supply a source-backed factory"
        )
    payload: dict[str, Any] = {
        "module": module,
        "qualified_name": qualified,
        "source_file_sha256": file_digest,
        "callable_source_sha256": (
            None if source is None else hashlib.sha256(source.encode("utf-8")).hexdigest()
        ),
    }
    explicit_identity = getattr(factory, "__vapa_content_id__", None)
    if explicit_identity is not None:
        if (
            not isinstance(explicit_identity, str)
            or len(explicit_identity) != 64
            or any(character not in "0123456789abcdef" for character in explicit_identity)
        ):
            raise ValueError("__vapa_content_id__ must be a lowercase SHA-256 digest")
        payload["explicit_content_id"] = explicit_identity
    else:
        routine = factory.__func__ if inspect.ismethod(factory) else factory
        if inspect.isfunction(routine):
            payload["defaults"] = _identity_value(routine.__defaults__ or ())
            payload["keyword_defaults"] = _identity_value(routine.__kwdefaults__ or {})
            closure: dict[str, Any] = {}
            if routine.__closure__ is not None:
                for name, cell in zip(
                    routine.__code__.co_freevars,
                    routine.__closure__,
                    strict=True,
                ):
                    try:
                        cell_value = cell.cell_contents
                    except ValueError as error:
                        raise ValueError(
                            f"cannot content-address empty closure cell {name!r}"
                        ) from error
                    closure[name] = _identity_value(cell_value)
            payload["closure"] = closure
            if inspect.ismethod(factory) and factory.__self__ is not None:
                receiver = factory.__self__
                receiver_type = receiver if isinstance(receiver, type) else type(receiver)
                receiver_identity: dict[str, Any] = {
                    "type": (f"{receiver_type.__module__}:{receiver_type.__qualname__}")
                }
                if not isinstance(receiver, type):
                    if is_dataclass(receiver):
                        receiver_identity["state"] = _identity_value(asdict(receiver))
                    elif hasattr(receiver, "__dict__"):
                        receiver_identity["state"] = _identity_value(vars(receiver))
                    else:
                        raise ValueError(
                            "cannot content-address bound-method receiver state; "
                            "supply __vapa_content_id__"
                        )
                payload["bound_receiver"] = receiver_identity
        elif hasattr(factory, "__dict__"):
            payload["instance_state"] = _identity_value(vars(factory))
    return f"{module}:{qualified}@sha256:{sha256_json(payload)}"


def checkpoint_identity(checkpoint_path: str | Path | None) -> dict[str, Any]:
    """Content-address a file or directory checkpoint without loading model code."""

    if checkpoint_path is None:
        return {"kind": "none"}
    unresolved = Path(checkpoint_path).expanduser()
    if unresolved.is_symlink():
        raise ValueError("checkpoint path cannot be a symbolic link")
    path = unresolved.resolve()
    if path.is_file():
        return {
            "kind": "file",
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    entries: list[dict[str, Any]] = []
    for candidate in path.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"checkpoint cannot contain symbolic links: {candidate}")
    for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        entries.append(
            {
                "path": child.relative_to(path).as_posix(),
                "sha256": sha256_file(child),
                "size_bytes": child.stat().st_size,
            }
        )
    return {"kind": "directory", "sha256": sha256_json(entries), "files": len(entries)}


def deterministic_instance_seed(seed: int, instance_id: str) -> int:
    """Derive a seed that is invariant to input order, sharding, and resume state."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError("instance_id must be a non-empty string")
    encoded = json.dumps(
        {"namespace": "vapa-evaluation-seed-v1", "seed": seed, "instance_id": instance_id},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def episode_fingerprint(episode: Episode) -> str:
    return sha256_json(cast(Any, episode_to_record(episode)))


def select_episodes(
    episodes: Iterable[Episode], *, max_instances: int | None = None
) -> tuple[Episode, ...]:
    """Validate, sort, and optionally truncate the deterministic evaluation set."""

    if max_instances is not None and (
        isinstance(max_instances, bool) or not isinstance(max_instances, int) or max_instances < 0
    ):
        raise ValueError("max_instances must be a non-negative integer or None")
    selected = list(episodes)
    seen: set[str] = set()
    for index, episode in enumerate(selected):
        if not isinstance(episode, Episode):
            raise TypeError(f"episodes[{index}] is not an Episode")
        identifier = episode.task.instance_id
        if identifier in seen:
            raise ValueError(f"duplicate inference instance_id: {identifier!r}")
        seen.add(identifier)
    selected.sort(key=lambda episode: episode.task.instance_id)
    if max_instances is not None:
        selected = selected[:max_instances]
    return tuple(selected)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_value(item) for key, item in value.items()}
    return None


@dataclass(frozen=True, slots=True)
class InferenceResult:
    schema_version: int
    instance_id: str
    episode_sha256: str
    patient_id: str
    task_id: str
    family: str
    seed: int
    prediction: Any
    reference: Any
    answer_evidence: tuple[str, ...]
    reference_evidence: tuple[str, ...]
    outcome_reward: float
    terminated: bool
    turn_count: int
    action_cost: int
    max_action_cost: int
    sampled_tokens: int
    binary_score: float | None = None
    status: str = "ok"
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.status == "ok" and self.outcome_reward > 0.0

    @property
    def reference_evidence_covered(self) -> bool | None:
        if not self.reference_evidence:
            return None
        return set(self.reference_evidence).issubset(self.answer_evidence)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "instance_id": self.instance_id,
            "episode_sha256": self.episode_sha256,
            "patient_id": self.patient_id,
            "task_id": self.task_id,
            "family": self.family,
            "seed": self.seed,
            "status": self.status,
            "error": self.error,
            "prediction": _json_value(self.prediction),
            "reference": _json_value(self.reference),
            "answer_evidence": list(self.answer_evidence),
            "reference_evidence": list(self.reference_evidence),
            "reference_evidence_covered": self.reference_evidence_covered,
            "outcome_reward": self.outcome_reward,
            "success": self.success,
            "terminated": self.terminated,
            "turn_count": self.turn_count,
            "action_cost": self.action_cost,
            "max_action_cost": self.max_action_cost,
            "sampled_tokens": self.sampled_tokens,
            "binary_score": self.binary_score,
        }


def evaluation_task_id(episode: Episode) -> str:
    """Return the immutable task stratum used for binary evaluation selection."""

    candidate = episode.task.metadata.get("evaluation_task_id", episode.task.family)
    if not isinstance(candidate, str) or not candidate.strip() or candidate != candidate.strip():
        raise ValueError(
            "episode metadata.evaluation_task_id must be a non-empty trimmed string when set"
        )
    return candidate


def binary_label(episode: Episode) -> int:
    """Normalize a binary endpoint label without accepting truthy values."""

    value = episode.gold_answer
    if isinstance(value, bool):
        raise ValueError("binary evaluation gold_answer must be 0 or 1, not boolean")
    if isinstance(value, int) and value in {0, 1}:
        return value
    if isinstance(value, str) and value in {"0", "1"}:
        return int(value)
    raise ValueError("binary evaluation gold_answer must be integer/string 0 or 1")


def _binary_score(policy: Policy, rollout: Rollout) -> float | None:
    """Read p(class=1) at the terminal Answer position, or mark format failure."""

    if not rollout.turns:
        return None
    final_turn = rollout.turns[-1]
    if (
        not rollout.terminated
        or not final_turn.accepted
        or final_turn.action is None
        or final_turn.action.kind is not ActionKind.ANSWER
    ):
        return None
    prediction = final_turn.action.arguments.get("prediction")
    if isinstance(prediction, bool) or not isinstance(prediction, int) or prediction not in {0, 1}:
        return None
    scorer = getattr(policy, "binary_answer_probability", None)
    if not callable(scorer):
        raise TypeError("binary readout requires policy.binary_answer_probability(observation)")
    score = scorer(final_turn.observation, final_turn.decision)
    if isinstance(score, bool) or not isinstance(score, int | float):
        raise TypeError("binary answer probability must be a finite number")
    converted = float(score)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError("binary answer probability must be finite and in [0, 1]")
    return converted


class InferenceEngine:
    """One loaded policy plus deterministic environment and scorer factories."""

    def __init__(
        self,
        policy: Policy,
        *,
        manager_factory: ManagerFactory | None = None,
        outcome_scorer: OutcomeScorer = exact_outcome_scorer,
        binary_readout: bool = False,
    ) -> None:
        self.policy = policy
        self.manager_factory = manager_factory or (lambda episode: StateManager(episode))
        self.runner = RolloutRunner(self.manager_factory, outcome_scorer=outcome_scorer)
        self.binary_readout = binary_readout
        if binary_readout and not callable(getattr(self.policy, "binary_answer_probability", None)):
            raise TypeError("binary readout requires policy.binary_answer_probability(observation)")

    def run(self, episode: Episode, *, seed: int, greedy: bool = True) -> InferenceResult:
        instance_seed = deterministic_instance_seed(seed, episode.task.instance_id)
        rollout: Rollout = self.runner.run(
            episode,
            self.policy,
            rollout_id=f"evaluation:{episode.task.instance_id}",
            seed=instance_seed,
            greedy=greedy,
        )
        max_action_cost = (
            rollout.turns[0].observation.budget_cap if rollout.turns else rollout.action_cost
        )
        score = None
        if self.binary_readout:
            binary_label(episode)
            score = _binary_score(self.policy, rollout)
        return InferenceResult(
            schema_version=2,
            instance_id=episode.task.instance_id,
            episode_sha256=episode_fingerprint(episode),
            patient_id=episode.task.patient_id,
            task_id=evaluation_task_id(episode),
            family=episode.task.family,
            seed=instance_seed,
            prediction=rollout.prediction,
            reference=episode.gold_answer,
            answer_evidence=rollout.answer_evidence,
            reference_evidence=episode.reference_evidence,
            outcome_reward=rollout.outcome_reward,
            terminated=rollout.terminated,
            turn_count=len(rollout.turns),
            action_cost=rollout.action_cost,
            max_action_cost=max_action_cost,
            sampled_tokens=rollout.sampled_tokens,
            binary_score=score,
        )


def build_inference_engine(
    *,
    checkpoint_path: str | Path | None = None,
    checkpoint_factory: CheckpointFactory = path_checkpoint_factory,
    policy_factory: PolicyFactory = heuristic_policy_factory,
    manager_factory: ManagerFactory | None = None,
    checkpoint_manager_factory: CheckpointManagerFactory | None = None,
    outcome_scorer: OutcomeScorer = exact_outcome_scorer,
    binary_readout: bool = False,
) -> InferenceEngine:
    """Compose explicit checkpoint and policy factories into an inference engine."""

    if manager_factory is not None and checkpoint_manager_factory is not None:
        raise ValueError("manager_factory and checkpoint_manager_factory are mutually exclusive")
    resolved = None if checkpoint_path is None else Path(checkpoint_path).resolve()
    if resolved is not None and not resolved.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {resolved}")
    if resolved is not None and manager_factory is None and checkpoint_manager_factory is None:
        raise ValueError(
            "checkpoint-backed inference requires an explicit manager_factory or "
            "checkpoint_manager_factory"
        )
    checkpoint = checkpoint_factory(resolved)
    policy = policy_factory(checkpoint)
    if not callable(getattr(policy, "sample", None)):
        raise TypeError("policy_factory did not return a Policy-like object")
    selected_manager_factory = (
        checkpoint_manager_factory(checkpoint)
        if checkpoint_manager_factory is not None
        else manager_factory
    )
    if selected_manager_factory is not None and not callable(selected_manager_factory):
        raise TypeError("checkpoint_manager_factory did not return a callable")
    return InferenceEngine(
        policy,
        manager_factory=selected_manager_factory,
        outcome_scorer=outcome_scorer,
        binary_readout=binary_readout,
    )


def infer_episodes(
    episodes: Iterable[Episode],
    *,
    seed: int = 0,
    max_instances: int | None = None,
    completed_instance_ids: Iterable[str] = (),
    greedy: bool = True,
    checkpoint_path: str | Path | None = None,
    checkpoint_factory: CheckpointFactory = path_checkpoint_factory,
    policy_factory: PolicyFactory = heuristic_policy_factory,
    manager_factory: ManagerFactory | None = None,
    checkpoint_manager_factory: CheckpointManagerFactory | None = None,
    outcome_scorer: OutcomeScorer = exact_outcome_scorer,
    binary_readout: bool = False,
) -> Iterator[InferenceResult]:
    """Yield pending results; no model is loaded when every item is complete."""

    selected = select_episodes(episodes, max_instances=max_instances)
    completed = frozenset(completed_instance_ids)
    unknown = completed - {episode.task.instance_id for episode in selected}
    if unknown:
        raise ValueError(f"completed_instance_ids contains unknown IDs: {sorted(unknown)}")
    pending = [episode for episode in selected if episode.task.instance_id not in completed]
    if not pending:
        return
    engine = build_inference_engine(
        checkpoint_path=checkpoint_path,
        checkpoint_factory=checkpoint_factory,
        policy_factory=policy_factory,
        manager_factory=manager_factory,
        checkpoint_manager_factory=checkpoint_manager_factory,
        outcome_scorer=outcome_scorer,
        binary_readout=binary_readout,
    )
    for episode in pending:
        yield engine.run(episode, seed=seed, greedy=greedy)
