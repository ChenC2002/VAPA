"""Explicit task-level outcome rules; no implicit clinical formulas or normalization.

An author-supplied manifest maps task IDs to exact or numeric comparison rules,
grounding policies, and explicit missing-answer behavior. This is a scoring engine,
not a claim to reconstruct the paper's unavailable task definitions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from vapa.artifacts import artifact_fingerprint, strict_json_loads
from vapa.evaluation.metrics import verified_match
from vapa.rollouts import Turn
from vapa.schemas import ActionKind, Episode, ReturnCode, TimeWindow, normalize_field


@dataclass(frozen=True)
class ScoreRule:
    method: str
    grounding: str
    absolute_tolerance: float
    relative_tolerance: float
    missing_value: str | None
    allow_missing_without_evidence: bool

    def __post_init__(self) -> None:
        if self.method not in {"exact", "numeric"}:
            raise ValueError("method must be exact or numeric")
        if self.grounding not in {"none", "any_reference", "all_references", "exact_references"}:
            raise ValueError("unknown grounding policy")
        for value in (self.absolute_tolerance, self.relative_tolerance):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("tolerances must be finite nonnegative numbers")
        if self.method == "exact" and (self.absolute_tolerance or self.relative_tolerance):
            raise ValueError("exact comparison does not use tolerances")
        if self.missing_value is not None and (
            not isinstance(self.missing_value, str) or not self.missing_value.strip()
        ):
            raise ValueError("missing_value must be an explicit string or null")
        if type(self.allow_missing_without_evidence) is not bool:
            raise ValueError("allow_missing_without_evidence must be boolean")
        if self.allow_missing_without_evidence and self.missing_value is None:
            raise ValueError("missing evidence exemption needs an explicit missing sentinel")


class TaskScorer:
    """Answer-only terminal reward (Sections 2/B.2), separate from grounded evaluation."""

    def __init__(self, manifest: dict) -> None:
        if set(manifest) != {"schema_version", "task_rules"}:
            raise ValueError("scorer manifest requires only schema_version and task_rules")
        if manifest["schema_version"] != "vapa-task-scorer-v1":
            raise ValueError("unsupported task scorer schema")
        raw = manifest["task_rules"]
        if not isinstance(raw, dict) or not raw:
            raise ValueError("task_rules must map explicit task IDs to rules")
        self.rules = {}
        for task_id, rule in raw.items():
            if not isinstance(task_id, str) or not task_id.strip() or not isinstance(rule, dict):
                raise ValueError("invalid task rule")
            try:
                self.rules[task_id] = ScoreRule(**rule)
            except TypeError as error:
                raise ValueError(
                    "each score rule must declare every field, without extras"
                ) from error
        self.manifest_sha256 = artifact_fingerprint(manifest)

    @classmethod
    def from_json(cls, path: str | Path) -> TaskScorer:
        payload = strict_json_loads(Path(path).read_bytes())
        if not isinstance(payload, dict):
            raise ValueError("scorer manifest must be an object")
        return cls(payload)

    def _rule(self, episode: Episode) -> ScoreRule:
        task_id = episode.task.metadata.get("task_type")
        if not isinstance(task_id, str) or task_id not in self.rules:
            raise ValueError("task_type has no declared scoring rule; no fallback is permitted")
        return self.rules[task_id]

    def __call__(self, episode: Episode, prediction: object, evidence: tuple[str, ...]) -> float:
        # Grounding belongs to process rewards and evaluation, not Rout.
        del evidence
        rule = self._rule(episode)
        gold = episode.gold_answer
        if gold is None:
            raise ValueError("a missing gold label is not a scored negative or a missing sentinel")
        missing = rule.missing_value is not None and gold == rule.missing_value
        if missing or rule.method == "exact":
            matched = type(prediction) is type(gold) and prediction == gold
        else:
            matched = verified_match(
                prediction,
                gold,
                abs_tolerance=rule.absolute_tolerance,
                relative_tolerance=rule.relative_tolerance,
            )
        return 1.0 if matched else -1.0

    def evaluation_scorer(self) -> GroundedTaskScorer:
        return GroundedTaskScorer(self)

    def grounded_reward(
        self,
        episode: Episode,
        prediction: object,
        evidence: tuple[str, ...],
        *,
        turns: tuple[Turn, ...] | None = None,
    ) -> float:
        rule = self._rule(episode)
        matched = self(episode, prediction, evidence) > 0
        admissible = {
            event.pointer
            for event in episode.admissible_events
            if episode.task.requested_window.contains(event.timestamp, episode.task.cutoff)
        }
        reference = set(episode.reference_evidence)
        if not reference <= admissible:
            raise ValueError("reference evidence is outside the task's patient/cutoff boundary")
        if any(not isinstance(pointer, str) or not pointer for pointer in evidence):
            return -1.0
        predicted = set(evidence)
        if not predicted <= admissible or len(predicted) != len(evidence):
            return -1.0
        if not matched:
            return -1.0
        missing = rule.missing_value is not None and episode.gold_answer == rule.missing_value
        if missing and rule.allow_missing_without_evidence and not reference and not predicted:
            # Appendix A.1 requires an actual absence-bearing tool return, not just
            # an empty gold pointer list. General multi-field missingness needs its
            # own task adapter; this engine can verify one explicit requested field.
            if turns is None:
                raise ValueError("grounded missingness requires the executed tool trace")
            if len(episode.task.requested_fields) != 1:
                raise ValueError("multi-field missingness requires a task-specific scorer")
            if any(turn.observation.task != episode.task for turn in turns):
                raise ValueError("grounding trace belongs to a different task")
            field = episode.task.requested_fields[0]
            if any(
                event.field == field and event.pointer in admissible
                for event in episode.admissible_events
            ):
                return -1.0
            for turn in turns:
                if (
                    turn.accepted
                    and turn.action is not None
                    and turn.action.kind is ActionKind.QUERY_FIELD
                    and normalize_field(turn.action.arguments["field"]) == field
                    and turn.tool_return.code in {ReturnCode.NOT_RECORDED, ReturnCode.OUT_OF_WINDOW}
                    and not turn.tool_return.events
                    and not turn.tool_return.evidence_pointers
                ):
                    window = TimeWindow.parse(turn.action.arguments["window"])
                    requested = episode.task.requested_window
                    if (window.lookback, window.lag) == (requested.lookback, requested.lag):
                        return 1.0
            return -1.0
        grounded = {
            "none": True,
            "any_reference": bool(reference & predicted),
            "all_references": bool(reference) and reference <= predicted,
            "exact_references": bool(reference) and reference == predicted,
        }[rule.grounding]
        return 1.0 if grounded else -1.0


@dataclass(frozen=True)
class GroundedTaskScorer:
    """Evaluation-only adapter with access to already executed tool returns."""

    scorer: TaskScorer

    def __call__(self, episode: Episode, prediction: object, evidence: tuple[str, ...]) -> float:
        return self.scorer.grounded_reward(episode, prediction, evidence)

    def score_with_trace(
        self,
        episode: Episode,
        prediction: object,
        evidence: tuple[str, ...],
        turns: tuple[Turn, ...],
    ) -> float:
        return self.scorer.grounded_reward(episode, prediction, evidence, turns=turns)
