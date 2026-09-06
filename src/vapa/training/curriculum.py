"""Actor-token-fraction curriculum from Appendix A.7."""

from __future__ import annotations

from dataclasses import dataclass

from vapa.schemas import ActionKind


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    start_fraction: float
    end_fraction: float
    max_history_quartile: int
    actions: frozenset[ActionKind]


STAGES = (
    CurriculumStage(
        "foundation",
        0.0,
        0.2,
        2,
        frozenset(
            {
                ActionKind.RETRIEVE,
                ActionKind.CALCULATE,
                ActionKind.UPDATE_MEMORY,
                ActionKind.ANSWER,
            }
        ),
    ),
    CurriculumStage(
        "status",
        0.2,
        0.5,
        3,
        frozenset(
            {
                ActionKind.RETRIEVE,
                ActionKind.QUERY_FIELD,
                ActionKind.CALCULATE,
                ActionKind.UPDATE_MEMORY,
                ActionKind.MARK_STATUS,
                ActionKind.ANSWER,
            }
        ),
    ),
    CurriculumStage(
        "full",
        0.5,
        1.0,
        4,
        frozenset(set(ActionKind) - {ActionKind.MALFORMED}),
    ),
)


def stage_at(sampled_tokens: int, total_tokens: int) -> CurriculumStage:
    if total_tokens <= 0 or sampled_tokens < 0:
        raise ValueError("token counts must be nonnegative and total positive")
    fraction = min(sampled_tokens / total_tokens, 1.0)
    for stage in STAGES:
        if stage.start_fraction <= fraction < stage.end_fraction:
            return stage
    return STAGES[-1]
