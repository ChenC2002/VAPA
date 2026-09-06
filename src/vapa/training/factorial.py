"""The manuscript's 2x2 reward-content by credit-placement design."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum

from vapa.config import ExperimentConfig


class Arm(str, Enum):
    A1_GRPO = "a1"
    A2_STEP = "a2"
    A3_PROCESS = "a3"
    A4_VAPA = "a4"


def configure_arm(base: ExperimentConfig, arm: Arm | str) -> ExperimentConfig:
    selected = Arm(arm)
    process = selected in {Arm.A3_PROCESS, Arm.A4_VAPA}
    step = selected in {Arm.A2_STEP, Arm.A4_VAPA}
    credit = replace(
        base.credit,
        process_weight=base.credit.process_weight if process else 0.0,
        local_weight=base.credit.local_weight if step else 0.0,
        use_process_rewards=process,
        use_step_credit=step,
    )
    replay = base.replay if step else replace(base.replay, max_fork_states=0, siblings_per_fork=0)
    result = replace(base, name=f"{selected.value}_{base.name}", credit=credit, replay=replay)
    result.validate()
    return result


@dataclass(frozen=True)
class FactorialContrasts:
    process_given_step: float
    step_given_process: float
    interaction: float
    joint: float


def factorial_contrasts(values: Mapping[Arm | str, float]) -> FactorialContrasts:
    normalized = {Arm(key): float(value) for key, value in values.items()}
    missing = set(Arm) - set(normalized)
    extra = set(normalized) - set(Arm)
    if missing or extra:
        raise ValueError(f"factorial arms mismatch; missing={missing}, extra={extra}")
    a1 = normalized[Arm.A1_GRPO]
    a2 = normalized[Arm.A2_STEP]
    a3 = normalized[Arm.A3_PROCESS]
    a4 = normalized[Arm.A4_VAPA]
    return FactorialContrasts(
        process_given_step=a4 - a2,
        step_given_process=a4 - a3,
        interaction=(a4 - a3) - (a2 - a1),
        joint=a4 - a1,
    )
