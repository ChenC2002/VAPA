"""VAPA replay, grouping, advantage, ledger, and objective utilities."""

from vapa.training.advantages import AdvantageSummary, assign_step_advantages
from vapa.training.checkpoint import (
    CheckpointContract,
    JsonStateStore,
    RuntimeState,
    TorchStateStore,
    read_manifest,
    resume_checkpoint,
    save_checkpoint,
)
from vapa.training.factorial import Arm, FactorialContrasts, configure_arm, factorial_contrasts
from vapa.training.groups import GroupTier, StepGroup, assign_step_groups
from vapa.training.ledger import GroupCharge, ReplayQuotaLedger, TokenLedger
from vapa.training.replay import ForkAnchor, build_fork_branches, select_fork_anchors
from vapa.training.runtime import (
    DistributedContext,
    IntactActionGroup,
    SFTExample,
    TrainStepReport,
    WarmupCosineScheduler,
    build_adamw,
    collect_vapa_action_groups,
    initialize_distributed,
    resolve_device,
    seed_everything,
    train_sft_step,
    train_vapa_update,
)
from vapa.training.trainer import InstanceBatch, InstanceBatchBuilder, UpdateBatch

__all__ = [
    "AdvantageSummary",
    "Arm",
    "CheckpointContract",
    "DistributedContext",
    "FactorialContrasts",
    "ForkAnchor",
    "GroupTier",
    "GroupCharge",
    "IntactActionGroup",
    "InstanceBatch",
    "InstanceBatchBuilder",
    "ReplayQuotaLedger",
    "RuntimeState",
    "SFTExample",
    "JsonStateStore",
    "TorchStateStore",
    "TokenLedger",
    "TrainStepReport",
    "UpdateBatch",
    "WarmupCosineScheduler",
    "StepGroup",
    "assign_step_advantages",
    "assign_step_groups",
    "build_fork_branches",
    "build_adamw",
    "collect_vapa_action_groups",
    "configure_arm",
    "factorial_contrasts",
    "initialize_distributed",
    "read_manifest",
    "resolve_device",
    "resume_checkpoint",
    "save_checkpoint",
    "seed_everything",
    "select_fork_anchors",
    "train_sft_step",
    "train_vapa_update",
]
