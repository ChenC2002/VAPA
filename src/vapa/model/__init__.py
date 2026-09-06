"""Optional language-model adapters for VAPA training."""

from vapa.model.protocols import (
    ActorModelAdapter,
    LossReport,
    OptimizerAdapter,
    OptionalDependencyError,
    SchedulerAdapter,
    TokenizedAction,
    TokenizerAdapter,
    VAPAAction,
)
from vapa.model.transformers import (
    TransformersActorAdapter,
    TransformersGenerationBackend,
    TransformersTokenizerAdapter,
    apply_lora,
    apply_shared_lora,
)

__all__ = [
    "ActorModelAdapter",
    "LossReport",
    "OptionalDependencyError",
    "OptimizerAdapter",
    "SchedulerAdapter",
    "TokenizedAction",
    "TokenizerAdapter",
    "TransformersActorAdapter",
    "TransformersGenerationBackend",
    "TransformersTokenizerAdapter",
    "VAPAAction",
    "apply_lora",
    "apply_shared_lora",
]
