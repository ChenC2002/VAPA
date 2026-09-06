"""Validated, dependency-free experiment configuration."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar, get_type_hints


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class EnvironmentConfig:
    memory_capacity: int = 8
    action_budget: int = 12
    turn_cap: int = 16
    context_tokens: int = 32_768
    max_turn_tokens: int = 512

    def validate(self) -> None:
        for name in (
            "memory_capacity",
            "action_budget",
            "turn_cap",
            "context_tokens",
            "max_turn_tokens",
        ):
            _integer(getattr(self, name), name, minimum=1)
        if self.turn_cap < self.action_budget + 1:
            raise ValueError("turn_cap must admit all costed actions plus ANSWER")
        if self.max_turn_tokens > self.context_tokens:
            raise ValueError("max_turn_tokens cannot exceed context_tokens")


@dataclass(frozen=True)
class CreditConfig:
    process_weight: float = 0.05
    cost_weight: float = 0.02
    process_discount: float = 1.0
    local_weight: float = 1.0
    epsilon: float = 1e-8
    use_process_rewards: bool = True
    use_step_credit: bool = True

    def validate(self) -> None:
        for name in (
            "process_weight",
            "cost_weight",
            "process_discount",
            "local_weight",
            "epsilon",
        ):
            _number(getattr(self, name), name)
        _boolean(self.use_process_rewards, "use_process_rewards")
        _boolean(self.use_step_credit, "use_step_credit")
        if not 0.0 < self.process_discount <= 1.0:
            raise ValueError("process_discount must be in (0, 1]")
        if min(self.process_weight, self.cost_weight, self.local_weight) < 0:
            raise ValueError("credit weights cannot be negative")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


@dataclass(frozen=True)
class ReplayConfig:
    base_group_size: int = 8
    max_fork_states: int = 2
    siblings_per_fork: int = 3
    memory_pressure_fraction: float = 0.75
    fork_enabled: bool = True
    reallocate_disabled_forks: bool = False

    def validate(self) -> None:
        for name in ("base_group_size", "max_fork_states", "siblings_per_fork"):
            _integer(getattr(self, name), name, minimum=0)
        _number(self.memory_pressure_fraction, "memory_pressure_fraction")
        _boolean(self.fork_enabled, "fork_enabled")
        _boolean(self.reallocate_disabled_forks, "reallocate_disabled_forks")
        if self.base_group_size < 2:
            raise ValueError("base_group_size must be at least two")
        if min(self.max_fork_states, self.siblings_per_fork) < 0:
            raise ValueError("replay counts cannot be negative")
        if not 0.0 <= self.memory_pressure_fraction <= 1.0:
            raise ValueError("memory_pressure_fraction must be in [0, 1]")
        if self.fork_enabled and self.reallocate_disabled_forks:
            raise ValueError("fork reallocation applies only when forks are disabled")


@dataclass(frozen=True)
class OptimizationConfig:
    sft_learning_rate: float = 1e-4
    rl_learning_rate: float = 5e-6
    sampled_token_budget: int = 2**25
    update_token_floor: int = 131_072
    warmup_fraction: float = 0.03
    final_lr_fraction: float = 0.10
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    kl_weight: float = 0.01
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0

    def validate(self) -> None:
        for name in ("sft_learning_rate", "rl_learning_rate"):
            _number(getattr(self, name), name)
        for name in ("sampled_token_budget", "update_token_floor", "lora_rank", "lora_alpha"):
            _integer(getattr(self, name), name, minimum=1)
        for name in (
            "warmup_fraction",
            "final_lr_fraction",
            "weight_decay",
            "gradient_clip",
            "kl_weight",
            "lora_dropout",
        ):
            _number(getattr(self, name), name)
        if min(self.sft_learning_rate, self.rl_learning_rate) <= 0:
            raise ValueError("learning rates must be positive")
        if min(self.sampled_token_budget, self.update_token_floor) <= 0:
            raise ValueError("token budgets must be positive")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if not 0.0 < self.final_lr_fraction <= 1.0:
            raise ValueError("final_lr_fraction must be in (0, 1]")
        if self.update_token_floor > self.sampled_token_budget:
            raise ValueError("update_token_floor cannot exceed sampled_token_budget")
        if min(self.weight_decay, self.kl_weight) < 0:
            raise ValueError("weight_decay and kl_weight cannot be negative")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")


@dataclass(frozen=True)
class ModelConfig:
    name: str = "Qwen/Qwen3.5-9B"
    thinking: bool = False
    reasoning_effort: str | None = None
    dequantize_mxfp4: bool = False
    dtype: str = "bfloat16"
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0

    def validate(self) -> None:
        _text(self.name, "model name")
        _text(self.dtype, "dtype")
        _boolean(self.thinking, "thinking")
        _boolean(self.dequantize_mxfp4, "dequantize_mxfp4")
        if self.reasoning_effort is not None:
            if not isinstance(self.reasoning_effort, str):
                raise TypeError("reasoning_effort must be a string or None")
            if self.reasoning_effort not in {"low", "medium", "high"}:
                raise ValueError("reasoning_effort must be 'low', 'medium', 'high', or None")
        if "gpt-oss" in self.name.casefold():
            if self.reasoning_effort is None:
                raise ValueError("gpt-oss requires an explicit reasoning_effort")
            if self.thinking:
                raise ValueError("gpt-oss uses reasoning_effort instead of thinking=true")
        elif self.reasoning_effort is not None:
            raise ValueError("reasoning_effort is supported only for gpt-oss models")
        if self.dequantize_mxfp4 and "gpt-oss" not in self.name.casefold():
            raise ValueError("dequantize_mxfp4 is supported only for gpt-oss models")
        _number(self.temperature, "temperature")
        _number(self.top_p, "top_p")
        _integer(self.top_k, "top_k", minimum=0)
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "a4_vapa"
    seed: int = 0
    environment: EnvironmentConfig = EnvironmentConfig()
    credit: CreditConfig = CreditConfig()
    replay: ReplayConfig = ReplayConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    model: ModelConfig = ModelConfig()

    def validate(self) -> None:
        _text(self.name, "experiment name")
        _integer(self.seed, "seed", minimum=0)
        for name, expected in (
            ("environment", EnvironmentConfig),
            ("credit", CreditConfig),
            ("replay", ReplayConfig),
            ("optimization", OptimizationConfig),
            ("model", ModelConfig),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        self.environment.validate()
        self.credit.validate()
        self.replay.validate()
        self.optimization.validate()
        self.model.validate()
        if (
            self.credit.use_step_credit
            and self.replay.fork_enabled
            and self.replay.siblings_per_fork < 1
        ):
            raise ValueError("step credit requires at least one fork sibling")


T = TypeVar("T")


def _construct(cls: type[T], values: dict[str, Any], section: str) -> T:
    expected = {field.name for field in fields(cls)}
    unknown = set(values) - expected
    if unknown:
        raise ValueError(f"unknown keys in [{section}]: {sorted(unknown)}")
    hints = get_type_hints(cls)
    converted: dict[str, Any] = {}
    for key, value in values.items():
        nested = hints.get(key)
        if (
            isinstance(value, dict)
            and isinstance(nested, type)
            and hasattr(nested, "__dataclass_fields__")
        ):
            converted[key] = _construct(nested, value, f"{section}.{key}")
        else:
            converted[key] = value
    return cls(**converted)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_raw(path: Path, seen: set[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"cyclic config inheritance at {resolved}")
    with resolved.open("rb") as stream:
        raw = tomllib.load(stream)
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    if not isinstance(parent, str) or not parent.strip():
        raise ValueError("extends must name a non-empty TOML path")
    inherited = _load_raw(resolved.parent / parent, seen | {resolved})
    return _merge(inherited, raw)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load TOML (with optional relative ``extends``) and reject unsafe values."""

    config_path = Path(path)
    raw = _load_raw(config_path, set())
    config = _construct(ExperimentConfig, raw, "root")
    config.validate()
    return config
