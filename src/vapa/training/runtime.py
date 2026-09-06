"""Production-oriented orchestration for optional action-token model training.

This module remains importable without PyTorch.  It converts the backend-neutral VAPA
rollouts into strict action-token examples and delegates tensor work to a model adapter.
"""

from __future__ import annotations

import math
import os
import random
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vapa.model.protocols import (
    ActorModelAdapter,
    OptimizerAdapter,
    OptionalDependencyError,
    SchedulerAdapter,
    TokenizedAction,
    TokenizerAdapter,
    VAPAAction,
)
from vapa.prompts import render_chat
from vapa.training.trainer import UpdateBatch


class TrainingDataError(ValueError):
    """Raised when rollout artifacts cannot support an actor update."""


@dataclass(frozen=True)
class SFTExample:
    """One demonstration whose prompt tokens are context-only."""

    messages: tuple[Mapping[str, str], ...]
    action_text: str
    group_id: str = ""

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("an SFT example requires prompt messages")
        if not isinstance(self.action_text, str) or not self.action_text:
            raise ValueError("an SFT action must be non-empty")


@dataclass(frozen=True)
class IntactActionGroup:
    """An atomic comparison group used as one gradient-accumulation microbatch."""

    group_id: str
    examples: tuple[VAPAAction, ...]

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("an intact action group requires an ID")
        if not self.examples:
            raise ValueError("an intact action group cannot be empty")

    @property
    def token_count(self) -> int:
        return sum(example.token_count for example in self.examples)


@dataclass(frozen=True)
class TrainStepReport:
    """Update metrics; learning_rates are the rates used by this optimizer step."""

    objective: str
    loss: float
    policy_loss: float
    kl: float
    action_tokens: int
    microbatches: int
    gradient_norm: float
    learning_rates: tuple[float, ...]


@dataclass(frozen=True)
class DistributedContext:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    def __post_init__(self) -> None:
        if min(self.rank, self.local_rank) < 0 or self.world_size < 1:
            raise ValueError("distributed ranks must be nonnegative and world_size positive")
        if self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @classmethod
    def from_environment(cls) -> DistributedContext:
        def integer(name: str, default: int) -> int:
            raw = os.environ.get(name)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError as error:
                raise ValueError(f"{name} must be an integer") from error

        return cls(
            rank=integer("RANK", 0),
            local_rank=integer("LOCAL_RANK", 0),
            world_size=integer("WORLD_SIZE", 1),
        )


def seed_everything(seed: int, *, rank: int = 0, deterministic: bool = False) -> int:
    """Seed Python and installed numerical backends, returning the rank-local seed."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    local_seed = seed + rank
    random.seed(local_seed)
    try:
        import numpy
    except ImportError:
        pass
    else:
        numpy.random.seed(local_seed % (2**32))
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(local_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(local_seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
    return local_seed


def resolve_device(requested: str = "auto", *, local_rank: int = 0) -> str:
    """Resolve an explicit device, failing clearly instead of silently changing hardware."""

    if not isinstance(requested, str) or not requested.strip():
        raise ValueError("device must be a non-empty string")
    requested = requested.lower()
    try:
        import torch
    except ImportError as error:
        if requested in {"auto", "cpu"}:
            return "cpu"
        raise OptionalDependencyError(
            "non-CPU device selection requires PyTorch; install the 'train' extra"
        ) from error
    if requested == "auto":
        if torch.cuda.is_available():
            return f"cuda:{local_rank}"
        mps = getattr(torch.backends, "mps", None)
        return "mps" if mps is not None and mps.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
    return str(torch.device(requested))


def initialize_distributed(
    context: DistributedContext,
    *,
    backend: str | None = None,
) -> None:
    """Initialize torch.distributed only for a genuinely multi-process launch."""

    if context.world_size == 1:
        return
    try:
        import torch
    except ImportError as error:
        raise OptionalDependencyError(
            "distributed training requires PyTorch; install the 'train' extra"
        ) from error
    selected = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    if not torch.distributed.is_available():
        raise RuntimeError("this PyTorch build does not provide distributed training")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=selected,
            rank=context.rank,
            world_size=context.world_size,
        )


def build_adamw(
    model: ActorModelAdapter,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.999),
    epsilon: float = 1e-8,
) -> OptimizerAdapter:
    """Construct torch AdamW without making torch a core package dependency."""

    for value, name in (
        (learning_rate, "learning_rate"),
        (weight_decay, "weight_decay"),
        (epsilon, "epsilon"),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if learning_rate == 0 or epsilon == 0:
        raise ValueError("learning_rate and epsilon must be positive")
    if len(betas) != 2 or not all(
        not isinstance(beta, bool) and isinstance(beta, int | float) and 0 <= beta < 1
        for beta in betas
    ):
        raise ValueError("betas must contain two values in [0, 1)")
    try:
        import torch
    except ImportError as error:
        raise OptionalDependencyError(
            "AdamW training requires PyTorch; install the 'train' extra"
        ) from error
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        eps=epsilon,
    )


class WarmupCosineScheduler:
    """Warm up linearly, then decay cosinely to a nonzero LR fraction."""

    def __init__(
        self,
        optimizer: OptimizerAdapter,
        *,
        total_steps: int,
        warmup_steps: int,
        final_lr_fraction: float = 0.1,
    ) -> None:
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 1:
            raise ValueError("total_steps must be a positive integer")
        if (
            isinstance(warmup_steps, bool)
            or not isinstance(warmup_steps, int)
            or not 0 <= warmup_steps < total_steps
        ):
            raise ValueError("warmup_steps must be an integer in [0, total_steps)")
        if not math.isfinite(final_lr_fraction) or not 0 < final_lr_fraction <= 1:
            raise ValueError("final_lr_fraction must be in (0, 1]")
        if not optimizer.param_groups:
            raise ValueError("optimizer must contain parameter groups")
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.final_lr_fraction = final_lr_fraction
        self.base_lrs = tuple(float(group["lr"]) for group in optimizer.param_groups)
        self.completed_steps = 0
        self._apply()

    def _factor(self) -> float:
        if self.warmup_steps and self.completed_steps < self.warmup_steps:
            return (self.completed_steps + 1) / self.warmup_steps
        decay_steps = self.total_steps - self.warmup_steps
        progress = min(1.0, max(0.0, self.completed_steps - self.warmup_steps) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final_lr_fraction + (1.0 - self.final_lr_fraction) * cosine

    def _apply(self) -> None:
        factor = self._factor()
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base_lr * factor

    def step(self) -> None:
        self.completed_steps = min(self.completed_steps + 1, self.total_steps)
        self._apply()

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "format_version": 1,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "final_lr_fraction": self.final_lr_fraction,
            "base_lrs": list(self.base_lrs),
            "completed_steps": self.completed_steps,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "format_version",
            "total_steps",
            "warmup_steps",
            "final_lr_fraction",
            "base_lrs",
            "completed_steps",
        }
        if set(state) != expected or state["format_version"] != 1:
            raise ValueError("scheduler state has an incompatible schema")
        contract = (
            state["total_steps"],
            state["warmup_steps"],
            float(state["final_lr_fraction"]),
        )
        if contract != (self.total_steps, self.warmup_steps, self.final_lr_fraction):
            raise ValueError("scheduler state does not match the configured schedule")
        base_lrs = tuple(float(value) for value in state["base_lrs"])
        if base_lrs != self.base_lrs:
            raise ValueError("scheduler state does not match optimizer base learning rates")
        completed = state["completed_steps"]
        if isinstance(completed, bool) or not isinstance(completed, int):
            raise ValueError("scheduler completed_steps must be an integer")
        if not 0 <= completed <= self.total_steps:
            raise ValueError("scheduler completed_steps is outside the schedule")
        self.completed_steps = completed
        self._apply()


def collect_vapa_action_groups(
    update: UpdateBatch,
    tokenizer: TokenizerAdapter,
    *,
    scaffold: str = "",
) -> tuple[IntactActionGroup, ...]:
    """Convert one normalized update without splitting any comparison group."""

    buckets: OrderedDict[str, list[VAPAAction]] = OrderedDict()
    expected_tokens = 0
    for instance in update.instances:
        for rollout in instance.base_rollouts + instance.branch_rollouts:
            for turn in rollout.turns:
                if not turn.loss_mask:
                    continue
                decision = turn.decision
                if decision is None or turn.action is None:
                    raise TrainingDataError("a loss-enabled turn lacks a parsed actor decision")
                if not decision.token_ids:
                    raise TrainingDataError(
                        "loss-enabled decisions require generated token IDs from the model backend"
                    )
                if not decision.behavior_log_probs:
                    raise TrainingDataError(
                        "loss-enabled decisions require behavior-policy token log probabilities"
                    )
                if len(decision.token_ids) != decision.token_count:
                    raise TrainingDataError("decision token IDs do not match token_count")
                if len(decision.behavior_log_probs) != decision.token_count:
                    raise TrainingDataError("behavior log probabilities do not match token_count")
                messages = render_chat(turn.observation, scaffold)
                prompt_ids = tokenizer.encode_messages(messages, add_generation_prompt=True)
                mask_factory = getattr(tokenizer, "action_prefix_mask", None)
                prefix_mask = (
                    mask_factory(messages, turn.observation.legal_actions)
                    if callable(mask_factory)
                    else None
                )
                example = VAPAAction(
                    tokens=TokenizedAction(
                        prompt_ids,
                        decision.token_ids,
                        prefix_mask=prefix_mask,
                    ),
                    behavior_log_probs=decision.behavior_log_probs,
                    advantage=turn.normalized_advantage,
                    rollout_id=rollout.rollout_id,
                    turn_index=turn.index,
                )
                group_id = turn.group_id or f"singleton:{rollout.rollout_id}:{turn.index}"
                buckets.setdefault(group_id, []).append(example)
                expected_tokens += decision.token_count
    if not buckets:
        raise TrainingDataError("the update contains no loss-enabled action tokens")
    if expected_tokens != update.trainable_tokens:
        raise TrainingDataError("runtime token accounting disagrees with the update batch")
    return tuple(
        IntactActionGroup(group_id, tuple(examples)) for group_id, examples in buckets.items()
    )


def _learning_rates(optimizer: OptimizerAdapter) -> tuple[float, ...]:
    return tuple(float(group["lr"]) for group in optimizer.param_groups)


def _finish_step(
    model: ActorModelAdapter,
    optimizer: OptimizerAdapter,
    scheduler: SchedulerAdapter | None,
    *,
    gradient_clip: float,
) -> tuple[float, tuple[float, ...]]:
    gradient_norm = model.clip_grad_norm(gradient_clip)
    if not math.isfinite(gradient_norm):
        optimizer.zero_grad()
        raise FloatingPointError("gradient norm is non-finite; optimizer step was skipped")
    learning_rates = _learning_rates(optimizer)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return gradient_norm, learning_rates


def train_vapa_update(
    update: UpdateBatch,
    *,
    tokenizer: TokenizerAdapter,
    actor: ActorModelAdapter,
    reference: ActorModelAdapter,
    optimizer: OptimizerAdapter,
    scheduler: SchedulerAdapter | None = None,
    kl_weight: float = 0.01,
    ratio_clip: float | None = None,
    kl_mode: str = "k3",
    gradient_clip: float = 1.0,
    scaffold: str = "",
) -> TrainStepReport:
    """Take one actor step, accumulating each comparison group as an intact unit."""

    if not math.isfinite(gradient_clip) or gradient_clip <= 0:
        raise ValueError("gradient_clip must be finite and positive")
    if actor is reference:
        raise ValueError("actor and reference must be distinct model adapters")
    groups = collect_vapa_action_groups(update, tokenizer, scaffold=scaffold)
    total_tokens = sum(group.token_count for group in groups)
    actor.train()
    reference.eval()
    optimizer.zero_grad()
    total = policy = kl = 0.0
    try:
        for group in groups:
            report = actor.vapa_loss(
                group.examples,
                reference=reference,
                kl_weight=kl_weight,
                ratio_clip=ratio_clip,
                kl_mode=kl_mode,
            )
            if report.token_count != group.token_count:
                raise RuntimeError("model loss token count disagrees with its intact group")
            weight = group.token_count / total_tokens
            report.backward(weight)
            total += weight * report.total
            policy += weight * report.policy
            kl += weight * report.kl
        gradient_norm, learning_rates = _finish_step(
            actor,
            optimizer,
            scheduler,
            gradient_clip=gradient_clip,
        )
    except Exception:
        optimizer.zero_grad()
        raise
    return TrainStepReport(
        objective="vapa",
        loss=total,
        policy_loss=policy,
        kl=kl,
        action_tokens=total_tokens,
        microbatches=len(groups),
        gradient_norm=gradient_norm,
        learning_rates=learning_rates,
    )


def train_sft_step(
    examples: Sequence[SFTExample],
    *,
    tokenizer: TokenizerAdapter,
    actor: ActorModelAdapter,
    optimizer: OptimizerAdapter,
    scheduler: SchedulerAdapter | None = None,
    gradient_clip: float = 1.0,
) -> TrainStepReport:
    """Take one action-only SFT step while preserving caller-declared groups."""

    if not examples:
        raise ValueError("an SFT step requires at least one example")
    if not math.isfinite(gradient_clip) or gradient_clip <= 0:
        raise ValueError("gradient_clip must be finite and positive")
    buckets: OrderedDict[str, list[TokenizedAction]] = OrderedDict()
    for index, example in enumerate(examples):
        prompt = tokenizer.encode_messages(example.messages, add_generation_prompt=True)
        tokenized = tokenizer.encode_action(example.messages, example.action_text)
        group_id = example.group_id or f"sft:{index}"
        if tokenized.prompt_ids != prompt:
            raise TrainingDataError("tokenizer returned inconsistent SFT prompt IDs")
        buckets.setdefault(group_id, []).append(tokenized)
    total_tokens = sum(item.token_count for items in buckets.values() for item in items)
    actor.train()
    optimizer.zero_grad()
    total = 0.0
    try:
        for items in buckets.values():
            report = actor.sft_loss(items)
            group_tokens = sum(item.token_count for item in items)
            if report.token_count != group_tokens:
                raise RuntimeError("model SFT loss token count disagrees with its intact group")
            weight = group_tokens / total_tokens
            report.backward(weight)
            total += weight * report.total
        gradient_norm, learning_rates = _finish_step(
            actor,
            optimizer,
            scheduler,
            gradient_clip=gradient_clip,
        )
    except Exception:
        optimizer.zero_grad()
        raise
    return TrainStepReport(
        objective="sft",
        loss=total,
        policy_loss=total,
        kl=0.0,
        action_tokens=total_tokens,
        microbatches=len(buckets),
        gradient_norm=gradient_norm,
        learning_rates=learning_rates,
    )
