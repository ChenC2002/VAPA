"""Dependency-neutral contracts for action-token language-model training.

The method layer deliberately does not require PyTorch.  Concrete model backends own
their tensor implementation and return :class:`LossReport` objects whose backward
callback lets the training runtime accumulate intact comparison groups.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from vapa.schemas import ActionKind


class OptionalDependencyError(RuntimeError):
    """Raised when an explicitly requested optional model backend is unavailable."""


def _token_ids(values: Iterable[int], name: str) -> tuple[int, ...]:
    result = tuple(values)
    if not result:
        raise ValueError(f"{name} cannot be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in result):
        raise ValueError(f"{name} must contain nonnegative integer token IDs")
    return result


@dataclass(frozen=True)
class ActionPrefixMask:
    """Tokenizer-level state machine for feasible canonical action-call prefixes.

    A template may offer either an immediate final channel or a bounded analysis-channel
    preamble. Analysis text stays sampled and trainable; the final-channel transition
    activates the legal call-prefix trie. The argument suffix remains unconstrained so
    malformed arguments retain the environment's accounting and masking behavior.
    """

    action_prefixes: tuple[tuple[int, ...], ...]
    vocab_size: int
    visible_prefixes: tuple[str, ...] = ()
    content_offsets: tuple[int, ...] = ()
    reasoning_start: tuple[int, ...] = ()
    reasoning_end: tuple[int, ...] = ()
    reasoning_action_prefixes: tuple[tuple[int, ...], ...] = ()
    reasoning_forbidden_tokens: tuple[int, ...] = ()
    action_forbidden_tokens: tuple[int, ...] = ()
    action_terminators: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.vocab_size, bool)
            or not isinstance(self.vocab_size, int)
            or self.vocab_size < 1
        ):
            raise ValueError("action-prefix vocab_size must be a positive integer")
        prefixes = tuple(
            _token_ids(prefix, f"action_prefixes[{index}]")
            for index, prefix in enumerate(self.action_prefixes)
        )
        if not prefixes:
            raise ValueError("an action-prefix mask requires at least one legal prefix")
        if len(set(prefixes)) != len(prefixes):
            raise ValueError("legal action prefixes must have distinct tokenizations")
        if any(token >= self.vocab_size for prefix in prefixes for token in prefix):
            raise ValueError("an action-prefix token is outside the tokenizer vocabulary")
        visible = tuple(self.visible_prefixes)
        offsets = tuple(self.content_offsets)
        if bool(visible) != bool(offsets) or (
            visible and (len(visible) != len(prefixes) or len(offsets) != len(prefixes))
        ):
            raise ValueError(
                "visible action prefixes and content offsets must align with every token prefix"
            )
        if any(not isinstance(value, str) or not value for value in visible):
            raise ValueError("visible action prefixes must be non-empty strings")
        if any(
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or not 0 <= offset < len(prefixes[index])
            for index, offset in enumerate(offsets)
        ):
            raise ValueError("action-prefix content offsets are outside their token prefixes")

        reasoning_parts = (
            bool(self.reasoning_start),
            bool(self.reasoning_end),
            bool(self.reasoning_action_prefixes),
            bool(self.reasoning_forbidden_tokens),
        )
        if any(reasoning_parts) and not all(reasoning_parts):
            raise ValueError("reasoning-channel mask metadata must be supplied together")
        reasoning_start: tuple[int, ...] = ()
        reasoning_end: tuple[int, ...] = ()
        reasoning_prefixes: tuple[tuple[int, ...], ...] = ()
        reasoning_forbidden: tuple[int, ...] = ()
        if all(reasoning_parts):
            if not visible:
                raise ValueError("reasoning-channel masks require visible action metadata")
            reasoning_start = _token_ids(self.reasoning_start, "reasoning_start")
            reasoning_end = _token_ids(self.reasoning_end, "reasoning_end")
            reasoning_prefixes = tuple(
                _token_ids(prefix, f"reasoning_action_prefixes[{index}]")
                for index, prefix in enumerate(self.reasoning_action_prefixes)
            )
            if len(reasoning_prefixes) != len(prefixes):
                raise ValueError("reasoning action prefixes must align with direct prefixes")
            reasoning_forbidden = tuple(
                sorted(
                    set(
                        _token_ids(
                            self.reasoning_forbidden_tokens,
                            "reasoning_forbidden_tokens",
                        )
                    )
                )
            )
            if any(
                token >= self.vocab_size
                for sequence in (reasoning_start, reasoning_end, *reasoning_prefixes)
                for token in sequence
            ) or any(token >= self.vocab_size for token in reasoning_forbidden):
                raise ValueError("a reasoning-channel token is outside the tokenizer vocabulary")
            if reasoning_end[0] in reasoning_forbidden:
                raise ValueError("reasoning support must allow its final-channel delimiter")

        action_forbidden = tuple(sorted(set(self.action_forbidden_tokens)))
        if any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < self.vocab_size
            for token in action_forbidden
        ):
            raise ValueError("action-forbidden tokens must be valid vocabulary IDs")
        action_terminators = tuple(
            sorted(
                {
                    _token_ids(terminator, f"action_terminators[{index}]")
                    for index, terminator in enumerate(self.action_terminators)
                },
                key=lambda item: (-len(item), item),
            )
        )
        if any(
            token >= self.vocab_size for terminator in action_terminators for token in terminator
        ):
            raise ValueError("an action terminator is outside the tokenizer vocabulary")
        if set(action_forbidden).intersection(
            token for terminator in action_terminators for token in terminator
        ):
            raise ValueError("action terminators cannot also be forbidden control tokens")

        def require_prefix_free(paths: Sequence[tuple[int, ...]], message: str) -> None:
            for index, prefix in enumerate(paths):
                for other_index, other in enumerate(paths):
                    if (
                        index != other_index
                        and len(prefix) <= len(other)
                        and other[: len(prefix)] == prefix
                    ):
                        raise ValueError(message)

        require_prefix_free(prefixes, "legal action-prefix tokenizations must be prefix-free")
        if reasoning_prefixes:
            require_prefix_free(
                reasoning_prefixes,
                "reasoning action-prefix tokenizations must be prefix-free",
            )
            require_prefix_free(
                (*prefixes, reasoning_start),
                "direct and reasoning-channel token prefixes must be unambiguous",
            )
        if visible:
            reasoning_values: Sequence[tuple[int, ...] | None] = (
                reasoning_prefixes if reasoning_prefixes else (None,) * len(prefixes)
            )
            ordered = tuple(sorted(zip(prefixes, visible, offsets, reasoning_values, strict=True)))
            object.__setattr__(self, "action_prefixes", tuple(item[0] for item in ordered))
            object.__setattr__(self, "visible_prefixes", tuple(item[1] for item in ordered))
            object.__setattr__(self, "content_offsets", tuple(item[2] for item in ordered))
            if reasoning_prefixes:
                object.__setattr__(
                    self,
                    "reasoning_action_prefixes",
                    tuple(item[3] for item in ordered),
                )
        else:
            object.__setattr__(self, "action_prefixes", tuple(sorted(prefixes)))
        if reasoning_prefixes:
            object.__setattr__(self, "reasoning_start", reasoning_start)
            object.__setattr__(self, "reasoning_end", reasoning_end)
            object.__setattr__(self, "reasoning_forbidden_tokens", reasoning_forbidden)
        object.__setattr__(self, "action_forbidden_tokens", action_forbidden)
        object.__setattr__(self, "action_terminators", action_terminators)

    @staticmethod
    def _trie_allowed(
        paths: Sequence[tuple[int, ...]], generated: tuple[int, ...]
    ) -> tuple[int, ...] | None:
        if any(
            len(generated) >= len(prefix) and generated[: len(prefix)] == prefix for prefix in paths
        ):
            return None
        matching = [
            prefix
            for prefix in paths
            if len(generated) < len(prefix) and prefix[: len(generated)] == generated
        ]
        return tuple(sorted({prefix[len(generated)] for prefix in matching}))

    def _reasoning_post(self, generated: tuple[int, ...]) -> tuple[int, tuple[int, ...]] | None:
        if (
            not self.reasoning_start
            or generated[: len(self.reasoning_start)] != self.reasoning_start
        ):
            return None
        tail = generated[len(self.reasoning_start) :]
        try:
            marker_index = tail.index(self.reasoning_end[0])
        except ValueError:
            return None
        if any(token in self.reasoning_forbidden_tokens for token in tail[:marker_index]):
            return None
        marker = tail[marker_index : marker_index + len(self.reasoning_end)]
        if marker != self.reasoning_end:
            return None
        offset = len(self.reasoning_start) + marker_index + len(self.reasoning_end)
        return offset, generated[offset:]

    def _reasoning_allowed_next(self, generated: tuple[int, ...]) -> tuple[int, ...] | None:
        tail = generated[len(self.reasoning_start) :]
        try:
            marker_index = tail.index(self.reasoning_end[0])
        except ValueError:
            return None
        progress = tail[marker_index:]
        marker_length = min(len(progress), len(self.reasoning_end))
        if progress[:marker_length] != self.reasoning_end[:marker_length]:
            return ()
        if len(progress) < len(self.reasoning_end):
            return (self.reasoning_end[len(progress)],)
        return self._trie_allowed(
            self.reasoning_action_prefixes,
            progress[len(self.reasoning_end) :],
        )

    def forbidden_next(self, generated_ids: Sequence[int]) -> tuple[int, ...]:
        """Return compact control-token exclusions for reasoning or action content."""

        generated = tuple(generated_ids)
        if any(
            len(generated) >= len(prefix) and generated[: len(prefix)] == prefix
            for prefix in self.action_prefixes
        ):
            return self.action_forbidden_tokens
        reasoning = self._reasoning_post(generated)
        if (
            reasoning is not None
            and self._trie_allowed(self.reasoning_action_prefixes, reasoning[1]) is None
        ):
            return self.action_forbidden_tokens
        if (
            self.reasoning_start
            and len(generated) >= len(self.reasoning_start)
            and generated[: len(self.reasoning_start)] == self.reasoning_start
        ):
            tail = generated[len(self.reasoning_start) :]
            if self.reasoning_end[0] not in tail:
                return self.reasoning_forbidden_tokens
        return ()

    def allowed_next(self, generated_ids: Sequence[int]) -> tuple[int, ...] | None:
        """Return allowed next IDs, or ``None`` after a legal prefix is complete."""

        generated = tuple(generated_ids)
        if any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in generated
        ):
            raise ValueError("generated prefix must contain nonnegative integer token IDs")
        if (
            self.reasoning_start
            and len(generated) >= len(self.reasoning_start)
            and generated[: len(self.reasoning_start)] == self.reasoning_start
        ):
            return self._reasoning_allowed_next(generated)
        paths = (
            (*self.action_prefixes, self.reasoning_start)
            if self.reasoning_start
            else self.action_prefixes
        )
        return self._trie_allowed(paths, generated)

    def accepts(self, action_ids: Sequence[int]) -> bool:
        """Whether the token sequence begins with one complete feasible prefix."""

        values = tuple(action_ids)
        if any(
            len(values) >= len(prefix) and values[: len(prefix)] == prefix
            for prefix in self.action_prefixes
        ):
            return True
        reasoning = self._reasoning_post(values)
        return (
            reasoning is not None
            and self._trie_allowed(self.reasoning_action_prefixes, reasoning[1]) is None
        )

    def visible_content(self, action_ids: Sequence[int]) -> tuple[int, str]:
        """Locate parser-visible content after any direct or reasoning preamble."""

        if not self.visible_prefixes:
            raise ValueError("the action-prefix mask lacks visible-text metadata")
        values = tuple(action_ids)
        direct_matches = [
            index
            for index, prefix in enumerate(self.action_prefixes)
            if len(values) >= len(prefix) and values[: len(prefix)] == prefix
        ]
        if len(direct_matches) == 1:
            index = direct_matches[0]
            return self.content_offsets[index], self.visible_prefixes[index]
        reasoning = self._reasoning_post(values)
        if reasoning is not None:
            offset, post = reasoning
            matches = [
                index
                for index, prefix in enumerate(self.reasoning_action_prefixes)
                if len(post) >= len(prefix) and post[: len(prefix)] == prefix
            ]
            if len(matches) == 1:
                index = matches[0]
                return offset, self.visible_prefixes[index]
        raise ValueError("generated tokens do not have one unambiguous legal action prefix")


@dataclass(frozen=True)
class TokenizedAction:
    """A causal-LM prompt and its action-only target tokens."""

    prompt_ids: tuple[int, ...]
    action_ids: tuple[int, ...]
    prefix_mask: ActionPrefixMask | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_ids", _token_ids(self.prompt_ids, "prompt_ids"))
        object.__setattr__(self, "action_ids", _token_ids(self.action_ids, "action_ids"))
        if self.prefix_mask is not None:
            if not isinstance(self.prefix_mask, ActionPrefixMask):
                raise TypeError("prefix_mask must be an ActionPrefixMask or None")
            if not self.prefix_mask.accepts(self.action_ids):
                raise ValueError("action tokens do not begin with a feasible canonical prefix")

    @property
    def token_count(self) -> int:
        return len(self.action_ids)


@dataclass(frozen=True)
class BinaryAnswerPosition:
    """Fixed next-token position for binary ``Answer(0`` / ``Answer(1`` scoring."""

    prompt_ids: tuple[int, ...]
    answer_prefix_ids: tuple[int, ...]
    negative_token_id: int
    positive_token_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_ids", _token_ids(self.prompt_ids, "prompt_ids"))
        object.__setattr__(
            self,
            "answer_prefix_ids",
            _token_ids(self.answer_prefix_ids, "answer_prefix_ids"),
        )
        for name in ("negative_token_id", "positive_token_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer token ID")
        if self.negative_token_id == self.positive_token_id:
            raise ValueError("binary class identifiers must have distinct token IDs")

    @property
    def context_ids(self) -> tuple[int, ...]:
        return self.prompt_ids + self.answer_prefix_ids


@dataclass(frozen=True)
class VAPAAction:
    """One trainable action with behavior-policy probabilities and assigned credit."""

    tokens: TokenizedAction
    behavior_log_probs: tuple[float, ...]
    advantage: float
    rollout_id: str = ""
    turn_index: int = -1

    def __post_init__(self) -> None:
        object.__setattr__(self, "behavior_log_probs", tuple(self.behavior_log_probs))
        if len(self.behavior_log_probs) != self.tokens.token_count:
            raise ValueError("behavior log probabilities must align with action tokens")
        if not all(
            not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)
            for value in self.behavior_log_probs
        ):
            raise ValueError("behavior log probabilities must be finite")
        if (
            isinstance(self.advantage, bool)
            or not isinstance(self.advantage, int | float)
            or not math.isfinite(self.advantage)
        ):
            raise ValueError("advantage must be finite")
        if isinstance(self.turn_index, bool) or not isinstance(self.turn_index, int):
            raise TypeError("turn_index must be an integer")

    @property
    def token_count(self) -> int:
        return self.tokens.token_count


@dataclass
class LossReport:
    """Detached metrics plus a single-use backend-owned backward operation."""

    total: float
    policy: float
    kl: float
    token_count: int
    _backward: Callable[[float], None] | None = field(default=None, repr=False)
    _used: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.total, self.policy, self.kl)):
            raise ValueError("loss metrics must be finite")
        if (
            isinstance(self.token_count, bool)
            or not isinstance(self.token_count, int)
            or self.token_count <= 0
        ):
            raise ValueError("a loss report must contain action tokens")

    def backward(self, scale: float = 1.0) -> None:
        """Backpropagate once, scaling a microbatch mean into an update-wide mean."""

        if self._used:
            raise RuntimeError("a loss report can only be backpropagated once")
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("backward scale must be finite and nonnegative")
        if self._backward is None:
            raise RuntimeError("this loss report is metrics-only")
        self._backward(scale)
        self._used = True


@runtime_checkable
class TokenizerAdapter(Protocol):
    """Minimal tokenizer surface needed by rollout-to-training conversion."""

    @property
    def fingerprint(self) -> str: ...

    def encode_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool = True,
    ) -> tuple[int, ...]: ...

    def encode_text(self, text: str) -> tuple[int, ...]: ...

    def encode_action(
        self,
        messages: Sequence[Mapping[str, str]],
        action_text: str,
    ) -> TokenizedAction: ...

    def action_prefix_mask(
        self,
        messages: Sequence[Mapping[str, str]],
        allowed_actions: Sequence[ActionKind],
    ) -> ActionPrefixMask: ...

    def save_pretrained(self, path: str) -> None: ...


@runtime_checkable
class ActorModelAdapter(Protocol):
    """Backend-owned differentiable action-token objectives."""

    @property
    def fingerprint(self) -> str: ...

    def train(self) -> None: ...

    def eval(self) -> None: ...

    def parameters(self) -> Iterable[Any]: ...

    def sft_loss(self, examples: Sequence[TokenizedAction]) -> LossReport: ...

    def vapa_loss(
        self,
        examples: Sequence[VAPAAction],
        *,
        reference: ActorModelAdapter,
        kl_weight: float,
        ratio_clip: float | None,
        kl_mode: str,
    ) -> LossReport: ...

    def clip_grad_norm(self, max_norm: float) -> float: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


@runtime_checkable
class OptimizerAdapter(Protocol):
    """Subset shared by torch optimizers and dependency-light test optimizers."""

    param_groups: list[dict[str, Any]]

    def zero_grad(self) -> None: ...

    def step(self) -> None: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


@runtime_checkable
class SchedulerAdapter(Protocol):
    def step(self) -> None: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
