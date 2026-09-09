"""Optional Hugging Face adapters for action-token SFT and VAPA optimization."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from vapa.actions import action_call_prefix
from vapa.model.protocols import (
    ActionPrefixMask,
    ActorModelAdapter,
    BinaryAnswerPosition,
    LossReport,
    OptionalDependencyError,
    TokenizedAction,
    VAPAAction,
)
from vapa.policies.text import GeneratedCandidate
from vapa.schemas import ActionKind


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised without the extra installed
        raise OptionalDependencyError(
            "PyTorch training is unavailable; install the project with the 'train' extra"
        ) from error
    return torch


def _require_transformers() -> Any:
    try:
        import transformers
    except ImportError as error:  # pragma: no cover - exercised without the extra installed
        raise OptionalDependencyError(
            "Transformers support is unavailable; install the project with the 'train' extra"
        ) from error
    return transformers


def _require_peft() -> Any:
    try:
        import peft
    except ImportError as error:  # pragma: no cover - exercised without the extra installed
        raise OptionalDependencyError(
            "LoRA support is unavailable; install the project with the 'train' extra"
        ) from error
    return peft


def _fingerprint(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _masked_logit_row(
    row: Any,
    allowed: tuple[int, ...] | None,
    mask: ActionPrefixMask,
    forbidden: tuple[int, ...] = (),
) -> Any:
    """Apply one prefix-trie support set without detaching differentiable logits."""

    values = row.float()
    width = int(values.shape[-1])
    if width != mask.vocab_size:
        raise ValueError(
            "model-logit vocabulary does not match the feasible-action tokenizer vocabulary"
        )
    if allowed is not None and forbidden:
        raise ValueError("a token position cannot have both allow and deny support")
    if allowed is None and not forbidden:
        return values
    if allowed is not None and not allowed:
        raise ValueError("generated tokens diverged from every feasible action prefix")
    torch = _require_torch()
    if allowed is None:
        excluded = torch.zeros_like(values, dtype=torch.bool)
        excluded[list(forbidden)] = True
    else:
        excluded = torch.ones_like(values, dtype=torch.bool)
        excluded[list(allowed)] = False
    return values.masked_fill(excluded, float("-inf"))


def _prefix_allowed_tokens_function(prompt_ids: tuple[int, ...], mask: ActionPrefixMask):
    """Build the Hugging Face callback for exactly one causal-LM prompt."""

    all_tokens = list(range(mask.vocab_size))

    def allowed_tokens(_batch_id: int, input_ids: Any) -> list[int]:
        raw = input_ids.tolist() if callable(getattr(input_ids, "tolist", None)) else input_ids
        values = tuple(int(token) for token in raw)
        if values[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError("generation callback received an unexpected prompt prefix")
        allowed = mask.allowed_next(values[len(prompt_ids) :])
        return all_tokens if allowed is None else list(allowed)

    return allowed_tokens


class _FeasibleActionLogitsProcessor:
    """Compact prefix mask for templates with a free-form reasoning channel."""

    def __init__(self, prompt_ids: tuple[int, ...], mask: ActionPrefixMask) -> None:
        self.prompt_ids = prompt_ids
        self.mask = mask

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        for row_index in range(int(input_ids.shape[0])):
            raw = input_ids[row_index].tolist()
            values = tuple(int(token) for token in raw)
            if values[: len(self.prompt_ids)] != self.prompt_ids:
                raise RuntimeError("generation processor received an unexpected prompt prefix")
            generated = values[len(self.prompt_ids) :]
            allowed = self.mask.allowed_next(generated)
            forbidden = self.mask.forbidden_next(generated)
            scores[row_index] = _masked_logit_row(
                scores[row_index],
                allowed,
                self.mask,
                forbidden,
            )
        return scores


class TransformersTokenizerAdapter:
    """Chat-template-aware wrapper around a tokenizer or multimodal processor."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        template_source: Any | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        if tokenizer is None or not callable(getattr(tokenizer, "encode", None)):
            raise TypeError("tokenizer must provide encode()")
        self.tokenizer = tokenizer
        self.template_source = tokenizer if template_source is None else template_source
        self.chat_template_kwargs = dict(chat_template_kwargs or {})

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        *,
        use_processor: bool | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
        **kwargs: Any,
    ) -> TransformersTokenizerAdapter:
        transformers = _require_transformers()
        if use_processor is None:
            use_processor = "qwen3.5" in name.casefold()
        if use_processor:
            processor_class = getattr(transformers, "AutoProcessor", None)
            if processor_class is None:
                raise OptionalDependencyError(
                    "this multimodal model requires AutoProcessor from a current Transformers"
                )
            processor = processor_class.from_pretrained(name, **kwargs)
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is None:
                raise TypeError("the loaded processor does not expose a text tokenizer")
            return cls(
                tokenizer,
                template_source=processor,
                chat_template_kwargs=chat_template_kwargs,
            )
        tokenizer = transformers.AutoTokenizer.from_pretrained(name, **kwargs)
        return cls(tokenizer, chat_template_kwargs=chat_template_kwargs)

    @property
    def fingerprint(self) -> str:
        special = {
            name: getattr(self.tokenizer, name, None)
            for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
        }
        # Names and vocabulary size alone cannot detect changed token IDs or a
        # modified local tokenizer. Bind the actual encoding rules as well.
        get_vocab = getattr(self.tokenizer, "get_vocab", None)
        vocabulary = get_vocab() if callable(get_vocab) else None
        backend = getattr(self.tokenizer, "backend_tokenizer", None)
        serialize = getattr(backend, "to_str", None)
        encoding_rules = json.loads(serialize()) if callable(serialize) else None
        if isinstance(encoding_rules, dict):
            # These are transient per-call settings, not tokenizer identity.
            encoding_rules.pop("padding", None)
            encoding_rules.pop("truncation", None)
        return _fingerprint(
            {
                "adapter": "transformers-tokenizer-v2",
                "vocabulary": vocabulary,
                "encoding_rules": encoding_rules,
                "name": getattr(self.tokenizer, "name_or_path", self.tokenizer.__class__.__name__),
                "class": self.tokenizer.__class__.__name__,
                "template_class": self.template_source.__class__.__name__,
                "vocab_size": getattr(self.tokenizer, "vocab_size", None),
                "special_tokens": special,
                "chat_template": getattr(
                    self.template_source,
                    "chat_template",
                    getattr(self.tokenizer, "chat_template", None),
                ),
                "commit": getattr(self.tokenizer, "_commit_hash", None),
                "chat_template_kwargs": self.chat_template_kwargs,
            }
        )

    def _has_chat_template(self) -> bool:
        return bool(
            getattr(
                self.template_source,
                "chat_template",
                getattr(self.tokenizer, "chat_template", None),
            )
        )

    @staticmethod
    def _fallback_text(
        messages: Sequence[Mapping[str, str]], *, add_generation_prompt: bool
    ) -> str:
        rendered = "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages
        )
        if add_generation_prompt:
            rendered += "\nassistant:"
        return rendered

    def encode_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool = True,
    ) -> tuple[int, ...]:
        normalized = [dict(message) for message in messages]
        template = getattr(self.template_source, "apply_chat_template", None)
        if callable(template) and self._has_chat_template():
            encoded = template(
                normalized,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                **self.chat_template_kwargs,
            )
        else:
            rendered = self._fallback_text(normalized, add_generation_prompt=add_generation_prompt)
            encoded = self.tokenizer.encode(rendered, add_special_tokens=True)
        result = tuple(int(token) for token in encoded)
        if not result:
            raise ValueError("the tokenizer produced an empty prompt")
        return result

    def encode_text(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str) or not text:
            raise ValueError("action text must be non-empty")
        encoded = tuple(
            int(token) for token in self.tokenizer.encode(text, add_special_tokens=False)
        )
        if not encoded:
            raise ValueError("the tokenizer produced no action tokens")
        return encoded

    def _vocabulary_size(self) -> int:
        try:
            size = len(self.tokenizer)
        except (TypeError, AttributeError):
            size = getattr(self.tokenizer, "vocab_size", None)
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("the tokenizer does not expose a usable vocabulary size")
        return size

    def _assistant_continuation_ids(
        self,
        messages: Sequence[Mapping[str, str]],
        content: str,
        *,
        assistant_fields: Mapping[str, str] | None = None,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        normalized = [dict(message) for message in messages]
        prompt_ids = self.encode_messages(normalized, add_generation_prompt=True)
        template = getattr(self.template_source, "apply_chat_template", None)
        if callable(template) and self._has_chat_template():
            assistant = {"role": "assistant", "content": content}
            if assistant_fields:
                if {"role", "content"}.intersection(assistant_fields):
                    raise ValueError("assistant_fields cannot replace role or content")
                assistant.update(assistant_fields)
            full_ids = tuple(
                int(token)
                for token in template(
                    normalized + [assistant],
                    tokenize=True,
                    add_generation_prompt=False,
                    **self.chat_template_kwargs,
                )
            )
            if full_ids[: len(prompt_ids)] != prompt_ids:
                if assistant_fields:
                    raise ValueError(
                        "tokenizer assistant-field boundary is unstable; "
                        "feasible-action masking is unsafe"
                    )
                probe = "VAPA completion probe 93c6e1"
                artificial: list[tuple[int, ...]] = []
                for assistant_content in ("", probe):
                    artificial.append(
                        tuple(
                            int(token)
                            for token in template(
                                normalized + [{"role": "assistant", "content": assistant_content}],
                                tokenize=True,
                                add_generation_prompt=False,
                                **self.chat_template_kwargs,
                            )
                        )
                    )
                empty_artificial, probe_artificial = artificial
                prefix_length = self._common_prefix_length(
                    empty_artificial,
                    probe_artificial,
                )
                suffix_length = self._common_suffix_length(
                    empty_artificial[prefix_length:],
                    probe_artificial[prefix_length:],
                )
                if prefix_length + suffix_length != len(empty_artificial):
                    raise ValueError("tokenizer cannot isolate its assistant completion suffix")
                completion_suffix = empty_artificial[-suffix_length:] if suffix_length else ()
                if full_ids[:prefix_length] != empty_artificial[:prefix_length] or (
                    completion_suffix and full_ids[-len(completion_suffix) :] != completion_suffix
                ):
                    raise ValueError("tokenizer assistant completion framing changes with content")
                rendered = template(
                    normalized,
                    tokenize=False,
                    add_generation_prompt=True,
                    **self.chat_template_kwargs,
                )
                if not isinstance(rendered, str):
                    raise ValueError("tokenizer cannot render a stable text generation prompt")
                generated_ids = tuple(
                    int(token)
                    for token in self.tokenizer.encode(
                        rendered + content,
                        add_special_tokens=False,
                    )
                )
                full_ids = generated_ids + completion_suffix
        else:
            prompt_text = self._fallback_text(normalized, add_generation_prompt=True)
            full_ids = tuple(
                int(token)
                for token in self.tokenizer.encode(
                    prompt_text + content,
                    add_special_tokens=True,
                )
            )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "tokenizer chat boundary is unstable; feasible-action masking is unsafe"
            )
        return prompt_ids, full_ids[len(prompt_ids) :]

    def _decode_tokens(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        decoder = getattr(self.tokenizer, "decode", None)
        if not callable(decoder):
            raise ValueError("the tokenizer cannot verify canonical action-prefix decoding")
        try:
            decoded = decoder(
                list(token_ids),
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            decoded = decoder(list(token_ids), skip_special_tokens=skip_special_tokens)
        if not isinstance(decoded, str):
            raise ValueError("the tokenizer returned a non-text action-prefix decoding")
        return decoded

    @staticmethod
    def _common_suffix_length(left: Sequence[int], right: Sequence[int]) -> int:
        length = 0
        limit = min(len(left), len(right))
        while length < limit and left[-(length + 1)] == right[-(length + 1)]:
            length += 1
        return length

    @staticmethod
    def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
        length = 0
        limit = min(len(left), len(right))
        while length < limit and left[length] == right[length]:
            length += 1
        return length

    def action_prefix_mask(
        self,
        messages: Sequence[Mapping[str, str]],
        allowed_actions: Sequence[ActionKind],
    ) -> ActionPrefixMask:
        """Build a strict token trie for the state-feasible canonical call names."""

        raw_actions = tuple(allowed_actions)
        if not raw_actions:
            raise ValueError("feasible-action masking requires at least one legal action")
        if any(not isinstance(action, ActionKind) for action in raw_actions):
            raise TypeError("allowed_actions must contain ActionKind values")
        actions = tuple(sorted(set(raw_actions), key=lambda item: item.value))
        vocab_size = self._vocabulary_size()
        prefix_ids: list[tuple[int, ...]] = []
        content_prefixes: list[tuple[int, ...]] = []
        visible_prefixes: list[str] = []
        special_ids = {
            int(token)
            for token in getattr(self.tokenizer, "all_special_ids", ())
            if isinstance(token, int) and not isinstance(token, bool)
        }
        for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
            token = getattr(self.tokenizer, name, None)
            if isinstance(token, int) and not isinstance(token, bool):
                special_ids.add(token)
            elif name == "eos_token_id" and isinstance(token, list | tuple):
                special_ids.update(
                    int(item)
                    for item in token
                    if isinstance(item, int) and not isinstance(item, bool)
                )
        _, empty_continuation = self._assistant_continuation_ids(messages, "")
        completion_suffix: tuple[int, ...] | None = None
        direct_preamble: tuple[int, ...] | None = None
        for action in actions:
            text = action_call_prefix(action)
            _, continuation = self._assistant_continuation_ids(messages, text)
            suffix_length = self._common_suffix_length(empty_continuation, continuation)
            suffix = continuation[-suffix_length:] if suffix_length else ()
            empty_core = (
                empty_continuation[:-suffix_length] if suffix_length else empty_continuation
            )
            encoded = continuation[:-suffix_length] if suffix_length else continuation
            if not encoded or encoded[: len(empty_core)] != empty_core:
                raise ValueError(
                    f"tokenizer cannot isolate the canonical {text!r} assistant preamble"
                )
            content_ids = encoded[len(empty_core) :]
            if not content_ids:
                raise ValueError(f"canonical action prefix {text!r} produced no content tokens")
            if special_ids.intersection(content_ids):
                raise ValueError(f"canonical action prefix {text!r} contains a special token")
            if self._decode_tokens(content_ids, skip_special_tokens=True) != text:
                raise ValueError(
                    f"canonical action prefix {text!r} does not decode exactly at the "
                    "assistant content boundary"
                )
            if completion_suffix is None:
                completion_suffix = suffix
                direct_preamble = empty_core
            elif suffix != completion_suffix or empty_core != direct_preamble:
                raise ValueError("assistant template framing changes across legal actions")
            prefix_ids.append(encoded)
            content_prefixes.append(content_ids)
            visible_prefixes.append(text)
        assert completion_suffix is not None and direct_preamble is not None

        reasoning_start: tuple[int, ...] = ()
        reasoning_end: tuple[int, ...] = ()
        reasoning_prefixes: tuple[tuple[int, ...], ...] = ()
        reasoning_forbidden: tuple[int, ...] = ()
        raw_template = getattr(
            self.template_source,
            "chat_template",
            getattr(self.tokenizer, "chat_template", ""),
        )
        template_text = str(raw_template)
        if "message.thinking" in template_text or '"thinking" in message' in template_text:
            probe = "VAPA reasoning probe 7f4a2c"
            _, reasoning_empty = self._assistant_continuation_ids(
                messages,
                "",
                assistant_fields={"thinking": ""},
            )
            _, reasoning_probe = self._assistant_continuation_ids(
                messages,
                "",
                assistant_fields={"thinking": probe},
            )
            if reasoning_empty != empty_continuation or reasoning_probe != empty_continuation:

                def strip_completion_suffix(values: tuple[int, ...]) -> tuple[int, ...]:
                    if completion_suffix:
                        if values[-len(completion_suffix) :] != completion_suffix:
                            raise ValueError(
                                "reasoning and final assistant turns use different terminators"
                            )
                        return values[: -len(completion_suffix)]
                    return values

                empty_reasoning_core = strip_completion_suffix(reasoning_empty)
                probe_reasoning_core = strip_completion_suffix(reasoning_probe)
                start_length = self._common_prefix_length(
                    empty_reasoning_core,
                    probe_reasoning_core,
                )
                end_length = self._common_suffix_length(
                    empty_reasoning_core[start_length:],
                    probe_reasoning_core[start_length:],
                )
                if (
                    start_length < 1
                    or end_length < 1
                    or start_length + end_length != len(empty_reasoning_core)
                ):
                    raise ValueError("assistant reasoning-channel boundaries are ambiguous")
                reasoning_start = empty_reasoning_core[:start_length]
                reasoning_end = empty_reasoning_core[start_length:]
                probe_ids = probe_reasoning_core[
                    start_length : len(probe_reasoning_core) - end_length
                ]
                if self._decode_tokens(probe_ids, skip_special_tokens=True) != probe:
                    raise ValueError("assistant reasoning content does not tokenize independently")
                if reasoning_end[0] not in special_ids:
                    raise ValueError(
                        "assistant reasoning delimiter is not an unambiguous special token"
                    )
                derived_reasoning_prefixes: list[tuple[int, ...]] = []
                expected_head = reasoning_start + reasoning_end
                for text, expected_content in zip(
                    visible_prefixes,
                    content_prefixes,
                    strict=True,
                ):
                    _, reasoning_action = self._assistant_continuation_ids(
                        messages,
                        text,
                        assistant_fields={"thinking": ""},
                    )
                    reasoning_action_core = strip_completion_suffix(reasoning_action)
                    if reasoning_action_core[: len(expected_head)] != expected_head:
                        raise ValueError(
                            "assistant final channel changes after the reasoning transition"
                        )
                    content = reasoning_action_core[len(expected_head) :]
                    if content != expected_content:
                        raise ValueError(
                            "assistant action tokenization differs after the reasoning channel"
                        )
                    derived_reasoning_prefixes.append(content)
                reasoning_prefixes = tuple(derived_reasoning_prefixes)
                reasoning_forbidden = tuple(
                    sorted(
                        token
                        for token in special_ids
                        if token != reasoning_end[0] and token < vocab_size
                    )
                )
        terminators = set()
        if completion_suffix:
            terminators.add(completion_suffix)
        raw_eos = getattr(self.tokenizer, "eos_token_id", None)
        eos_tokens = raw_eos if isinstance(raw_eos, list | tuple) else (raw_eos,)
        terminators.update(
            (int(token),)
            for token in eos_tokens
            if isinstance(token, int) and not isinstance(token, bool) and 0 <= token < vocab_size
        )
        terminator_controls = {
            token for terminator in terminators for token in terminator if token in special_ids
        }
        action_forbidden = tuple(
            sorted(token for token in special_ids - terminator_controls if 0 <= token < vocab_size)
        )
        return ActionPrefixMask(
            tuple(prefix_ids),
            vocab_size,
            tuple(visible_prefixes),
            tuple(len(direct_preamble) for _ in prefix_ids),
            reasoning_start,
            reasoning_end,
            reasoning_prefixes,
            reasoning_forbidden,
            action_forbidden,
            tuple(terminators),
        )

    def decode_generated_action(
        self,
        token_ids: Sequence[int],
        prefix_mask: ActionPrefixMask,
    ) -> str:
        """Decode parser-visible action text, or a malformed sentinel if truncated.

        Prefix masking guarantees that a completed prefix is legal, but a model may
        exhaust ``max_new_tokens`` while it is still reasoning, changing channels, or
        spelling the call name.  Such samples remain real, chargeable rollouts.  The
        sentinel makes only their parser view malformed; the candidate still carries
        every sampled token and behavior log probability.
        """

        try:
            content_offset, visible_prefix = prefix_mask.visible_content(token_ids)
        except ValueError:
            return "<incomplete-feasible-action>"
        content_ids = tuple(token_ids)[content_offset:]
        for terminator in prefix_mask.action_terminators:
            if (
                len(content_ids) >= len(terminator)
                and content_ids[-len(terminator) :] == terminator
            ):
                content_ids = content_ids[: -len(terminator)]
                break
        control_ids = set(prefix_mask.action_forbidden_tokens)
        control_ids.update(
            token for terminator in prefix_mask.action_terminators for token in terminator
        )
        if control_ids.intersection(content_ids):
            return "<incomplete-feasible-action>"
        decoded = self._decode_tokens(
            content_ids,
            skip_special_tokens=False,
        )
        if not decoded.startswith(visible_prefix):
            return "<incomplete-feasible-action>"
        return decoded

    def binary_answer_position(
        self,
        messages: Sequence[Mapping[str, str]],
        action_ids: Sequence[int] | None = None,
    ) -> BinaryAnswerPosition:
        """Resolve a strict binary prediction position, optionally from a sampled action.

        Supplying ``action_ids`` preserves the actual same-turn reasoning and channel
        tokens sampled before ``Answer(``.  The direct-final position remains the
        backward-compatible default for callers that have no terminal candidate.
        """

        visible_prefix = action_call_prefix(ActionKind.ANSWER)
        mask = self.action_prefix_mask(messages, (ActionKind.ANSWER,))
        direct_prefix = mask.action_prefixes[0]
        content_offset = mask.content_offsets[0]
        prompt_ids = self.encode_messages(messages, add_generation_prompt=True)
        _, empty_continuation = self._assistant_continuation_ids(messages, "")
        class_tokens: list[int] = []
        special_ids = {
            int(token)
            for token in getattr(self.tokenizer, "all_special_ids", ())
            if isinstance(token, int) and not isinstance(token, bool)
        }
        for identifier in ("0", "1"):
            _, continuation = self._assistant_continuation_ids(
                messages,
                visible_prefix + identifier,
            )
            suffix_length = self._common_suffix_length(empty_continuation, continuation)
            path = continuation[:-suffix_length] if suffix_length else continuation
            if path[: len(direct_prefix)] != direct_prefix:
                raise ValueError(
                    f"binary identifier {identifier!r} changes the Answer prediction boundary"
                )
            identifier_ids = path[len(direct_prefix) :]
            if len(identifier_ids) != 1:
                raise ValueError(
                    f"binary identifier {identifier!r} is not exactly one token at the "
                    "Answer prediction boundary"
                )
            token = identifier_ids[0]
            if token in special_ids:
                raise ValueError(f"binary identifier {identifier!r} is a special token")
            if (
                self._decode_tokens(path[content_offset:], skip_special_tokens=True)
                != visible_prefix + identifier
            ):
                raise ValueError(
                    f"binary identifier {identifier!r} does not decode exactly at the "
                    "Answer prediction boundary"
                )
            class_tokens.append(token)
        answer_prefix_ids = direct_prefix
        if action_ids is not None:
            values = tuple(action_ids)
            if not values or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in values
            ):
                raise ValueError(
                    "binary readout action_ids must contain nonnegative integer token IDs"
                )
            sampled_content_offset, sampled_prefix = mask.visible_content(values)
            if sampled_prefix != visible_prefix:
                raise ValueError("binary readout candidate is not a canonical Answer action")
            content_prefix_ids = direct_prefix[content_offset:]
            class_offset = sampled_content_offset + len(content_prefix_ids)
            if len(values) <= class_offset:
                raise ValueError("binary readout candidate has no sampled class identifier")
            if values[class_offset] not in class_tokens:
                raise ValueError(
                    "binary readout candidate does not use a canonical single-token 0/1 identifier"
                )
            answer_prefix_ids = values[:class_offset]
        return BinaryAnswerPosition(
            prompt_ids,
            answer_prefix_ids,
            class_tokens[0],
            class_tokens[1],
        )

    def encode_action(
        self,
        messages: Sequence[Mapping[str, str]],
        action_text: str,
    ) -> TokenizedAction:
        if not isinstance(action_text, str) or not action_text:
            raise ValueError("action text must be non-empty")
        prompt_ids, action_ids = self._assistant_continuation_ids(messages, action_text)
        if not action_ids:
            raise ValueError("the tokenizer produced no supervised action tokens")
        return TokenizedAction(prompt_ids, action_ids)

    def save_pretrained(self, path: str) -> None:
        save = getattr(self.template_source, "save_pretrained", None)
        if not callable(save):
            raise TypeError("tokenizer does not provide save_pretrained()")
        save(path)


class TransformersActorAdapter:
    """Text-logit model adapter whose objectives cover action tokens only.

    The implementation evaluates each intact action separately.  This favors clear
    masking semantics and backend portability; high-throughput deployments can supply
    another :class:`~vapa.model.protocols.ActorModelAdapter` with packed kernels.
    """

    def __init__(
        self,
        model: Any,
        *,
        device: str | None = None,
        provenance: Mapping[str, object] | None = None,
        adapter_name: str | None = None,
        adapter_trainable: bool = True,
    ) -> None:
        if not callable(model):
            raise TypeError("model must be callable")
        if adapter_name is not None and (
            not isinstance(adapter_name, str)
            or not adapter_name.strip()
            or adapter_name != adapter_name.strip()
        ):
            raise ValueError("adapter_name must be a non-empty, trimmed string or None")
        if not isinstance(adapter_trainable, bool):
            raise TypeError("adapter_trainable must be a boolean")
        if adapter_name is not None and not callable(getattr(model, "set_adapter", None)):
            raise TypeError("named-adapter models must provide set_adapter()")
        self.model = model
        self.device = device
        self.provenance = dict(provenance or {})
        self.adapter_name = adapter_name
        self.adapter_trainable = adapter_trainable
        self._training = bool(getattr(model, "training", False))

    @property
    def shares_backbone(self) -> bool:
        """Whether this wrapper addresses one named adapter on a PEFT backbone."""

        return self.adapter_name is not None

    def _activate_adapter(self) -> None:
        if self.adapter_name is None:
            return
        self.model.set_adapter(
            self.adapter_name,
            inference_mode=not self.adapter_trainable,
        )
        if not self.adapter_trainable:
            for parameter in self.model.parameters():
                if getattr(parameter, "requires_grad", False):
                    requires_grad = getattr(parameter, "requires_grad_", None)
                    if not callable(requires_grad):
                        raise TypeError("adapter parameter does not support requires_grad_()")
                    requires_grad(False)

    def _prepare_forward(self) -> None:
        if self.adapter_name is None:
            return
        self._activate_adapter()
        mode = getattr(self.model, "train", None)
        if callable(mode):
            mode(self._training)

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        *,
        device: str | None = None,
        dtype: str | None = None,
        model_kind: str = "auto",
        dequantize_mxfp4: bool = False,
        **kwargs: Any,
    ) -> TransformersActorAdapter:
        if not isinstance(dequantize_mxfp4, bool):
            raise TypeError("dequantize_mxfp4 must be a boolean")
        if dequantize_mxfp4 and "gpt-oss" not in name.casefold():
            raise ValueError("dequantize_mxfp4 is supported only for gpt-oss models")
        torch = _require_torch()
        transformers = _require_transformers()
        if model_kind not in {"auto", "multimodal", "causal"}:
            raise ValueError("model_kind must be 'auto', 'multimodal', or 'causal'")
        if dtype is not None:
            try:
                selected_dtype = getattr(torch, dtype)
            except AttributeError as error:
                raise ValueError(f"unknown torch dtype: {dtype}") from error
            transformers_major = int(str(transformers.__version__).split(".", maxsplit=1)[0])
            kwargs["dtype" if transformers_major >= 5 else "torch_dtype"] = selected_dtype
        if dequantize_mxfp4:
            if "quantization_config" in kwargs:
                raise ValueError(
                    "dequantize_mxfp4 cannot be combined with an explicit quantization_config"
                )
            mxfp4_config = getattr(transformers, "Mxfp4Config", None)
            if mxfp4_config is None:
                raise OptionalDependencyError(
                    "GPT-OSS MXFP4 dequantization requires Transformers with Mxfp4Config"
                )
            kwargs["quantization_config"] = mxfp4_config(dequantize=True)
        config_kwargs = {
            key: kwargs[key]
            for key in ("revision", "token", "trust_remote_code", "cache_dir", "local_files_only")
            if key in kwargs
        }
        config = transformers.AutoConfig.from_pretrained(name, **config_kwargs)
        inferred_multimodal = "qwen3_5" in str(getattr(config, "model_type", "")).casefold()
        inferred_multimodal = inferred_multimodal or "qwen3.5" in name.casefold()
        selected_kind = (
            ("multimodal" if inferred_multimodal else "causal")
            if model_kind == "auto"
            else model_kind
        )
        if selected_kind == "multimodal":
            model_class = getattr(transformers, "AutoModelForMultimodalLM", None)
            if model_class is None:
                raise OptionalDependencyError(
                    "Qwen3.5 requires AutoModelForMultimodalLM; install Transformers >=5.13"
                )
        else:
            model_class = transformers.AutoModelForCausalLM
        model = model_class.from_pretrained(name, config=config, **kwargs)
        if device is not None:
            model.to(device)
        return cls(
            model,
            device=device,
            provenance={
                "name": name,
                "revision": kwargs.get("revision"),
                "commit": getattr(config, "_commit_hash", None),
                "model_kind": selected_kind,
                "dtype": dtype,
                "dequantize_mxfp4": dequantize_mxfp4,
            },
        )

    @property
    def fingerprint(self) -> str:
        config = getattr(self.model, "config", None)
        return _fingerprint(
            {
                "adapter": "transformers-causal-lm-v1",
                "class": self.model.__class__.__name__,
                "name": getattr(config, "_name_or_path", self.model.__class__.__name__),
                "architectures": getattr(config, "architectures", None),
                "vocab_size": getattr(config, "vocab_size", None),
                "config": config.to_dict() if callable(getattr(config, "to_dict", None)) else None,
                "provenance": self.provenance,
            }
        )

    def train(self) -> None:
        self._training = True
        if self.adapter_name is None:
            self.model.train()
        else:
            self._prepare_forward()

    def eval(self) -> None:
        self._training = False
        if self.adapter_name is None:
            self.model.eval()
        else:
            self._activate_adapter()
            self.model.eval()

    def parameters(self) -> Iterable[Any]:
        if self.adapter_name is not None:
            self._activate_adapter()
            return tuple(
                parameter
                for parameter in self.model.parameters()
                if getattr(parameter, "requires_grad", False)
            )
        return self.model.parameters()

    def _model_device(self) -> Any:
        torch = _require_torch()
        if self.device is not None:
            return torch.device(self.device)
        try:
            return next(iter(self.model.parameters())).device
        except StopIteration:
            return torch.device("cpu")

    def _action_logits(self, example: TokenizedAction) -> Any:
        torch = _require_torch()
        self._prepare_forward()
        sequence = example.prompt_ids + example.action_ids
        input_ids = torch.tensor([sequence], dtype=torch.long, device=self._model_device())
        output = self.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        return output.logits[0, len(example.prompt_ids) - 1 : len(sequence) - 1]

    @staticmethod
    def _mask_action_logits(example: TokenizedAction, logits: Any, *, offset: int = 0) -> Any:
        """Apply the same legal support at each sampled history for both policies."""

        if example.prefix_mask is None:
            return logits.float()
        torch = _require_torch()
        rows = []
        for index, row in enumerate(logits, start=offset):
            prefix = example.action_ids[:index]
            allowed = example.prefix_mask.allowed_next(prefix)
            forbidden = example.prefix_mask.forbidden_next(prefix)
            if allowed is not None and example.action_ids[index] not in allowed:
                raise ValueError("action target diverges from its feasible canonical prefix")
            if example.action_ids[index] in forbidden:
                raise ValueError("action target uses an illegal reasoning-channel control token")
            rows.append(_masked_logit_row(row, allowed, example.prefix_mask, forbidden))
        return torch.stack(rows)

    def _action_log_probs(
        self,
        examples: Sequence[TokenizedAction],
        *,
        requires_grad: bool,
    ) -> list[Any]:
        torch = _require_torch()
        if not examples:
            raise ValueError("an action-token loss requires at least one example")
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        result: list[Any] = []
        with context:
            for example in examples:
                normalized_logits = self._mask_action_logits(example, self._action_logits(example))
                targets = torch.tensor(
                    example.action_ids, dtype=torch.long, device=self._model_device()
                )
                log_probs = torch.log_softmax(normalized_logits, dim=-1)
                result.append(log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1))
        return result

    def binary_answer_probability(self, position: BinaryAnswerPosition) -> float:
        """Return binary-softmax ``p(class=1)`` at a validated Answer position."""

        if not isinstance(position, BinaryAnswerPosition):
            raise TypeError("position must be a BinaryAnswerPosition")
        torch = _require_torch()
        input_ids = torch.tensor(
            [position.context_ids],
            dtype=torch.long,
            device=self._model_device(),
        )
        attention_mask = torch.ones_like(input_ids)
        self.eval()
        with torch.no_grad():
            self._prepare_forward()
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            next_logits = output.logits[0, -1].float()
            width = int(next_logits.shape[-1])
            if max(position.negative_token_id, position.positive_token_id) >= width:
                raise ValueError("binary class token is outside the model vocabulary")
            selected = torch.stack(
                (
                    next_logits[position.negative_token_id],
                    next_logits[position.positive_token_id],
                )
            )
            positive_log_probability = torch.log_softmax(selected, dim=-1)[1]
        probability = math.exp(float(positive_log_probability.detach().cpu().item()))
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError("model returned an invalid binary Answer probability")
        return probability

    @staticmethod
    def _metric(value: Any) -> float:
        return float(value.detach().float().cpu().item())

    def sft_loss(self, examples: Sequence[TokenizedAction]) -> LossReport:
        torch = _require_torch()
        log_probs = self._action_log_probs(examples, requires_grad=True)
        token_count = sum(item.numel() for item in log_probs)
        loss = -torch.cat(log_probs).mean()
        value = self._metric(loss)
        return LossReport(
            total=value,
            policy=value,
            kl=0.0,
            token_count=token_count,
            _backward=lambda scale: (loss * scale).backward(),
        )

    def vapa_loss(
        self,
        examples: Sequence[VAPAAction],
        *,
        reference: ActorModelAdapter,
        kl_weight: float,
        ratio_clip: float | None,
        kl_mode: str,
    ) -> LossReport:
        torch = _require_torch()
        if not isinstance(reference, TransformersActorAdapter):
            raise TypeError("the Transformers actor requires a Transformers reference adapter")
        shared_backbone = reference.model is self.model
        if shared_backbone and (
            self.adapter_name is None
            or reference.adapter_name is None
            or self.adapter_name == reference.adapter_name
            or not self.adapter_trainable
            or reference.adapter_trainable
        ):
            raise ValueError(
                "a shared actor/reference backbone requires distinct trainable/frozen "
                "named adapters"
            )
        if not math.isfinite(kl_weight) or kl_weight < 0:
            raise ValueError("kl_weight must be finite and nonnegative")
        if ratio_clip is not None:
            raise ValueError("Eq. 9 uses a log-policy objective without ratio clipping")
        if kl_mode != "forward":
            raise ValueError("Eq. 9 requires full-vocabulary forward KL")
        if not examples:
            raise ValueError("an action-token loss requires at least one example")
        from torch.utils.checkpoint import checkpoint

        policy_terms: list[Any] = []
        kl_terms: list[Any] = []
        token_count = sum(example.token_count for example in examples)
        for example in examples:
            try:
                with torch.no_grad():
                    reference_logits = reference._action_logits(example.tokens)
            finally:
                # Shared PEFT activation is global: restore the trainable actor before
                # its forward pass and the deferred backward callback.
                if shared_backbone:
                    self._prepare_forward()
            actor_logits = self._action_logits(example.tokens)
            if actor_logits.shape != reference_logits.shape:
                raise ValueError("actor and reference must use the same vocabulary")
            for offset in range(0, example.token_count, 64):
                stop = min(offset + 64, example.token_count)

                def chunk_loss(
                    actor_chunk: Any,
                    reference_chunk: Any,
                    *,
                    item: VAPAAction = example,
                    start: int = offset,
                    end: int = stop,
                ) -> tuple[Any, Any]:
                    # FP32 renormalization on identical grammar support. Bind the
                    # history/offset here: checkpoint recomputes this after the loop.
                    actor_log = torch.log_softmax(
                        self._mask_action_logits(item.tokens, actor_chunk, offset=start), dim=-1
                    )
                    reference_log = torch.log_softmax(
                        self._mask_action_logits(item.tokens, reference_chunk, offset=start), dim=-1
                    )
                    targets = torch.tensor(
                        item.tokens.action_ids[start:end], dtype=torch.long, device=actor_log.device
                    )
                    sampled_log = actor_log.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                    # Excluded vocabulary entries have zero mass. Zero their log terms
                    # before subtraction to avoid NaN from (-inf)-(-inf) in autograd.
                    excluded = torch.isneginf(actor_log)
                    difference = actor_log.masked_fill(excluded, 0.0) - reference_log.masked_fill(
                        excluded, 0.0
                    )
                    return (
                        -item.advantage * sampled_log.sum(),
                        (actor_log.exp() * difference).sum(),
                    )

                # Recompute only the vocabulary normalization during backward instead
                # of retaining all FP32 token-by-vocabulary distributions at once.
                policy_chunk, kl_chunk = checkpoint(
                    chunk_loss,
                    actor_logits[offset:stop],
                    reference_logits[offset:stop].to(device=actor_logits.device),
                    use_reentrant=False,
                )
                policy_terms.append(policy_chunk)
                kl_terms.append(kl_chunk)
        policy = torch.stack(policy_terms).sum() / token_count
        kl = torch.stack(kl_terms).sum() / token_count
        total = policy + kl_weight * kl
        return LossReport(
            total=self._metric(total),
            policy=self._metric(policy),
            kl=self._metric(kl),
            token_count=token_count,
            _backward=lambda scale: (total * scale).backward(),
        )

    def clip_grad_norm(self, max_norm: float) -> float:
        torch = _require_torch()
        if not math.isfinite(max_norm) or max_norm <= 0:
            raise ValueError("max_norm must be finite and positive")
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
        return float(norm.detach().float().cpu().item())

    def state_dict(self) -> Mapping[str, Any]:
        if self.adapter_name is not None:
            peft = _require_peft()
            state = peft.get_peft_model_state_dict(
                self.model,
                adapter_name=self.adapter_name,
                save_embedding_layers=False,
            )
            if not isinstance(state, Mapping) or not state:
                raise RuntimeError("PEFT returned an empty or malformed adapter state")
            return state
        return self.model.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if self.adapter_name is not None:
            if not isinstance(state, Mapping) or not state:
                raise ValueError("adapter state must be a non-empty mapping")
            peft = _require_peft()
            expected = peft.get_peft_model_state_dict(
                self.model,
                adapter_name=self.adapter_name,
                save_embedding_layers=False,
            )
            if set(state) != set(expected):
                raise ValueError("adapter state keys do not match the configured LoRA topology")
            result = peft.set_peft_model_state_dict(
                self.model,
                dict(state),
                adapter_name=self.adapter_name,
            )
            if getattr(result, "unexpected_keys", ()):
                raise ValueError("adapter state contains unexpected PEFT parameter keys")
            self._activate_adapter()
            return
        self.model.load_state_dict(state)


class TransformersGenerationBackend:
    """Seeded text generation that records the sampled action-token probabilities."""

    def __init__(
        self,
        actor: TransformersActorAdapter,
        tokenizer: TransformersTokenizerAdapter,
        *,
        context_tokens: int = 32_768,
    ) -> None:
        if (
            isinstance(context_tokens, bool)
            or not isinstance(context_tokens, int)
            or context_tokens < 1
        ):
            raise ValueError("context_tokens must be a positive integer")
        self.actor = actor
        self.tokenizer = tokenizer
        self.context_tokens = context_tokens

    @staticmethod
    def _validate_generation(
        *,
        n: int,
        seed: int,
        greedy: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        allowed_actions: Sequence[ActionKind],
    ) -> None:
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        if greedy and n != 1:
            raise ValueError("greedy generation supports exactly one candidate")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(greedy, bool):
            raise TypeError("greedy must be a boolean")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if not allowed_actions:
            raise ValueError("generation requires at least one legal action")

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        n: int,
        seed: int,
        greedy: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        allowed_actions: tuple[ActionKind, ...],
    ) -> list[GeneratedCandidate]:
        self._validate_generation(
            n=n,
            seed=seed,
            greedy=greedy,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            allowed_actions=allowed_actions,
        )
        torch = _require_torch()
        prompt_ids = self.tokenizer.encode_messages(messages, add_generation_prompt=True)
        prefix_mask = self.tokenizer.action_prefix_mask(messages, allowed_actions)
        if len(prompt_ids) + max_tokens > self.context_tokens:
            raise ValueError(
                "tokenized prompt plus max_tokens exceeds the configured context window"
            )
        device = self.actor._model_device()
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        pad_token_id = getattr(self.tokenizer.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer.tokenizer, "eos_token_id", None)
        generation_kwargs: dict[str, object] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_tokens,
            "do_sample": not greedy,
            "num_return_sequences": n,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if prefix_mask.reasoning_start:
            generation_kwargs["logits_processor"] = [
                _FeasibleActionLogitsProcessor(prompt_ids, prefix_mask)
            ]
        else:
            generation_kwargs["prefix_allowed_tokens_fn"] = _prefix_allowed_tokens_function(
                prompt_ids, prefix_mask
            )
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        if not greedy:
            if temperature == 0:
                raise ValueError("sampled generation requires temperature > 0")
            generation_kwargs.update(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        self.actor.eval()
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                output = self.actor.model.generate(**generation_kwargs)
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        scores = tuple(output.scores)
        if not scores:
            raise RuntimeError("the model generated no actor tokens")
        if output.sequences.shape[0] != n:
            raise RuntimeError("the model returned an unexpected number of sequences")
        eos = getattr(self.tokenizer.tokenizer, "eos_token_id", None)
        eos_ids = {int(item) for item in eos} if isinstance(eos, list | tuple) else {eos}
        eos_ids.discard(None)
        pad = getattr(self.tokenizer.tokenizer, "pad_token_id", None)
        candidates: list[GeneratedCandidate] = []
        for row in range(n):
            generated = [int(item) for item in output.sequences[row, len(prompt_ids) :]]
            effective_length = min(len(generated), len(scores))
            for index, token in enumerate(generated[:effective_length]):
                if token in eos_ids:
                    effective_length = index + 1
                    break
                if pad is not None and token == pad:
                    effective_length = index
                    break
            generated = generated[:effective_length]
            if not generated:
                raise RuntimeError("the model generated an empty candidate")
            log_prob_values: list[float] = []
            for index, token in enumerate(generated):
                allowed = prefix_mask.allowed_next(generated[:index])
                forbidden = prefix_mask.forbidden_next(generated[:index])
                if allowed is not None and token not in allowed:
                    raise RuntimeError("generation escaped the feasible-action prefix mask")
                if token in forbidden:
                    raise RuntimeError("generation used an illegal reasoning control token")
                normalized = torch.log_softmax(
                    _masked_logit_row(
                        scores[index][row],
                        allowed,
                        prefix_mask,
                        forbidden,
                    ),
                    dim=-1,
                )
                log_prob_values.append(float(normalized[token].detach().cpu().item()))
            log_probs = tuple(log_prob_values)
            text = self.tokenizer.decode_generated_action(generated, prefix_mask)
            if not text:
                raise RuntimeError("the masked generation has no parser-visible action text")
            candidates.append(GeneratedCandidate(text, tuple(generated), log_probs))
        return candidates

    def binary_answer_probability(
        self,
        messages: list[dict[str, str]],
        *,
        action_ids: Sequence[int] | None = None,
    ) -> float:
        """Score class 1 at a direct or sampled binary Answer prediction position."""

        position = self.tokenizer.binary_answer_position(messages, action_ids)
        if len(position.context_ids) >= self.context_tokens:
            raise ValueError("binary Answer scoring position exceeds the context window")
        return self.actor.binary_answer_probability(position)


def _lora_config(
    *,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: Sequence[str] | None,
) -> tuple[Any, Any]:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("LoRA rank must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, int) or alpha < 1:
        raise ValueError("LoRA alpha must be a positive integer")
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("LoRA dropout must be in [0, 1)")
    if target_modules is not None and (
        not target_modules or any(not isinstance(item, str) or not item for item in target_modules)
    ):
        raise ValueError("target_modules must contain non-empty module names")
    peft = _require_peft()
    config = peft.LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=None if target_modules is None else list(target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return peft, config


def apply_lora(
    actor: TransformersActorAdapter,
    *,
    rank: int,
    alpha: int,
    dropout: float = 0.0,
    target_modules: Sequence[str] | None = None,
) -> TransformersActorAdapter:
    """Attach one checkpoint-compact PEFT LoRA adapter for SFT/inference."""

    if actor.adapter_name is not None:
        raise ValueError("the actor already addresses a named PEFT adapter")
    peft, config = _lora_config(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
    )
    actor.model = peft.get_peft_model(actor.model, config)
    actor.adapter_name = "default"
    actor.adapter_trainable = True
    actor.provenance["lora"] = {
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "target_modules": None if target_modules is None else list(target_modules),
    }
    actor._activate_adapter()
    return actor


def apply_shared_lora(
    actor: TransformersActorAdapter,
    *,
    rank: int,
    alpha: int,
    dropout: float = 0.0,
    target_modules: Sequence[str] | None = None,
    actor_adapter_name: str = "actor",
    reference_adapter_name: str = "reference",
) -> tuple[TransformersActorAdapter, TransformersActorAdapter]:
    """Create actor/reference LoRA snapshots on exactly one frozen base backbone.

    The returned wrappers are distinct policy snapshots but share the heavyweight
    model object.  Forward calls activate their named adapter sequentially.  Their
    ``state_dict`` surface contains only the selected adapter, so SFT initialization
    and RL checkpoints never duplicate the frozen base weights.
    """

    if actor.adapter_name is not None:
        raise ValueError("the actor already addresses a named PEFT adapter")
    for value, name in (
        (actor_adapter_name, "actor_adapter_name"),
        (reference_adapter_name, "reference_adapter_name"),
    ):
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"{name} must be a non-empty, trimmed string")
    if actor_adapter_name == reference_adapter_name:
        raise ValueError("actor and reference adapter names must be distinct")
    peft, config = _lora_config(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
    )
    _, reference_config = _lora_config(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
    )
    shared_model = peft.get_peft_model(
        actor.model,
        config,
        adapter_name=actor_adapter_name,
    )
    add_adapter = getattr(shared_model, "add_adapter", None)
    if not callable(add_adapter):
        raise TypeError("PEFT shared LoRA models must provide add_adapter()")
    add_adapter(reference_adapter_name, reference_config)
    provenance = dict(actor.provenance)
    provenance["lora"] = {
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "target_modules": None if target_modules is None else list(target_modules),
    }
    actor.model = shared_model
    actor.provenance = dict(provenance)
    actor.adapter_name = actor_adapter_name
    actor.adapter_trainable = True
    reference = TransformersActorAdapter(
        shared_model,
        device=actor.device,
        provenance=provenance,
        adapter_name=reference_adapter_name,
        adapter_trainable=False,
    )
    reference.load_state_dict(actor.state_dict())
    actor._activate_adapter()
    return actor, reference
