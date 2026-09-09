from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import vapa.model.transformers as transformers_backend
from vapa.actions import format_action, make_action
from vapa.model.protocols import (
    ActionPrefixMask,
    LossReport,
    OptionalDependencyError,
    TokenizedAction,
    VAPAAction,
)
from vapa.policies.base import PolicyDecision
from vapa.policies.text import TextPolicy
from vapa.rollouts import Rollout, Turn
from vapa.schemas import ActionKind, Observation, ReturnCode, TaskSpec, ToolReturn
from vapa.training.checkpoint import (
    CheckpointContract,
    JsonStateStore,
    RuntimeState,
    fingerprint_payload,
    resume_checkpoint,
    save_checkpoint,
)
from vapa.training.runtime import (
    DistributedContext,
    SFTExample,
    WarmupCosineScheduler,
    collect_vapa_action_groups,
    seed_everything,
    train_sft_step,
    train_vapa_update,
)
from vapa.training.trainer import InstanceBatch, UpdateBatch


class FakeTokenizer:
    fingerprint = fingerprint_payload({"adapter": "fake-tokenizer-v1"})

    @staticmethod
    def _encode(text: str) -> tuple[int, ...]:
        return tuple(byte + 1 for byte in text.encode("utf-8")) or (1,)

    def encode_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool = True,
    ) -> tuple[int, ...]:
        rendered = "|".join(
            f"{message.get('role', '')}:{message.get('content', '')}" for message in messages
        )
        if add_generation_prompt:
            rendered += "|assistant:"
        return self._encode(rendered)

    def encode_text(self, text: str) -> tuple[int, ...]:
        return self._encode(text)

    def encode_action(
        self,
        messages: Sequence[Mapping[str, str]],
        action_text: str,
    ) -> TokenizedAction:
        return TokenizedAction(
            self.encode_messages(messages, add_generation_prompt=True),
            self.encode_text(action_text),
        )

    def save_pretrained(self, path: str) -> None:
        Path(path, "tokenizer.json").write_text("{}\n", encoding="utf-8")


def test_transformers_tokenizer_identity_binds_mapping_and_encoding_rules() -> None:
    class Backend:
        rule = "Whitespace"
        padding = None

        def to_str(self):
            return json.dumps({"pre_tokenizer": self.rule, "padding": self.padding})

    class Tokenizer:
        vocab_size = 2
        name_or_path = "same-name"
        vocab = {"a": 0, "b": 1}
        backend_tokenizer = Backend()

        def encode(self, text, **kwargs):
            return [self.vocab[text]]

        def get_vocab(self):
            return self.vocab

    tokenizer = Tokenizer()
    adapter = transformers_backend.TransformersTokenizerAdapter(tokenizer)
    original = adapter.fingerprint
    tokenizer.backend_tokenizer.padding = {"length": 128}
    assert adapter.fingerprint == original
    tokenizer.vocab = {"a": 1, "b": 0}
    assert adapter.fingerprint != original
    tokenizer.vocab = {"a": 0, "b": 1}
    tokenizer.backend_tokenizer.rule = "ByteLevel"
    assert adapter.fingerprint != original


class FakeModel:
    def __init__(self, weight: float, name: str = "actor") -> None:
        self.weight = weight
        self.grad = 0.0
        self.name = name
        self.mode = "eval"
        self.vapa_group_tokens: list[int] = []
        self.backward_scales: list[float] = []

    @property
    def fingerprint(self) -> str:
        return fingerprint_payload({"adapter": "fake-model-v1", "name": self.name})

    def train(self) -> None:
        self.mode = "train"

    def eval(self) -> None:
        self.mode = "eval"

    def parameters(self):
        return (self,)

    def sft_loss(self, examples: Sequence[TokenizedAction]) -> LossReport:
        token_count = sum(example.token_count for example in examples)
        total = self.weight * self.weight

        def backward(scale: float) -> None:
            self.backward_scales.append(scale)
            self.grad += scale * 2.0 * self.weight

        return LossReport(total, total, 0.0, token_count, backward)

    def vapa_loss(
        self,
        examples: Sequence[VAPAAction],
        *,
        reference: FakeModel,
        kl_weight: float,
        ratio_clip: float | None,
        kl_mode: str,
    ) -> LossReport:
        assert ratio_clip is None
        assert kl_mode == "forward"
        assert reference.mode == "eval"
        token_count = sum(example.token_count for example in examples)
        mean_advantage = (
            sum(example.advantage * example.token_count for example in examples) / token_count
        )
        policy = -self.weight * mean_advantage
        kl = (self.weight - reference.weight) ** 2
        total = policy + kl_weight * kl
        derivative = -mean_advantage + 2 * kl_weight * (self.weight - reference.weight)
        self.vapa_group_tokens.append(token_count)

        def backward(scale: float) -> None:
            self.backward_scales.append(scale)
            self.grad += scale * derivative

        return LossReport(total, policy, kl, token_count, backward)

    def clip_grad_norm(self, max_norm: float) -> float:
        norm = abs(self.grad)
        if norm > max_norm:
            self.grad *= max_norm / norm
        return norm

    def state_dict(self) -> Mapping[str, Any]:
        return {"weight": self.weight, "name": self.name}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.weight = float(state["weight"])
        self.name = str(state["name"])


class FakeOptimizer:
    def __init__(self, model: FakeModel, learning_rate: float = 0.1) -> None:
        self.model = model
        self.param_groups = [{"lr": learning_rate}]
        self.steps = 0

    def zero_grad(self) -> None:
        self.model.grad = 0.0

    def step(self) -> None:
        self.model.weight -= self.param_groups[0]["lr"] * self.model.grad
        self.steps += 1

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "steps": self.steps,
            "param_groups": [dict(group) for group in self.param_groups],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.steps = int(state["steps"])
        self.param_groups = [dict(group) for group in state["param_groups"]]


def _observation() -> Observation:
    task = TaskSpec(
        instance_id="train-1",
        patient_id="patient-1",
        instruction="Return the requested value.",
        cutoff=datetime(2025, 1, 10, tzinfo=UTC),
        family="synthetic",
    )
    return Observation(
        task=task,
        last_return=None,
        memory=(),
        history=(),
        budget_remaining=4,
        budget_cap=4,
        turn=0,
        turn_cap=8,
        memory_capacity=8,
        legal_actions=(ActionKind.QUERY_FIELD, ActionKind.ANSWER),
    )


def _update() -> UpdateBatch:
    observation = _observation()
    specifications = (
        ("g1", (11, 12), (-0.2, -0.3), 1.0),
        ("g1", (13,), (-0.4,), -0.5),
        ("g2", (14, 15), (-0.1, -0.1), 0.25),
    )
    turns = []
    for index, (group, tokens, log_probs, advantage) in enumerate(specifications):
        action = make_action(ActionKind.QUERY_FIELD, field="value", window="all")
        turns.append(
            Turn(
                index=index,
                observation=observation,
                decision=PolicyDecision(
                    action=action,
                    token_count=len(tokens),
                    token_ids=tokens,
                    behavior_log_probs=log_probs,
                ),
                action=action,
                tool_return=ToolReturn(ReturnCode.OK),
                cost=1,
                accepted=True,
                group_id=group,
                group_tier="test",
                normalized_advantage=advantage,
            )
        )
    rollout = Rollout("rollout-1", "train-1", turns, sampled_tokens=5)
    instance = InstanceBatch("train-1", "occurrence-1", (rollout,), (), (), None)
    return UpdateBatch((instance,), None)


def test_dependency_light_vapa_and_action_sft_end_to_end():
    tokenizer = FakeTokenizer()
    actor = FakeModel(0.5)
    reference = FakeModel(0.25, "reference")
    optimizer = FakeOptimizer(actor)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=4,
        warmup_steps=1,
        final_lr_fraction=0.1,
    )

    groups = collect_vapa_action_groups(_update(), tokenizer)
    assert [(group.group_id, group.token_count) for group in groups] == [("g1", 3), ("g2", 2)]
    before = actor.weight
    vapa = train_vapa_update(
        _update(),
        tokenizer=tokenizer,
        actor=actor,
        reference=reference,
        optimizer=optimizer,
        scheduler=scheduler,
        kl_weight=0.01,
    )

    assert vapa.objective == "vapa"
    assert vapa.action_tokens == 5
    assert vapa.microbatches == 2
    assert actor.vapa_group_tokens == [3, 2]
    assert actor.backward_scales == pytest.approx([0.6, 0.4])
    assert actor.weight != before
    assert optimizer.steps == 1

    actor.backward_scales.clear()
    sft = train_sft_step(
        (
            SFTExample(({"role": "user", "content": "one"},), "AB", "same"),
            SFTExample(({"role": "user", "content": "two"},), "C", "same"),
            SFTExample(({"role": "user", "content": "three"},), "DE"),
        ),
        tokenizer=tokenizer,
        actor=actor,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    assert sft.objective == "sft"
    assert sft.action_tokens == 5
    assert sft.microbatches == 2
    assert actor.backward_scales == pytest.approx([0.6, 0.4])
    assert optimizer.steps == 2


@pytest.mark.parametrize("objective", ["sft", "vapa"])
def test_training_metrics_record_the_applied_learning_rate(objective: str):
    actor = FakeModel(0.5)
    optimizer = FakeOptimizer(actor, learning_rate=0.1)
    scheduler = WarmupCosineScheduler(optimizer, total_steps=4, warmup_steps=2)
    applied = tuple(group["lr"] for group in optimizer.param_groups)
    kwargs = dict(tokenizer=FakeTokenizer(), actor=actor, optimizer=optimizer, scheduler=scheduler)
    if objective == "sft":
        report = train_sft_step(
            (SFTExample(({"role": "user", "content": "prompt"},), "AB"),), **kwargs
        )
    else:
        report = train_vapa_update(_update(), reference=FakeModel(0.25, "reference"), **kwargs)
    assert report.learning_rates == applied == (0.05,)
    assert optimizer.param_groups[0]["lr"] == 0.1


def test_checkpoint_round_trip_verifies_contract_and_integrity(tmp_path: Path):
    model = FakeModel(0.75)
    reference = FakeModel(0.25, "reference")
    tokenizer = FakeTokenizer()
    optimizer = FakeOptimizer(model)
    optimizer.steps = 3
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=10,
        warmup_steps=2,
        final_lr_fraction=0.1,
    )
    scheduler.step()
    contract = CheckpointContract(
        run_id="run-1",
        run_manifest_fingerprint=fingerprint_payload({"run": "manifest"}),
        config_fingerprint=fingerprint_payload({"seed": 7}),
        model_fingerprint=model.fingerprint,
        reference_model_fingerprint=reference.fingerprint,
        tokenizer_fingerprint=FakeTokenizer.fingerprint,
        optimizer_name="fake",
        scheduler_name="warmup-cosine",
        state_format="json-v1",
    )
    runtime = RuntimeState(3, 100, 80, 7, {"curriculum": "middle"})
    path = tmp_path / "step-3"
    save_checkpoint(
        path,
        contract=contract,
        runtime=runtime,
        model=model,
        reference=reference,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        store=JsonStateStore(),
    )

    model.weight = 99.0
    optimizer.steps = 99
    scheduler.step()
    loaded = resume_checkpoint(
        path,
        expected=contract,
        model=model,
        reference=reference,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        store=JsonStateStore(),
    )
    assert loaded == runtime
    assert model.weight == 0.75
    assert optimizer.steps == 3
    assert scheduler.completed_steps == 1

    mismatched = CheckpointContract(
        **{
            **contract.__dict__,
            "tokenizer_fingerprint": fingerprint_payload({"adapter": "different-tokenizer"}),
        }
    )
    model.weight = 12.0
    with pytest.raises(ValueError, match="contract"):
        resume_checkpoint(
            path,
            expected=mismatched,
            model=model,
            reference=reference,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            store=JsonStateStore(),
        )
    assert model.weight == 12.0

    model_file = next(path.glob("model.*"))
    model_file.write_text(json.dumps({"weight": -1, "name": "bad"}), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        resume_checkpoint(
            path,
            expected=contract,
            model=model,
            reference=reference,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            store=JsonStateStore(),
        )
    assert model.weight == 12.0


def test_scheduler_seed_and_distributed_helpers(monkeypatch: pytest.MonkeyPatch):
    model = FakeModel(1.0)
    optimizer = FakeOptimizer(model, learning_rate=1.0)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=4,
        warmup_steps=2,
        final_lr_fraction=0.1,
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)
    for _ in range(4):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)

    assert seed_everything(13, rank=2) == 15
    first = random_value = __import__("random").random()
    seed_everything(13, rank=2)
    assert __import__("random").random() == random_value == first

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    context = DistributedContext.from_environment()
    assert context == DistributedContext(rank=1, local_rank=0, world_size=2)
    assert not context.is_main_process
    assert math.isfinite(optimizer.param_groups[0]["lr"])


def test_transformers_auto_loader_selects_qwen_multimodal_without_download(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, str]] = []

    class Config:
        def __init__(self, name: str) -> None:
            self.model_type = "qwen3_5" if "Qwen3.5" in name else "gpt_oss"
            self._commit_hash = "commit"

        def to_dict(self):
            return {"model_type": self.model_type}

    class AutoConfig:
        @staticmethod
        def from_pretrained(name: str, **kwargs):
            del kwargs
            return Config(name)

    class LoadedModel:
        def __init__(self, config: Config) -> None:
            self.config = config

        def __call__(self, **kwargs):
            return kwargs

        def to(self, device: str) -> None:
            del device

    class Multimodal:
        @staticmethod
        def from_pretrained(name: str, *, config: Config, **kwargs):
            del kwargs
            calls.append((name, "multimodal"))
            return LoadedModel(config)

    class Causal:
        @staticmethod
        def from_pretrained(name: str, *, config: Config, **kwargs):
            del kwargs
            calls.append((name, "causal"))
            return LoadedModel(config)

    class Processor:
        chat_template = "fake-template"

        class Tokenizer:
            name_or_path = "fake-qwen-tokenizer"

            @staticmethod
            def encode(text: str, *, add_special_tokens: bool):
                del add_special_tokens
                return [ord(character) for character in text]

            @staticmethod
            def decode(tokens, *, skip_special_tokens: bool):
                del skip_special_tokens
                return "".join(chr(token) for token in tokens)

        tokenizer = Tokenizer()

        @staticmethod
        def apply_chat_template(messages, *, tokenize, add_generation_prompt, **kwargs):
            assert tokenize
            assert kwargs == {"enable_thinking": False}
            if add_generation_prompt:
                return [1, 2]
            assert messages[-1]["role"] == "assistant"
            return [1, 2, 3]

    class AutoProcessor:
        @staticmethod
        def from_pretrained(name: str, **kwargs):
            del kwargs
            calls.append((name, "processor"))
            return Processor()

    fake_transformers = type(
        "FakeTransformers",
        (),
        {
            "__version__": "5.13.0",
            "AutoConfig": AutoConfig,
            "AutoModelForMultimodalLM": Multimodal,
            "AutoModelForCausalLM": Causal,
            "AutoProcessor": AutoProcessor,
        },
    )
    fake_torch = type("FakeTorch", (), {})
    monkeypatch.setattr(transformers_backend, "_require_transformers", lambda: fake_transformers)
    monkeypatch.setattr(transformers_backend, "_require_torch", lambda: fake_torch)

    qwen = transformers_backend.TransformersActorAdapter.from_pretrained("Qwen/Qwen3.5-9B")
    gpt = transformers_backend.TransformersActorAdapter.from_pretrained("openai/gpt-oss-20b")
    assert calls == [
        ("Qwen/Qwen3.5-9B", "multimodal"),
        ("openai/gpt-oss-20b", "causal"),
    ]
    assert qwen.provenance["model_kind"] == "multimodal"
    assert gpt.provenance["model_kind"] == "causal"

    tokenizer = transformers_backend.TransformersTokenizerAdapter.from_pretrained(
        "Qwen/Qwen3.5-9B",
        chat_template_kwargs={"enable_thinking": False},
    )
    encoded = tokenizer.encode_action(({"role": "user", "content": "question"},), "answer")
    assert encoded == TokenizedAction((1, 2), (3,))
    assert calls[-1] == ("Qwen/Qwen3.5-9B", "processor")

    delattr(fake_transformers, "AutoModelForMultimodalLM")
    with pytest.raises(OptionalDependencyError, match="Transformers >=5.13"):
        transformers_backend.TransformersActorAdapter.from_pretrained("Qwen/Qwen3.5-9B")


class _NumericTensor:
    """Small nested-list tensor used to test masking without importing torch."""

    def __init__(self, data):
        self.data = data
        self.device = "cpu"

    @property
    def shape(self):
        dimensions = []
        value = self.data
        while isinstance(value, list):
            dimensions.append(len(value))
            value = value[0] if value else None
        return tuple(dimensions)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for value in self.data:
            yield _NumericTensor(value) if isinstance(value, list) else value

    def __getitem__(self, key):
        if isinstance(key, tuple):
            value = self
            for item in key:
                value = value[item]
            return value
        value = self.data[key]
        return _NumericTensor(value)

    def __setitem__(self, key, value):
        assigned = value.data if isinstance(value, _NumericTensor) else value
        if isinstance(key, list):
            for index in key:
                self.data[index] = assigned
            return
        self.data[key] = assigned

    def float(self):
        return _NumericTensor(_map_nested(self.data, float))

    def masked_fill(self, mask, value):
        def fill(data, flags):
            if isinstance(data, list):
                return [fill(item, flag) for item, flag in zip(data, flags, strict=True)]
            return value if flags else data

        return _NumericTensor(fill(self.data, mask.data))

    def unsqueeze(self, dimension):
        assert dimension == -1
        return _NumericTensor([[item] for item in self.data])

    def squeeze(self, dimension):
        assert dimension == -1
        return _NumericTensor([item[0] for item in self.data])

    def gather(self, dimension, indices):
        assert dimension == -1
        return _NumericTensor(
            [[row[index_row[0]]] for row, index_row in zip(self.data, indices.data, strict=True)]
        )

    def tolist(self):
        return self.data

    def detach(self):
        return self

    def cpu(self):
        return self

    def item(self):
        return self.data

    def numel(self):
        def count(value):
            return sum(count(item) for item in value) if isinstance(value, list) else 1

        return count(self.data)


def _map_nested(value, function):
    if isinstance(value, list | tuple):
        return [_map_nested(item, function) for item in value]
    return function(value)


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


class _FakeTorch:
    long = "long"
    bool = "bool"

    class random:
        @staticmethod
        def get_rng_state():
            return "state"

        @staticmethod
        def set_rng_state(_state):
            return None

    class cuda:
        @staticmethod
        def is_available():
            return False

    @staticmethod
    def device(value):
        return value

    @staticmethod
    def tensor(data, **_kwargs):
        return _NumericTensor(_map_nested(data, lambda value: value))

    @staticmethod
    def ones_like(tensor, **_kwargs):
        return _NumericTensor(_map_nested(tensor.data, lambda _value: True))

    @staticmethod
    def zeros_like(tensor, **_kwargs):
        return _NumericTensor(_map_nested(tensor.data, lambda _value: False))

    @staticmethod
    def stack(rows):
        return _NumericTensor([row.data for row in rows])

    @staticmethod
    def log_softmax(tensor, *, dim):
        assert dim == -1

        def normalize(row):
            if row and isinstance(row[0], list):
                return [normalize(item) for item in row]
            finite = [value for value in row if math.isfinite(value)]
            peak = max(finite)
            normalizer = peak + math.log(sum(math.exp(value - peak) for value in finite))
            return [value - normalizer if math.isfinite(value) else float("-inf") for value in row]

        return _NumericTensor(normalize(tensor.data))

    @staticmethod
    def no_grad():
        return _NoGrad()

    @staticmethod
    def enable_grad():
        return _NoGrad()

    @staticmethod
    def manual_seed(_seed):
        return None


class _CharacterTokenizer:
    vocab_size = 128
    all_special_ids = ()
    eos_token_id = None
    pad_token_id = None
    bos_token_id = None
    unk_token_id = None

    def __len__(self):
        return self.vocab_size

    @staticmethod
    def encode(text, *, add_special_tokens):
        del add_special_tokens
        return list(text.encode("ascii"))

    @staticmethod
    def decode(tokens, *, skip_special_tokens, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        return bytes(tokens).decode("ascii")


class _GPTOSSChannelTokenizer:
    """Dependency-free shape of the pinned GPT-OSS assistant channel template."""

    vocab_size = 256
    chat_template = 'gpt-oss-channel-template: if "thinking" in message'
    _assistant = 200
    _final = 201
    _message = 202
    _return = 203
    _user = 204
    _end = 205
    _analysis = 206
    all_special_ids = (_assistant, _final, _message, _return, _user, _end, _analysis)
    eos_token_id = None
    pad_token_id = None
    bos_token_id = None
    unk_token_id = None

    def __len__(self):
        return self.vocab_size

    @staticmethod
    def encode(text, *, add_special_tokens):
        del add_special_tokens
        return list(text.encode("ascii"))

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        reasoning_effort,
    ):
        assert tokenize and reasoning_effort == "low"
        rendered = []
        for index, message in enumerate(messages):
            if message["role"] == "user":
                rendered.extend((self._user, *message["content"].encode("ascii"), self._end))
            elif message["role"] == "assistant":
                assert index == len(messages) - 1 and not add_generation_prompt
                rendered.append(self._assistant)
                if "thinking" in message:
                    rendered.extend(
                        (
                            self._analysis,
                            self._message,
                            *message["thinking"].encode("ascii"),
                            self._end,
                            self._assistant,
                        )
                    )
                rendered.extend(
                    (
                        self._final,
                        self._message,
                        *message["content"].encode("ascii"),
                        self._return,
                    )
                )
            else:  # pragma: no cover - this focused fake only needs one user turn
                raise AssertionError(message["role"])
        if add_generation_prompt:
            rendered.append(self._assistant)
        return rendered

    @classmethod
    def decode(cls, tokens, *, skip_special_tokens, clean_up_tokenization_spaces=False):
        del clean_up_tokenization_spaces
        values = list(tokens)
        if skip_special_tokens:
            values = [value for value in values if value not in cls.all_special_ids]
            return bytes(values).decode("ascii")
        names = {
            cls._assistant: "<|start-assistant|>",
            cls._final: "<|channel-final|>",
            cls._message: "<|message|>",
            cls._return: "<|return|>",
            cls._user: "<|start-user|>",
            cls._end: "<|end|>",
            cls._analysis: "<|channel-analysis|>",
        }
        return "".join(names.get(value, chr(value)) for value in values)


class _QwenClosedThinkingTokenizer:
    """Faithful boundary shape for Qwen3.5 with ``enable_thinking=false``."""

    vocab_size = 256
    chat_template = "qwen-template using enable_thinking"
    _user = 220
    _end = 221
    _assistant = 222
    _think_start = 223
    _think_end = 224
    all_special_ids = (_user, _end, _assistant, _think_start, _think_end)
    eos_token_id = None
    pad_token_id = None
    bos_token_id = None
    unk_token_id = None
    _tags = {
        "<U>": _user,
        "<E>": _end,
        "<A>": _assistant,
        "<TS>": _think_start,
        "<TE>": _think_end,
    }

    def __len__(self):
        return self.vocab_size

    @classmethod
    def encode(cls, text, *, add_special_tokens):
        del add_special_tokens
        result = []
        index = 0
        while index < len(text):
            match = next((tag for tag in cls._tags if text.startswith(tag, index)), None)
            if match is None:
                result.append(ord(text[index]))
                index += 1
            else:
                result.append(cls._tags[match])
                index += len(match)
        return result

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert enable_thinking is False
        rendered = ""
        for index, message in enumerate(messages):
            if message["role"] == "user":
                rendered += f"<U>{message['content']}<E>"
            elif message["role"] == "assistant":
                assert index == len(messages) - 1 and not add_generation_prompt
                rendered += f"<A>{message['content']}<E>"
            else:  # pragma: no cover - this focused fake only needs one user turn
                raise AssertionError(message["role"])
        if add_generation_prompt:
            rendered += "<A><TS><TE>"
        return self.encode(rendered, add_special_tokens=False) if tokenize else rendered

    @classmethod
    def decode(cls, tokens, *, skip_special_tokens, clean_up_tokenization_spaces=False):
        del clean_up_tokenization_spaces
        values = list(tokens)
        if skip_special_tokens:
            values = [value for value in values if value not in cls.all_special_ids]
            return bytes(values).decode("ascii")
        names = {value: key for key, value in cls._tags.items()}
        return "".join(names.get(value, chr(value)) for value in values)


def _logit_row(vocab_size):
    return [((index * 17) % 23) / 10.0 for index in range(vocab_size)]


def test_transformers_feasible_prefix_mask_matches_behavior_and_training(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(transformers_backend, "_require_torch", lambda: fake_torch)
    tokenizer = transformers_backend.TransformersTokenizerAdapter(_CharacterTokenizer())
    messages = [{"role": "user", "content": "question"}]
    prompt_ids = tokenizer.encode_messages(messages, add_generation_prompt=True)
    prefix_mask = tokenizer.action_prefix_mask(messages, (ActionKind.ANSWER,))
    legal_prefix = tokenizer.encode_text("Answer(")
    illegal_prefix = tokenizer.encode_text("Retrieve(")
    generated = legal_prefix + tokenizer.encode_text("bad)")
    raw_row = _logit_row(prefix_mask.vocab_size)

    first_support = prefix_mask.allowed_next(())
    assert first_support == (legal_prefix[0],)
    assert illegal_prefix[0] not in first_support
    first_distribution = fake_torch.log_softmax(
        transformers_backend._masked_logit_row(_NumericTensor(raw_row), first_support, prefix_mask),
        dim=-1,
    )
    assert first_distribution[illegal_prefix[0]].item() == float("-inf")
    assert first_distribution[legal_prefix[0]].item() == pytest.approx(0.0)

    class GenerateModel:
        config = type("Config", (), {"vocab_size": prefix_mask.vocab_size})()

        def __init__(self):
            self.callback = None

        def __call__(self, *, input_ids, attention_mask):
            del attention_mask
            length = input_ids.shape[-1]
            logits = [[list(raw_row) for _ in range(length)]]
            return type("Output", (), {"logits": _NumericTensor(logits)})()

        def generate(self, **kwargs):
            self.callback = kwargs["prefix_allowed_tokens_fn"]
            prompt = kwargs["input_ids"].data[0]
            emitted = []
            for token in generated:
                allowed = self.callback(0, _NumericTensor(prompt + emitted))
                assert token in allowed
                emitted.append(token)
            assert illegal_prefix[0] not in self.callback(0, _NumericTensor(prompt))
            scores = tuple(_NumericTensor([list(raw_row)]) for _ in emitted)
            sequences = _NumericTensor([prompt + emitted])
            return type("Output", (), {"scores": scores, "sequences": sequences})()

        def eval(self):
            return None

        def train(self):
            return None

        def parameters(self):
            return ()

    model = GenerateModel()
    actor = transformers_backend.TransformersActorAdapter(model, device="cpu")
    backend = transformers_backend.TransformersGenerationBackend(
        actor, tokenizer, context_tokens=4096
    )
    candidate = backend.generate(
        messages,
        n=1,
        seed=7,
        greedy=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        max_tokens=len(generated),
        allowed_actions=(ActionKind.ANSWER,),
    )[0]
    assert candidate.text.startswith("Answer(")
    assert candidate.text == "Answer(bad)"

    tokens = TokenizedAction(prompt_ids, generated, prefix_mask=prefix_mask)
    training_log_probs = actor._action_log_probs((tokens,), requires_grad=False)[0].tolist()
    assert candidate.log_probs == pytest.approx(training_log_probs)
    assert candidate.log_probs[: len(legal_prefix)] == pytest.approx([0.0] * len(legal_prefix))


def test_qwen_generation_only_thinking_preamble_keeps_action_boundary_stable(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(transformers_backend, "_require_torch", lambda: fake_torch)
    tokenizer = transformers_backend.TransformersTokenizerAdapter(
        _QwenClosedThinkingTokenizer(),
        chat_template_kwargs={"enable_thinking": False},
    )
    messages = [{"role": "user", "content": "question"}]
    prompt_ids = tokenizer.encode_messages(messages, add_generation_prompt=True)
    assert prompt_ids[-3:] == (
        _QwenClosedThinkingTokenizer._assistant,
        _QwenClosedThinkingTokenizer._think_start,
        _QwenClosedThinkingTokenizer._think_end,
    )
    prefix_mask = tokenizer.action_prefix_mask(messages, (ActionKind.ANSWER,))
    generated = tokenizer.encode_text("Answer(bad)")
    raw_row = _logit_row(prefix_mask.vocab_size)
    assert prefix_mask.action_prefixes == (tokenizer.encode_text("Answer("),)
    assert tokenizer.encode_action(messages, "Answer(ok, [])") == TokenizedAction(
        prompt_ids,
        tokenizer.encode_text("Answer(ok, [])") + (_QwenClosedThinkingTokenizer._end,),
    )

    class GenerateModel:
        config = type("Config", (), {"vocab_size": prefix_mask.vocab_size})()

        def __call__(self, *, input_ids, attention_mask):
            del attention_mask
            logits = [[list(raw_row) for _ in range(input_ids.shape[-1])]]
            return type("Output", (), {"logits": _NumericTensor(logits)})()

        def generate(self, **kwargs):
            callback = kwargs["prefix_allowed_tokens_fn"]
            prompt = kwargs["input_ids"].data[0]
            emitted = []
            for token in generated:
                assert token in callback(0, _NumericTensor(prompt + emitted))
                emitted.append(token)
            return type(
                "Output",
                (),
                {
                    "scores": tuple(_NumericTensor([list(raw_row)]) for _ in emitted),
                    "sequences": _NumericTensor([prompt + emitted]),
                },
            )()

        def eval(self):
            return None

        def train(self):
            return None

        def parameters(self):
            return ()

    actor = transformers_backend.TransformersActorAdapter(GenerateModel(), device="cpu")
    candidate = transformers_backend.TransformersGenerationBackend(
        actor, tokenizer, context_tokens=4096
    ).generate(
        messages,
        n=1,
        seed=9,
        greedy=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        max_tokens=len(generated),
        allowed_actions=(ActionKind.ANSWER,),
    )[0]

    assert candidate.text == "Answer(bad)"
    tokens = TokenizedAction(prompt_ids, generated, prefix_mask=prefix_mask)
    assert candidate.log_probs == pytest.approx(
        actor._action_log_probs((tokens,), requires_grad=False)[0].tolist()
    )


def test_gpt_oss_template_preamble_is_masked_trained_and_hidden_from_parser(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(transformers_backend, "_require_torch", lambda: fake_torch)
    raw_tokenizer = _GPTOSSChannelTokenizer()
    tokenizer = transformers_backend.TransformersTokenizerAdapter(
        raw_tokenizer,
        chat_template_kwargs={"reasoning_effort": "low"},
    )
    messages = [{"role": "user", "content": "question"}]
    prompt_ids = tokenizer.encode_messages(messages, add_generation_prompt=True)
    prefix_mask = tokenizer.action_prefix_mask(messages, (ActionKind.ANSWER,))
    visible_prefix = tokenizer.encode_text("Answer(")
    template_preamble = (raw_tokenizer._final, raw_tokenizer._message)
    reasoning_start = (raw_tokenizer._analysis, raw_tokenizer._message)
    reasoning_text = tokenizer.encode_text("check")
    reasoning_end = (
        raw_tokenizer._end,
        raw_tokenizer._assistant,
        raw_tokenizer._final,
        raw_tokenizer._message,
    )
    generated = (
        reasoning_start
        + reasoning_text
        + reasoning_end
        + visible_prefix
        + tokenizer.encode_text("bad)")
        + (raw_tokenizer._return,)
    )
    raw_row = _logit_row(prefix_mask.vocab_size)

    assert prefix_mask.action_prefixes == (template_preamble + visible_prefix,)
    assert prefix_mask.reasoning_start == reasoning_start
    assert prefix_mask.reasoning_end == reasoning_end
    assert prefix_mask.action_terminators == ((raw_tokenizer._return,),)
    assert prefix_mask.allowed_next(()) == tuple(
        sorted((raw_tokenizer._final, raw_tokenizer._analysis))
    )
    assert prefix_mask.allowed_next(reasoning_start) is None
    assert prefix_mask.forbidden_next(reasoning_start) == prefix_mask.reasoning_forbidden_tokens
    action_support = prefix_mask.allowed_next(reasoning_start + reasoning_text + reasoning_end)
    assert action_support == (visible_prefix[0],)
    retrieve_token = tokenizer.encode_text("Retrieve(")[0]
    assert retrieve_token not in action_support
    action_distribution = fake_torch.log_softmax(
        transformers_backend._masked_logit_row(
            _NumericTensor(raw_row), action_support, prefix_mask
        ),
        dim=-1,
    )
    assert action_distribution[retrieve_token].item() == float("-inf")
    argument_prefix = template_preamble + visible_prefix + tokenizer.encode_text("1")
    assert raw_tokenizer._analysis in prefix_mask.forbidden_next(argument_prefix)
    assert (
        tokenizer.decode_generated_action(
            argument_prefix
            + (raw_tokenizer._return,)
            + tokenizer.encode_text(", [])")
            + (raw_tokenizer._return,),
            prefix_mask,
        )
        == "<incomplete-feasible-action>"
    )

    class GenerateModel:
        config = type("Config", (), {"vocab_size": prefix_mask.vocab_size})()

        def __call__(self, *, input_ids, attention_mask):
            del attention_mask
            logits = [[list(raw_row) for _ in range(input_ids.shape[-1])]]
            return type("Output", (), {"logits": _NumericTensor(logits)})()

        def generate(self, **kwargs):
            processor = kwargs["logits_processor"][0]
            prompt = kwargs["input_ids"].data[0]
            emitted = []
            scores = []
            for token in generated:
                processed = processor(
                    _NumericTensor([prompt + emitted]),
                    _NumericTensor([list(raw_row)]),
                )
                assert math.isfinite(processed[0][token].item())
                scores.append(processed[0].data)
                emitted.append(token)
            return type(
                "Output",
                (),
                {
                    "scores": tuple(_NumericTensor([row]) for row in scores),
                    "sequences": _NumericTensor([prompt + emitted]),
                },
            )()

        def eval(self):
            return None

        def train(self):
            return None

        def parameters(self):
            return ()

    actor = transformers_backend.TransformersActorAdapter(GenerateModel(), device="cpu")
    backend = transformers_backend.TransformersGenerationBackend(
        actor, tokenizer, context_tokens=4096
    )
    candidate = backend.generate(
        messages,
        n=1,
        seed=11,
        greedy=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        max_tokens=len(generated),
        allowed_actions=(ActionKind.ANSWER,),
    )[0]

    assert candidate.text == "Answer(bad)"
    assert "channel" not in candidate.text
    assert tuple(candidate.token_ids[: len(reasoning_start + reasoning_text)]) == (
        reasoning_start + reasoning_text
    )
    tokens = TokenizedAction(prompt_ids, generated, prefix_mask=prefix_mask)
    training_log_probs = actor._action_log_probs((tokens,), requires_grad=False)[0].tolist()
    assert candidate.log_probs == pytest.approx(training_log_probs)
    action_offset = len(reasoning_start + reasoning_text + reasoning_end)
    masked_action_log_probs = candidate.log_probs[
        action_offset : action_offset + len(visible_prefix)
    ]
    assert masked_action_log_probs == pytest.approx([0.0] * len(visible_prefix))


def test_gpt_oss_truncated_reasoning_remains_a_charged_malformed_candidate(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(transformers_backend, "_require_torch", lambda: fake_torch)
    raw_tokenizer = _GPTOSSChannelTokenizer()
    tokenizer = transformers_backend.TransformersTokenizerAdapter(
        raw_tokenizer,
        chat_template_kwargs={"reasoning_effort": "low"},
    )
    messages = [{"role": "user", "content": "question"}]
    prefix_mask = tokenizer.action_prefix_mask(messages, (ActionKind.ANSWER,))
    generated = (
        raw_tokenizer._analysis,
        raw_tokenizer._message,
        *tokenizer.encode_text("unfinished reasoning"),
    )
    raw_row = _logit_row(prefix_mask.vocab_size)

    class TruncatedReasoningModel:
        config = type("Config", (), {"vocab_size": prefix_mask.vocab_size})()

        def __call__(self, *, input_ids, attention_mask):
            del attention_mask
            logits = [[list(raw_row) for _ in range(input_ids.shape[-1])]]
            return type("Output", (), {"logits": _NumericTensor(logits)})()

        def generate(self, **kwargs):
            processor = kwargs["logits_processor"][0]
            prompt = kwargs["input_ids"].data[0]
            emitted = []
            scores = []
            for token in generated:
                processed = processor(
                    _NumericTensor([prompt + emitted]),
                    _NumericTensor([list(raw_row)]),
                )
                assert math.isfinite(processed[0][token].item())
                scores.append(processed[0].data)
                emitted.append(token)
            return type(
                "Output",
                (),
                {
                    "scores": tuple(_NumericTensor([row]) for row in scores),
                    "sequences": _NumericTensor([prompt + emitted]),
                },
            )()

        def eval(self):
            return None

        def train(self):
            return None

        def parameters(self):
            return ()

    actor = transformers_backend.TransformersActorAdapter(TruncatedReasoningModel(), device="cpu")
    backend = transformers_backend.TransformersGenerationBackend(
        actor, tokenizer, context_tokens=4096
    )
    candidate = backend.generate(
        messages,
        n=1,
        seed=12,
        greedy=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        max_tokens=len(generated),
        allowed_actions=(ActionKind.ANSWER,),
    )[0]

    assert candidate.text == "<incomplete-feasible-action>"
    assert candidate.token_ids == generated
    assert len(candidate.log_probs) == len(generated)

    class CandidateBackend:
        @staticmethod
        def generate(_messages, **kwargs):
            return [candidate] * kwargs["n"]

    policy = TextPolicy(CandidateBackend())
    decision = policy.sample(_observation(), rng=random.Random(5))[0]
    assert decision.action is None
    assert decision.token_ids == generated
    assert decision.token_count == len(generated)


def test_binary_answer_probability_uses_strict_fixed_position_and_text_policy(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(transformers_backend, "_require_torch", lambda: fake_torch)
    tokenizer = transformers_backend.TransformersTokenizerAdapter(
        _QwenClosedThinkingTokenizer(),
        chat_template_kwargs={"enable_thinking": False},
    )
    messages = [{"role": "user", "content": "binary question"}]
    position = tokenizer.binary_answer_position(messages)
    assert position.answer_prefix_ids == tokenizer.encode_text("Answer(")
    assert position.negative_token_id == ord("0")
    assert position.positive_token_id == ord("1")

    class BinaryModel:
        config = type("Config", (), {"vocab_size": 256})()

        def __init__(self):
            self.last_input = None

        def __call__(self, *, input_ids, attention_mask):
            del attention_mask
            self.last_input = input_ids.tolist()[0]
            row = [0.0] * 256
            row[ord("0")] = 1.0
            row[ord("1")] = 2.0
            logits = [[list(row) for _ in self.last_input]]
            return type("Output", (), {"logits": _NumericTensor(logits)})()

        def eval(self):
            return None

        def train(self):
            return None

        def parameters(self):
            return ()

    model = BinaryModel()
    actor = transformers_backend.TransformersActorAdapter(model, device="cpu")
    backend = transformers_backend.TransformersGenerationBackend(
        actor, tokenizer, context_tokens=4096
    )
    probability = backend.binary_answer_probability(messages)
    assert probability == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))
    assert model.last_input == list(position.context_ids)

    class PolicyBackend:
        def __init__(self):
            self.messages = None

        def binary_answer_probability(self, rendered):
            self.messages = rendered
            return 0.75

    policy_backend = PolicyBackend()
    policy = TextPolicy(policy_backend, scaffold="binary-scaffold")
    assert policy.binary_answer_probability(_observation()) == pytest.approx(0.75)
    assert "binary-scaffold" in policy_backend.messages[0]["content"]


def test_binary_answer_probability_keeps_sampled_gpt_reasoning_context(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(transformers_backend, "_require_torch", lambda: fake_torch)
    raw_tokenizer = _GPTOSSChannelTokenizer()
    tokenizer = transformers_backend.TransformersTokenizerAdapter(
        raw_tokenizer,
        chat_template_kwargs={"reasoning_effort": "low"},
    )
    messages = [{"role": "user", "content": "binary question"}]
    prompt_ids = tokenizer.encode_messages(messages, add_generation_prompt=True)
    mask = tokenizer.action_prefix_mask(messages, (ActionKind.ANSWER,))
    generated = (
        mask.reasoning_start
        + tokenizer.encode_text("reason first")
        + mask.reasoning_end
        + tokenizer.encode_text("Answer(1, [])")
        + (raw_tokenizer._return,)
    )
    class_offset = generated.index(ord("1"))
    position = tokenizer.binary_answer_position(messages, generated)
    assert position.context_ids == prompt_ids + generated[:class_offset]
    assert position.negative_token_id == ord("0")
    assert position.positive_token_id == ord("1")

    class BinaryModel:
        config = type("Config", (), {"vocab_size": 256})()

        def __init__(self):
            self.last_input = None

        def __call__(self, *, input_ids, attention_mask):
            del attention_mask
            self.last_input = input_ids.tolist()[0]
            row = [0.0] * 256
            row[ord("0")] = -1.0
            row[ord("1")] = 1.0
            logits = [[list(row) for _ in self.last_input]]
            return type("Output", (), {"logits": _NumericTensor(logits)})()

        def eval(self):
            return None

        def train(self):
            return None

        def parameters(self):
            return ()

    model = BinaryModel()
    backend = transformers_backend.TransformersGenerationBackend(
        transformers_backend.TransformersActorAdapter(model, device="cpu"),
        tokenizer,
        context_tokens=4096,
    )
    assert backend.binary_answer_probability(messages, action_ids=generated) == pytest.approx(
        1.0 / (1.0 + math.exp(-2.0))
    )
    assert model.last_input == list(position.context_ids)

    class PolicyBackend:
        def __init__(self):
            self.action_ids = None

        def binary_answer_probability(self, _messages, *, action_ids=None):
            self.action_ids = action_ids
            return 0.8

    policy_backend = PolicyBackend()
    policy = TextPolicy(policy_backend)
    decision = PolicyDecision(
        action=make_action(ActionKind.ANSWER, prediction=1, evidence=[]),
        token_count=len(generated),
        token_ids=generated,
        behavior_log_probs=(-0.1,) * len(generated),
    )
    assert policy.binary_answer_probability(_observation(), decision) == pytest.approx(0.8)
    assert policy_backend.action_ids == generated


def test_binary_answer_position_fails_closed_for_split_or_unstable_identifiers():
    class SplitZeroTokenizer(_QwenClosedThinkingTokenizer):
        @classmethod
        def encode(cls, text, *, add_special_tokens):
            result = super().encode(text, add_special_tokens=add_special_tokens)
            if text.endswith("Answer(0"):
                return [*result[:-1], result[-1], result[-1]]
            return result

    split = transformers_backend.TransformersTokenizerAdapter(
        SplitZeroTokenizer(),
        chat_template_kwargs={"enable_thinking": False},
    )
    with pytest.raises(ValueError, match="not exactly one token"):
        split.binary_answer_position([{"role": "user", "content": "question"}])

    class UnstableBoundaryTokenizer(_QwenClosedThinkingTokenizer):
        @classmethod
        def encode(cls, text, *, add_special_tokens):
            result = super().encode(text, add_special_tokens=add_special_tokens)
            if text.endswith("Answer(") and "<TS><TE>" in text:
                result[0] = ord("X")
            return result

    unstable = transformers_backend.TransformersTokenizerAdapter(
        UnstableBoundaryTokenizer(),
        chat_template_kwargs={"enable_thinking": False},
    )
    with pytest.raises(ValueError, match="boundary is unstable"):
        unstable.binary_answer_position([{"role": "user", "content": "question"}])


def test_rollout_conversion_carries_the_feasible_prefix_mask():
    tokenizer = transformers_backend.TransformersTokenizerAdapter(_CharacterTokenizer())
    observation = _observation()
    action = make_action(ActionKind.QUERY_FIELD, field="value", window="all")
    action_ids = tokenizer.encode_text(format_action(action))
    decision = PolicyDecision(
        action=action,
        token_count=len(action_ids),
        token_ids=action_ids,
        behavior_log_probs=(-0.25,) * len(action_ids),
    )
    turn = Turn(
        index=0,
        observation=observation,
        decision=decision,
        action=action,
        tool_return=ToolReturn(ReturnCode.OK),
        cost=1,
        accepted=True,
        group_id="masked",
        normalized_advantage=0.5,
    )
    rollout = Rollout("masked-rollout", "train-1", [turn], sampled_tokens=len(action_ids))
    instance = InstanceBatch("train-1", "occurrence-1", (rollout,), (), (), None)

    examples = collect_vapa_action_groups(UpdateBatch((instance,), None), tokenizer)

    prefix_mask = examples[0].examples[0].tokens.prefix_mask
    assert prefix_mask is not None
    assert prefix_mask.accepts(action_ids)
    assert prefix_mask.allowed_next(()) == tuple(
        sorted(
            {
                tokenizer.encode_text("QueryField(")[0],
                tokenizer.encode_text("Answer(")[0],
            }
        )
    )


def test_action_prefix_mask_rejects_ambiguous_or_noncanonical_tokenization():
    with pytest.raises(ValueError, match="prefix-free"):
        ActionPrefixMask(((1,), (1, 2)), vocab_size=8)

    class AmbiguousTokenizer(_CharacterTokenizer):
        @staticmethod
        def encode(text, *, add_special_tokens):
            del add_special_tokens
            for prefix in ("Answer(", "Retrieve("):
                if text.endswith(prefix):
                    return [*text[: -len(prefix)].encode("ascii"), 7]
            return list(text.encode("ascii"))

        @staticmethod
        def decode(tokens, *, skip_special_tokens, clean_up_tokenization_spaces=False):
            del skip_special_tokens, clean_up_tokenization_spaces
            return "Answer(" if tokens == [7] else bytes(tokens).decode("ascii")

    tokenizer = transformers_backend.TransformersTokenizerAdapter(AmbiguousTokenizer())
    with pytest.raises(ValueError, match="canonical|distinct|preserve"):
        tokenizer.action_prefix_mask(
            [{"role": "user", "content": "question"}],
            (ActionKind.ANSWER, ActionKind.RETRIEVE),
        )
