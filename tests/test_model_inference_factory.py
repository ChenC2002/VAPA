from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from vapa.artifacts import artifact_fingerprint
from vapa.environment.state_manager import StateManager
from vapa.inference import build_inference_engine, import_factory
from vapa.model.inference import (
    CheckpointEnvironmentSpec,
    InferenceDependencies,
    LoadedTransformerCheckpoint,
    TransformersInferenceSpec,
    build_transformers_policy,
    checkpoint_factory,
    checkpoint_manager_factory,
    load_transformers_checkpoint,
    make_inference_factories,
    policy_factory,
)
from vapa.policies.text import TextPolicy
from vapa.provenance import package_code_fingerprint
from vapa.schemas import Episode, TaskSpec
from vapa.training.checkpoint import (
    CheckpointContract,
    JsonStateStore,
    RuntimeState,
    fingerprint_payload,
    save_checkpoint,
)

ACTOR_FINGERPRINT = fingerprint_payload({"component": "actor-with-lora"})
BASE_FINGERPRINT = fingerprint_payload({"component": "base-actor"})
REFERENCE_FINGERPRINT = fingerprint_payload({"component": "reference"})
TOKENIZER_FINGERPRINT = fingerprint_payload({"component": "tokenizer"})
REVISION = "a" * 40
TOKENIZER_REVISION = "b" * 40


def _qwen_spec(*, lora: bool = True) -> TransformersInferenceSpec:
    return TransformersInferenceSpec(
        model_name="Qwen/Qwen3.5-9B",
        model_revision=REVISION,
        tokenizer_name="Qwen/Qwen3.5-9B",
        tokenizer_revision=TOKENIZER_REVISION,
        model_kind="multimodal",
        use_processor=True,
        dtype="bfloat16",
        device="cpu",
        lora_enabled=lora,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.0,
        lora_target_modules=("q_proj", "v_proj") if lora else None,
        temperature=0.8,
        top_p=0.95,
        top_k=4,
        max_tokens=96,
    )


class _SavedActor:
    def __init__(self, fingerprint: str, state: dict[str, object]) -> None:
        self.fingerprint = fingerprint
        self.state = state

    def state_dict(self):
        return dict(self.state)


class _SavedTokenizer:
    fingerprint = TOKENIZER_FINGERPRINT


class _SavedOptimizer:
    param_groups = []

    @staticmethod
    def state_dict():
        return {"steps": 3}


def _write_checkpoint(
    path: Path,
    spec: TransformersInferenceSpec,
    *,
    actor_fingerprint: str = ACTOR_FINGERPRINT,
    environment_spec: CheckpointEnvironmentSpec | None = None,
) -> CheckpointContract:
    contract = CheckpointContract(
        run_id="factory-test",
        run_manifest_fingerprint=fingerprint_payload({"run": "manifest"}),
        config_fingerprint=fingerprint_payload({"config": "test"}),
        model_fingerprint=actor_fingerprint,
        reference_model_fingerprint=REFERENCE_FINGERPRINT,
        tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
        optimizer_name="fake",
        scheduler_name="none",
        state_format="json-v1",
    )
    save_checkpoint(
        path,
        contract=contract,
        runtime=RuntimeState(
            global_step=3,
            sampled_tokens=100,
            trainable_tokens=80,
            seed=7,
            extra={
                "inference_spec": spec.to_dict(),
                **(
                    {}
                    if environment_spec is None
                    else {"environment_spec": environment_spec.to_dict()}
                ),
            },
        ),
        model=_SavedActor(actor_fingerprint, {"weight": 7}),
        reference=_SavedActor(REFERENCE_FINGERPRINT, {}),
        tokenizer=_SavedTokenizer(),
        optimizer=_SavedOptimizer(),
        store=JsonStateStore(),
    )
    return contract


def test_inference_spec_round_trip_is_strict_and_requires_immutable_revisions():
    spec = _qwen_spec()
    assert TransformersInferenceSpec.from_dict(spec.to_dict()) == spec

    mutable = spec.to_dict()
    mutable["model_revision"] = "main"
    with pytest.raises(ValueError, match="immutable"):
        TransformersInferenceSpec.from_dict(mutable)

    unknown = spec.to_dict()
    unknown["trust_remote_code"] = True
    with pytest.raises(ValueError, match="unexpected fields"):
        TransformersInferenceSpec.from_dict(unknown)

    with pytest.raises(ValueError, match="use_processor=true"):
        replace(spec, use_processor=False)
    with pytest.raises(ValueError, match="explicit lora_target_modules"):
        replace(spec, lora_target_modules=None)
    with pytest.raises(ValueError, match="only for gpt-oss"):
        replace(spec, dequantize_mxfp4=True)

    missing_mxfp4 = spec.to_dict()
    del missing_mxfp4["dequantize_mxfp4"]
    with pytest.raises(ValueError, match="missing fields: dequantize_mxfp4"):
        TransformersInferenceSpec.from_dict(missing_mxfp4)


def test_checkpoint_factory_verifies_contract_runtime_and_every_checksum(tmp_path: Path):
    checkpoint_path = tmp_path / "step-3"
    spec = _qwen_spec()
    contract = _write_checkpoint(checkpoint_path, spec)

    loaded = load_transformers_checkpoint(
        checkpoint_path,
        expected_contract=contract,
        expected_spec=spec,
    )
    assert loaded.path == checkpoint_path.resolve()
    assert loaded.actor_state == {"weight": 7}
    assert loaded.manifest.contract == contract

    optimizer_path = next(checkpoint_path.glob("optimizer.*"))
    optimizer_path.write_text(json.dumps({"steps": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match="optimizer.*integrity"):
        load_transformers_checkpoint(checkpoint_path)


def test_bound_factory_rejects_contract_or_spec_before_model_loading(tmp_path: Path):
    checkpoint_path = tmp_path / "step-3"
    spec = _qwen_spec()
    contract = _write_checkpoint(checkpoint_path, spec)
    model_loads: list[str] = []

    def actor_loader(name: str, **kwargs):
        del kwargs
        model_loads.append(name)
        raise AssertionError("actor loader must not run")

    dependencies = InferenceDependencies(actor_loader=actor_loader)
    wrong_contract = replace(contract, run_id="another-run")
    factories = make_inference_factories(
        expected_contract=wrong_contract,
        expected_spec=spec,
        dependencies=dependencies,
    )
    with pytest.raises(ValueError, match="expected contract"):
        factories.checkpoint_factory(checkpoint_path)
    assert model_loads == []

    factories = make_inference_factories(
        expected_contract=contract,
        expected_spec=replace(spec, model_revision="c" * 40),
        dependencies=dependencies,
    )
    with pytest.raises(ValueError, match="expected spec"):
        factories.checkpoint_factory(checkpoint_path)
    assert model_loads == []


def test_policy_factory_loads_processor_and_lora_before_restoring_actor(tmp_path: Path):
    checkpoint_path = tmp_path / "step-3"
    spec = _qwen_spec()
    _write_checkpoint(checkpoint_path, spec)
    events: list[object] = []

    class RuntimeTokenizer:
        fingerprint = TOKENIZER_FINGERPRINT

    class RuntimeActor:
        lora_applied = False
        restored = False

        @property
        def fingerprint(self):
            return ACTOR_FINGERPRINT if self.lora_applied else BASE_FINGERPRINT

        def load_state_dict(self, state):
            assert self.lora_applied
            events.append(("restore", dict(state)))
            self.restored = True

        def eval(self):
            assert self.restored
            events.append("eval")

    actor = RuntimeActor()

    def tokenizer_loader(name: str, **kwargs):
        events.append(("tokenizer", name, kwargs))
        return RuntimeTokenizer()

    def actor_loader(name: str, **kwargs):
        events.append(("actor", name, kwargs))
        return actor

    def lora_loader(runtime_actor, **kwargs):
        assert runtime_actor is actor
        events.append(("lora", kwargs))
        actor.lora_applied = True
        return actor

    class Backend:
        pass

    backend = Backend()

    def backend_factory(runtime_actor, tokenizer, *, context_tokens):
        assert runtime_actor is actor and runtime_actor.restored
        assert isinstance(tokenizer, RuntimeTokenizer)
        events.append(("backend", context_tokens))
        return backend

    dependencies = InferenceDependencies(
        actor_loader=actor_loader,
        tokenizer_loader=tokenizer_loader,
        lora_loader=lora_loader,
        backend_factory=backend_factory,
    )
    factories = make_inference_factories(expected_spec=spec, dependencies=dependencies)
    with pytest.raises(ValueError, match="checkpoint-backed inference requires"):
        build_inference_engine(
            checkpoint_path=checkpoint_path,
            checkpoint_factory=factories.checkpoint_factory,
            policy_factory=factories.policy_factory,
        )
    engine = build_inference_engine(
        checkpoint_path=checkpoint_path,
        checkpoint_factory=factories.checkpoint_factory,
        policy_factory=factories.policy_factory,
        manager_factory=lambda episode: StateManager(episode),
    )
    policy = engine.policy

    assert isinstance(policy, TextPolicy)
    assert policy.backend is backend
    assert events[-1] == ("backend", 32_768)
    assert (policy.temperature, policy.top_p, policy.top_k, policy.max_tokens) == (
        0.8,
        0.95,
        4,
        96,
    )
    assert [event if isinstance(event, str) else event[0] for event in events] == [
        "tokenizer",
        "actor",
        "lora",
        "restore",
        "eval",
        "backend",
    ]
    tokenizer_kwargs = events[0][2]
    assert tokenizer_kwargs == {
        "revision": TOKENIZER_REVISION,
        "use_processor": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "trust_remote_code": False,
    }
    actor_kwargs = events[1][2]
    assert actor_kwargs == {
        "revision": REVISION,
        "device": "cpu",
        "dtype": "bfloat16",
        "model_kind": "multimodal",
        "dequantize_mxfp4": False,
        "trust_remote_code": False,
        "use_safetensors": True,
    }
    assert events[2][1]["target_modules"] == ("q_proj", "v_proj")


def test_causal_fallback_skips_processor_and_lora(tmp_path: Path):
    spec = replace(
        _qwen_spec(lora=False),
        model_name="openai/gpt-oss-20b",
        tokenizer_name="openai/gpt-oss-20b",
        model_kind="causal",
        use_processor=False,
        device="auto",
        reasoning_effort="low",
        dequantize_mxfp4=True,
    )
    assert TransformersInferenceSpec.from_dict(spec.to_dict()) == spec
    checkpoint_path = tmp_path / "causal"
    _write_checkpoint(checkpoint_path, spec, actor_fingerprint=BASE_FINGERPRINT)
    loaded = load_transformers_checkpoint(checkpoint_path)
    calls: list[tuple[str, dict[str, object]]] = []

    class Tokenizer:
        fingerprint = TOKENIZER_FINGERPRINT

    class Actor:
        fingerprint = BASE_FINGERPRINT

        def load_state_dict(self, state):
            assert state == {"weight": 7}

        def eval(self):
            return None

    def tokenizer_loader(name: str, **kwargs):
        calls.append((name, kwargs))
        return Tokenizer()

    def actor_loader(name: str, **kwargs):
        calls.append((name, kwargs))
        return Actor()

    def must_not_apply_lora(*args, **kwargs):
        raise AssertionError("LoRA must not be applied")

    policy = build_transformers_policy(
        loaded,
        dependencies=InferenceDependencies(
            tokenizer_loader=tokenizer_loader,
            actor_loader=actor_loader,
            lora_loader=must_not_apply_lora,
            backend_factory=lambda actor, tokenizer, *, context_tokens: object(),
            device_resolver=lambda requested: "cuda:2" if requested == "auto" else requested,
        ),
    )
    assert isinstance(policy, TextPolicy)
    assert calls[0][1]["use_processor"] is False
    assert calls[0][1]["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert calls[1][1]["model_kind"] == "causal"
    assert calls[1][1]["device"] == "cuda:2"
    assert calls[1][1]["dequantize_mxfp4"] is True


def test_default_factories_are_importable_for_evaluate_script(tmp_path: Path):
    checkpoint_path = tmp_path / "step-3"
    _write_checkpoint(checkpoint_path, _qwen_spec())

    assert import_factory("vapa.model.inference:checkpoint_factory") is checkpoint_factory
    assert import_factory("vapa.model.inference:policy_factory") is policy_factory
    assert (
        import_factory("vapa.model.inference:checkpoint_manager_factory")
        is checkpoint_manager_factory
    )
    loaded = checkpoint_factory(checkpoint_path)
    assert isinstance(loaded, LoadedTransformerCheckpoint)
    with pytest.raises(ValueError, match="requires --checkpoint"):
        checkpoint_factory(None)


def test_checkpoint_manager_factory_restores_exact_environment(tmp_path: Path):
    calculator_manifest = {
        "schema_version": 1,
        "paper_exact": False,
        "calculators": [],
    }
    environment = CheckpointEnvironmentSpec(
        memory_capacity=5,
        action_budget=7,
        turn_cap=9,
        retrieval_limit=3,
        implementation_sha256=package_code_fingerprint(),
        calculator_manifest=calculator_manifest,
        calculator_manifest_sha256=artifact_fingerprint(calculator_manifest),
    )
    assert CheckpointEnvironmentSpec.from_dict(environment.to_dict()) == environment
    checkpoint_path = tmp_path / "rl-checkpoint"
    _write_checkpoint(checkpoint_path, _qwen_spec(), environment_spec=environment)
    loaded = load_transformers_checkpoint(checkpoint_path)
    build_manager = checkpoint_manager_factory(loaded)
    episode = Episode(
        TaskSpec(
            instance_id="environment-test",
            patient_id="patient-1",
            instruction="Answer.",
            cutoff=datetime(2025, 1, 1, tzinfo=UTC),
            family="synthetic",
        ),
        (),
        gold_answer="ok",
    )
    manager = build_manager(episode)
    assert (
        manager.memory_capacity,
        manager.action_budget,
        manager.turn_cap,
        manager.retrieval_limit,
        len(manager.calculators),
    ) == (5, 7, 9, 3, 0)

    invalid = environment.to_dict()
    invalid["calculator_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        CheckpointEnvironmentSpec.from_dict(invalid)

    wrong_code = environment.to_dict()
    wrong_code["implementation_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="installed VAPA code"):
        CheckpointEnvironmentSpec.from_dict(wrong_code)


def test_policy_contract_fingerprints_are_checked_before_state_restore(tmp_path: Path):
    checkpoint_path = tmp_path / "step-3"
    _write_checkpoint(checkpoint_path, _qwen_spec())
    loaded = load_transformers_checkpoint(checkpoint_path)
    restored = False

    class Tokenizer:
        fingerprint = fingerprint_payload({"wrong": "tokenizer"})

    class Actor:
        fingerprint = ACTOR_FINGERPRINT

        def load_state_dict(self, state):
            nonlocal restored
            restored = True

        def eval(self):
            return None

    with pytest.raises(ValueError, match="tokenizer.*contract"):
        build_transformers_policy(
            loaded,
            dependencies=InferenceDependencies(
                tokenizer_loader=lambda name, **kwargs: Tokenizer(),
                actor_loader=lambda name, **kwargs: Actor(),
            ),
        )
    assert not restored
