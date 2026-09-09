from __future__ import annotations

from types import SimpleNamespace

import pytest

import vapa.model.transformers as transformers_backend


class _Parameter:
    def __init__(self, value: float, *, requires_grad: bool) -> None:
        self.value = value
        self.requires_grad = requires_grad

    def requires_grad_(self, enabled: bool):
        self.requires_grad = enabled
        return self


class _BaseModel:
    training = True
    config = SimpleNamespace(
        _name_or_path="fake/shared-base",
        architectures=("FakeForCausalLM",),
        vocab_size=16,
        to_dict=lambda: {"model_type": "fake"},
    )

    def __init__(self) -> None:
        self.base_weight = _Parameter(42.0, requires_grad=False)

    def __call__(self, **_kwargs):  # pragma: no cover - activation is tested directly
        raise AssertionError("a model forward is outside this dependency-light test")

    def parameters(self):
        return (self.base_weight,)

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self):
        return {"base.weight": self.base_weight.value}


class _SharedPeftModel:
    def __init__(self, base: _BaseModel, adapter_name: str) -> None:
        self.base = base
        self.config = base.config
        self.training = base.training
        self.adapters: dict[str, _Parameter] = {}
        self.active_adapter = ""
        self.add_adapter(adapter_name, object())
        self.set_adapter(adapter_name)

    def __call__(self, **_kwargs):  # pragma: no cover - activation is tested directly
        raise AssertionError("a model forward is outside this dependency-light test")

    def add_adapter(self, name: str, _config: object) -> None:
        if name in self.adapters:
            raise ValueError(name)
        self.adapters[name] = _Parameter(0.0, requires_grad=False)

    def set_adapter(self, name: str, *, inference_mode: bool = False) -> None:
        if name not in self.adapters:
            raise ValueError(name)
        self.active_adapter = name
        for adapter_name, parameter in self.adapters.items():
            parameter.requires_grad_(adapter_name == name and not inference_mode)

    def parameters(self):
        return (self.base.base_weight, *self.adapters.values())

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self):
        return {
            "base.weight": self.base.base_weight.value,
            **{
                f"adapter.{name}.weight": parameter.value
                for name, parameter in self.adapters.items()
            },
        }


class _FakePeft:
    @staticmethod
    def LoraConfig(**kwargs):
        return dict(kwargs)

    @staticmethod
    def get_peft_model(base, _config, *, adapter_name="default"):
        return _SharedPeftModel(base, adapter_name)

    @staticmethod
    def get_peft_model_state_dict(model, *, adapter_name, save_embedding_layers):
        assert save_embedding_layers is False
        return {"lora.weight": model.adapters[adapter_name].value}

    @staticmethod
    def set_peft_model_state_dict(model, state, *, adapter_name):
        model.adapters[adapter_name].value = state["lora.weight"]


def test_shared_lora_uses_one_backbone_and_adapter_only_strict_state(monkeypatch):
    monkeypatch.setattr(transformers_backend, "_require_peft", lambda: _FakePeft)
    actor = transformers_backend.TransformersActorAdapter(
        _BaseModel(),
        device="cpu",
        provenance={"revision": "a" * 40},
    )

    actor, reference = transformers_backend.apply_shared_lora(
        actor,
        rank=8,
        alpha=16,
        target_modules=("q_proj", "v_proj"),
    )

    assert actor is not reference
    assert actor.model is reference.model
    assert actor.adapter_name == "actor"
    assert reference.adapter_name == "reference"
    assert actor.fingerprint == reference.fingerprint
    single = transformers_backend.apply_lora(
        transformers_backend.TransformersActorAdapter(
            _BaseModel(),
            device="cpu",
            provenance={"revision": "a" * 40},
        ),
        rank=8,
        alpha=16,
        target_modules=("q_proj", "v_proj"),
    )
    assert actor.fingerprint == single.fingerprint
    assert actor.state_dict() == reference.state_dict() == {"lora.weight": 0.0}
    assert "base.weight" not in actor.state_dict()

    actor.load_state_dict({"lora.weight": 3.5})
    assert actor.state_dict() == {"lora.weight": 3.5}
    assert reference.state_dict() == {"lora.weight": 0.0}
    with pytest.raises(ValueError, match="state keys"):
        actor.load_state_dict({"wrong.weight": 1.0})

    actor_parameters = tuple(actor.parameters())
    assert actor_parameters == (actor.model.adapters["actor"],)
    assert actor_parameters[0].requires_grad is True
    assert actor.model.base.base_weight.requires_grad is False
    assert tuple(reference.parameters()) == ()
    assert reference.model.adapters["reference"].requires_grad is False


def test_shared_lora_restores_actor_after_frozen_reference_activation(monkeypatch):
    monkeypatch.setattr(transformers_backend, "_require_peft", lambda: _FakePeft)
    actor, reference = transformers_backend.apply_shared_lora(
        transformers_backend.TransformersActorAdapter(_BaseModel()),
        rank=4,
        alpha=8,
    )

    actor.train()
    assert actor.model.active_adapter == "actor"
    assert actor.model.training is True
    reference.eval()
    assert actor.model.active_adapter == "reference"
    assert actor.model.training is False
    assert reference.model.adapters["reference"].requires_grad is False
    actor._prepare_forward()
    assert actor.model.active_adapter == "actor"
    assert actor.model.training is True
    assert actor.model.adapters["actor"].requires_grad is True

    monkeypatch.setattr(transformers_backend, "_require_torch", lambda: object())
    with pytest.raises(ValueError, match="requires at least one example"):
        actor.vapa_loss(
            (),
            reference=reference,
            kl_weight=0.01,
            ratio_clip=None,
            kl_mode="forward",
        )


def test_single_lora_state_is_adapter_only(monkeypatch):
    monkeypatch.setattr(transformers_backend, "_require_peft", lambda: _FakePeft)
    actor = transformers_backend.apply_lora(
        transformers_backend.TransformersActorAdapter(_BaseModel()),
        rank=2,
        alpha=4,
    )

    assert actor.adapter_name == "default"
    assert actor.state_dict() == {"lora.weight": 0.0}
    assert "base.weight" not in actor.state_dict()
