from __future__ import annotations

from typing import Any

import pytest

import vapa.model.transformers as transformers_backend
from vapa.model.protocols import OptionalDependencyError


class _Config:
    model_type = "gpt_oss"
    _commit_hash = "immutable-commit"


class _LoadedModel:
    def __init__(self) -> None:
        self.config = _Config()

    def __call__(self, **kwargs: object) -> object:
        return kwargs


def _fake_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_mxfp4: bool,
) -> tuple[type[Any], list[dict[str, object]]]:
    model_calls: list[dict[str, object]] = []

    class AutoConfig:
        @staticmethod
        def from_pretrained(name: str, **kwargs: object) -> _Config:
            del name, kwargs
            return _Config()

    class Causal:
        @staticmethod
        def from_pretrained(name: str, **kwargs: object) -> _LoadedModel:
            model_calls.append({"name": name, **kwargs})
            return _LoadedModel()

    class Mxfp4Config:
        def __init__(self, *, dequantize: bool) -> None:
            self.dequantize = dequantize

    attributes: dict[str, object] = {
        "__version__": "5.13.0",
        "AutoConfig": AutoConfig,
        "AutoModelForCausalLM": Causal,
    }
    if include_mxfp4:
        attributes["Mxfp4Config"] = Mxfp4Config
    fake_transformers = type("FakeTransformers", (), attributes)
    fake_torch = type("FakeTorch", (), {})
    monkeypatch.setattr(transformers_backend, "_require_transformers", lambda: fake_transformers)
    monkeypatch.setattr(transformers_backend, "_require_torch", lambda: fake_torch)
    return fake_transformers, model_calls


def test_gpt_oss_mxfp4_request_constructs_and_forwards_dequantization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, calls = _fake_dependencies(monkeypatch, include_mxfp4=True)

    actor = transformers_backend.TransformersActorAdapter.from_pretrained(
        "openai/gpt-oss-20b",
        dequantize_mxfp4=True,
        use_safetensors=True,
    )

    quantization = calls[0]["quantization_config"]
    assert quantization.dequantize is True
    assert calls[0]["use_safetensors"] is True
    assert actor.provenance["dequantize_mxfp4"] is True


def test_false_or_non_gpt_path_never_forwards_mxfp4_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, calls = _fake_dependencies(monkeypatch, include_mxfp4=True)

    actor = transformers_backend.TransformersActorAdapter.from_pretrained(
        "openai/gpt-oss-20b",
        dequantize_mxfp4=False,
    )
    assert "quantization_config" not in calls[0]
    assert actor.provenance["dequantize_mxfp4"] is False

    with pytest.raises(ValueError, match="only for gpt-oss"):
        transformers_backend.TransformersActorAdapter.from_pretrained(
            "Qwen/Qwen3.5-9B",
            dequantize_mxfp4=True,
        )
    assert len(calls) == 1


def test_mxfp4_request_fails_closed_when_transformers_lacks_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_dependencies(monkeypatch, include_mxfp4=False)

    with pytest.raises(OptionalDependencyError, match="Mxfp4Config"):
        transformers_backend.TransformersActorAdapter.from_pretrained(
            "openai/gpt-oss-20b",
            dequantize_mxfp4=True,
        )


def test_mxfp4_request_rejects_conflicting_quantization_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_dependencies(monkeypatch, include_mxfp4=True)

    with pytest.raises(ValueError, match="explicit quantization_config"):
        transformers_backend.TransformersActorAdapter.from_pretrained(
            "openai/gpt-oss-20b",
            dequantize_mxfp4=True,
            quantization_config=object(),
        )
