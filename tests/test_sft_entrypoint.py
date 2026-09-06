from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from vapa.artifacts import strict_json_loads
from vapa.model.protocols import LossReport, TokenizedAction
from vapa.training.checkpoint import JsonStateStore, fingerprint_payload, read_manifest
from vapa.training.sft_train import (
    SFTDataError,
    SFTTrainConfig,
    build_parser,
    build_sft_schedule,
    config_from_namespace,
    load_sft_config,
    load_sft_demonstrations,
    train_sft,
    validate_sft_run,
)


def _record(action: str, group: str, prompt: str = "Choose an action.") -> str:
    return json.dumps(
        {
            "messages": [{"role": "user", "content": prompt}],
            "action": action,
            "group": group,
        },
        sort_keys=True,
    )


def _write_data(path: Path) -> Path:
    path.write_text(
        "\n".join(
            (
                _record("Answer(1, [])", "g1"),
                _record("Answer(22, [])", "g1"),
                _record("Answer(333, [])", "g2"),
                _record("Answer(4444, [])", "g3"),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return path


class FakeTokenizer:
    fingerprint = fingerprint_payload({"adapter": "sft-test-tokenizer-v1"})

    def encode_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool = True,
    ) -> tuple[int, ...]:
        del messages, add_generation_prompt
        return (90,)

    def encode_text(self, text: str) -> tuple[int, ...]:
        prediction = text.split("(", 1)[1].split(",", 1)[0]
        return tuple(range(1, len(prediction) + 1))

    def encode_action(
        self,
        messages: Sequence[Mapping[str, str]],
        action_text: str,
    ) -> TokenizedAction:
        return TokenizedAction(self.encode_messages(messages), self.encode_text(action_text))

    def save_pretrained(self, path: str) -> None:
        Path(path, "tokenizer.json").write_text("{}\n", encoding="utf-8")


class FakeActor:
    def __init__(self) -> None:
        self.weight = 1.0
        self.grad = 0.0
        self.mode = "eval"
        self.group_sizes: list[int] = []

    @property
    def fingerprint(self) -> str:
        return fingerprint_payload({"adapter": "sft-test-actor-v1"})

    def train(self) -> None:
        self.mode = "train"

    def eval(self) -> None:
        self.mode = "eval"

    def parameters(self):
        return (self,)

    def sft_loss(self, examples: Sequence[TokenizedAction]) -> LossReport:
        tokens = sum(example.token_count for example in examples)
        self.group_sizes.append(tokens)
        loss = self.weight**2

        def backward(scale: float) -> None:
            self.grad += 2 * self.weight * scale

        return LossReport(loss, loss, 0.0, tokens, backward)

    def vapa_loss(self, *args: Any, **kwargs: Any) -> LossReport:
        raise AssertionError("SFT must not invoke the VAPA objective")

    def clip_grad_norm(self, max_norm: float) -> float:
        norm = abs(self.grad)
        if norm > max_norm:
            self.grad *= max_norm / norm
        return norm

    def state_dict(self) -> Mapping[str, Any]:
        return {"weight": self.weight}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.weight = float(state["weight"])


class FakeOptimizer:
    def __init__(self, actor: FakeActor, learning_rate: float) -> None:
        self.actor = actor
        self.param_groups = [{"lr": learning_rate}]
        self.steps = 0

    def zero_grad(self) -> None:
        self.actor.grad = 0.0

    def step(self) -> None:
        self.actor.weight -= self.param_groups[0]["lr"] * self.actor.grad
        self.steps += 1

    def state_dict(self) -> Mapping[str, Any]:
        return {"steps": self.steps, "param_groups": self.param_groups}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.steps = int(state["steps"])
        self.param_groups = [dict(item) for item in state["param_groups"]]


def _config(data: Path, output: Path, *, resume: Path | None = None) -> SFTTrainConfig:
    return SFTTrainConfig(
        data_path=data,
        output_dir=output,
        run_id="unit-sft",
        model_name="test/model",
        model_revision="a" * 40,
        tokenizer_name="test/tokenizer",
        tokenizer_revision="b" * 40,
        model_kind="causal",
        use_processor=False,
        dtype="float32",
        device="cpu",
        epochs=2,
        action_token_floor=4,
        learning_rate=0.1,
        warmup_fraction=0.25,
        checkpoint_every=1,
        seed=17,
        shuffle=False,
        use_lora=False,
        resume_from=resume,
    )


def _factories():
    created: list[FakeActor] = []

    def tokenizer_factory(config: SFTTrainConfig) -> FakeTokenizer:
        assert config.model_name == "test/model"
        return FakeTokenizer()

    def actor_factory(config: SFTTrainConfig, device: str) -> FakeActor:
        assert config.device == device == "cpu"
        actor = FakeActor()
        created.append(actor)
        return actor

    def optimizer_factory(actor: FakeActor, config: SFTTrainConfig) -> FakeOptimizer:
        return FakeOptimizer(actor, config.learning_rate)

    return created, tokenizer_factory, actor_factory, optimizer_factory


def test_strict_jsonl_loader_and_diagnostics(tmp_path: Path):
    valid = _write_data(tmp_path / "valid.jsonl")
    examples = load_sft_demonstrations(valid)
    assert len(examples) == 4
    assert examples[0].group_id == examples[1].group_id == "g1"

    impossible_memory = _record(
        'UpdateMemory({"field":"x","item_id":"m1","value":1,"status":"observed",'
        '"validity_scope":"all","evidence_pointers":["e#1"]})',
        "g",
    )
    invalid_payloads = (
        '{"messages":[{"role":"user","content":"x"}],"action":"Answer(1, [])",'
        '"action":"Answer(2, [])","group":"g"}\n',
        '{"messages":[{"role":"user","content":"x"}],"action":"Answer(1, [])",'
        '"group":"g","score":NaN}\n',
        '{"messages":[{"role":"user","content":"x"}],"action":"not an action","group":"g"}\n',
        impossible_memory + "\n",
        _record("Answer(1, [])", "g") + "\n\n" + _record("Answer(2, [])", "h") + "\n",
    )
    expected = ("duplicate", "non-finite", "invalid VAPA action", "observed", "blank JSONL")
    for index, (payload, message) in enumerate(zip(invalid_payloads, expected, strict=True)):
        path = tmp_path / f"bad-{index}.jsonl"
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(SFTDataError, match=message):
            load_sft_demonstrations(path)


def test_schedule_is_deterministic_and_never_splits_a_group(tmp_path: Path):
    examples = load_sft_demonstrations(_write_data(tmp_path / "data.jsonl"))
    tokenizer = FakeTokenizer()
    first = build_sft_schedule(
        examples,
        tokenizer,
        epochs=2,
        action_token_floor=4,
        seed=9,
        shuffle=True,
    )
    second = build_sft_schedule(
        examples,
        tokenizer,
        epochs=2,
        action_token_floor=4,
        seed=9,
        shuffle=True,
    )
    assert [(item.epoch, item.group_ids, item.action_tokens) for item in first] == [
        (item.epoch, item.group_ids, item.action_tokens) for item in second
    ]
    for epoch in range(2):
        seen: list[str] = []
        for batch in first:
            if batch.epoch == epoch:
                seen.extend(batch.group_ids)
        assert sorted(seen) == ["g1", "g2", "g3"]
        assert len(seen) == len(set(seen))

    with pytest.raises(SFTDataError, match="exceeds context_tokens"):
        build_sft_schedule(
            examples,
            tokenizer,
            epochs=1,
            action_token_floor=4,
            seed=0,
            context_tokens=1,
        )


def test_dry_run_uses_no_factories_or_optional_dependencies(tmp_path: Path):
    config = _config(_write_data(tmp_path / "data.jsonl"), tmp_path / "unused")
    report = validate_sft_run(config)
    assert report.examples == 4
    assert report.groups == 3
    assert report.epochs == 2
    assert report.planned_updates > 0
    assert report.token_count_basis == "utf8-byte-estimate"
    assert not config.output_dir.exists()


def test_sft_rejects_sensitive_artifacts_in_tracked_public_paths(tmp_path: Path):
    data = _write_data(tmp_path / "data.jsonl")
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tracked public location"):
        train_sft(_config(data, repository / "docs" / "patient-run"))


def test_sft_freezes_guarded_output_before_injected_factories(tmp_path: Path):
    data = _write_data(tmp_path / "data.jsonl")
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    outside = tmp_path / "protected-run"
    output_link = repository / "run-link"
    output_link.symlink_to(outside, target_is_directory=True)
    _, tokenizer_factory, base_actor_factory, optimizer_factory = _factories()

    def retargeting_actor_factory(config: SFTTrainConfig, device: str):
        output_link.unlink()
        output_link.symlink_to(repository / "docs" / "patient-run", target_is_directory=True)
        return base_actor_factory(config, device)

    result = train_sft(
        _config(data, output_link),
        tokenizer_factory=tokenizer_factory,
        actor_factory=retargeting_actor_factory,
        optimizer_factory=optimizer_factory,
        state_store=JsonStateStore(),
        stop_after_updates=1,
    )

    assert Path(result.output_dir) == outside.resolve()
    assert (outside / "run_manifest.json").is_file()
    assert not (repository / "docs" / "patient-run").exists()


def test_train_checkpoint_metrics_and_strict_resume(tmp_path: Path):
    data = _write_data(tmp_path / "data.jsonl")
    output = tmp_path / "run"
    created, tokenizer_factory, actor_factory, optimizer_factory = _factories()
    partial = train_sft(
        _config(data, output),
        tokenizer_factory=tokenizer_factory,
        actor_factory=actor_factory,
        optimizer_factory=optimizer_factory,
        state_store=JsonStateStore(),
        stop_after_updates=1,
    )
    assert partial.status == "stopped"
    assert partial.completed_updates == 1
    assert partial.total_updates == 4
    checkpoint = Path(partial.last_checkpoint or "")
    assert checkpoint.name == "step-00000001"
    run_manifest = strict_json_loads((output / "run_manifest.json").read_bytes())
    assert len(run_manifest["implementation_sha256"]) == 64
    manifest = read_manifest(checkpoint)
    inference_spec = manifest.runtime.extra["inference_spec"]
    assert inference_spec["model_revision"] == "a" * 40
    assert inference_spec["lora_enabled"] is False

    resumed_created, tokenizer_factory, actor_factory, optimizer_factory = _factories()
    resumed = train_sft(
        _config(data, output, resume=checkpoint),
        tokenizer_factory=tokenizer_factory,
        actor_factory=actor_factory,
        optimizer_factory=optimizer_factory,
        state_store=JsonStateStore(),
    )
    assert resumed.status == "complete"
    assert resumed.completed_updates == resumed.total_updates == 4
    assert Path(resumed.last_checkpoint or "").name == "step-00000004"
    metrics = [
        strict_json_loads(line)
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [item["step"] for item in metrics] == [1, 2, 3, 4]
    assert metrics[-1]["cumulative_action_tokens"] == resumed.action_tokens
    assert created[0].weight != 1.0
    assert resumed_created[0].weight != 1.0

    _, tokenizer_factory, actor_factory, optimizer_factory = _factories()
    with pytest.raises(FileExistsError, match="resume"):
        train_sft(
            _config(data, output),
            tokenizer_factory=tokenizer_factory,
            actor_factory=actor_factory,
            optimizer_factory=optimizer_factory,
            state_store=JsonStateStore(),
        )


def test_sft_resume_binds_installed_implementation_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import vapa.training.sft_train as sft_module

    data = _write_data(tmp_path / "data.jsonl")
    output = tmp_path / "run"
    factories = _factories()
    monkeypatch.setattr(sft_module, "package_code_fingerprint", lambda: "a" * 64)
    partial = train_sft(
        _config(data, output),
        tokenizer_factory=factories[1],
        actor_factory=factories[2],
        optimizer_factory=factories[3],
        state_store=JsonStateStore(),
        stop_after_updates=1,
    )
    monkeypatch.setattr(sft_module, "package_code_fingerprint", lambda: "b" * 64)
    resumed_factories = _factories()
    with pytest.raises(ValueError, match="run manifest"):
        train_sft(
            _config(data, output, resume=Path(partial.last_checkpoint or "")),
            tokenizer_factory=resumed_factories[1],
            actor_factory=resumed_factories[2],
            optimizer_factory=resumed_factories[3],
            state_store=JsonStateStore(),
        )


def test_resume_rejects_changed_data_before_state_is_loaded(tmp_path: Path):
    data = _write_data(tmp_path / "data.jsonl")
    output = tmp_path / "run"
    _, tokenizer_factory, actor_factory, optimizer_factory = _factories()
    partial = train_sft(
        _config(data, output),
        tokenizer_factory=tokenizer_factory,
        actor_factory=actor_factory,
        optimizer_factory=optimizer_factory,
        state_store=JsonStateStore(),
        stop_after_updates=1,
    )
    data.write_text(data.read_text(encoding="utf-8") + _record("Answer(5, [])", "g4") + "\n")
    _, tokenizer_factory, actor_factory, optimizer_factory = _factories()
    with pytest.raises(ValueError, match="manifest"):
        train_sft(
            _config(data, output, resume=Path(partial.last_checkpoint or "")),
            tokenizer_factory=tokenizer_factory,
            actor_factory=actor_factory,
            optimizer_factory=optimizer_factory,
            state_store=JsonStateStore(),
        )


def test_toml_config_cli_overrides_and_direct_dry_run(tmp_path: Path):
    data = _write_data(tmp_path / "data.jsonl")
    config_path = tmp_path / "sft.toml"
    config_path.write_text(
        "[sft]\n"
        'data_path = "data.jsonl"\n'
        'output_dir = "configured-output"\n'
        "epochs = 3\n"
        "action_token_floor = 20\n",
        encoding="utf-8",
    )
    loaded = load_sft_config(config_path, overrides={"epochs": 2, "output_dir": tmp_path / "cli"})
    assert loaded.data_path == data
    assert loaded.output_dir == tmp_path / "cli"
    assert loaded.epochs == 2

    arguments = build_parser().parse_args(
        ["--config", str(config_path), "--epochs", "4", "--no-shuffle", "--dry-run"]
    )
    from_cli = config_from_namespace(arguments)
    assert from_cli.epochs == 4
    assert from_cli.shuffle is False

    script = Path(__file__).resolve().parents[1] / "scripts" / "train_sft.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--data", str(data), "--dry-run", "--epochs", "1"],
        check=True,
        capture_output=True,
        text=True,
    )
    report = strict_json_loads(completed.stdout)
    assert report["examples"] == 4
    assert report["scope"] == "data-and-schedule-only"
    assert report["training_ready"] is False
    assert report["readiness_issues"]
    assert report["token_count_basis"] == "utf8-byte-estimate"


def test_gpt_oss_sft_mxfp4_setting_is_loaded_forwarded_and_checkpoint_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vapa.model.transformers as transformer_module
    import vapa.training.sft_train as sft_module

    config_path = tmp_path / "gpt-oss-sft.toml"
    config_path.write_text(
        "[sft]\n"
        'model_name = "openai/gpt-oss-20b"\n'
        f'model_revision = "{"a" * 40}"\n'
        'model_kind = "causal"\n'
        "use_processor = false\n"
        'reasoning_effort = "low"\n'
        "dequantize_mxfp4 = true\n"
        "use_lora = false\n",
        encoding="utf-8",
    )
    config = load_sft_config(config_path)
    captured: dict[str, object] = {}
    actor = object()

    def fake_from_pretrained(name: str, **kwargs: object) -> object:
        captured["name"] = name
        captured.update(kwargs)
        return actor

    monkeypatch.setattr(
        transformer_module.TransformersActorAdapter,
        "from_pretrained",
        fake_from_pretrained,
    )
    assert sft_module._default_actor_factory(config, "cpu") is actor
    assert captured["dequantize_mxfp4"] is True
    assert captured["use_safetensors"] is True
    assert sft_module._inference_spec(config)["dequantize_mxfp4"] is True


def test_config_fingerprint_ignores_resume_location(tmp_path: Path):
    base = _config(tmp_path / "data", tmp_path / "output")
    resumed = replace(base, resume_from=tmp_path / "some-checkpoint")
    assert fingerprint_payload(base.fingerprint_payload()) == fingerprint_payload(
        resumed.fingerprint_payload()
    )
