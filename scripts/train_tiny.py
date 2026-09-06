#!/usr/bin/env python3
"""Real CPU LoRA/SFT training and shared-reference loss checks on public fixtures.

No pretrained weights, external datasets, or network are used. This is an optimizer
and checkpoint integration test, NOT a clinical benchmark or full VAPA RL run.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import sys
import uuid
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vapa.artifacts import (  # noqa: E402
    artifact_fingerprint,
    atomic_write_text,
    canonical_json_dumps,
    fingerprint_file,
    strict_json_loads,
    validate_output_paths,
)
from vapa.data.io import load_jsonl  # noqa: E402


def validate_results(result: dict, events: list[dict]) -> None:
    if (
        result.get("schema_version") != "vapa-tiny-training-v1"
        or result.get("kind") != "executed_synthetic_model_training"
        or result.get("paper_reproduction") is not False
        or result.get("full_rl_rollout_training") is not False
    ):
        raise ValueError("invalid tiny-training identity or scope")
    if result["trace_sha256"] != artifact_fingerprint(events):
        raise ValueError("training trace checksum mismatch")
    sft = [event for event in events if event["event"] == "sft_update"]
    if len(events) != len(sft) + 1 or events[-1]["event"] != "vapa_objective_check":
        raise ValueError("training trace must end with exactly one VAPA objective check")
    if [row["step"] for row in sft] != list(range(1, result["sft"]["optimizer_steps"] + 1)):
        raise ValueError("training trace has missing or duplicate steps")
    if len(sft) < 2 or not all(math.isfinite(row["loss"]) for row in sft):
        raise ValueError("training trace needs finite measured losses")
    if (
        sft[0]["loss"] != result["sft"]["first_update_loss"]
        or sft[-1]["loss"] != result["sft"]["last_update_loss"]
    ):
        raise ValueError("training summary and trace disagree")
    total_tokens = 0
    for row in sft:
        if type(row["action_tokens"]) is not int or row["action_tokens"] < 1:
            raise ValueError("invalid training token count")
        total_tokens += row["action_tokens"]
        if row["cumulative_action_tokens"] != total_tokens:
            raise ValueError("inconsistent cumulative training tokens")
    if result["sft"]["action_tokens"] != total_tokens:
        raise ValueError("training summary token count disagrees with trace")
    if sft[-1]["loss"] >= sft[0]["loss"]:
        raise ValueError("measured SFT loss did not decrease")
    gradient = events[-1]["gradient_norm"]
    if not math.isfinite(gradient) or gradient <= 0:
        raise ValueError("invalid measured VAPA gradient")
    for check in (
        "resume_matches_uninterrupted",
        "sft_loss_decreased",
        "actor_changed",
        "reference_unchanged",
        "backbone_unchanged",
        "nonzero_finite_gradient",
    ):
        if result["checks"].get(check) is not True:
            raise ValueError(f"training check failed: {check}")


def run(output: Path, *, seed: int = 7, steps: int = 16) -> tuple[dict, list[dict]]:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from vapa.model.inference import load_transformers_checkpoint
    from vapa.model.protocols import VAPAAction
    from vapa.model.transformers import (
        TransformersActorAdapter,
        TransformersTokenizerAdapter,
        apply_shared_lora,
    )
    from vapa.provenance import package_code_fingerprint
    from vapa.training.runtime import build_adamw, seed_everything
    from vapa.training.sft_train import SFTTrainConfig, load_sft_demonstrations, train_sft

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError("steps must be an integer of at least two")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    # New run only; never merge checkpoints or overwrite a previous experiment.
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    seed_everything(seed, deterministic=True)
    data = ROOT / "examples/tiny_sft.jsonl"
    examples = load_sft_demonstrations(data)
    raw = Tokenizer(models.WordLevel(unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    raw.train_from_iterator(
        [
            TransformersTokenizerAdapter._fallback_text(
                example.messages, add_generation_prompt=True
            )
            + " "
            + example.action_text
            for example in examples
        ],
        trainers.WordLevelTrainer(special_tokens=["<unk>", "<pad>", "<eos>"]),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>"
    )
    model_config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=256,
        n_embd=32,
        n_layer=1,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
    )
    base = GPT2LMHeadModel(model_config)
    parameter_count = sum(parameter.numel() for parameter in base.parameters())
    base_dir = output / "random-base"
    base.save_pretrained(base_dir)
    tokenizer.save_pretrained(base_dir)
    base_sha = fingerprint_file(base_dir / "model.safetensors").sha256
    del base
    config = SFTTrainConfig(
        data_path=data,
        output_dir=output / "sft",
        content_kind="public",
        run_id="tiny-cpu-sft",
        model_name=str(base_dir),
        model_revision=base_sha,
        tokenizer_revision=base_sha,
        model_kind="causal",
        use_processor=False,
        dtype="float32",
        device="cpu",
        context_tokens=256,
        generation_max_tokens=64,
        epochs=steps,
        action_token_floor=512,
        learning_rate=0.01,
        lora_rank=4,
        lora_alpha=8,
        lora_target_modules=("c_attn", "c_proj", "c_fc"),
        seed=seed,
        shuffle=False,
        deterministic=True,
        checkpoint_every=steps,
        local_files_only=True,
    )
    interrupted = train_sft(config, stop_after_updates=steps // 2)
    resumed = train_sft(replace(config, resume_from=Path(interrupted.last_checkpoint)))
    uninterrupted = train_sft(replace(config, output_dir=output / "sft-uninterrupted"))
    loaded = load_transformers_checkpoint(resumed.last_checkpoint)
    complete = load_transformers_checkpoint(uninterrupted.last_checkpoint)
    resume_equal = all(
        torch.equal(value, complete.actor_state[key]) for key, value in loaded.actor_state.items()
    )
    metrics = load_jsonl(output / "sft/metrics.jsonl")
    comparison_metrics = load_jsonl(output / "sft-uninterrupted/metrics.jsonl")
    resume_equal = resume_equal and metrics == comparison_metrics

    # Exercise the actual differentiable shared-backbone VAPA objective on a fixed
    # supervised action batch. Advantages here are test inputs, not environment returns.
    actor = TransformersActorAdapter(
        GPT2LMHeadModel.from_pretrained(base_dir, local_files_only=True), device="cpu"
    )
    actor, reference = apply_shared_lora(
        actor, rank=4, alpha=8, target_modules=config.lora_target_modules
    )
    actor.load_state_dict(loaded.actor_state)
    reference.load_state_dict(loaded.actor_state)
    adapter = TransformersTokenizerAdapter(tokenizer)
    encoded = [adapter.encode_action(example.messages, example.action_text) for example in examples]

    def snapshot(model):
        return {key: value.detach().clone() for key, value in model.state_dict().items()}

    actor_before, reference_before = snapshot(actor), snapshot(reference)
    backbone_before = {
        name: parameter.detach().clone()
        for name, parameter in actor.model.named_parameters()
        if "lora_" not in name
    }
    actor.eval()
    before_loss = actor.sft_loss(encoded).total
    behavior = actor._action_log_probs(encoded, requires_grad=False)
    batch = [
        VAPAAction(tokens, tuple(row.tolist()), advantage)
        for tokens, row, advantage in zip(encoded, behavior, (1.0, -0.5, 0.25), strict=True)
    ]
    optimizer = build_adamw(actor, learning_rate=0.001, weight_decay=0.01)
    actor.train()
    reference.eval()
    optimizer.zero_grad()
    loss = actor.vapa_loss(
        batch, reference=reference, kl_weight=0.01, ratio_clip=None, kl_mode="k3"
    )
    loss.backward()
    gradient_norm = actor.clip_grad_norm(1.0)
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise RuntimeError("shared-LoRA objective has no finite nonzero gradient")
    optimizer.step()
    actor_after, reference_after = snapshot(actor), snapshot(reference)
    checks = {
        "resume_matches_uninterrupted": resume_equal,
        "sft_loss_decreased": metrics[-1]["loss"] < metrics[0]["loss"],
        "actor_changed": any(
            not torch.equal(value, actor_after[key]) for key, value in actor_before.items()
        ),
        "reference_unchanged": all(
            torch.equal(value, reference_after[key]) for key, value in reference_before.items()
        ),
        "backbone_unchanged": all(
            torch.equal(value, dict(actor.model.named_parameters())[key])
            for key, value in backbone_before.items()
        ),
        "nonzero_finite_gradient": True,
    }
    trace = [{"event": "sft_update", **row} for row in metrics]
    trace.append(
        {
            "event": "vapa_objective_check",
            "optimizer_steps": 1,
            "loss": loss.total,
            "policy_loss": loss.policy,
            "kl": loss.kl,
            "gradient_norm": gradient_norm,
            "action_tokens": loss.token_count,
            "advantages": [1.0, -0.5, 0.25],
            "advantages_source": "fixed_test_inputs_not_environment_rewards",
        }
    )
    result = {
        "schema_version": "vapa-tiny-training-v1",
        "kind": "executed_synthetic_model_training",
        "paper_reproduction": False,
        "full_rl_rollout_training": False,
        "seed": seed,
        "model": {
            "architecture": "GPT2LMHeadModel",
            "initialization": "random_from_config_no_pretrained_weights",
            "parameters": parameter_count,
            "layers": 1,
            "hidden_size": 32,
            "heads": 2,
            "context_tokens": 256,
            "vocab_size": len(tokenizer),
            "lora_rank": 4,
            "lora_alpha": 8,
            "lora_targets": list(config.lora_target_modules),
            "base_weights_sha256": base_sha,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "architecture": platform.machine(),
            "device": "cpu",
            "dtype": "float32",
            "threads": 1,
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "peft",
                    "tokenizers",
                    "numpy",
                    "safetensors",
                    "accelerate",
                )
            },
        },
        "provenance": {
            "fixture": "examples/tiny_sft.jsonl",
            "fixture_sha256": fingerprint_file(data).sha256,
            "implementation_sha256": package_code_fingerprint(),
            "runner_sha256": fingerprint_file(Path(__file__)).sha256,
        },
        "sft": {
            "optimizer_steps": resumed.completed_updates,
            "action_tokens": resumed.action_tokens,
            "examples": len(examples),
            "first_update_loss": metrics[0]["loss"],
            "last_update_loss": metrics[-1]["loss"],
            "final_training_set_nll": before_loss,
            "learning_rate": config.learning_rate,
            "evaluation_scope": "training_fixture_only_no_held_out_accuracy",
        },
        "checks": checks,
        "trace_sha256": artifact_fingerprint(trace),
        "limitations": [
            "Tiny random model and three synthetic demonstrations, not Qwen3.5-9B or gpt-oss-20b.",
            "SFT losses are measured on training examples; "
            "no generalization or clinical-performance claim.",
            "The VAPA optimizer check uses fixed test advantages and does not run "
            "rollout sampling, replay, or benchmark evaluation.",
            "Resume equivalence is verified on this recorded software/platform; "
            "other platforms may differ numerically.",
        ],
    }
    validate_results(result, trace)
    atomic_write_text(
        output / "events.jsonl", "".join(canonical_json_dumps(row) + "\n" for row in trace)
    )
    atomic_write_text(output / "results.json", json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result, trace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="replace public aggregate summary and measured optimizer trace",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check saved public results without training dependencies",
    )
    args = parser.parse_args()
    if args.check:
        if args.publish or args.output:
            parser.error("--check cannot be combined with training options")
        result = strict_json_loads((ROOT / "results/training_results.json").read_bytes())
        trace = load_jsonl(ROOT / "logs/training_results.jsonl")
        validate_results(result, trace)
        print("validated measured tiny-training summary and optimizer trace")
        return 0
    output = args.output or ROOT / "runs/tiny-training" / f"run-{uuid.uuid4().hex[:12]}"
    result, trace = run(output)
    if args.publish:
        validate_output_paths(
            [ROOT / "logs/training_results.jsonl", ROOT / "results/training_results.json"],
            inputs=[output / "events.jsonl", output / "results.json"],
            overwrite=True,
        )
        atomic_write_text(
            ROOT / "logs/training_results.jsonl",
            "".join(canonical_json_dumps(row) + "\n" for row in trace),
        )
        atomic_write_text(
            ROOT / "results/training_results.json",
            json.dumps(result, indent=2, allow_nan=False) + "\n",
        )
    print(f"completed actual tiny-model training: {output}")
    print(
        f"training loss: {result['sft']['first_update_loss']:.6f} -> "
        f"{result['sft']['last_update_loss']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
