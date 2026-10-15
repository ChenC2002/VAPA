#!/usr/bin/env python3
"""Opt-in CUDA integration benchmark using public prompts and one real LoRA update.

Run in the pinned Linux environment with VLLM_USE_V2_MODEL_RUNNER=0. This checks
backend integration, not task accuracy. Output includes measurements, an adapter
checkpoint, and exact model/software identifiers; no external telemetry is enabled.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import math
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vapa.artifacts import atomic_write_text, canonical_json_dumps, fingerprint_file  # noqa: E402
from vapa.model.protocols import TokenizedAction, VAPAAction  # noqa: E402
from vapa.policies.text import GenerationRequest  # noqa: E402
from vapa.provenance import package_code_fingerprint  # noqa: E402
from vapa.schemas import ActionKind  # noqa: E402
from vapa.training.evidence import software_environment  # noqa: E402


def loss_backward(actor, reference, examples, budget):
    actor.forward_batch_tokens = budget
    actor.model.zero_grad(set_to_none=True)
    actor.train()
    reference.eval()
    loss = actor.vapa_loss(
        examples, reference=reference, kl_weight=0.01, ratio_clip=None, kl_mode="forward"
    )
    loss.backward()
    return loss.total


def gradients(actor):
    return {
        name: p.grad.detach().float().cpu().clone()
        for name, p in actor.model.named_parameters()
        if p.grad is not None
    }


def relative_gradient_error(left, right):
    if left.keys() != right.keys():
        raise RuntimeError("batching changed the trainable parameter set")
    difference = sum((left[n].double() - right[n].double()).square().sum().item() for n in left)
    baseline = sum(g.double().square().sum().item() for g in left.values())
    return math.sqrt(difference / max(baseline, 1e-30))


def check_full_precision(args, examples):
    """Separate mathematical parity from batch-dependent BF16 kernel rounding."""
    import torch

    from vapa.model.transformers import TransformersActorAdapter, apply_shared_lora

    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    actor = TransformersActorAdapter.from_pretrained(
        args.model, revision=args.revision, device="cuda:0", dtype="float32", model_kind="auto"
    )
    actor, reference = apply_shared_lora(
        actor, rank=16, alpha=32, dropout=0.0, target_modules=("q_proj", "v_proj")
    )
    serial = loss_backward(actor, reference, examples, 0)
    baseline = gradients(actor)
    batched = loss_backward(actor, reference, examples, args.train_batch_tokens)
    return {
        "dtype": "float32",
        "tf32": False,
        "loss": {"serial": serial, "batched": batched},
        "gradient_relative_l2_error": relative_gradient_error(baseline, gradients(actor)),
    }


def run(args):
    import torch

    from vapa.model.transformers import (
        TransformersActorAdapter,
        TransformersTokenizerAdapter,
        apply_shared_lora,
    )
    from vapa.model.vllm_backend import VLLMGenerationBackend

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        raise ValueError("revision must be an immutable 40-character model commit")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    actor = TransformersActorAdapter.from_pretrained(
        args.model, revision=args.revision, device="cuda:0", dtype="bfloat16", model_kind="auto"
    )
    actor.model.config.use_cache = False
    actor, reference = apply_shared_lora(
        actor, rank=16, alpha=32, dropout=0.0, target_modules=("q_proj", "v_proj")
    )
    tokenizer = TransformersTokenizerAdapter.from_pretrained(
        args.model, revision=args.revision, chat_template_kwargs={"enable_thinking": False}
    )
    backend = VLLMGenerationBackend(
        actor,
        tokenizer,
        model=args.model,
        revision=args.revision,
        context_tokens=2048,
        max_lora_rank=16,
        max_num_seqs=args.requests,
    )

    def timed(function):
        torch.cuda.synchronize()
        started = time.perf_counter()
        value = function()
        torch.cuda.synchronize()
        return value, time.perf_counter() - started

    def reference_hash():
        digest = hashlib.sha256()
        for name, parameter in actor.model.named_parameters():
            if ".reference." in name:
                digest.update(name.encode())
                digest.update(
                    parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                )
        return digest.hexdigest()

    requests = [
        GenerationRequest(
            messages=[{"role": "user", "content": f"Return Answer({index % 2}, [])."}],
            n=1,
            seed=args.seed + index,
            greedy=False,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            max_tokens=args.max_tokens,
            allowed_actions=(ActionKind.ANSWER,),
        )
        for index in range(args.requests)
    ]
    report = {
        "schema_version": "vapa-cuda-validation-v1",
        "scope": "public-prompt backend integration and one LoRA optimizer update",
        "model": args.model,
        "revision": args.revision,
        "seed": args.seed,
        "implementation_sha256": package_code_fingerprint(),
        "runner_sha256": fingerprint_file(Path(__file__)).sha256,
        "environment": software_environment(),
        "vllm": importlib.metadata.version("vllm"),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "nvidia_driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        )
        .splitlines()[0]
        .strip(),
        "settings": {
            "dtype": "bfloat16",
            "lora_rank": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "requests": args.requests,
            "max_tokens": args.max_tokens,
            "repeats": args.repeats,
            "train_batch_tokens": args.train_batch_tokens,
            "learning_rate": 1e-4,
            "kl_weight": 0.01,
        },
    }
    try:
        _, report["engine_start_seconds"] = timed(lambda: backend.prepare_sampling(0))
        initial_receipt = backend.adapter_receipt
        backend.generate_batch(requests)  # Warm JIT kernels; excluded from timing.
        measurements, final_outputs = {}, {}
        for mode in ("serial", "batched"):
            trials = []
            for _ in range(args.repeats):
                if not backend.engine.reset_prefix_cache():
                    raise RuntimeError("cannot reset cache between benchmark trials")
                outputs, seconds = timed(
                    lambda mode=mode: (
                        [backend.generate_batch([r])[0] for r in requests]
                        if mode == "serial"
                        else backend.generate_batch(requests)
                    )
                )
                count = sum(len(c.token_ids) for group in outputs for c in group)
                trials.append(
                    {
                        "seconds": seconds,
                        "sampled_tokens": count,
                        "tokens_per_second": count / seconds,
                    }
                )
            measurements[mode] = {
                "trials": trials,
                "median_tokens_per_second": statistics.median(
                    t["tokens_per_second"] for t in trials
                ),
            }
            final_outputs[mode] = outputs
        report["generation"] = measurements
        report["generation"]["identical_sequences"] = sum(
            left[0].token_ids == right[0].token_ids
            for left, right in zip(final_outputs["serial"], final_outputs["batched"], strict=True)
        )
        _, report["sleep_seconds"] = timed(backend.prepare_training)
        actor.eval()
        actions, behavior = [], []
        for req, group in zip(requests, final_outputs["batched"], strict=True):
            candidate = group[0]
            actions.append(
                TokenizedAction(
                    candidate.prompt_ids,
                    candidate.token_ids,
                    tokenizer.action_prefix_mask(req.messages, req.allowed_actions),
                )
            )
            behavior.append(candidate.log_probs)
        actor.forward_batch_tokens = args.train_batch_tokens
        actual = actor._action_log_probs(actions, requires_grad=False)
        differences = [
            abs(a - b)
            for row, stored in zip(actual, behavior, strict=True)
            for a, b in zip(row.tolist(), stored, strict=True)
        ]
        report["behavior_logprob_drift"] = {
            "tokens": len(differences),
            "mean_abs": statistics.mean(differences),
            "max_abs": max(differences),
        }
        del actual
        examples = [
            VAPAAction(action, log_probs, 1.0 if index % 2 == 0 else -0.5)
            for index, (action, log_probs) in enumerate(zip(actions, behavior, strict=True))
        ]
        reference_before = reference_hash()
        serial_grads, losses, train_seconds = {}, {}, {}
        torch.cuda.reset_peak_memory_stats()
        for mode, budget in (("serial", 0), ("batched", args.train_batch_tokens)):
            losses[mode], train_seconds[mode] = timed(
                lambda budget=budget: loss_backward(actor, reference, examples, budget)
            )
            grads = gradients(actor)
            if mode == "serial":
                serial_grads = grads
            else:
                relative_error = relative_gradient_error(serial_grads, grads)
        parameters = tuple(actor.parameters())
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
        optimizer = torch.optim.AdamW(parameters, lr=1e-4)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        report["training"] = {
            "loss": losses,
            "forward_backward_seconds": train_seconds,
            "gradient_relative_l2_error": relative_error,
            "batch_gradient_within_5_percent": relative_error < 0.05,
            "gradient_norm": gradient_norm,
            "optimizer_steps": 1,
            "eligible_action_tokens": sum(e.token_count for e in examples),
            "actor_process_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        _, report["wake_sync_seconds"] = timed(lambda: backend.prepare_sampling(1))
        updated = backend.generate_batch([replace(requests[0], seed=args.seed + 100)])[0]
        report["initial_adapter"] = initial_receipt
        report["updated_adapter"] = backend.adapter_receipt
        report["checks"] = {
            "behavior_alignment": max(differences) < 0.5,
            "finite_batch_numerics": all(
                math.isfinite(v) for v in (*losses.values(), relative_error)
            ),
            "nonzero_finite_gradient": math.isfinite(gradient_norm) and gradient_norm > 0,
            "reference_unchanged": reference_before == reference_hash(),
            "adapter_changed": initial_receipt["adapter_sha256"]
            != backend.adapter_receipt["adapter_sha256"],
            "updated_generation": bool(updated),
        }
        actor.model.save_pretrained(
            args.output / "checkpoint",
            selected_adapters=["actor"],
            safe_serialization=True,
            save_embedding_layers=False,
        )
        report["checkpoint_files"] = {
            path.relative_to(args.output).as_posix(): fingerprint_file(path).sha256
            for path in sorted((args.output / "checkpoint").rglob("*"))
            if path.is_file()
        }
        atomic_write_text(args.output / "report.json", canonical_json_dumps(report) + "\n")
        return report, examples
    finally:
        backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("revision")
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--train-batch-tokens", type=int, default=1024)
    args = parser.parse_args()
    if min(args.requests, args.max_tokens, args.repeats, args.train_batch_tokens) < 1:
        parser.error("request, token, repeat, and training-batch limits must be positive")
    report, examples = run(args)
    # vLLM registers its own Transformers config classes, including Qwen vision
    # configs. A fresh interpreter keeps the independent HF diagnostic isolated
    # from those registrations. Only public token IDs and metadata cross over.
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
        parity = pool.submit(check_full_precision, args, examples).result()
    report["full_precision_parity"] = parity
    report["checks"]["full_precision_loss_alignment"] = (
        abs(parity["loss"]["serial"] - parity["loss"]["batched"]) < 1e-4
    )
    report["checks"]["full_precision_gradient_alignment"] = (
        parity["gradient_relative_l2_error"] < 1e-3
    )
    report["training"]["batching_default"] = (
        "serial; BF16 batch differences require workload validation"
    )
    atomic_write_text(args.output / "report.json", canonical_json_dumps(report) + "\n")
    print(canonical_json_dumps(report))
    if not all(report["checks"].values()):
        raise RuntimeError("CUDA acceptance checks failed; inspect report.json")


if __name__ == "__main__":
    main()
