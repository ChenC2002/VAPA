"""Optional, synchronous LoRA rollout engine; optimization stays in Transformers.

The integration is pinned to vLLM 0.31.0's V1 model runner. Set
VLLM_USE_V2_MODEL_RUNNER=0 before starting Python. GPU acceptance tests are
opt-in; CPU contract tests do not establish model compatibility or speedups.
"""

from __future__ import annotations

import importlib.metadata
import math
import os
import platform
import tempfile
from dataclasses import asdict
from pathlib import Path

from vapa.artifacts import artifact_fingerprint
from vapa.data.io import sha256_file
from vapa.model.protocols import OptionalDependencyError
from vapa.policies.text import GeneratedCandidate, GenerationRequest

VLLM_VERSION = "0.31.0"


def _release_training_cache() -> None:
    """Return unused PyTorch allocations before the sampler allocates or wakes."""
    import torch

    torch.cuda.empty_cache()


def load_vllm():
    """Fail before allocating GPU memory when the optional stack is unsupported."""
    if platform.system() != "Linux":
        raise OptionalDependencyError("vLLM rollouts require a separate Linux/CUDA environment")
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "0":
        raise OptionalDependencyError("set VLLM_USE_V2_MODEL_RUNNER=0 before starting Python")
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as error:
        raise OptionalDependencyError(
            "install vapa-ehr[train,vllm] in a Linux/CUDA environment"
        ) from error
    if version != VLLM_VERSION:
        raise OptionalDependencyError(
            f"this integration requires vllm=={VLLM_VERSION}, found {version}"
        )
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    if not torch.cuda.is_available():
        raise OptionalDependencyError("vLLM rollouts require CUDA")
    return LLM, SamplingParams, LoRARequest


class VLLMGenerationBackend:
    """One immutable adapter per sampling phase, with a synchronous update barrier."""

    def __init__(
        self,
        actor,
        tokenizer,
        *,
        model: str,
        revision: str,
        context_tokens: int,
        max_lora_rank: int,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.4,
        max_num_seqs: int = 32,
    ):
        if not getattr(actor, "adapter_name", None):
            raise ValueError("vLLM weight synchronization requires a named LoRA actor")
        if (
            isinstance(gpu_memory_utilization, bool)
            or not isinstance(gpu_memory_utilization, int | float)
            or not math.isfinite(gpu_memory_utilization)
            or not 0 < gpu_memory_utilization < 1
        ):
            raise ValueError("vLLM gpu_memory_utilization must be between zero and one")
        if type(max_num_seqs) is not int or max_num_seqs < 1:
            raise ValueError("vLLM max_num_seqs must be a positive integer")
        self.actor, self.tokenizer = actor, tokenizer
        self.engine_options = dict(
            model=model,
            revision=revision,
            dtype=dtype,
            max_model_len=context_tokens,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            max_loras=1,
            max_cpu_loras=1,
            enable_prefix_caching=True,
            enable_sleep_mode=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            logprobs_mode="processed_logprobs",
            generation_config="vllm",
            # Multimodal backbones may require a tokenizer even in text-only
            # mode. Requests still carry our exact prompt_token_ids: vLLM never
            # renders a chat template or retokenizes these prompts.
            skip_tokenizer_init=False,
            tokenizer=model,
            tokenizer_revision=revision,
            language_model_only=True,
            trust_remote_code=False,
            load_format="safetensors",
            logits_processors=["vapa.model.vllm_mask:ActionPrefixLogitsProcessor"],
        )
        self.engine = None
        self.version = None
        self.adapter = None
        self.adapter_receipt = None
        self._snapshot = None
        self._asleep = False
        self._closed = False
        self._generating = False
        self._next_id = 1
        self._params_type = self._lora_type = None

    def prepare_sampling(self, version: int) -> None:
        if self._closed or self._generating:
            raise RuntimeError("cannot synchronize a closed or busy rollout engine")
        if type(version) is not int or version < 0:
            raise ValueError("actor version must be a nonnegative optimizer step")
        if self.version is not None and version < self.version:
            raise ValueError("actor versions must not move backwards in a live engine")
        if self.engine is None:
            llm_type, self._params_type, self._lora_type = load_vllm()
            _release_training_cache()
            self.engine = llm_type(**self.engine_options)
        if self._asleep:
            _release_training_cache()
            self.engine.wake_up()
            self._asleep = False
        if version == self.version:
            return
        # Export only the actor adapter, never the shared frozen-reference adapter.
        snapshot = tempfile.TemporaryDirectory(prefix="vapa-rollout-adapter-")
        try:
            self.actor.eval()
            self.actor.model.save_pretrained(
                snapshot.name,
                selected_adapters=[self.actor.adapter_name],
                safe_serialization=True,
                save_embedding_layers=False,
            )
            path = Path(snapshot.name)
            if self.actor.adapter_name != "default":
                path /= self.actor.adapter_name
            required = (path / "adapter_config.json", path / "adapter_model.safetensors")
            if not all(file.is_file() for file in required):
                raise RuntimeError(
                    "actor export did not produce a complete safetensors LoRA adapter"
                )
            files = {file.name: sha256_file(file) for file in required}
            adapter = self._lora_type(f"actor-{self._next_id}", self._next_id, str(path))
            if not self.engine.reset_prefix_cache():
                raise RuntimeError("vLLM could not clear stale prefix cache at the update barrier")
            # generate(lora_request=...) loads the snapshot using vLLM's bounded
            # adapter cache. Do not duplicate its loading/eviction implementation.
        except BaseException:
            snapshot.cleanup()
            self.close()
            raise
        if self._snapshot is not None:
            self._snapshot.cleanup()
        self._snapshot, self.adapter, self.version = snapshot, adapter, version
        self._next_id += 1
        self.adapter_receipt = {
            "actor_step": version,
            "adapter_sha256": artifact_fingerprint(files),
            "files": files,
            "vllm_version": VLLM_VERSION,
            "logprobs_mode": "processed_logprobs",
            "policy_id": f"step-{version}:{artifact_fingerprint(files)}",
        }

    def prepare_training(self) -> None:
        if self._generating:
            raise RuntimeError("generation must finish before optimization")
        if self.engine is not None and not self._asleep:
            # Level 1 offloads engine weights to CPU AND discards KV cache.
            self.engine.sleep(level=1)
            self._asleep = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.engine is not None:
                # vLLM 0.31's offline LLM has no public close method. Keep this
                # version-pinned lifecycle exception isolated from training code.
                self.engine.llm_engine.engine_core.shutdown()
        finally:
            if self._snapshot is not None:
                self._snapshot.cleanup()

    def generate(self, messages, **kwargs):
        return self.generate_batch([GenerationRequest(messages=messages, **kwargs)])[0]

    def generate_batch(self, requests: list[GenerationRequest]) -> list[list[GeneratedCandidate]]:
        if not requests:
            return []
        if self._closed or self.engine is None or self.adapter is None or self._asleep:
            raise RuntimeError("prepare_sampling(actor_step) must precede vLLM generation")
        if self._generating:
            raise RuntimeError("concurrent calls into the rollout engine are not supported")
        from vapa.model.transformers import TransformersGenerationBackend

        prompts, params, masks = [], [], []
        for request in requests:
            values = asdict(request)
            values.pop("messages")
            TransformersGenerationBackend._validate_generation(**values)
            if (request.greedy and request.temperature != 0.0) or (
                not request.greedy and request.temperature == 0.0
            ):
                raise ValueError("greedy and temperature must specify the same sampling mode")
            prompt_ids = tuple(
                self.tokenizer.encode_messages(request.messages, add_generation_prompt=True)
            )
            if len(prompt_ids) + request.max_tokens > self.engine_options["max_model_len"]:
                raise ValueError(
                    "tokenized prompt plus max_tokens exceeds the configured context window"
                )
            mask = self.tokenizer.action_prefix_mask(request.messages, request.allowed_actions)
            eos = getattr(self.tokenizer.tokenizer, "eos_token_id", None)
            stop_ids = (
                list(eos) if isinstance(eos, list | tuple) else ([] if eos is None else [eos])
            )
            prompts.append({"prompt_token_ids": list(prompt_ids)})
            masks.append(mask)
            params.append(
                self._params_type(
                    n=request.n,
                    seed=request.seed,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k or -1,
                    max_tokens=request.max_tokens,
                    logprobs=0,
                    watermarking=False,
                    detokenize=False,
                    ignore_eos=True,
                    stop_token_ids=stop_ids,
                    extra_args={"vapa_action_mask": asdict(mask)},
                )
            )
        self._generating = True
        try:
            outputs = self.engine.generate(
                prompts, params, lora_request=self.adapter, use_tqdm=False
            )
        except BaseException:
            # Includes lazy adapter-load failures: never reuse a potentially
            # partially updated engine or silently fall back to the base model.
            self.close()
            raise
        finally:
            self._generating = False
        if len(outputs) != len(requests):
            raise RuntimeError("vLLM returned an incomplete request batch")
        result = []
        for request, prompt, mask, output in zip(requests, prompts, masks, outputs, strict=True):
            prompt_ids = tuple(prompt["prompt_token_ids"])
            if tuple(output.prompt_token_ids or ()) != prompt_ids:
                raise RuntimeError("vLLM changed or reordered input prompt token IDs")
            candidates = sorted(output.outputs, key=lambda item: item.index)
            if [item.index for item in candidates] != list(range(request.n)):
                raise RuntimeError("vLLM returned missing or duplicate candidate indices")
            group = []
            for candidate in candidates:
                ids = tuple(candidate.token_ids)
                if (
                    not ids
                    or len(ids) > request.max_tokens
                    or candidate.finish_reason not in {"stop", "length"}
                ):
                    raise RuntimeError("vLLM returned an empty, aborted, or oversized candidate")
                if candidate.logprobs is None or len(candidate.logprobs) != len(ids):
                    raise RuntimeError("vLLM omitted sampled-token behavior log probabilities")
                probabilities = []
                for index, token in enumerate(ids):
                    allowed, forbidden = (
                        mask.allowed_next(ids[:index]),
                        mask.forbidden_next(ids[:index]),
                    )
                    if (
                        token < 0
                        or token >= mask.vocab_size
                        or (allowed is not None and token not in allowed)
                        or token in forbidden
                    ):
                        raise RuntimeError(
                            "vLLM generation escaped the feasible-action prefix mask"
                        )
                    item = candidate.logprobs[index].get(token)
                    if item is None or not math.isfinite(item.logprob) or item.logprob > 1e-6:
                        raise RuntimeError("vLLM returned an invalid sampled-token log probability")
                    probabilities.append(float(item.logprob))
                text = self.tokenizer.decode_generated_action(ids, mask)
                if not text:
                    raise RuntimeError("vLLM generation has no parser-visible action text")
                group.append(
                    GeneratedCandidate(
                        text,
                        ids,
                        tuple(probabilities),
                        prompt_ids,
                        self.adapter_receipt["policy_id"],
                    )
                )
            result.append(group)
        return result

    def behavior_diagnostics(self, update, *, scaffold: str) -> dict[str, object]:
        """Audit policy-vs-behavior drift before updating; never reweight the loss."""
        from vapa.training.runtime import collect_vapa_action_groups

        self.actor.eval()
        # Matched-control queues may retain samples across optimizer steps. Record
        # the actual source policy of each retained decision, not just today's adapter.
        policy_tokens: dict[str, int] = {}
        for instance in update.instances:
            for rollout in instance.base_rollouts + instance.branch_rollouts:
                for turn in rollout.turns:
                    if turn.loss_mask and turn.decision is not None:
                        identity = turn.decision.behavior_policy_id or "unrecorded"
                        policy_tokens[identity] = (
                            policy_tokens.get(identity, 0) + turn.decision.token_count
                        )
        count, total, absolute, maximum = 0, 0.0, 0.0, 0.0
        groups = collect_vapa_action_groups(update, self.tokenizer, scaffold=scaffold)
        for group in groups:
            for action in group.examples:
                values = self.actor._action_log_probs([action.tokens], requires_grad=False)[0]
                for actual, behavior in zip(
                    values.tolist(), action.behavior_log_probs, strict=True
                ):
                    difference = actual - behavior
                    if not math.isfinite(difference):
                        raise RuntimeError("non-finite policy/behavior log-probability drift")
                    count += 1
                    total += difference
                    absolute += abs(difference)
                    maximum = max(maximum, abs(difference))
        return {
            "scope": "loss-enabled action tokens; unwarped training policy minus behavior",
            "tokens": count,
            "mean": total / count if count else None,
            "mean_abs": absolute / count if count else None,
            "max_abs": maximum,
            "current_actor_adapter": self.adapter_receipt,
            "behavior_policy_tokens": policy_tokens,
        }
