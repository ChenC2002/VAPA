"""CPU contracts plus an explicitly requested, real CUDA acceptance test."""

import importlib
import json
import os
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vapa.model import vllm_backend
from vapa.model.protocols import ActionPrefixMask, OptionalDependencyError
from vapa.policies.text import GenerationRequest
from vapa.schemas import ActionKind


class Tokenizer:
    tokenizer = SimpleNamespace(eos_token_id=4)

    def encode_messages(self, messages, *, add_generation_prompt=True):
        assert add_generation_prompt
        return (7, 8)

    def action_prefix_mask(self, messages, allowed_actions):
        return ActionPrefixMask(((1, 2),), 10)

    def decode_generated_action(self, ids, mask):
        return 'Answer(prediction="ok", evidence=[])'


class Actor:
    adapter_name = "actor"
    weight = 1.0

    def __init__(self):
        self.model = self
        self.exports = 0

    def eval(self):
        pass

    def save_pretrained(self, path, **kwargs):
        assert kwargs == dict(
            selected_adapters=["actor"], safe_serialization=True, save_embedding_layers=False
        )
        self.exports += 1
        directory = Path(path) / "actor"
        directory.mkdir()
        (directory / "adapter_config.json").write_text("{}")
        (directory / "adapter_model.safetensors").write_bytes(str(self.weight).encode())


class Engine:
    def __init__(self, **kwargs):
        assert kwargs["logprobs_mode"] == "processed_logprobs"
        assert kwargs["skip_tokenizer_init"] is True
        assert kwargs["generation_config"] == "vllm"
        self.llm_engine = self.engine_core = self
        self.calls = []

    def add_lora(self, adapter):
        assert Path(adapter.lora_path, "adapter_model.safetensors").is_file()
        self.calls.append(("add", adapter.lora_int_id))
        return True

    def remove_lora(self, adapter_id):
        self.calls.append(("remove", adapter_id))
        return True

    def reset_prefix_cache(self):
        self.calls.append(("reset",))
        return True

    def sleep(self, level):
        self.calls.append(("sleep", level))

    def wake_up(self):
        self.calls.append(("wake",))

    def shutdown(self):
        self.calls.append(("shutdown",))

    def generate(self, prompts, params, **kwargs):
        self.last_requests = (prompts, params, kwargs)
        return [
            SimpleNamespace(
                prompt_token_ids=p["prompt_token_ids"],
                outputs=[
                    SimpleNamespace(
                        index=i,
                        token_ids=[1, 2, 4],
                        finish_reason="stop",
                        logprobs=[{t: SimpleNamespace(logprob=-0.5)} for t in [1, 2, 4]],
                    )
                    for i in reversed(range(param.n))
                ],
            )
            for p, param in zip(prompts, params, strict=True)
        ]


def request(n=1, seed=7):
    return GenerationRequest(
        [{"role": "user", "content": "test"}], n, seed, False, 1.0, 1.0, 0, 8, (ActionKind.ANSWER,)
    )


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(vllm_backend, "_release_training_cache", lambda: None)

    def lora(name, identifier, path):
        return SimpleNamespace(lora_int_id=identifier, lora_path=path)

    monkeypatch.setattr(vllm_backend, "load_vllm", lambda: (Engine, SimpleNamespace, lora))
    backend = vllm_backend.VLLMGenerationBackend(
        Actor(),
        Tokenizer(),
        model="fixture",
        revision="a" * 40,
        context_tokens=100,
        max_lora_rank=16,
    )
    yield backend
    backend.close()


def test_adapter_barrier_initial_resume_updates_sleep_and_cleanup(backend):
    with pytest.raises(RuntimeError, match="prepare_sampling"):
        backend.generate_batch([request()])
    # A resumed actor is exported at its restored step, not at construction.
    backend.actor.weight = 9.0
    backend.prepare_sampling(8)
    first = backend.adapter_receipt
    first_path = Path(backend.adapter.lora_path)
    backend.prepare_sampling(8)
    assert backend.actor.exports == 1
    backend.prepare_training()
    with pytest.raises(RuntimeError, match="prepare_sampling"):
        backend.generate_batch([request()])
    backend.actor.weight = 10.0
    backend.prepare_sampling(9)
    assert backend.actor.exports == 2
    assert first["adapter_sha256"] != backend.adapter_receipt["adapter_sha256"]
    assert not first_path.exists()
    assert backend.engine.calls == [
        ("reset",),
        ("add", 1),
        ("sleep", 1),
        ("wake",),
        ("remove", 1),
        ("reset",),
        ("add", 2),
    ]
    with pytest.raises(ValueError, match="backwards"):
        backend.prepare_sampling(0)
    last_path = Path(backend.adapter.lora_path)
    backend.close()
    backend.close()
    assert not last_path.exists()
    assert backend.engine.calls.count(("shutdown",)) == 1


def test_generation_is_token_in_and_preserves_groups_eos_probabilities(backend):
    backend.prepare_sampling(0)
    outputs = backend.generate_batch([request(n=3), request(n=1, seed=19)])
    assert list(map(len, outputs)) == [3, 1]
    candidate = outputs[0][0]
    assert candidate.token_ids == (1, 2, 4)  # EOS is sampled and charged.
    assert candidate.prompt_ids == (7, 8)
    assert candidate.behavior_policy_id == backend.adapter_receipt["policy_id"]
    assert candidate.log_probs == (-0.5,) * 3
    prompts, params, kwargs = backend.engine.last_requests
    assert prompts == [{"prompt_token_ids": [7, 8]}] * 2
    assert [p.seed for p in params] == [7, 19]
    assert [p.n for p in params] == [3, 1]
    assert params[0].top_k == -1
    assert params[0].stop_token_ids == [4]
    assert params[0].logprobs == 0
    assert kwargs["lora_request"] is backend.adapter
    assert backend.generate_batch([]) == []


@pytest.mark.parametrize("corruption", ["prompt", "index", "probability", "mask", "aborted"])
def test_generation_rejects_corrupt_engine_outputs(backend, corruption):
    backend.prepare_sampling(0)
    original = backend.engine.generate

    def corrupt(*args, **kwargs):
        outputs = original(*args, **kwargs)
        output, candidate = outputs[0], outputs[0].outputs[0]
        if corruption == "prompt":
            output.prompt_token_ids = [9]
        elif corruption == "index":
            candidate.index = 4
        elif corruption == "probability":
            candidate.logprobs[0][1].logprob = float("nan")
        elif corruption == "mask":
            candidate.token_ids[0] = 0
        else:
            candidate.finish_reason = "abort"
        return outputs

    backend.engine.generate = corrupt
    with pytest.raises(RuntimeError):
        backend.generate_batch([request()])


def test_generation_rejects_context_overflow(backend):
    backend.prepare_sampling(0)
    with pytest.raises(ValueError, match="context"):
        backend.generate_batch([replace(request(), max_tokens=100)])


def test_failed_adapter_sync_closes_engine_and_removes_temporary_exports(backend):
    backend.prepare_sampling(0)
    previous_path = Path(backend.adapter.lora_path)
    backend.engine.add_lora = lambda adapter: False
    with pytest.raises(RuntimeError, match="did not load"):
        backend.prepare_sampling(1)
    assert backend._closed
    assert not previous_path.exists()
    with pytest.raises(RuntimeError, match="closed"):
        backend.prepare_sampling(1)


@pytest.mark.parametrize("n,greedy,temperature", [(1, False, 0.0), (1, True, 1.0), (2, True, 0.0)])
def test_generation_rejects_inconsistent_sampling_modes(backend, n, greedy, temperature):
    backend.prepare_sampling(0)
    with pytest.raises(ValueError):
        backend.generate_batch([replace(request(n), greedy=greedy, temperature=temperature)])


def test_missing_vllm_is_actionable_without_importing_it(monkeypatch):
    monkeypatch.setattr(vllm_backend.platform, "system", lambda: "Darwin")
    with pytest.raises(OptionalDependencyError, match="Linux/CUDA"):
        vllm_backend.load_vllm()
    monkeypatch.setattr(vllm_backend.platform, "system", lambda: "Linux")
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    with pytest.raises(OptionalDependencyError, match="V2_MODEL_RUNNER=0"):
        vllm_backend.load_vllm()


def test_vllm_processor_and_hf_mask_have_identical_support(monkeypatch):
    torch = pytest.importorskip("torch")
    from vapa.model.transformers import _masked_logit_row

    # Load our worker bridge without installing vLLM in a CPU environment.
    module_name = "vllm.v1.sample.logits_processor"
    stub = ModuleType(module_name)
    stub.AdapterLogitsProcessor = object
    monkeypatch.setitem(sys.modules, module_name, stub)
    module = importlib.import_module("vapa.model.vllm_mask")
    try:
        mask = ActionPrefixMask(
            ((0, 1), (0, 2)),
            16,
            visible_prefixes=("A(", "B("),
            content_offsets=(1, 1),
            reasoning_start=(3,),
            reasoning_end=(4, 5),
            reasoning_action_prefixes=((1,), (2,)),
            reasoning_forbidden_tokens=(3, 5),
            action_forbidden_tokens=(3, 4, 5),
            action_terminators=((15,),),
        )
        processor = module.ActionPrefixLogitsProcessor()
        params = SimpleNamespace(
            extra_args={"vapa_action_mask": json.loads(json.dumps(asdict(mask)))}
        )
        apply = processor.new_req_logits_processor(params)
        prefixes = [
            (),
            (0,),
            (0, 1),
            (0, 2),
            (3,),
            (3, 6, 7),
            (3, 6, 4),
            (3, 6, 4, 5),
            (3, 6, 4, 5, 1),
            (0, 1, 9),
            (0, 1, 15),
        ]
        # Walk random legal paths, including unrestricted argument/reasoning text.
        rng = random.Random(10)
        for _ in range(30):
            prefix = ()
            for _ in range(12):
                prefixes.append(prefix)
                allowed = mask.allowed_next(prefix)
                choices = (
                    list(allowed)
                    if allowed is not None
                    else [t for t in range(mask.vocab_size) if t not in mask.forbidden_next(prefix)]
                )
                if not choices:
                    break
                prefix += (rng.choice(choices),)
        for prefix in prefixes:
            row = torch.arange(mask.vocab_size).float()
            if mask.allowed_next(prefix) == ():
                with pytest.raises(ValueError):
                    apply(list(prefix), row)
                continue
            expected = _masked_logit_row(
                row, mask.allowed_next(prefix), mask, mask.forbidden_next(prefix)
            )
            assert torch.equal(apply(list(prefix), row.clone()), expected)
        with pytest.raises(ValueError, match="action-prefix mask"):
            processor.new_req_logits_processor(SimpleNamespace(extra_args={}))
    finally:
        sys.modules.pop("vapa.model.vllm_mask", None)


@pytest.mark.vllm
@pytest.mark.skipif(
    not os.environ.get("VAPA_VLLM_TEST_MODEL"),
    reason="set VAPA_VLLM_TEST_MODEL and VAPA_VLLM_TEST_REVISION on Linux/CUDA",
)
def test_real_vllm_mask_adapter_update_and_behavior_probabilities():
    """Explicit model download/large-GPU opt-in; never enabled in the CPU suite."""
    torch = pytest.importorskip("torch")
    from vapa.model.protocols import TokenizedAction
    from vapa.model.transformers import (
        TransformersActorAdapter,
        TransformersTokenizerAdapter,
        apply_shared_lora,
    )

    revision = os.environ["VAPA_VLLM_TEST_REVISION"]
    assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
    model = os.environ["VAPA_VLLM_TEST_MODEL"]
    actor = TransformersActorAdapter.from_pretrained(
        model, revision=revision, device="cuda:0", dtype="bfloat16", model_kind="causal"
    )
    actor, reference = apply_shared_lora(
        actor, rank=16, alpha=32, dropout=0.0, target_modules=("q_proj", "v_proj")
    )
    tokenizer = TransformersTokenizerAdapter.from_pretrained(model, revision=revision)
    backend = vllm_backend.VLLMGenerationBackend(
        actor, tokenizer, model=model, revision=revision, context_tokens=2048, max_lora_rank=16
    )
    try:
        backend.prepare_sampling(0)
        first_receipt = backend.adapter_receipt
        req = replace(
            request(n=2),
            messages=[{"role": "user", "content": "Return Answer(1, [])."}],
            max_tokens=64,
        )
        candidates = backend.generate_batch([req, replace(req, seed=19)])
        backend.prepare_training()
        mask = tokenizer.action_prefix_mask(req.messages, req.allowed_actions)
        actor.eval()

        def assert_alignment(candidates):
            for candidate in candidates:
                tokens = TokenizedAction(
                    candidate.prompt_ids, candidate.token_ids, prefix_mask=mask
                )
                actual = actor._action_log_probs([tokens], requires_grad=False)[0]
                assert torch.isfinite(actual).all()
                assert (
                    max(
                        abs(a - b)
                        for a, b in zip(actual.tolist(), candidate.log_probs, strict=True)
                    )
                    < 0.5
                )

        assert_alignment(candidates[0])
        with torch.no_grad():
            for name, parameter in actor.model.named_parameters():
                if ".actor." in name and "lora_B" in name:
                    parameter.add_(0.01)
        backend.prepare_sampling(1)
        assert first_receipt["adapter_sha256"] != backend.adapter_receipt["adapter_sha256"]
        updated = backend.generate_batch([req])[0]
        assert len(updated) == 2
        backend.prepare_training()
        assert_alignment(updated)
    finally:
        backend.close()
