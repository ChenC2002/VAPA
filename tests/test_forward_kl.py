"""Numerical regression tests for the paper's masked Eq. 9 objective."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from vapa.model.protocols import ActionPrefixMask, TokenizedAction, VAPAAction
from vapa.model.transformers import TransformersActorAdapter
from vapa.training.loss import forward_kl
from vapa.training.rl_train import RLRunSettings


def test_advantage_scales_and_normalization_use_fp32_population_convention() -> None:
    from vapa.training.advantages import _fp32, _normalize, _population_std, _validate_weights

    assert _population_std([]) == 0.0
    assert _population_std([1.0, 1.0]) == 0.0
    assert _population_std([-1.0, 1.0]) == 1.0  # population, not sample SD
    assert _population_std([-2.0, 0.0, 2.0]) == _fp32(math.sqrt(_fp32(8.0 / 3.0)))
    assert _normalize(1.0, 1.0, 1e-8) == 1.0  # epsilon rounds away in FP32 here
    with pytest.raises(ValueError, match="FP32"):
        _validate_weights(epsilon=1e-100)
    with pytest.raises(ValueError, match="FP32"):
        _population_std([1e100])


@pytest.mark.parametrize(
    "settings", [{"ratio_clip": 0.2}, {"kl_mode": "k3"}, {"kl_mode": "log_ratio"}]
)
def test_paper_settings_reject_legacy_surrogates(settings: dict) -> None:
    with pytest.raises(ValueError, match="Eq. 9"):
        RLRunSettings(**settings)


@pytest.mark.parametrize(
    "policy,reference",
    [([], []), ([0.0], [-1.0]), ([math.nan], [0.0]), ([0.0, -math.inf], [math.log(0.5)] * 2)],
)
def test_reference_kl_rejects_invalid_distributions(policy, reference) -> None:
    with pytest.raises(ValueError):
        forward_kl(policy, reference)


@pytest.mark.model
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("kl_weight", [0.0, 0.01, 1.0])
def test_chunked_loss_and_gradients_match_dense_forward_kl(masked: bool, kl_weight: float) -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(17)

    class LogitModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.randn(90, 7))

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=self.logits[: input_ids.shape[1]].unsqueeze(0))

    actor_model, reference_model = LogitModel(), LogitModel()
    actor = TransformersActorAdapter(actor_model, device="cpu")
    reference = TransformersActorAdapter(reference_model, device="cpu")
    mask = ActionPrefixMask(((1, 2), (1, 3)), 7, action_forbidden_tokens=(6,)) if masked else None
    examples = [
        VAPAAction(TokenizedAction((0,), (1, 2) + (4,) * 70, mask), (-0.7,) * 72, 1.25),
        VAPAAction(TokenizedAction((0, 5), (1, 3, 5), mask), (-2.0,) * 3, -0.5),
    ]
    # Independent dense calculation selects legal indices explicitly. It neither
    # uses the adapter's masking implementation nor its chunk/checkpoint reduction.
    policy_terms, kl_terms = [], []
    for example in examples:
        for index, target in enumerate(example.tokens.action_ids):
            position = len(example.tokens.prompt_ids) - 1 + index
            legal = (
                ([1] if index == 0 else [2, 3] if index == 1 else list(range(6)))
                if masked
                else list(range(7))
            )
            p_log = actor_model.logits[position, legal].log_softmax(-1)
            q_log = reference_model.logits[position, legal].detach().log_softmax(-1)
            policy_terms.append(-example.advantage * p_log[legal.index(target)])
            kl_terms.append((p_log.exp() * (p_log - q_log)).sum())
    dense_policy = torch.stack(policy_terms).mean()
    dense_kl = torch.stack(kl_terms).mean()
    dense = dense_policy + kl_weight * dense_kl
    expected_grad = torch.autograd.grad(dense, actor_model.logits)[0]
    report = actor.vapa_loss(
        examples, reference=reference, kl_weight=kl_weight, ratio_clip=None, kl_mode="forward"
    )
    assert report.token_count == 75
    assert report.policy == pytest.approx(dense_policy.item(), abs=1e-6)
    assert report.kl == pytest.approx(dense_kl.item(), abs=1e-6)
    assert report.total == pytest.approx(dense.item(), abs=1e-6)
    report.backward()
    torch.testing.assert_close(actor_model.logits.grad, expected_grad, rtol=2e-5, atol=1e-7)
    assert torch.isfinite(actor_model.logits.grad).all()
    assert reference_model.logits.grad is None
    if masked:
        assert torch.count_nonzero(actor_model.logits.grad[:, 6]) == 0
    # Sampling probabilities are provenance only: Eq. 9 has no importance ratio.
    changed_behavior = [
        VAPAAction(item.tokens, (-10.0,) * item.token_count, item.advantage) for item in examples
    ]
    assert actor.vapa_loss(
        changed_behavior,
        reference=reference,
        kl_weight=kl_weight,
        ratio_clip=None,
        kl_mode="forward",
    ).total == pytest.approx(report.total)
