"""Right-padded model forwards must preserve action-only losses and gradients."""

import pytest

from vapa.model.protocols import ActionPrefixMask, TokenizedAction, VAPAAction
from vapa.model.transformers import TransformersActorAdapter, apply_shared_lora


@pytest.mark.model
@pytest.mark.parametrize("shared", [False, True])
def test_padded_training_batches_match_serial_losses_gradients_and_reference(shared):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(38)
    config = transformers.GPT2Config(
        vocab_size=17,
        n_positions=32,
        n_embd=16,
        n_layer=1,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )
    actor = TransformersActorAdapter(transformers.GPT2LMHeadModel(config), device="cpu")
    if shared:
        pytest.importorskip("peft")
        actor, reference = apply_shared_lora(
            actor,
            rank=2,
            alpha=4,
            target_modules=("c_attn",),
        )
    else:
        reference = TransformersActorAdapter(transformers.GPT2LMHeadModel(config), device="cpu")
    mask = ActionPrefixMask(((1, 2), (1, 3)), 17, action_forbidden_tokens=(16,))
    tokens = (
        TokenizedAction((4, 5), (1, 2, 8), mask),
        TokenizedAction((4,), (1, 3, 6, 7, 8, 9), mask),
        TokenizedAction((4, 5, 6, 7), (1, 2), mask),
    )
    examples = tuple(
        VAPAAction(t, (-1.0,) * len(t.action_ids), a)
        for t, a in zip(tokens, (1.2, -0.5, 0.2), strict=True)
    )
    baseline = None
    for budget in (0, 1, 14, 64):
        actor.forward_batch_tokens = budget
        actor.model.zero_grad(set_to_none=True)
        actor.train()
        reference.eval()
        report = actor.vapa_loss(
            examples,
            reference=reference,
            kl_weight=0.2,
            ratio_clip=None,
            kl_mode="forward",
        )
        report.backward()
        gradients = {
            name: p.grad.clone() for name, p in actor.model.named_parameters() if p.grad is not None
        }
        if baseline is None:
            baseline = report.total, gradients
        else:
            assert report.total == pytest.approx(baseline[0], abs=1e-6)
            assert gradients.keys() == baseline[1].keys()
            for name in gradients:
                torch.testing.assert_close(gradients[name], baseline[1][name], rtol=2e-4, atol=1e-6)
        assert report.token_count == 11
        assert all(
            p.grad is None
            for n, p in reference.model.named_parameters()
            if not shared or ".reference." in n
        )
    assert list(actor._action_batch_slices(tokens)) == [(0, 3)]
    actor.forward_batch_tokens = 14
    assert list(actor._action_batch_slices(tokens)) == [(0, 2), (2, 3)]
    serial_probs = []
    for budget in (0, 64):
        actor.forward_batch_tokens = budget
        serial_probs.append(actor._action_log_probs(tokens, requires_grad=False))
    for serial, batched in zip(*serial_probs, strict=True):
        torch.testing.assert_close(serial, batched)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_forward_batch_token_limit_is_validated(value):
    with pytest.raises(ValueError, match="forward_batch_tokens"):
        TransformersActorAdapter(lambda **kwargs: None, forward_batch_tokens=value)
