"""Gradient checks and behavior tests for the v2 upgrade: Dropout,
TiedHead, gradient clipping, and the LR schedule. Reuses the
central-difference `check_gradient` helper from test_backprop.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpt_from_scratch.layers import Dropout, Linear, TiedHead
from gpt_from_scratch.loss import cross_entropy_loss
from gpt_from_scratch.model import TinyGPT
from gpt_from_scratch.optim import Adam, clip_grad_norm, cosine_lr_with_warmup
from gpt_from_scratch.tests.test_backprop import check_gradient


# ---------- Dropout ----------

def test_dropout_eval_mode_is_identity():
    rng = np.random.default_rng(0)
    d = Dropout(p=0.5)
    x = rng.normal(size=(3, 4))
    out = d.forward(x, training=False, rng=rng)
    assert np.array_equal(out, x)


def test_dropout_zero_p_is_identity_even_in_training():
    rng = np.random.default_rng(0)
    d = Dropout(p=0.0)
    x = rng.normal(size=(3, 4))
    out = d.forward(x, training=True, rng=rng)
    assert np.array_equal(out, x)


def test_dropout_rejects_invalid_p():
    with pytest.raises(ValueError):
        Dropout(p=1.0)
    with pytest.raises(ValueError):
        Dropout(p=-0.1)


def test_dropout_training_zeroes_some_fraction():
    rng = np.random.default_rng(0)
    d = Dropout(p=0.5)
    x = np.ones((200, 50))
    out = d.forward(x, training=True, rng=rng)
    frac_zeroed = np.mean(out == 0.0)
    assert 0.35 < frac_zeroed < 0.65  # should be close to p=0.5


def test_dropout_backward_matches_numerical_gradient():
    """Gradient check with a FIXED mask: reuse the same rng seed on every
    perturbed forward call so the random mask doesn't change out from
    under the finite-difference comparison (that would compare gradients
    of two different functions)."""
    x = np.random.default_rng(1).normal(size=(4, 5))
    dout = np.random.default_rng(2).normal(size=(4, 5))
    d = Dropout(p=0.4)

    def loss_fn():
        y = d.forward(x, training=True, rng=np.random.default_rng(42))
        return float(np.sum(y * dout))

    d.forward(x, training=True, rng=np.random.default_rng(42))
    dx = d.backward(dout)
    check_gradient(x, dx, loss_fn, np.random.default_rng(3), n_checks=5)


# ---------- TiedHead ----------

def test_tied_head_backward_matches_numerical_gradient():
    rng = np.random.default_rng(4)
    vocab, d_model = 6, 5
    token_emb = rng.normal(size=(vocab, d_model))
    head = TiedHead(token_emb, vocab)
    x = rng.normal(size=(2, 3, d_model))
    dout = rng.normal(size=(2, 3, vocab))

    def loss_fn():
        return float(np.sum(head.forward(x) * dout))

    head.forward(x)
    dx = head.backward(dout)

    check_gradient(x, dx, loss_fn, rng)
    check_gradient(head.bias, head.dbias, loss_fn, rng)
    # token_emb gradient check: perturb token_emb directly (loss_fn reads
    # it via closure since `head.token_emb is token_emb`)
    check_gradient(token_emb, head.d_token_emb_contrib, loss_fn, rng, n_checks=6)


def test_tied_head_shares_the_same_array_not_a_copy():
    rng = np.random.default_rng(5)
    token_emb = rng.normal(size=(4, 3))
    head = TiedHead(token_emb, 4)
    assert head.token_emb is token_emb


# ---------- TinyGPT with tie_weights / dropout end-to-end gradient check ----------

def test_tied_and_dropout_model_backward_matches_numerical_gradient():
    """Integration check: with dropout_p=0 (so the check is deterministic)
    but tie_weights=True, verify the full model's gradient w.r.t.
    token_emb correctly sums BOTH contributions (embedding lookup + head
    projection) - this is exactly the kind of bug (double counting, or
    forgetting one of the two paths) that would slip through a shape-only
    test but not a numerical gradient check."""
    rng = np.random.default_rng(6)
    vocab = 6
    model = TinyGPT(vocab_size=vocab, d_model=8, n_heads=2, n_layers=2, max_seq_len=5, seed=6, tie_weights=True)
    batch, seq = 2, 4
    token_ids = rng.integers(0, vocab, size=(batch, seq))
    targets = rng.integers(0, vocab, size=(batch, seq))

    def loss_fn():
        logits = model.forward(token_ids, training=False)
        loss, _ = cross_entropy_loss(logits, targets)
        return loss

    logits = model.forward(token_ids, training=False)
    loss, dlogits = cross_entropy_loss(logits, targets)
    model.backward(dlogits)

    check_gradient(model.token_emb, model.d_token_emb, loss_fn, rng, n_checks=8)
    check_gradient(model.head.bias, model.head.dbias, loss_fn, rng, n_checks=4)


def test_model_with_dropout_trains_and_reduces_loss():
    rng = np.random.default_rng(7)
    model = TinyGPT(vocab_size=8, d_model=16, n_heads=2, n_layers=2, max_seq_len=8, seed=7, dropout_p=0.1)
    opt = Adam(lr=1e-2)
    ids = rng.integers(0, 8, size=(4, 6))
    targets = rng.integers(0, 8, size=(4, 6))
    losses = []
    for _ in range(80):
        logits = model.forward(ids, training=True, rng=rng)
        loss, dlogits = cross_entropy_loss(logits, targets)
        model.backward(dlogits)
        opt.step(model.parameters_and_grads())
        losses.append(loss)
    assert losses[-1] < losses[0]


# ---------- gradient clipping ----------

def test_clip_grad_norm_rescales_to_max_norm():
    rng = np.random.default_rng(8)
    p1, g1 = rng.normal(size=(3, 3)), rng.normal(size=(3, 3)) * 50
    p2, g2 = rng.normal(size=(4,)), rng.normal(size=(4,)) * 50
    pre_norm = clip_grad_norm([(p1, g1), (p2, g2)], max_norm=1.0)
    post_norm = np.sqrt(np.sum(g1 ** 2) + np.sum(g2 ** 2))
    assert pre_norm > 1.0
    assert np.isclose(post_norm, 1.0, atol=1e-4)


def test_clip_grad_norm_leaves_small_grads_untouched():
    rng = np.random.default_rng(9)
    p1, g1 = rng.normal(size=(3,)), rng.normal(size=(3,)) * 0.01
    g1_orig = g1.copy()
    clip_grad_norm([(p1, g1)], max_norm=10.0)
    assert np.array_equal(g1, g1_orig)


def test_clip_grad_norm_rejects_nonpositive_max_norm():
    with pytest.raises(ValueError):
        clip_grad_norm([(np.zeros(2), np.zeros(2))], max_norm=0)


# ---------- LR schedule ----------

def test_lr_schedule_warmup_increases_to_peak():
    lrs = [cosine_lr_with_warmup(s, max_lr=1e-3, warmup_steps=10, max_steps=100) for s in range(10)]
    assert all(a <= b for a, b in zip(lrs, lrs[1:]))  # monotonically increasing
    assert np.isclose(lrs[-1], 1e-3 * 10 / 10)  # step 9 -> (9+1)/10 * max_lr = max_lr


def test_lr_schedule_peaks_at_warmup_end():
    lr_at_warmup_end = cosine_lr_with_warmup(10, max_lr=1e-3, warmup_steps=10, max_steps=100)
    assert np.isclose(lr_at_warmup_end, 1e-3)


def test_lr_schedule_decays_after_warmup():
    lr_mid = cosine_lr_with_warmup(50, max_lr=1e-3, warmup_steps=10, max_steps=100)
    lr_end = cosine_lr_with_warmup(99, max_lr=1e-3, warmup_steps=10, max_steps=100)
    assert lr_mid > lr_end


def test_lr_schedule_floors_at_min_lr_ratio_past_max_steps():
    lr_way_past = cosine_lr_with_warmup(1000, max_lr=1e-3, warmup_steps=10, max_steps=100, min_lr_ratio=0.1)
    assert np.isclose(lr_way_past, 1e-4)


def test_lr_schedule_rejects_bad_args():
    with pytest.raises(ValueError):
        cosine_lr_with_warmup(0, max_lr=1e-3, warmup_steps=-1, max_steps=100)
    with pytest.raises(ValueError):
        cosine_lr_with_warmup(0, max_lr=1e-3, warmup_steps=10, max_steps=0)
