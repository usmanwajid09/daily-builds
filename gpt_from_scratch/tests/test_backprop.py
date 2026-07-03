"""Gradient checks: compare each layer's analytic backward() against a
central-difference numerical gradient of a scalar loss. This is the
standard way to catch sign errors / wrong formulas in hand-written
backprop - see check_gradient() below.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpt_from_scratch.attention import CausalSelfAttention
from gpt_from_scratch.block import FeedForward, TransformerBlock
from gpt_from_scratch.layers import Linear, LayerNorm, gelu, gelu_backward
from gpt_from_scratch.loss import cross_entropy_loss
from gpt_from_scratch.model import TinyGPT


def check_gradient(param, grad_analytic, loss_fn, rng, eps=1e-5, rtol=1e-2, atol=1e-4, n_checks=6):
    """Perturb a handful of random entries of `param` and compare the
    central-difference numerical gradient of loss_fn() (which must read
    `param`'s current values via closure) against grad_analytic."""
    flat_size = param.size
    n = min(n_checks, flat_size)
    flat_indices = rng.choice(flat_size, size=n, replace=False)
    flat_view = param.reshape(-1)
    grad_flat = grad_analytic.reshape(-1)

    for flat_i in flat_indices:
        orig = flat_view[flat_i]
        flat_view[flat_i] = orig + eps
        loss_plus = loss_fn()
        flat_view[flat_i] = orig - eps
        loss_minus = loss_fn()
        flat_view[flat_i] = orig

        numeric = (loss_plus - loss_minus) / (2 * eps)
        analytic = grad_flat[flat_i]
        assert np.isclose(numeric, analytic, rtol=rtol, atol=atol), (
            f"gradient mismatch at flat index {flat_i}: "
            f"numeric={numeric!r}, analytic={analytic!r}"
        )


def test_gelu_backward_matches_numerical_gradient():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 4))
    dout = rng.normal(size=(3, 4))

    def loss_fn():
        return float(np.sum(gelu(x) * dout))

    dx = gelu_backward(x, dout)
    check_gradient(x, dx, loss_fn, rng)


def test_linear_backward_matches_numerical_gradient():
    rng = np.random.default_rng(1)
    lin = Linear(4, 3, rng)
    x = rng.normal(size=(2, 5, 4))
    dout = rng.normal(size=(2, 5, 3))

    def loss_fn():
        return float(np.sum(lin.forward(x) * dout))

    lin.forward(x)
    dx = lin.backward(dout)

    check_gradient(x, dx, loss_fn, rng)
    check_gradient(lin.W, lin.dW, loss_fn, rng)
    check_gradient(lin.b, lin.db, loss_fn, rng)


def test_layernorm_backward_matches_numerical_gradient():
    rng = np.random.default_rng(2)
    ln = LayerNorm(6)
    x = rng.normal(size=(2, 4, 6))
    dout = rng.normal(size=(2, 4, 6))

    def loss_fn():
        return float(np.sum(ln.forward(x) * dout))

    ln.forward(x)
    dx = ln.backward(dout)

    check_gradient(x, dx, loss_fn, rng)
    check_gradient(ln.gamma, ln.dgamma, loss_fn, rng, n_checks=6)
    check_gradient(ln.beta, ln.dbeta, loss_fn, rng, n_checks=6)


def test_attention_backward_matches_numerical_gradient():
    rng = np.random.default_rng(3)
    attn = CausalSelfAttention(d_model=8, n_heads=2, rng=rng)
    x = rng.normal(size=(2, 4, 8))
    dout = rng.normal(size=(2, 4, 8))

    def loss_fn():
        return float(np.sum(attn.forward(x) * dout))

    attn.forward(x)
    dx = attn.backward(dout)

    check_gradient(x, dx, loss_fn, rng)
    for w, dw in attn.parameters_and_grads():
        check_gradient(w, dw, loss_fn, rng, n_checks=4)


def test_feedforward_backward_matches_numerical_gradient():
    rng = np.random.default_rng(4)
    ffn = FeedForward(d_model=6, rng=rng)
    x = rng.normal(size=(2, 3, 6))
    dout = rng.normal(size=(2, 3, 6))

    def loss_fn():
        return float(np.sum(ffn.forward(x) * dout))

    ffn.forward(x)
    dx = ffn.backward(dout)

    check_gradient(x, dx, loss_fn, rng)
    for w, dw in ffn.parameters_and_grads():
        check_gradient(w, dw, loss_fn, rng, n_checks=4)


def test_transformer_block_backward_matches_numerical_gradient():
    rng = np.random.default_rng(5)
    block = TransformerBlock(d_model=8, n_heads=2, rng=rng)
    x = rng.normal(size=(2, 4, 8))
    dout = rng.normal(size=(2, 4, 8))

    def loss_fn():
        return float(np.sum(block.forward(x) * dout))

    block.forward(x)
    dx = block.backward(dout)

    check_gradient(x, dx, loss_fn, rng)
    for w, dw in block.parameters_and_grads():
        check_gradient(w, dw, loss_fn, rng, n_checks=3)


def test_cross_entropy_backward_matches_numerical_gradient():
    rng = np.random.default_rng(6)
    logits = rng.normal(size=(2, 3, 5))
    targets = rng.integers(0, 5, size=(2, 3))

    def loss_fn():
        loss, _ = cross_entropy_loss(logits, targets)
        return loss

    _, dlogits = cross_entropy_loss(logits, targets)
    check_gradient(logits, dlogits, loss_fn, rng, n_checks=8)


def test_full_model_backward_matches_numerical_gradient():
    """End-to-end integration check: embeddings -> blocks -> final layernorm
    -> head -> cross-entropy, gradients checked at every stage."""
    rng = np.random.default_rng(7)
    vocab = 6
    model = TinyGPT(vocab_size=vocab, d_model=8, n_heads=2, n_layers=2, max_seq_len=5, seed=7)
    batch, seq = 2, 4
    token_ids = rng.integers(0, vocab, size=(batch, seq))
    targets = rng.integers(0, vocab, size=(batch, seq))

    def loss_fn():
        logits = model.forward(token_ids)
        loss, _ = cross_entropy_loss(logits, targets)
        return loss

    logits = model.forward(token_ids)
    loss, dlogits = cross_entropy_loss(logits, targets)
    model.backward(dlogits)

    check_gradient(model.token_emb, model.d_token_emb, loss_fn, rng, n_checks=8)
    check_gradient(model.pos_emb, model.d_pos_emb, loss_fn, rng, n_checks=4)
    check_gradient(model.blocks[0].attn.query.W, model.blocks[0].attn.query.dW, loss_fn, rng, n_checks=5)
    check_gradient(model.blocks[-1].ffn.fc2.W, model.blocks[-1].ffn.fc2.dW, loss_fn, rng, n_checks=5)
    check_gradient(model.ln_final.gamma, model.ln_final.dgamma, loss_fn, rng, n_checks=3)
    check_gradient(model.head.W, model.head.dW, loss_fn, rng, n_checks=5)
