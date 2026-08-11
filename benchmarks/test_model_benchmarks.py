"""Benchmarks for the building blocks of the pragya transformer.

These benchmarks only depend on PyTorch: they exercise the attention block, the
positional embedding and the decoder layers without loading the GPT-2
tokenizer.
"""

import pytest
import torch

from attention import Attention
from embedding import Embedding
from layer import Layer

from conftest import BATCH_SIZE, D_FF, D_MODEL, EPS, N_HEADS, N_LAYERS, SEQ_LEN

VOCAB_LEN = 50261


@pytest.fixture(scope="module")
def attention():
    return Attention(D_MODEL, N_HEADS).eval()


@pytest.fixture(scope="module")
def layer():
    return Layer(D_MODEL, D_FF, N_HEADS, EPS).eval()


@pytest.fixture(scope="module")
def decoder_layers():
    return torch.nn.ModuleList(
        [Layer(D_MODEL, D_FF, N_HEADS, EPS) for _ in range(N_LAYERS)]
    ).eval()


@pytest.fixture(scope="module")
def embedding():
    return Embedding(VOCAB_LEN, SEQ_LEN, D_MODEL).eval()


@pytest.fixture(scope="module")
def token_ids(generator):
    return torch.randint(
        0, VOCAB_LEN, (BATCH_SIZE, SEQ_LEN), generator=generator, dtype=torch.long
    )


def test_embedding_lookup(benchmark, embedding, token_ids):
    """Token embedding scaling plus the sinusoidal positional encoding."""
    with torch.no_grad():
        out = benchmark(embedding.get_embedding, token_ids)
    assert out.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)


def test_attention_forward(benchmark, attention, hidden_states, padding_mask, lookahead_mask):
    """Single masked multi-head attention block."""
    with torch.no_grad():
        out = benchmark(attention.forward, hidden_states, padding_mask, lookahead_mask)
    assert out.shape == hidden_states.shape


@pytest.mark.parametrize("seq_len", [32, 128])
def test_attention_forward_seq_len(benchmark, attention, generator, seq_len):
    """Attention cost as a function of the context length (quadratic term)."""
    x = torch.randn(BATCH_SIZE, seq_len, D_MODEL, generator=generator)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask[mask == 1] = float("-inf")
    lookahead = mask.view(1, 1, seq_len, seq_len).expand(BATCH_SIZE, -1, -1, -1)
    padding = torch.zeros(BATCH_SIZE, 1, seq_len, seq_len)

    with torch.no_grad():
        out = benchmark(attention.forward, x, padding, lookahead)
    assert out.shape == x.shape


def test_layer_forward(benchmark, layer, hidden_states, padding_mask, lookahead_mask):
    """One decoder layer: pre-norm attention plus the feed-forward network."""
    with torch.no_grad():
        out = benchmark(layer.forward, hidden_states, padding_mask, lookahead_mask)
    assert out.shape == hidden_states.shape


def test_decoder_stack_forward(
    benchmark, embedding, decoder_layers, token_ids, padding_mask, lookahead_mask
):
    """Full decoder pass: embedding followed by the stack of decoder layers."""

    def run():
        x = embedding.get_embedding(token_ids)
        for decoder_layer in decoder_layers:
            x = decoder_layer(x, padding_mask, lookahead_mask)
        return x

    with torch.no_grad():
        out = benchmark(run)
    assert out.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)


def test_layer_backward(benchmark, layer, hidden_states, padding_mask, lookahead_mask):
    """Training step of a decoder layer: forward pass plus back-propagation."""

    def run():
        layer.zero_grad(set_to_none=True)
        x = hidden_states.detach().requires_grad_(True)
        out = layer(x, padding_mask, lookahead_mask)
        out.sum().backward()
        return x.grad

    grad = benchmark(run)
    assert grad.shape == hidden_states.shape
