"""End-to-end benchmarks of the ``Transformer`` model.

The masks and the greedy decoding loop are exercised on a reduced model so that
a single iteration stays fast, while keeping the vocabulary of the real
tokenizer (the decoding loop compares generated ids with the special tokens).
"""

import pytest
import torch

import tokenizer_finetune
from transformer import Transformer

GEN_MAX_SEQ_LEN = 8
GEN_D_MODEL = 64
GEN_N_HEADS = 4
GEN_D_FF = 256
GEN_N_LAYERS = 2
GEN_EPS = 1e-9

MASK_BATCH_SIZE = 8
MASK_SEQ_LEN = 127


@pytest.fixture(scope="module")
def tokenization():
    return tokenizer_finetune.Tokenization(GEN_MAX_SEQ_LEN)


@pytest.fixture(scope="module")
def model(tokenization):
    torch.manual_seed(0)
    model = Transformer(
        tokenization.get_vocab_len(),
        GEN_MAX_SEQ_LEN,
        GEN_N_HEADS,
        GEN_D_MODEL,
        GEN_D_FF,
        GEN_N_LAYERS,
        GEN_EPS,
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def prompt_tokens(tokenization):
    return tokenization.tokenize("what is the world?")


def test_lookahead_mask(benchmark, model):
    """Causal mask built for every forward pass."""
    mask = benchmark(model.lookahead_mask, MASK_BATCH_SIZE, MASK_SEQ_LEN)
    assert mask.shape == (MASK_BATCH_SIZE, 1, MASK_SEQ_LEN, MASK_SEQ_LEN)


def test_padding_mask(benchmark, model, tokenization):
    """Padding mask built from a batch of tokenized sentences."""
    tokens = tokenization.tokenize(["what is the world?"] * MASK_BATCH_SIZE)
    mask = benchmark(model.padding_mask, tokens)
    assert mask.shape == (MASK_BATCH_SIZE, 1, GEN_MAX_SEQ_LEN, GEN_MAX_SEQ_LEN)


def test_generate_greedy(benchmark, model, prompt_tokens):
    """Autoregressive greedy decoding, including the repetition penalty."""

    def run():
        # generate_greedy fills the input tensor in place, so it is cloned to
        # keep every iteration identical.
        with torch.no_grad():
            return model.generate_greedy(prompt_tokens.clone())

    answer = benchmark(run)
    assert isinstance(answer, str)
