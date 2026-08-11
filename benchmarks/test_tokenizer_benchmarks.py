"""Benchmarks for the tokenization pipelines.

Both pipelines wrap the GPT-2 tokenizer extended with the ``<pad>``, ``<bos>``,
``<eos>`` and ``<sep>`` special tokens. The fine-tuning variant additionally
appends a separator to every sentence.
"""

import pytest

import tokenizer_finetune
import tokenizer_pretraining

MAX_SEQ_LEN = 127

SENTENCE = (
    "Artificial intelligence is the ability of a machine to perform tasks that "
    "usually require human reasoning, such as understanding language."
)
CORPUS = [
    "what is artificial intelligence technology?",
    "what is love?",
    "what do you think about Europe?",
    "what is democracy?",
    "which is the best country in the world?",
    "what is the difference between animals and humans?",
    "what is the meaning of emotion?",
    SENTENCE,
] * 4


@pytest.fixture(scope="module")
def pretraining_tokenizer():
    return tokenizer_pretraining.Tokenization(MAX_SEQ_LEN)


@pytest.fixture(scope="module")
def finetune_tokenizer():
    return tokenizer_finetune.Tokenization(MAX_SEQ_LEN)


def test_pretraining_process_sentence(benchmark, pretraining_tokenizer):
    """Encode a single sentence and pad it to the context length."""
    token_ids = benchmark(pretraining_tokenizer.process_sentence, SENTENCE)
    assert len(token_ids) == MAX_SEQ_LEN


def test_pretraining_tokenize_corpus(benchmark, pretraining_tokenizer):
    """Tokenize a batch of sentences into a padded tensor."""
    tokens = benchmark(pretraining_tokenizer.tokenize, CORPUS)
    assert tokens.shape == (len(CORPUS), MAX_SEQ_LEN)


def test_finetune_tokenize_corpus(benchmark, finetune_tokenizer):
    """Fine-tuning pipeline: same as above with an appended separator token."""
    tokens = benchmark(finetune_tokenizer.tokenize, CORPUS)
    assert tokens.shape == (len(CORPUS), MAX_SEQ_LEN)


def test_pretraining_decode(benchmark, pretraining_tokenizer):
    """Decode a generated sequence back to text."""
    tokens = pretraining_tokenizer.tokenize(SENTENCE)
    text = benchmark(pretraining_tokenizer.decode, tokens[0])
    assert text
