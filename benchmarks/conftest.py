import os
import sys

import pytest
import torch

# The model modules (attention.py, embedding.py, ...) live at the repository
# root, so make them importable from the benchmarks directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Keep the measurements deterministic: a single thread avoids the scheduling
# noise introduced by the intra-op thread pool of PyTorch.
torch.set_num_threads(1)


# Small but representative model configuration. The released checkpoint uses
# d_model=768, n_heads=12, d_ff=3072 and n_layers=12; the benchmarks keep the
# same shapes with smaller dimensions so a single iteration stays in the
# millisecond range.
D_MODEL = 128
N_HEADS = 4
D_FF = 512
N_LAYERS = 4
BATCH_SIZE = 2
SEQ_LEN = 64
EPS = 1e-9


@pytest.fixture(scope="session")
def generator():
    return torch.Generator().manual_seed(0)


@pytest.fixture(scope="session")
def hidden_states(generator):
    return torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, generator=generator)


@pytest.fixture(scope="session")
def lookahead_mask():
    mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1)
    mask[mask == 1] = float("-inf")
    return mask.view(1, 1, SEQ_LEN, SEQ_LEN).expand(BATCH_SIZE, -1, -1, -1)


@pytest.fixture(scope="session")
def padding_mask():
    return torch.zeros(BATCH_SIZE, 1, SEQ_LEN, SEQ_LEN)
