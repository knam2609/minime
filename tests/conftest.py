"""Make ``src/`` importable without requiring an editable install.

Keeps a fresh clone one ``pytest`` away from a green suite.
"""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _cpu_and_seeded():
    """Every test runs on CPU with a fixed seed: fast, and comparable anywhere."""
    torch.manual_seed(0)
    torch.set_num_threads(2)
    yield


@pytest.fixture
def tiny_cfg():
    from minime.config import ModelConfig

    return ModelConfig(
        vocab_size=97, d_model=32, n_heads=4, n_layers=2, context_length=16, dropout=0.0
    )


@pytest.fixture
def repo_root():
    return REPO_ROOT
