"""The overfitting gate, kept in the normal test suite.

Fast enough on CPU to run every time, and it is the single most informative
test in the repository: it exercises the mask, the loss, the target shift and
the optimiser wiring together, end to end.
"""

from __future__ import annotations

import pytest

from minime.config import ModelConfig
from minime.training.overfit import run_overfit


@pytest.mark.slow
def test_tiny_model_memorises_a_single_batch():
    result = run_overfit(steps=250, lr=3e-3, batch_size=4, device="cpu")
    assert result.initial_loss > 3.0, "should start near the uniform prior"
    assert result.passed, f"model failed to memorise a batch: {result.summary()}"
    assert result.steps_to_threshold is not None and result.steps_to_threshold < 200


@pytest.mark.slow
@pytest.mark.parametrize("norm,pos,ffn", [("rmsnorm", "rope", "swiglu"), ("layernorm", "learned", "gelu")])
def test_both_architecture_stacks_can_overfit(norm, pos, ffn):
    """Neither arm of the architecture ablation may be broken."""
    cfg = ModelConfig(
        vocab_size=64, d_model=64, n_heads=4, n_layers=2, context_length=64,
        dropout=0.0, norm=norm, pos=pos, ffn=ffn,
    )
    result = run_overfit(cfg=cfg, steps=250, lr=3e-3, batch_size=4, device="cpu")
    assert result.passed, f"{norm}/{pos}/{ffn} failed: {result.summary()}"


@pytest.mark.slow
def test_overfitting_target_is_not_trivially_copyable():
    """A model with no context cannot solve the task.

    Guards the test itself: if the fixture were predictable from the current
    token alone, a broken causal mask would still pass.
    """
    crippled = ModelConfig(
        vocab_size=64, d_model=64, n_heads=4, n_layers=2, context_length=64, dropout=0.0
    )
    result = run_overfit(cfg=crippled, steps=5, lr=3e-3, batch_size=4, device="cpu")
    assert result.losses[0] > 3.0
