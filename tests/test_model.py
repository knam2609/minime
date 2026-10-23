"""Correctness tests for the Transformer.

The two that matter most are the causal-mask tests: a leaky mask lets the model
see the future, which makes the loss curve look *better*, not worse.  Nothing
downstream would catch it.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from minime.config import ModelConfig
from minime.model import MiniMeLM, RotaryCache, apply_rotary, build_model

ARCH_COMBOS = list(itertools.product(["rmsnorm", "layernorm"], ["rope", "learned"], ["swiglu", "gelu"]))


def _model(cfg: ModelConfig) -> MiniMeLM:
    model = build_model(cfg)
    model.eval()
    return model


# --------------------------------------------------------------- causality


def test_future_tokens_cannot_change_past_logits(tiny_cfg):
    """Perturb the suffix; the prefix logits must be bit-identical."""
    model = _model(tiny_cfg)
    idx = torch.randint(0, tiny_cfg.vocab_size, (2, tiny_cfg.context_length))
    cut = 7

    logits_a, _ = model(idx)
    idx_b = idx.clone()
    idx_b[:, cut:] = (idx_b[:, cut:] + 13) % tiny_cfg.vocab_size
    logits_b, _ = model(idx_b)

    torch.testing.assert_close(logits_a[:, :cut], logits_b[:, :cut], rtol=0, atol=0)
    assert not torch.allclose(logits_a[:, cut:], logits_b[:, cut:]), "suffix should have changed"


def test_causal_mask_gradient_is_exactly_zero_for_future_positions(tiny_cfg):
    """The stricter form: d logits[t] / d embedding[t+k] == 0 for every k > 0."""
    model = build_model(tiny_cfg)
    captured = {}

    def hook(_module, _inp, out):
        out.retain_grad()
        captured["emb"] = out

    handle = model.tok_emb.register_forward_hook(hook)
    idx = torch.randint(0, tiny_cfg.vocab_size, (1, tiny_cfg.context_length))
    logits, _ = model(idx)
    target_t = 5
    logits[0, target_t].sum().backward()
    handle.remove()

    grad = captured["emb"].grad
    assert grad is not None
    future = grad[0, target_t + 1 :]
    assert future.abs().max().item() == 0.0, "future positions leaked into an earlier logit"
    assert grad[0, : target_t + 1].abs().max().item() > 0.0, "past positions should matter"


@pytest.mark.parametrize("norm,pos,ffn", ARCH_COMBOS)
def test_sdpa_matches_manual_attention(norm, pos, ffn):
    """The fused kernel and the textbook implementation must agree exactly enough."""
    cfg = ModelConfig(
        vocab_size=64, d_model=32, n_heads=4, n_layers=2, context_length=12,
        dropout=0.0, norm=norm, pos=pos, ffn=ffn,
    )
    model = _model(cfg)
    idx = torch.randint(0, cfg.vocab_size, (3, cfg.context_length))

    with torch.no_grad():
        fused, _ = model(idx, use_sdpa=True)
        manual, _ = model(idx, use_sdpa=False)
    torch.testing.assert_close(fused, manual, rtol=1e-4, atol=1e-5)


# ------------------------------------------------------------------ shapes


def test_head_split_merge_roundtrip(tiny_cfg):
    from minime.model import CausalSelfAttention

    attn = CausalSelfAttention(tiny_cfg)
    x = torch.randn(2, tiny_cfg.context_length, tiny_cfg.d_model)
    split = attn._split_heads(x)
    assert split.shape == (2, tiny_cfg.n_heads, tiny_cfg.context_length, tiny_cfg.head_dim)
    torch.testing.assert_close(attn._merge_heads(split), x)


@pytest.mark.parametrize("norm,pos,ffn", ARCH_COMBOS)
def test_forward_shapes_and_finiteness(norm, pos, ffn):
    cfg = ModelConfig(
        vocab_size=64, d_model=32, n_heads=4, n_layers=2, context_length=12,
        dropout=0.0, norm=norm, pos=pos, ffn=ffn,
    )
    model = _model(cfg)
    idx = torch.randint(0, cfg.vocab_size, (3, cfg.context_length))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (3, cfg.context_length, cfg.vocab_size)
    assert torch.isfinite(logits).all()
    assert loss is not None and torch.isfinite(loss)


def test_context_length_is_enforced(tiny_cfg):
    model = _model(tiny_cfg)
    too_long = torch.randint(0, tiny_cfg.vocab_size, (1, tiny_cfg.context_length + 1))
    with pytest.raises(ValueError, match="exceeds context"):
        model(too_long)


# -------------------------------------------------------------------- init


@pytest.mark.parametrize("vocab_size", [97, 1024, 16384])
def test_initial_loss_is_near_uniform_entropy(vocab_size):
    """At init the model knows nothing, so loss should be ~ln(V).

    Much lower means the init is leaking signal; much higher means the logit
    scale has blown up.  Either way the run is starting from a bad place.
    """
    cfg = ModelConfig(
        vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=3, context_length=16, dropout=0.0
    )
    model = _model(cfg)
    idx = torch.randint(0, vocab_size, (8, cfg.context_length))
    # Targets must be independent of the inputs.  Re-using `idx` as its own
    # target asks the model to copy the current token, which weight tying makes
    # easy at init -- the loss then sits well below ln(V) for a good reason.
    targets = torch.randint(0, vocab_size, (8, cfg.context_length))
    with torch.no_grad():
        _, loss = model(idx, targets=targets)
    assert loss.item() == pytest.approx(math.log(vocab_size), abs=0.25)


def test_residual_projections_are_scaled_down():
    shallow = build_model(ModelConfig(vocab_size=64, d_model=64, n_heads=4, n_layers=2, context_length=8))
    deep = build_model(ModelConfig(vocab_size=64, d_model=64, n_heads=4, n_layers=16, context_length=8))
    shallow_std = shallow.blocks[0].attn.out_proj.weight.std().item()
    deep_std = deep.blocks[0].attn.out_proj.weight.std().item()
    assert deep_std < shallow_std, "deeper models must initialise residual writes smaller"


def test_weight_tying_shares_storage(tiny_cfg):
    model = build_model(tiny_cfg)
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()

    untied = build_model(ModelConfig(**{**tiny_cfg.__dict__, "tie_embeddings": False}))
    assert untied.lm_head.weight.data_ptr() != untied.tok_emb.weight.data_ptr()
    assert untied.num_parameters() > model.num_parameters()


def test_param_groups_partition_every_parameter(tiny_cfg):
    model = build_model(tiny_cfg)
    groups = model.param_groups(weight_decay=0.1)
    grouped = [p for g in groups for p in g["params"]]
    unique_model_params = {id(p) for p in model.parameters()}
    assert len({id(p) for p in grouped}) == len(grouped), "a parameter was placed in two groups"
    assert {id(p) for p in grouped} == unique_model_params
    assert all(p.dim() >= 2 for p in groups[0]["params"])
    assert groups[1]["weight_decay"] == 0.0


# -------------------------------------------------------------------- RoPE


def test_rope_attention_logit_depends_only_on_relative_offset():
    """The defining property of RoPE, and the thing a wrong pairing breaks."""
    head_dim, seq = 16, 32
    cache = RotaryCache(head_dim, seq)
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    def logit(m: int, n: int) -> float:
        cos_m, sin_m = cache.get(1, offset=m)
        cos_n, sin_n = cache.get(1, offset=n)
        return float((apply_rotary(q, cos_m, sin_m) * apply_rotary(k, cos_n, sin_n)).sum())

    for offset in (0, 1, 5):
        values = [logit(m, m - offset) for m in range(offset, offset + 6)]
        assert max(values) - min(values) < 1e-4, f"offset {offset} is not translation invariant"


def test_rope_is_norm_preserving():
    cache = RotaryCache(16, 8)
    x = torch.randn(2, 3, 8, 16)
    cos, sin = cache.get(8)
    rotated = apply_rotary(x, cos, sin)
    torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1), rtol=1e-5, atol=1e-5)


def test_rope_rejects_sequences_beyond_the_cache():
    cache = RotaryCache(16, 8)
    with pytest.raises(ValueError, match="exceeds the RoPE cache"):
        cache.get(9)


# -------------------------------------------------------------------- norms


def test_rmsnorm_matches_reference_formula():
    from minime.model import RMSNorm

    norm = RMSNorm(16)
    with torch.no_grad():
        norm.weight.normal_()
    x = torch.randn(4, 7, 16) * 5.0
    expected = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps) * norm.weight
    torch.testing.assert_close(norm(x), expected, rtol=1e-5, atol=1e-5)


def test_rmsnorm_output_has_unit_rms_before_gain():
    from minime.model import RMSNorm

    norm = RMSNorm(64)
    x = torch.randn(8, 64) * 100.0
    rms = norm(x).pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------- generation


def test_generate_extends_sequence_and_respects_context(tiny_cfg):
    model = _model(tiny_cfg)
    prompt = torch.randint(0, tiny_cfg.vocab_size, (2, 4))
    out = model.generate(prompt, max_new_tokens=25, temperature=0.8, top_k=10)
    assert out.shape == (2, 29)
    assert torch.equal(out[:, :4], prompt)
    assert out.max().item() < tiny_cfg.vocab_size


def test_greedy_generation_is_deterministic(tiny_cfg):
    model = _model(tiny_cfg)
    prompt = torch.randint(0, tiny_cfg.vocab_size, (1, 3))
    a = model.generate(prompt, max_new_tokens=8, temperature=0.0)
    b = model.generate(prompt, max_new_tokens=8, temperature=0.0)
    assert torch.equal(a, b)


def test_generation_restores_training_mode(tiny_cfg):
    model = build_model(tiny_cfg)
    model.train()
    model.generate(torch.zeros(1, 2, dtype=torch.long), max_new_tokens=2)
    assert model.training


def test_tied_head_favours_copying_the_current_token_at_init():
    """Documents the shift convention that `forward` warns about.

    With tied embeddings the residual stream still carries the input embedding,
    so at init the model scores the *current* token highly.  Predicting the
    current token is therefore much easier than chance, while predicting an
    independent token is not.  A training loss below ln(V) at step 0 is the
    signature of an accidentally unshifted target.
    """
    cfg = ModelConfig(vocab_size=1024, d_model=64, n_heads=4, n_layers=3, context_length=16)
    model = _model(cfg)
    idx = torch.randint(0, cfg.vocab_size, (8, cfg.context_length))
    independent = torch.randint(0, cfg.vocab_size, (8, cfg.context_length))
    with torch.no_grad():
        _, self_loss = model(idx, targets=idx)
        _, indep_loss = model(idx, targets=independent)
    assert self_loss.item() < indep_loss.item() - 0.5
    assert indep_loss.item() == pytest.approx(math.log(cfg.vocab_size), abs=0.25)
