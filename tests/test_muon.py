"""Muon (M7): the orthogonalisation, the hybrid split, and the AdamW half.

The E4 comparison is only fair if the Muon arm differs from the AdamW arm in
the hidden matrices and nowhere else, so the AdamW half is held to
``torch.optim.AdamW``'s trajectory and the Muon half to ``torch.optim.Muon``.
"""

from __future__ import annotations

import json

import pytest
import torch

from minime.config import ModelConfig, TrainingConfig
from minime.model import build_model, iter_named_2d
from minime.training.checkpoint import load_checkpoint, save_checkpoint
from minime.training.logging import read_metrics
from minime.training.optim import MuonWithAdamW, build_optimizer, orthogonalise
from minime.training.overfit import _structured_tokens

# ------------------------------------------------------ Newton-Schulz


def _with_condition_number(rows: int, cols: int, cond: float) -> torch.Tensor:
    k = min(rows, cols)
    u, _ = torch.linalg.qr(torch.randn(rows, k))
    v, _ = torch.linalg.qr(torch.randn(cols, k))
    return u @ torch.diag(torch.logspace(0, -torch.log10(torch.tensor(cond)).item(), k)) @ v.T


@pytest.mark.parametrize("shape", [(32, 32), (96, 32), (32, 96)])
def test_orthogonalise_pushes_singular_values_towards_one(shape):
    g = _with_condition_number(*shape, cond=100.0)
    before = torch.linalg.svdvals(g)
    assert before.max() / before.min() > 50

    out = orthogonalise(g)
    after = torch.linalg.svdvals(out)
    assert out.shape == g.shape
    # The quintic trades exact convergence for speed: ~[0.7, 1.2], not 1.
    assert after.min() > 0.6 and after.max() < 1.25

    u, _, vh = torch.linalg.svd(g, full_matrices=False)
    polar = u @ vh
    cosine = (out * polar).sum() / (out.norm() * polar.norm())
    assert cosine > 0.97, "the update must point where U V^T points"


def test_orthogonalise_is_batched_over_leading_dims():
    g = torch.randn(3, 16, 24)
    batched = orthogonalise(g)
    for i in range(3):
        torch.testing.assert_close(batched[i], orthogonalise(g[i]))


def test_orthogonalise_ignores_the_input_scale():
    g = torch.randn(16, 24)
    torch.testing.assert_close(orthogonalise(g), orthogonalise(1e-4 * g), rtol=1e-4, atol=1e-5)


def test_orthogonalise_rejects_a_vector():
    with pytest.raises(ValueError, match="matrix"):
        orthogonalise(torch.randn(8))


def _one_param_muon(shape, **group):
    p = torch.nn.Parameter(torch.randn(*shape))
    opt = MuonWithAdamW([{"params": [p], "use_muon": True, **group}])
    return p, opt


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch has no reference Muon")
@pytest.mark.parametrize("shape", [(48, 16), (16, 48)])
def test_muon_matches_the_torch_reference(shape):
    """Momentum, Nesterov, decoupled decay and the shape scale, three steps.

    torch runs Newton-Schulz in bfloat16 and we run it in float32, so the
    tolerance is a few percent: enough to catch a wrong transpose, scale or
    momentum rule, which are off by far more.
    """
    torch.manual_seed(0)
    ours, opt = _one_param_muon(shape, lr=0.02, momentum=0.95, weight_decay=0.1)
    ref = torch.nn.Parameter(ours.detach().clone())
    ref_opt = torch.optim.Muon([ref], lr=0.02, momentum=0.95, weight_decay=0.1, nesterov=True)
    start = ours.detach().clone()

    for _ in range(3):
        grad = torch.randn(shape)
        ours.grad, ref.grad = grad.clone(), grad.clone()
        opt.step()
        ref_opt.step()

    ours_delta, ref_delta = ours.detach() - start, ref.detach() - start
    assert (ours_delta - ref_delta).norm() / ref_delta.norm() < 0.05


@pytest.mark.parametrize("shape", [(48, 16), (16, 48)])
def test_one_step_is_the_scaled_orthogonalised_gradient(shape):
    """With no momentum and no decay a step is exactly -lr * scale * NS(g)."""
    p, opt = _one_param_muon(shape, lr=0.1, momentum=0.0, weight_decay=0.0)
    start = p.detach().clone()
    p.grad = torch.randn(shape)
    opt.step()

    rows, cols = shape
    scale = max(1.0, rows / cols) ** 0.5
    torch.testing.assert_close(p.detach() - start, -0.1 * scale * orthogonalise(p.grad))


def test_fused_weight_is_orthogonalised_block_by_block():
    """A fused q/k/v weight is three square maps, each with scale 1."""
    d = 16
    p, opt = _one_param_muon((3 * d, d), lr=0.1, momentum=0.0, blocks=3)
    start = p.detach().clone()
    p.grad = torch.randn(3 * d, d)
    opt.step()

    expected = torch.cat([orthogonalise(block) for block in p.grad.split(d)])
    torch.testing.assert_close(p.detach() - start, -0.1 * expected)
    # Orthogonalising the stack instead would be a different update.
    assert not torch.allclose(expected, 3**0.5 * orthogonalise(p.grad), atol=1e-3)


def test_muon_rejects_what_it_cannot_orthogonalise():
    with pytest.raises(ValueError, match="2D"):
        MuonWithAdamW([{"params": [torch.nn.Parameter(torch.ones(4))], "use_muon": True}])
    with pytest.raises(ValueError, match="row blocks"):
        MuonWithAdamW([{"params": [torch.nn.Parameter(torch.ones(4, 4))], "use_muon": True, "blocks": 3}])
    with pytest.raises(ValueError, match="unknown"):
        MuonWithAdamW([torch.nn.Parameter(torch.ones(4, 4))], lr_muon=0.1)


# ------------------------------------------------------- the hybrid


def _muon_cfg(**kwargs) -> TrainingConfig:
    defaults = dict(optimizer="muon", learning_rate=1e-3, muon_lr=0.02, weight_decay=0.1)
    return TrainingConfig(**{**defaults, **kwargs})


ARCHITECTURES = [
    {},
    {"tie_embeddings": False},
    {"pos": "learned"},
    {"bias": True, "ffn": "gelu", "norm": "layernorm"},
]


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_hybrid_assigns_every_parameter_exactly_once(tiny_cfg, arch):
    model = build_model(ModelConfig(**{**tiny_cfg.__dict__, **arch}))
    opt = build_optimizer(model, _muon_cfg())

    placed = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(placed) == len(set(placed)), "a parameter is in two groups"
    assert set(placed) == {id(p) for p in model.parameters()}

    muon = {id(p) for g in opt.param_groups if g["use_muon"] for p in g["params"]}
    assert muon == {id(p) for _, p in iter_named_2d(model)}
    for name, param in model.named_parameters():
        if "emb" in name or "lm_head" in name or param.dim() < 2:
            assert id(param) not in muon, f"{name} must stay on AdamW"


def test_hybrid_group_settings(tiny_cfg):
    model = build_model(tiny_cfg)
    opt = build_optimizer(model, _muon_cfg(muon_momentum=0.9, muon_weight_decay=0.01))
    qkv = {id(b.attn.qkv_proj.weight) for b in model.blocks}

    for group in opt.param_groups:
        ids = {id(p) for p in group["params"]}
        if group["use_muon"]:
            assert (group["lr"], group["momentum"], group["weight_decay"]) == (0.02, 0.9, 0.01)
            assert group["blocks"] == (3 if ids <= qkv else 1)
            assert group["blocks"] == 1 or ids == qkv
        else:
            assert group["lr"] == 1e-3
            # The embedding keeps the decay it has in the AdamW arm; norms none.
            if id(model.tok_emb.weight) in ids:
                assert group["weight_decay"] == 0.1
            else:
                assert group["weight_decay"] == 0.0


def test_adamw_half_follows_torch_adamw(tiny_cfg):
    """Embedding, head and norms must train identically in both E4 arms."""
    torch.manual_seed(0)
    model = build_model(tiny_cfg)
    twin = build_model(tiny_cfg)
    twin.load_state_dict(model.state_dict())

    cfg = _muon_cfg()
    hybrid = build_optimizer(model, cfg)
    aux_ids = {id(p) for g in hybrid.param_groups if not g["use_muon"] for p in g["params"]}
    names = [n for n, p in model.named_parameters() if id(p) in aux_ids]
    assert names, "no AdamW parameters to compare"

    twin_params = dict(twin.named_parameters())
    reference = torch.optim.AdamW(
        [
            {"params": [twin_params[n] for n in names if twin_params[n].dim() >= 2], "weight_decay": 0.1},
            {"params": [twin_params[n] for n in names if twin_params[n].dim() < 2], "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate, betas=cfg.betas, eps=1e-8,
    )

    for _ in range(5):
        grads = {n: torch.randn_like(p) for n, p in model.named_parameters()}
        for n, p in model.named_parameters():
            p.grad = grads[n].clone()
        for n, p in twin.named_parameters():
            p.grad = grads[n].clone()
        hybrid.step()
        reference.step()

    params = dict(model.named_parameters())
    for n in names:
        torch.testing.assert_close(params[n], twin_params[n], rtol=1e-6, atol=1e-7, msg=n)


def test_muon_state_round_trips_through_a_checkpoint(tmp_path, tiny_cfg):
    torch.manual_seed(0)
    model = build_model(tiny_cfg)
    opt = build_optimizer(model, _muon_cfg())
    idx = torch.randint(0, tiny_cfg.vocab_size, (2, tiny_cfg.context_length))

    def step(m, o):
        _, loss = m(idx, targets=idx)
        o.zero_grad(set_to_none=True)
        loss.backward()
        o.step()

    for _ in range(3):
        step(model, opt)
    save_checkpoint(tmp_path / "c.pt", model=model, optimizer=opt, step=3)

    fresh = build_model(tiny_cfg)
    fresh_opt = build_optimizer(fresh, _muon_cfg())
    load_checkpoint(tmp_path / "c.pt", model=fresh, optimizer=fresh_opt)

    saved, loaded = opt.state_dict(), fresh_opt.state_dict()
    assert saved["param_groups"] == loaded["param_groups"]
    kinds = {k for state in loaded["state"].values() for k in state}
    assert kinds == {"momentum_buffer", "step", "exp_avg", "exp_avg_sq"}
    for key, state in saved["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, loaded["state"][key][name], rtol=0, atol=0)

    step(model, opt)
    step(fresh, fresh_opt)
    for a, b in zip(model.parameters(), fresh.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.slow
def test_a_tiny_model_memorises_a_batch_with_muon():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, d_model=64, n_heads=4, n_layers=2, context_length=64)
    tokens = _structured_tokens(4, cfg.context_length + 1, cfg.vocab_size, seed=0)
    x, y = tokens[:, :-1], tokens[:, 1:]
    model = build_model(cfg)
    opt = build_optimizer(model, _muon_cfg(weight_decay=0.0))

    losses = []
    for _ in range(150):
        _, loss = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < 0.1 * losses[0], losses[::15]


@pytest.mark.slow
def test_a_muon_run_logs_both_learning_rates(tmp_path):
    from test_training import _make_trainer

    trainer, cfg = _make_trainer(tmp_path, optimizer="muon", muon_lr=0.02, learning_rate=1e-3)
    trainer.train()
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["config"]["training"]["optimizer"] == "muon"

    train = [r for r in read_metrics(tmp_path / "run" / "metrics.jsonl") if r.get("phase") == "train"]
    assert train
    for r in train:
        assert r["muon_lr"] / r["lr"] == pytest.approx(0.02 / 1e-3)
