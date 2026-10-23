"""Muon for the hidden 2D weights, AdamW for everything else -- one optimiser.

Muon (Keller Jordan, 2024) keeps a Nesterov momentum buffer per weight matrix
and, instead of stepping along it, steps along its nearest semi-orthogonal
matrix: every singular value of the update is pushed towards one, so no
direction in the matrix is starved or dominated.  The orthogonalisation is a
quintic Newton-Schulz iteration -- five matmul-only steps, cheap enough to run
every step.

Muon is defined for matrices that map one hidden space to another.  The token
embedding (tied to the output head here), the norm gains and any biases are
not that, so they stay on AdamW.  Both halves live in *one* ``Optimizer`` so
that GradScaler, gradient clipping, checkpointing and the LR schedule handle
the Muon arm exactly as they handle the AdamW arm; and the AdamW half is
tested against ``torch.optim.AdamW``, so the only difference between the E4
arms is what happens to the hidden matrices.
"""

from __future__ import annotations

import math

import torch

# The quintic's coefficients, chosen to maximise the slope at zero so that
# small singular values grow fast.  The price is that the iteration does not
# converge to exactly one: after five steps singular values sit in roughly
# [0.7, 1.2], which Muon's authors found costs nothing.
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5


def orthogonalise(g: torch.Tensor, steps: int = NS_STEPS, eps: float = 1e-7) -> torch.Tensor:
    """Approximate ``U @ V.T`` for ``g = U S V.T``, batched over leading dims.

    Runs in float32 on every device.  The reference implementation uses
    bfloat16, which the T4 this project trains on does not have; at these
    matrix sizes the iteration costs about one percent of a training step
    even in float32.
    """
    if g.ndim < 2:
        raise ValueError(f"orthogonalise needs a matrix, got shape {tuple(g.shape)}")
    a, b, c = NS_COEFFICIENTS
    x = g.float()
    # Iterate on the wide orientation: the Gram matrix is then the small one.
    tall = x.size(-2) > x.size(-1)
    if tall:
        x = x.mT
    # Frobenius norm >= spectral norm, so this puts every singular value in
    # [0, 1], inside the iteration's basin.
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if tall:
        x = x.mT
    return x.to(g.dtype)


class MuonWithAdamW(torch.optim.Optimizer):
    """Muon on groups with ``use_muon=True``, AdamW on the rest.

    Muon group keys: ``lr``, ``momentum``, ``nesterov``, ``weight_decay`` and
    ``blocks`` -- the number of equal row blocks a fused weight is split into
    before orthogonalising (3 for a fused q/k/v projection, which is three
    matrices sharing one kernel, not one matrix).

    AdamW group keys: ``lr``, ``betas``, ``eps``, ``weight_decay``.

    Muon's step is scaled by ``sqrt(max(1, rows / cols))`` per block, so that
    the update's RMS does not shrink for matrices taller than they are wide.
    Weight decay is decoupled in both halves: ``p *= 1 - lr * weight_decay``.
    """

    def __init__(self, params, **defaults) -> None:
        base = {
            "use_muon": False,
            "lr": 0.02,
            "momentum": 0.95,
            "nesterov": True,
            "blocks": 1,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "weight_decay": 0.0,
        }
        unknown = set(defaults) - set(base)
        if unknown:
            raise ValueError(f"unknown MuonWithAdamW option(s) {sorted(unknown)}")
        super().__init__(params, {**base, **defaults})

        for group in self.param_groups:
            if not group["use_muon"]:
                continue
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(f"Muon needs 2D parameters, got shape {tuple(p.shape)}")
                if p.size(0) % group["blocks"]:
                    raise ValueError(
                        f"a {tuple(p.shape)} weight cannot be split into "
                        f"{group['blocks']} equal row blocks"
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon(group)
            else:
                self._adamw(group)
        return loss

    def _muon(self, group: dict) -> None:
        lr, momentum, blocks = group["lr"], group["momentum"], group["blocks"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state["momentum_buffer"] = torch.zeros_like(p)
            buf = state["momentum_buffer"]
            buf.lerp_(p.grad, 1 - momentum)
            update = p.grad.lerp(buf, momentum) if group["nesterov"] else buf

            rows, cols = p.size(0) // blocks, p.size(1)
            update = orthogonalise(update.reshape(blocks, rows, cols)).reshape_as(p)

            p.mul_(1 - lr * group["weight_decay"])
            p.add_(update, alpha=-lr * math.sqrt(max(1.0, rows / cols)))

    def _adamw(self, group: dict) -> None:
        # Mirrors torch.optim.AdamW's single-tensor path operation for
        # operation; tests/test_muon.py holds the two to the same trajectory.
        beta1, beta2 = group["betas"]
        lr, eps = group["lr"], group["eps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                # A CPU float tensor, as torch's AdamW keeps it, so that
                # Optimizer.load_state_dict treats it the same way.
                state["step"] = torch.tensor(0.0)
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            step = float(state["step"])
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

            p.mul_(1 - lr * group["weight_decay"])
            exp_avg.lerp_(p.grad, 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

            denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2**step)).add_(eps)
            p.addcdiv_(exp_avg, denom, value=-lr / (1 - beta1**step))


__all__ = ["MuonWithAdamW", "NS_COEFFICIENTS", "NS_STEPS", "orthogonalise"]
