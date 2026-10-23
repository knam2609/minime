#!/usr/bin/env python3
"""Prove the model can memorise a single batch before spending money on a GPU.

    python scripts/overfit_test.py --config configs/tiny.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minime.config import load_config  # noqa: E402
from minime.training.overfit import run_overfit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="optional model config")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--plot", type=Path, default=Path("reports/overfit.png"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    model_cfg = None
    if args.config:
        cfg = load_config(args.config)
        model_cfg = cfg.model
        # A small vocab keeps the softmax honest at this size; the point is the
        # optimisation path, not the embedding table.
        model_cfg.vocab_size = min(model_cfg.vocab_size, 256)

    print(f"overfitting a single batch on {args.device} ...")
    result = run_overfit(
        cfg=model_cfg,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        threshold=args.threshold,
        device=args.device,
        log_every=max(1, args.steps // 6),
    )
    print(result.summary())

    if not args.no_plot:
        _plot(result, args.plot)
        print(f"plot -> {args.plot}")

    if result.passed:
        print("PASS: the model can memorise a batch; larger training is justified.")
        return 0
    print("FAIL: the model cannot memorise a batch. Do not start a real run.")
    return 1


def _plot(result, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result.losses, linewidth=1.5)
    ax.axhline(result.threshold, linestyle="--", linewidth=1, color="crimson")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (log scale)")
    ax.set_title(f"Overfitting test: {result.summary()}", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
