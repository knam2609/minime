#!/usr/bin/env python3
"""The M10 report's frontier: personal CE against general-weighted CE on 512-token windows.

    python scripts/plot_frontier.py        # -> reports/m10/frontier.png

One panel per split, each from its common-context table: val
(``reports/m10/common512.json``) and, once the test phase has run, test
(``reports/m10/test.json``).  Every run is a small point, every group's mean
a large one; the dashed line joins the group means no other mean beats on
both axes.  Lower is better on both.  Descriptive, like the tables: the
probes are the controlled study, the flagship and its adaptation single runs
of another size, budget and context.  P1 never saw the owner's messages; its
personal CE (~7.4) is far above the rest, so it is marked at the top edge
with its value rather than stretching the axis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from minime.training.m10 import non_dominated  # noqa: E402
from minime.training.noise_floor import PRIMARY_EVAL_METRICS  # noqa: E402

PERSONAL, GENERAL = PRIMARY_EVAL_METRICS
STYLE = {   # group -> (colour, marker, label)
    "P1": ("0.55", "o", "P1: general only"),
    "P4": ("tab:blue", "o", "P4: chat from step 0"),
    "E1 25/75": ("tab:orange", "o", "E1 25/75: P1 + adaptation"),
    "E1 50/50": ("tab:green", "o", "E1 50/50: P1 + adaptation"),
    "flagship": ("black", "s", "flagship (49M)"),
    "adaptation": ("tab:red", "*", "flagship + C (the final model)"),
}
OFF_SCALE = ("P1",)     # drawn at the top edge, its value written beside it


def panels(val: Path, test: Path) -> dict[str, dict]:
    out = {"val": json.loads(val.read_text())}
    if test.exists():
        out["test"] = json.loads(test.read_text())["common_context"]
    return out


def plot(tables: dict[str, dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    on_scale = [run["values"][PERSONAL] for t in tables.values() for g, group in t["groups"].items()
                if g not in OFF_SCALE for run in group["runs"].values()]
    xs = [run["values"][GENERAL] for t in tables.values() for group in t["groups"].values()
          for run in group["runs"].values()]
    pad_y, pad_x = 0.06 * (max(on_scale) - min(on_scale)), 0.06 * (max(xs) - min(xs))
    top = max(on_scale) + 3 * pad_y
    fig, axes = plt.subplots(1, len(tables), figsize=(6.2 * len(tables), 5.2), sharex=True, sharey=True,
                             squeeze=False)
    for ax, (split, table) in zip(axes[0], tables.items()):
        means = {g: (group["mean"][GENERAL], group["mean"][PERSONAL]) for g, group in table["groups"].items()}
        front = non_dominated(means)
        ax.plot([means[g][0] for g in front], [means[g][1] for g in front], "--", color="0.6", lw=1, zorder=1,
                 label="no other mean lower on both")
        for g, group in table["groups"].items():
            colour, marker, label = STYLE[g]
            x, y = means[g]
            if g in OFF_SCALE:
                ax.annotate(f"{g}: personal {y:.2f}", xy=(x, top), xytext=(x, top - 2.2 * pad_y),
                            ha="center", fontsize=8, color=colour,
                            arrowprops={"arrowstyle": "->", "color": colour})
                ax.scatter([x], [top], marker=marker, s=60, color=colour, zorder=3, clip_on=False, label=label)
                continue
            runs = list(group["runs"].values())
            if len(runs) > 1:
                ax.scatter([r["values"][GENERAL] for r in runs], [r["values"][PERSONAL] for r in runs],
                           marker=marker, s=14, color=colour, alpha=0.35, zorder=2, linewidths=0)
            ax.scatter([x], [y], marker=marker, s=160 if marker == "*" else 70, color=colour, zorder=3,
                       edgecolors="white", linewidths=0.6, label=label)
        ax.set_title(f"{split} split, {table['context_length']}-token windows")
        ax.set_xlabel("general-weighted CE (nats/token, lower is better)")
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("personal CE: the owner's held-out messages (lower is better)")
    axes[0][0].set_ylim(min(on_scale) - pad_y, top)
    axes[0][0].set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4 if len(tables) > 1 else 2, fontsize=8, frameon=False)
    fig.suptitle("MiniMe: voice fit against general ability (group means large, single runs small)", fontsize=11)
    fig.tight_layout(rect=(0, 0.09 if len(tables) > 1 else 0.16, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val", type=Path, default=REPO / "reports" / "m10" / "common512.json")
    parser.add_argument("--test", type=Path, default=REPO / "reports" / "m10" / "test.json")
    parser.add_argument("--out", type=Path, default=REPO / "reports" / "m10" / "frontier.png")
    args = parser.parse_args()
    tables = panels(args.val, args.test)
    plot(tables, args.out)
    print(f"{', '.join(tables)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
