#!/usr/bin/env python3
"""Evaluate a checkpoint: one ``eval.json`` of numbers and hashes.

    python scripts/evaluate.py --checkpoint experiments/probe-s1337/last.pt \\
        --personal $MINIME_DATA_ROOT/shards/personal \\
        --general $MINIME_DATA_ROOT/shards/general \\
        --tokenizer tokenizer/minime-16k.json

    # a subset of families, or a smaller eval for a smoke run
    ... --families perplexity,voice --set voice.prompts=64

    # the personal test split, once, for the final report
    ... --split test --final-report

Writes ``eval-<checkpoint>.json`` beside the checkpoint (``--out`` to
change it) and the generated text to ``samples/eval-<checkpoint>.jsonl``
there, which is gitignored (``--no-samples`` to skip it).  A scoring context
other than the model's own (``perplexity.context_length``) adds ``-ctx<N>`` to
both names, so it never replaces the native evaluation:

    # the common-context CE of the M10 report
    ... --eval-config configs/eval_common512.yaml   # -> eval-last-ctx512.json  Only aggregates are
printed.  The families and every setting are documented in
``minime.evaluation.harness``; the defaults are ``configs/eval.yaml``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minime.config import parse_cli_overrides  # noqa: E402
from minime.evaluation.harness import (  # noqa: E402
    load_eval_config,
    load_model,
    run_eval,
    write_outputs,
)
from minime.tokenizer import load_tokenizer  # noqa: E402
from minime.utils.device import resolve  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "eval.yaml"


def _fmt(x, digits=4):
    return "-" if x is None else f"{x:.{digits}f}"


def print_summary(record: dict) -> None:
    ppl = record.get("perplexity")
    if ppl:
        if ppl["context_length"] != ppl["native_context_length"]:
            print(f"CE scored on {ppl['context_length']}-token windows "
                  f"(the model's own context is {ppl['native_context_length']})")
        if "personal" in ppl:
            p = ppl["personal"]
            print(f"personal {record['split']:4s}  loss {_fmt(p['loss'])}  ppl {_fmt(p['ppl'], 2)}  "
                  f"bpb {_fmt(p['bits_per_byte'])}  ({p['tokens']:,} tokens)")
        for name, g in (ppl.get("general") or {}).items():
            print(f"general  {name:9s} loss {_fmt(g['loss'])}  ppl {_fmt(g['ppl'], 2)}  "
                  f"bpb {_fmt(g['bits_per_byte'])}  ({g['tokens']:,} tokens)")
        if "general_weighted" in ppl:
            print(f"general  weighted  loss {_fmt(ppl['general_weighted']['loss'])}")
    voice = record.get("voice")
    if voice:
        disc = voice["authorship"]["discriminator"]
        attr = voice["authorship"]["attribution"]
        delta = voice["stylometry"]["delta"]["questioned"]
        print(f"voice    impersonation AUC {_fmt(disc['auc'], 3)} ± {_fmt(disc['auc_std'], 3)} "
              f"(0.5 = indistinguishable)")
        print(f"voice    attributed to me: generated {_fmt(attr['generated']['attributed_to_target'], 3)}"
              f"  real {_fmt(attr['real']['attributed_to_target'], 3)}")
        print(f"voice    Delta nearest is me: generated {_fmt(delta['generated']['nearest_is_target'], 3)}"
              f"  real {_fmt(delta['real']['nearest_is_target'], 3)}")
        g = voice["generation"]
        print(f"voice    EOT rate {_fmt(g['eot_rate'], 3)}  mean tokens {_fmt(g['mean_tokens'], 1)}  "
              f"entropy {_fmt(g['entropy'], 3)}")
    deg = record.get("degeneration")
    if deg:
        gen, real = deg["generated"], deg["real"]
        print(f"degen    rep-4 {_fmt(gen['rep_4'], 3)} (real {_fmt(real['rep_4'], 3)})  "
              f"distinct-2 {_fmt(gen['distinct_2'], 3)} (real {_fmt(real['distinct_2'], 3)})  "
              f"self-BLEU {_fmt(gen['self_bleu'], 3)} (real {_fmt(real['self_bleu'], 3)})")
    code = record.get("code")
    if code:
        print(f"code     valid {_fmt(code['valid'], 3)}  terminated {_fmt(code['terminated'], 3)}  "
              f"reference valid {_fmt(code['reference_valid'], 3)}  ({code['prompts']} functions)")
    bl = record.get("blimp")
    if bl:
        fields = "  ".join(f"{k} {_fmt(v, 3)}" for k, v in bl["by_field"].items())
        print(f"blimp    accuracy {_fmt(bl['accuracy'], 3)}  ({bl['pairs']:,} pairs)  {fields}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--personal", type=Path, default=None, help="personal shard directory")
    parser.add_argument("--general", type=Path, default=None, help="general shard directory")
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--families", default=None, help="comma-separated subset of the config's families")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[],
                        help="eval config overrides, e.g. voice.prompts=64 seed=7")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--final-report", action="store_true",
                        help="required with --split test: the test split is read once, for the report")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-samples", action="store_true")
    args = parser.parse_args()

    if args.split == "test" and not args.final_report:
        parser.error("--split test reads the held-back test split; add --final-report if this is the final report")

    overrides = parse_cli_overrides(args.overrides)
    if args.families:
        overrides["families"] = [f.strip() for f in args.families.split(",") if f.strip()]
    cfg = load_eval_config(args.eval_config, overrides)

    spec = resolve(args.device, args.precision)
    tokenizer = load_tokenizer(args.tokenizer)
    model, info = load_model(args.checkpoint, spec.device)
    print(f"checkpoint {args.checkpoint} (step {info['step']}, {info['params'] / 1e6:.2f}M params) "
          f"| {tokenizer.id} | {spec.device} {spec.precision} | eval {cfg.fingerprint()} | split {args.split}")
    if tokenizer.dev:
        print("NOTE: dev tokenizer -- this evaluation is pipeline work, not an experiment result")

    record, samples = run_eval(
        model, tokenizer, cfg, personal=args.personal, general=args.general, split=args.split,
        unlock_test=args.final_report, spec=spec, checkpoint_info=info,
    )
    stem = args.checkpoint.stem
    if cfg.perplexity.context_length is not None and "perplexity" in cfg.families:
        stem += f"-ctx{cfg.perplexity.context_length}"
    stem += "-test" if args.split == "test" else ""
    out = args.out or args.checkpoint.parent / f"eval-{stem}.json"
    samples_path = None if args.no_samples else args.checkpoint.parent / "samples" / f"eval-{stem}.jsonl"
    write_outputs(record, samples, out, samples_path)

    print_summary(record)
    print(f"seconds: {json.dumps(record['seconds'])}")
    print(f"eval -> {out}" + (f"  samples -> {samples_path} (gitignored)" if samples_path and samples else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
