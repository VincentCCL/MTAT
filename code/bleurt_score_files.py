#!/usr/bin/env python3
"""
Compute corpus-level BLEURT for a hypothesis file against a reference file.

Usage:
  python bleurt_score.py reference.txt hypothesis.txt

Dependencies:
  pip install evaluate bleurt tensorflow
"""

import argparse
import sys
from pathlib import Path

import evaluate


def read_lines(path: Path) -> list[str]:
    # Keep empty lines (they matter for alignment), but strip trailing newlines/spaces
    return [line.rstrip("\n").rstrip("\r") for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute BLEURT for ref/hyp text files (one sentence per line).")
    ap.add_argument("reference", type=Path, help="Reference file (one sentence per line).")
    ap.add_argument("hypothesis", type=Path, help="Hypothesis file (one sentence per line).")
    ap.add_argument(
        "--checkpoint",
        default="BLEURT-20-D12",
        help=(
            "BLEURT checkpoint name used by HF Evaluate (default: BLEURT-20-D12). "
            "Other options include: BLEURT-20, BLEURT-20-D6, BLEURT-20-D3, "
            "bleurt-base-128, bleurt-large-512, bleurt-tiny-128, etc."
        ),
    )
    ap.add_argument("--batch-size", type=int, default=16, help="BLEURT batch size (default: 16).")
    ap.add_argument("--print-per-sentence", action="store_true", help="Print per-sentence BLEURT scores.")
    args = ap.parse_args()

    if not args.reference.exists():
        print(f"ERROR: reference file not found: {args.reference}", file=sys.stderr)
        return 2
    if not args.hypothesis.exists():
        print(f"ERROR: hypothesis file not found: {args.hypothesis}", file=sys.stderr)
        return 2

    refs = read_lines(args.reference)
    hyps = read_lines(args.hypothesis)

    if len(refs) != len(hyps):
        print(
            f"ERROR: line count mismatch: reference={len(refs)} vs hypothesis={len(hyps)}.\n"
            "Make sure both files have exactly one sentence per line and the same number of lines.",
            file=sys.stderr,
        )
        return 2

    # Load BLEURT metric (downloads checkpoint if needed)
    bleurt = evaluate.load("bleurt", config_name=args.checkpoint)

    # Compute scores (Evaluate forwards batch_size to the underlying BLEURT scorer)
    result = bleurt.compute(
        predictions=hyps,
        references=refs,
        batch_size=args.batch_size,
    )
    scores = result["scores"]
    mean_score = sum(scores) / len(scores) if scores else float("nan")

    print(f"BLEURT checkpoint: {args.checkpoint}")
    print(f"Sentences: {len(scores)}")
    print(f"Mean BLEURT: {mean_score:.6f}")

    if args.print_per_sentence:
        for i, s in enumerate(scores, start=1):
            print(f"{i}\t{s:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())