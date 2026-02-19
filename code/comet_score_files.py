#!/usr/bin/env python3
"""
Compute a COMET score from three aligned text files:
- source file
- MT output file
- reference file

Usage:
  python comet_score_files.py src.txt mt.txt ref.txt

Requirements:
  pip install unbabel-comet
"""

import argparse
import torch
from comet import download_model, load_from_checkpoint


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    # basic sanitation
    return [ln.replace("\n", " ").strip() for ln in lines]


def main():
    parser = argparse.ArgumentParser(description="Compute COMET score from text files")
    parser.add_argument("src", help="Source file (one sentence per line)")
    parser.add_argument("mt", help="MT output file (one sentence per line)")
    parser.add_argument("ref", help="Reference file (one sentence per line)")
    parser.add_argument(
        "--model",
        default="Unbabel/wmt22-comet-da",
        help="COMET model to use (default: Unbabel/wmt22-comet-da)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for prediction (default: 8)",
    )
    args = parser.parse_args()

    src_lines = read_lines(args.src)
    mt_lines = read_lines(args.mt)
    ref_lines = read_lines(args.ref)

    if not (len(src_lines) == len(mt_lines) == len(ref_lines)):
        raise ValueError(
            f"Line count mismatch: "
            f"src={len(src_lines)}, mt={len(mt_lines)}, ref={len(ref_lines)}"
        )

    data = [
        {"src": s, "mt": m, "ref": r}
        for s, m, r in zip(src_lines, mt_lines, ref_lines)
    ]

    gpus = 1 if torch.cuda.is_available() else 0

    model_path = download_model(args.model)
    model = load_from_checkpoint(model_path)

    out = model.predict(data, batch_size=args.batch_size, gpus=gpus)

    print(f"COMET model: {args.model}")
    print(f"Sentences: {len(data)}")
    print(f"System score: {out.system_score:.4f}")


if __name__ == "__main__":
    main()
