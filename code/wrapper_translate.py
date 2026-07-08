#!/usr/bin/env python3
import argparse
import csv
import subprocess
from pathlib import Path


SRC_VAL = "/home/nobackup/corpora/DeKamer/tmx/moses/250320.fr.dev"
TGT_VAL = "/home/nobackup/corpora/DeKamer/tmx/moses/250320.nl.dev"

SRC_SP = "/home/nobackup/corpora/DeKamer/tmx/moses/250320.sp.model.32000.model"
TGT_SP = "/home/nobackup/corpora/DeKamer/tmx/moses/250320.sp.model.32000.model"


def as_float(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--mtat", default="mtat.py")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--metric", default="val_nll")
    ap.add_argument("--direction", choices=["min", "max"], default="min")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.tsv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    usable = []
    for row in rows:
        score = as_float(row.get(args.metric, ""))
        if score is None:
            continue

        run_dir = (
            row.get("run_dir")
            or row.get("save")
            or row.get("model_dir")
            or row.get("output_dir")
        )
        if not run_dir:
            continue

        usable.append((score, row, Path(run_dir)))

    reverse = args.direction == "max"
    usable.sort(key=lambda x: x[0], reverse=reverse)
    selected = usable[: args.top_n]
    print(f"Read {len(rows)} rows from TSV")
    print(f"Found {len(usable)} usable rows with metric {args.metric}")
    print(f"Selected {len(selected)} models")

    for rank, (score, row, model_dir) in enumerate(selected, start=1):
        name = f"rank{rank:02d}_{args.metric}_{score:.4f}"
        hyp_file = out_dir / f"{name}.hyp.nl"
        log_file = out_dir / f"{name}.translate.log"

        cmd = [
            "python", args.mtat, "translate",
            "--model-type", "transformer-scratch",
            "--model", str(model_dir),
            "--src-file", SRC_VAL,
            "--out-file", str(hyp_file),
            "--ref-file", TGT_VAL,
            "--src-lang", "fr",
            "--tgt-lang", "nl",
            "--metrics", "bleu,chrf,ter",
        ]

        print("\n===", name, "===")
        print(" ".join(cmd))

        if args.execute:
            with open(log_file, "w", encoding="utf-8") as log:
                subprocess.run(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )


if __name__ == "__main__":
    main()