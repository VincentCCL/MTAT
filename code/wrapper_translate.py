#!/usr/bin/env python3
import argparse
import csv
import subprocess
from pathlib import Path
import os
import shlex


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
    ap.add_argument("--direction", choices=["min", "max"], default="min")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--metric", default="metric_val_nll")
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
        ckpt = model_dir / "best.pt"

        if not ckpt.exists():
            alt_dir = model_dir.parent / model_dir.name.replace(
                "transformer_scratch_",
                "transformer_scratch2_",
                1,
            )
            alt_ckpt = alt_dir / "best.pt"

            if alt_ckpt.exists():
                print(f"Using renamed run dir: {alt_ckpt}")
                ckpt = alt_ckpt
            else:
                print(f"SKIP: neither {ckpt} nor {alt_ckpt} exists")
                continue

        cmd = [
            "python", args.mtat, "translate",
            "--model-type", "transformer-scratch",
            "--model-dir", str(ckpt),
            "--src-file", SRC_VAL,
            "--out-file", str(hyp_file),
            "--ref-file", TGT_VAL,
            "--src-lang", "fr",
            "--tgt-lang", "nl",
            "--metrics", "bleu,chrf,ter",
            "--batch-size", "32",
        ]

        print("\n===", name, "===")
        print(" ".join(cmd))

        if args.execute:
            cmd_str = " ".join(shlex.quote(x) for x in cmd)

            tee_cmd = (
                f"set -o pipefail; "
                f"{cmd_str} 2>&1 | tee {shlex.quote(str(log_file))}"
            )

            proc = subprocess.run(
                ["bash", "-c", tee_cmd],
                env=os.environ.copy(),
            )

            if proc.returncode != 0:
                print(f"FAILED: {name}")
            else:
                print(f"OK: {name}")
            


if __name__ == "__main__":
    main()
