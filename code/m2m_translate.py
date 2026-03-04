#!/usr/bin/env python3
"""
Translate a file with an M2M-100 model directory (fine-tuned or pretrained).
Optionally compute BLEU/chrF if a reference file is provided.

Example:
python m2m_translate.py \
  --model-dir m2m_en_nl \
  --src-lang en --tgt-lang nl \
  --src-file test.en --out-file test.m2m.nl \
  --ref-file test.nl --metrics bleu,chrf \
  --batch-size 32 --max-src-len 128 --max-gen-len 128 --num-beams 4
"""

import argparse
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    import sacrebleu
except Exception:
    sacrebleu = None


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def write_lines(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def batched(xs: List[str], bs: int):
    for i in range(0, len(xs), bs):
        yield xs[i : i + bs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--src-lang", required=True, help="e.g. en, nl, fr, de ...")
    ap.add_argument("--tgt-lang", required=True, help="e.g. en, nl, fr, de ...")
    ap.add_argument("--src-file", required=True)
    ap.add_argument("--out-file", required=True)

    ap.add_argument("--ref-file", default=None)
    ap.add_argument("--metrics", default="", help="comma-separated: bleu,chrf")

    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-src-len", type=int, default=128)
    ap.add_argument("--max-gen-len", type=int, default=128)
    ap.add_argument("--num-beams", type=int, default=4)

    args = ap.parse_args()

    src_lines = read_lines(args.src_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    tokenizer.src_lang = args.src_lang
    forced_bos_token_id = tokenizer.get_lang_id(args.tgt_lang)

    hyps: List[str] = []

    for batch in batched(src_lines, args.batch_size):
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_src_len,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                num_beams=args.num_beams,
                max_new_tokens=args.max_gen_len,
                forced_bos_token_id=forced_bos_token_id,
            )

        batch_hyps = tokenizer.batch_decode(out, skip_special_tokens=True)
        hyps.extend(batch_hyps)

    write_lines(args.out_file, hyps)
    print(f"Wrote {len(hyps)} translations to {args.out_file}")

    # Optional evaluation
    if args.ref_file and args.metrics:
        if sacrebleu is None:
            raise RuntimeError("sacrebleu not installed but --metrics was requested. Install with: pip install sacrebleu")

        refs = read_lines(args.ref_file)
        assert len(refs) == len(hyps), "ref and hypothesis line counts differ"

        requested = {m.strip().lower() for m in args.metrics.split(",") if m.strip()}
        scores = {}

        if "bleu" in requested:
            scores["bleu"] = float(sacrebleu.corpus_bleu(hyps, [refs]).score)
        if "chrf" in requested:
            scores["chrf"] = float(sacrebleu.corpus_chrf(hyps, [refs]).score)

        print("Scores:", scores)


if __name__ == "__main__":
    main()