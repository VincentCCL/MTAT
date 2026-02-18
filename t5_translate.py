#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def read_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(lines, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def batch_iter(xs, bs):
    for i in range(0, len(xs), bs):
        yield xs[i : i + bs]


def compute_sacrebleu(sys_out, refs, metrics):
    """
    sys_out: list[str]
    refs: list[str]
    metrics: set[str] subset of {"bleu", "chrf"}
    """
    try:
        import sacrebleu
    except ImportError as e:
        raise RuntimeError(
            "sacrebleu is not installed. Install it with:\n"
            "  pip install sacrebleu"
        ) from e

    # SacreBLEU expects list of reference streams (even for single-ref)
    ref_list = [refs]

    results = {}

    if "bleu" in metrics:
        bleu_metric = sacrebleu.metrics.BLEU()
        bleu_score = bleu_metric.corpus_score(sys_out, ref_list)
        # signature may be available via get_signature() depending on version
        sig = bleu_metric.get_signature() if hasattr(bleu_metric, "get_signature") else None
        results["BLEU"] = {"score": float(bleu_score.score), "signature": str(sig) if sig else ""}

    if "chrf" in metrics:
        chrf_metric = sacrebleu.metrics.CHRF()
        chrf_score = chrf_metric.corpus_score(sys_out, ref_list)
        sig = chrf_metric.get_signature() if hasattr(chrf_metric, "get_signature") else None
        results["chrF"] = {"score": float(chrf_score.score), "signature": str(sig) if sig else ""}

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Directory passed as --save in finetune_pretrained_t5.py")
    ap.add_argument("--src-file", required=True, help="Source file: one sentence per line")
    ap.add_argument("--out-file", required=True, help="Output translations file")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-src-len", type=int, default=128)
    ap.add_argument("--max-gen-len", type=int, default=128)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument(
        "--device",
        default=None,
        choices=[None, "cpu", "cuda"],
        nargs="?",
        help="Force device (cpu/cuda). Default: auto",
    )

    # Prefix control (should match training)
    ap.add_argument("--prefix", default="translate English to Dutch: ", help="Prefix prepended to every input line")
    ap.add_argument("--no-prefix", action="store_true", help="Do not prepend a prefix")

    # Optional scoring
    ap.add_argument("--ref-file", default=None, help="Optional reference file (one sentence per line)")
    ap.add_argument("--metrics", default="bleu", help="Comma-separated metrics: bleu,chrf. Default: bleu")

    args = ap.parse_args()

    model_dir = Path(args.model_dir)

    # We saved tokenizer under model_dir/tokenizer in the fine-tune script.
    tok_path = model_dir / "tokenizer"
    if tok_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tok_path), use_fast=True)
    else:
        # Fallback: tokenizer saved in model-dir root (common HF convention)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    model.to(device)
    model.eval()

    prefix = "" if args.no_prefix else args.prefix

    src_lines = read_lines(args.src_file)
    outputs = []

    with torch.no_grad():
        for batch in batch_iter(src_lines, args.batch_size):
            batch_in = [prefix + s for s in batch]

            enc = tokenizer(
                batch_in,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_src_len,
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            gen = model.generate(
                **enc,
                num_beams=args.num_beams,
                max_length=args.max_gen_len,
            )

            texts = tokenizer.batch_decode(
                gen,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            outputs.extend(texts)

    write_lines(outputs, args.out_file)
    print(f"Wrote {len(outputs)} translations to {args.out_file}")

    if args.ref_file is not None:
        ref_lines = read_lines(args.ref_file)
        if len(ref_lines) != len(outputs):
            raise ValueError(
                f"Line count mismatch: {len(outputs)} system outputs vs {len(ref_lines)} references."
            )

        metrics = {m.strip().lower() for m in args.metrics.split(",") if m.strip()}
        supported = {"bleu", "chrf"}
        unknown = sorted(metrics - supported)
        if unknown:
            raise ValueError(f"Unknown metric(s): {unknown}. Supported: {sorted(supported)}")

        scores = compute_sacrebleu(outputs, ref_lines, metrics)
        for name, obj in scores.items():
            print(f"{name}: {obj['score']:.2f}")
            print(f"  signature: {obj['signature']}")


if __name__ == "__main__":
    main()

