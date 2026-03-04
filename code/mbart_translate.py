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
            "sacrebleu is not installed. Install it with:\n  pip install sacrebleu"
        ) from e

    # SacreBLEU expects list of reference streams (even for single-ref)
    ref_list = [refs]
    results = {}

    if "bleu" in metrics:
        bleu_metric = sacrebleu.metrics.BLEU()
        bleu_score = bleu_metric.corpus_score(sys_out, ref_list)
        sig = bleu_metric.get_signature() if hasattr(bleu_metric, "get_signature") else None
        results["BLEU"] = {"score": float(bleu_score.score), "signature": str(sig) if sig else ""}

    if "chrf" in metrics:
        chrf_metric = sacrebleu.metrics.CHRF()
        chrf_score = chrf_metric.corpus_score(sys_out, ref_list)
        sig = chrf_metric.get_signature() if hasattr(chrf_metric, "get_signature") else None
        results["chrF"] = {"score": float(chrf_score.score), "signature": str(sig) if sig else ""}

    return results


def _forced_bos_id(tokenizer, tgt_lang: str):
    """
    mBART uses a forced BOS token corresponding to the target language code.
    Works for MBartTokenizer/MBartTokenizerFast and most AutoTokenizer configs.
    """
    if hasattr(tokenizer, "lang_code_to_id") and tgt_lang in tokenizer.lang_code_to_id:
        return tokenizer.lang_code_to_id[tgt_lang]

    # Fallback (some tokenizers may not expose lang_code_to_id)
    tok_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if tok_id is None or tok_id == tokenizer.unk_token_id:
        raise ValueError(
            f"Cannot resolve tgt language code '{tgt_lang}' to a token id. "
            "For MBART-50 use codes like en_XX, nl_XX, fr_XX, de_DE, ..."
        )
    return tok_id


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model-dir", required=True, help="Directory passed as --save in the mBART fine-tune script")
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

    # mBART language codes (MBART-50 examples: en_XX, nl_XX, fr_XX, de_DE, ...)
    ap.add_argument("--src-lang", required=True, help="Source language code (e.g., en_XX)")
    ap.add_argument("--tgt-lang", required=True, help="Target language code (e.g., nl_XX)")

    # Optional scoring
    ap.add_argument("--ref-file", default=None, help="Optional reference file (one sentence per line)")
    ap.add_argument("--metrics", default="bleu", help="Comma-separated metrics: bleu,chrf. Default: bleu")

    args = ap.parse_args()
    model_dir = Path(args.model_dir)

    # Mirror the t5_translate.py convention: tokenizer in model_dir/tokenizer if present
    tok_path = model_dir / "tokenizer"
    if tok_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tok_path), use_fast=True)
    else:
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

    # mBART: set source language on tokenizer (important for correct encoding)
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = args.src_lang

    forced_bos_token_id = _forced_bos_id(tokenizer, args.tgt_lang)

    src_lines = read_lines(args.src_file)
    outputs = []

    with torch.no_grad():
        for batch in batch_iter(src_lines, args.batch_size):
            enc = tokenizer(
                batch,
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
                forced_bos_token_id=forced_bos_token_id,
            )

            texts = tokenizer.batch_decode(
                gen, skip_special_tokens=True, clean_up_tokenization_spaces=True
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
            print(f" signature: {obj['signature']}")


if __name__ == "__main__":
    main()