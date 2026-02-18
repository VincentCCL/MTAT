#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from transformers import EncoderDecoderModel, T5TokenizerFast, T5Tokenizer


def load_tok(path: str):
    # Saved via tok.save_pretrained(...). Try fast first.
    try:
        return T5TokenizerFast.from_pretrained(path)
    except Exception:
        return T5Tokenizer.from_pretrained(path)


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

    # SacreBLEU expects list of references (even for a single ref)
    ref_list = [refs]

    results = {}

    if "bleu" in metrics:
        bleu = sacrebleu.corpus_bleu(sys_out, ref_list)
        # bleu.score is float; bleu.signature is reproducibility string
        results["BLEU"] = {"score": bleu.score, "signature": bleu.signature}

    if "chrf" in metrics:
        chrf = sacrebleu.corpus_chrf(sys_out, ref_list)
        results["chrF"] = {"score": chrf.score, "signature": chrf.signature}

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Directory passed as --save in transformer.py")
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

    # NEW: scoring options
    ap.add_argument(
        "--ref-file",
        default=None,
        help="Optional reference file (one sentence per line). If provided, compute SacreBLEU/chrF.",
    )
    ap.add_argument(
        "--metrics",
        default="bleu",
        help="Comma-separated metrics to compute when --ref-file is set. Supported: bleu,chrf. Default: bleu",
    )

    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    tok_src = load_tok(str(model_dir / "tokenizer_src"))
    tok_tgt = load_tok(str(model_dir / "tokenizer_tgt"))
    model = EncoderDecoderModel.from_pretrained(str(model_dir))

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    model.to(device)
    model.eval()

    src_lines = read_lines(args.src_file)
    outputs = []

    with torch.no_grad():
        for batch in batch_iter(src_lines, args.batch_size):
            enc = tok_src(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_src_len,
                add_special_tokens=True,
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            gen = model.generate(
                **enc,
                num_beams=args.num_beams,
                max_length=args.max_gen_len,  # total length cap
            )

            texts = tok_tgt.batch_decode(
                gen,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            outputs.extend(texts)

    write_lines(outputs, args.out_file)
    print(f"Wrote {len(outputs)} translations to {args.out_file}")

    # NEW: compute SacreBLEU if ref provided
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

        # Print in a student-friendly way
        for name, obj in scores.items():
            print(f"{name}: {obj['score']:.2f}")
            print(f"  signature: {obj['signature']}")


if __name__ == "__main__":
    main()
