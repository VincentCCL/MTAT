#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
finetune_pretrained_mbart.py

Fine-tune an mBART(-50) model on parallel data (one sentence per line).
Designed to mirror the option style of finetune_pretrained_t5.py, but with
mBART-specific language control via --src-lang and --tgt-lang.

Example:
  python finetune_pretrained_mbart.py \
    --pretrained-model facebook/mbart-large-50-many-to-many-mmt \
    --src-lang en_XX --tgt-lang nl_XX \
    --src-file train.en --tgt-file train.nl \
    --src-val dev.en --tgt-val dev.nl \
    --epochs 3 --batch-size 4 --lr 3e-5 \
    --save mbart_en_nl \
    --eval-metrics --history-json mbart.hist \
    --show-val-examples 5 \
    --max-src-len 128 --max-tgt-len 128 \
    --num-beams 4 --max-gen-len 128
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

# Optional: sacrebleu for BLEU/chrF
try:
    import sacrebleu  # type: ignore
except Exception:
    sacrebleu = None


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    # keep empty lines if present (but align!)
    return lines


def build_parallel(src_path: str, tgt_path: str) -> Tuple[List[str], List[str]]:
    src = read_lines(src_path)
    tgt = read_lines(tgt_path)
    if len(src) != len(tgt):
        raise ValueError(
            f"Source and target have different number of lines: {len(src)} vs {len(tgt)}"
        )
    return src, tgt


def make_dataset_dict(src: List[str], tgt: List[str]) -> Dict[str, List[str]]:
    return {"src": src, "tgt": tgt}


def compute_sacrebleu_metrics(
    preds: List[str], refs: List[str], requested: List[str]
) -> Dict[str, float]:
    if sacrebleu is None:
        raise RuntimeError(
            "sacrebleu is not installed. Install with: pip install sacrebleu"
        )

    out: Dict[str, float] = {}
    # sacrebleu expects list of hypotheses and list of reference lists
    ref_lists = [refs]

    if "bleu" in requested:
        out["bleu"] = float(sacrebleu.corpus_bleu(preds, ref_lists).score)
    if "chrf" in requested or "chrf++" in requested:
        # Use chrF2 (default) or CHRF++? sacrebleu uses chrF by default.
        out["chrf"] = float(sacrebleu.corpus_chrf(preds, ref_lists).score)
    return out


def parse_metrics_list(metrics_str: str) -> List[str]:
    # Accept "bleu,chrf" or "bleu chrf"
    if not metrics_str:
        return []
    parts = []
    for chunk in metrics_str.replace(",", " ").split():
        parts.append(chunk.strip().lower())
    return [p for p in parts if p]


def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--src-file", required=True, help="Source training file (one sentence per line)")
    ap.add_argument("--tgt-file", required=True, help="Target training file (one sentence per line)")
    ap.add_argument("--src-val", required=True, help="Source validation file")
    ap.add_argument("--tgt-val", required=True, help="Target validation file")

    # Model / languages
    ap.add_argument(
        "--pretrained-model",
        default="facebook/mbart-large-50-many-to-many-mmt",
        help="HF model id for mBART (recommended: mbart-large-50-many-to-many-mmt)",
    )
    ap.add_argument(
        "--src-lang",
        required=True,
        help="mBART language code for the source language, e.g. en_XX",
    )
    ap.add_argument(
        "--tgt-lang",
        required=True,
        help="mBART language code for the target language, e.g. nl_XX",
    )

    # Training hyperparameters
    ap.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    ap.add_argument("--batch-size", type=int, default=4, help="Per-device train batch size")
    ap.add_argument("--eval-batch-size", type=int, default=None, help="Per-device eval batch size (defaults to --batch-size)")
    ap.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    ap.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    ap.add_argument("--warmup-ratio", type=float, default=0.0, help="Warmup ratio (0..1)")
    ap.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--fp16", action="store_true", help="Use fp16 mixed precision (if supported)")

    # Length / generation
    ap.add_argument("--max-src-len", type=int, default=128, help="Max source length (tokens)")
    ap.add_argument("--max-tgt-len", type=int, default=128, help="Max target length (tokens)")
    ap.add_argument("--num-beams", type=int, default=4, help="Beam size for generation")
    ap.add_argument("--max-gen-len", type=int, default=128, help="Max generation length (tokens)")

    # Logging / saving
    ap.add_argument("--save", required=True, help="Output directory for the fine-tuned model")
    ap.add_argument("--history-json", default=None, help="Write Trainer log history to this JSON file")
    ap.add_argument("--show-val-examples", type=int, default=0, help="Print N validation examples after training")
    ap.add_argument(
        "--eval-metrics",
        action="store_true",
        help="Compute evaluation metrics during validation (BLEU/chrF via sacrebleu)",
    )
    ap.add_argument(
        "--metrics",
        default="bleu,chrf",
        help="Which metrics to compute if --eval-metrics is set (default: bleu,chrf)",
    )
    ap.add_argument("--logging-steps", type=int, default=50, help="Logging steps")
    ap.add_argument("--save-steps", type=int, default=500, help="Checkpoint save steps")
    ap.add_argument("--eval-steps", type=int, default=500, help="Evaluation steps")
    ap.add_argument("--save-total-limit", type=int, default=2, help="Max number of checkpoints kept")

    args = ap.parse_args()

    # Seed
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save, exist_ok=True)

    # Load tokenizer/model
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model)

    # mBART language control
    # For mbart-large-50*, tokenizer has .src_lang and .lang_code_to_id
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = args.src_lang
    else:
        raise ValueError("Tokenizer does not support src_lang; are you using an mBART model?")

    if not hasattr(tokenizer, "lang_code_to_id"):
        raise ValueError("Tokenizer does not provide lang_code_to_id; are you using mbart-large-50?")

    if args.tgt_lang not in tokenizer.lang_code_to_id:
        raise ValueError(
            f"Unknown --tgt-lang {args.tgt_lang}. Not in tokenizer.lang_code_to_id."
        )
    forced_bos_token_id = tokenizer.lang_code_to_id[args.tgt_lang]

    # Read data
    train_src, train_tgt = build_parallel(args.src_file, args.tgt_file)
    val_src, val_tgt = build_parallel(args.src_val, args.tgt_val)

    # Build HF datasets (avoid hard dependency on datasets; use simple lists + map via Trainer)
    # We'll tokenize on the fly with a tiny wrapper dataset class.

    @dataclass
    class LinePairDataset:
        src: List[str]
        tgt: List[str]
        tokenizer: any
        max_src_len: int
        max_tgt_len: int

        def __len__(self):
            return len(self.src)

        def __getitem__(self, idx: int) -> Dict[str, List[int]]:
            s = self.src[idx]
            t = self.tgt[idx]

            # Encode source. For mBART, src_lang determines the language code used internally.
            model_inputs = self.tokenizer(
                s,
                max_length=self.max_src_len,
                truncation=True,
            )

            # Encode target (labels)
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    t,
                    max_length=self.max_tgt_len,
                    truncation=True,
                )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

    train_ds = LinePairDataset(train_src, train_tgt, tokenizer, args.max_src_len, args.max_tgt_len)
    val_ds = LinePairDataset(val_src, val_tgt, tokenizer, args.max_src_len, args.max_tgt_len)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    metrics_requested = parse_metrics_list(args.metrics) if args.eval_metrics else []

    def compute_metrics(eval_pred):
        # Only called if predict_with_generate=True
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        # Decode predictions
        pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in labels to pad_token_id so we can decode
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

        pred_str = [p.strip() for p in pred_str]
        label_str = [r.strip() for r in label_str]

        out = {}
        if metrics_requested:
            out.update(compute_sacrebleu_metrics(pred_str, label_str, metrics_requested))
        return out

    eval_bs = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.save,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        predict_with_generate=True,
        generation_num_beams=args.num_beams,
        generation_max_length=args.max_gen_len,
        save_total_limit=args.save_total_limit,
        report_to=[],  # no wandb by default (student-friendly)
        load_best_model_at_end=False,
        metric_for_best_model=None,
        forced_bos_token_id=forced_bos_token_id,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if args.eval_metrics else None,
    )

    # Train
    trainer.train()

    # Final eval
    eval_out = trainer.evaluate()
    print("\nFinal evaluation:")
    for k, v in eval_out.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    # Save model/tokenizer
    trainer.save_model(args.save)
    tokenizer.save_pretrained(args.save)

    # Save history if requested
    if args.history_json:
        with open(args.history_json, "w", encoding="utf-8") as f:
            json.dump(trainer.state.log_history, f, indent=2, ensure_ascii=False)
        print(f"\nWrote training history to {args.history_json}")

    # Show a few validation examples (generated)
    if args.show_val_examples and args.show_val_examples > 0:
        n = min(args.show_val_examples, len(val_src))
        print(f"\nValidation examples (n={n}):")
        # Sample first n (deterministic)
        for i in range(n):
            src_sent = val_src[i]
            ref_sent = val_tgt[i]
            enc = tokenizer(
                src_sent,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_src_len,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            gen = model.generate(
                **enc,
                num_beams=args.num_beams,
                max_length=args.max_gen_len,
                forced_bos_token_id=forced_bos_token_id,
            )
            hyp = tokenizer.decode(gen[0], skip_special_tokens=True).strip()
            print("-" * 60)
            print("SRC:", src_sent)
            print("HYP:", hyp)
            print("REF:", ref_sent)

    print("\nDone.")


if __name__ == "__main__":
    main()
