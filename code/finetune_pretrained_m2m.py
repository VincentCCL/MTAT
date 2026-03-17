#!/usr/bin/env python3
"""
Fine-tune M2M-100 on a parallel corpus (plain text files, 1 sentence per line).

Example:
python finetune_pretrained_m2m.py \
  --src-file train.en --tgt-file train.nl \
  --src-val dev.en --tgt-val dev.nl \
  --src-lang en --tgt-lang nl \
  --epochs 3 --batch-size 8 --lr 5e-5 \
  --save m2m_en_nl \
  --eval-metrics --history-json m2m.hist \
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
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

# Optional metrics
try:
    import sacrebleu
except Exception:
    sacrebleu = None


def read_lines(path: str, lower: bool = False) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    if lower:
        lines = [ln.lower() for ln in lines]
    return lines


@dataclass
class ParallelDataset(torch.utils.data.Dataset):
    src: List[str]
    tgt: List[str]
    tokenizer: any
    max_src_len: int
    max_tgt_len: int

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src_text = self.src[idx]
        tgt_text = self.tgt[idx]

        model_inputs = self.tokenizer(
            src_text,
            max_length=self.max_src_len,
            truncation=True,
        )

        labels = self.tokenizer(
           text_target=tgt_text,
            max_length=self.max_tgt_len,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def build_compute_metrics_fn(tokenizer, tgt_lang: str):
    """
    Returns a compute_metrics(pred) function for Seq2SeqTrainer.
    We compute corpus BLEU and chrF using sacrebleu if available.
    """
    if sacrebleu is None:
        return None

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 in labels (ignored positions) so we can decode
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
        ref_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # sacrebleu expects list of references (list-of-list)
        bleu = sacrebleu.corpus_bleu(pred_str, [ref_str]).score
        chrf = sacrebleu.corpus_chrf(pred_str, [ref_str]).score

        return {"bleu": float(bleu), "chrf": float(chrf)}

    return compute_metrics


def sample_and_show_examples(
    model,
    tokenizer,
    src_lines: List[str],
    tgt_lines: List[str],
    src_lang: str,
    tgt_lang: str,
    n: int,
    max_src_len: int,
    num_beams: int,
    max_gen_len: int,
):
    n = min(n, len(src_lines))
    idxs = list(range(len(src_lines)))
    random.shuffle(idxs)
    idxs = idxs[:n]

    model.eval()
    device = model.device

    # For M2M: set src language on tokenizer; target language via forced_bos_token_id
    tokenizer.src_lang = src_lang
    forced_bos_token_id = tokenizer.get_lang_id(tgt_lang)

    print("\n=== Sample validation examples ===")
    for i in idxs:
        src = src_lines[i]
        ref = tgt_lines[i]
        enc = tokenizer(
            src,
            return_tensors="pt",
            truncation=True,
            max_length=max_src_len,
        ).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                num_beams=num_beams,
                max_new_tokens=max_gen_len,
                forced_bos_token_id=forced_bos_token_id,
            )
        hyp = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        print(f"\nSRC: {src}\nREF: {ref}\nHYP: {hyp}")


def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--src-file", required=True)
    ap.add_argument("--tgt-file", required=True)
    ap.add_argument("--src-val", required=True)
    ap.add_argument("--tgt-val", required=True)
    ap.add_argument("--lower", action="store_true")

    # Model + langs
    ap.add_argument("--pretrained-model", default="facebook/m2m100_418M")
    ap.add_argument("--src-lang", required=True, help="M2M language code, e.g. en, nl, fr, de, ...")
    ap.add_argument("--tgt-lang", required=True, help="M2M language code, e.g. en, nl, fr, de, ...")

    # Training hyperparams (similar knobs as your other scripts)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # Lengths / decoding
    ap.add_argument("--max-src-len", type=int, default=128)
    ap.add_argument("--max-tgt-len", type=int, default=128)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-gen-len", type=int, default=128)

    # Output
    ap.add_argument("--save", required=True, help="Output directory for the fine-tuned model")
    ap.add_argument("--history-json", default=None)
    ap.add_argument("--eval-metrics", action="store_true")
    ap.add_argument("--show-val-examples", type=int, default=0)

    # Logging/eval cadence (keep simple defaults)
    ap.add_argument("--logging-steps", type=int, default=200)
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--save-steps", type=int, default=500)

    args = ap.parse_args()
    set_seed(args.seed)

    # Read data
    train_src = read_lines(args.src_file, lower=args.lower)
    train_tgt = read_lines(args.tgt_file, lower=args.lower)
    val_src = read_lines(args.src_val, lower=args.lower)
    val_tgt = read_lines(args.tgt_val, lower=args.lower)

    assert len(train_src) == len(train_tgt), "Train src/tgt line counts differ"
    assert len(val_src) == len(val_tgt), "Val src/tgt line counts differ"

    # Load
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model)

    tokenizer.src_lang = args.src_lang
    tokenizer.tgt_lang = args.tgt_lang
    forced_bos_token_id = tokenizer.get_lang_id(args.tgt_lang)

    # Datasets
    train_ds = ParallelDataset(train_src, train_tgt, tokenizer, args.max_src_len, args.max_tgt_len)
    val_ds = ParallelDataset(val_src, val_tgt, tokenizer, args.max_src_len, args.max_tgt_len)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    compute_metrics = None
    if args.eval_metrics:
        if sacrebleu is None:
            print("WARNING: sacrebleu not installed; metrics will be skipped.")
        else:
            compute_metrics = build_compute_metrics_fn(tokenizer, args.tgt_lang)

    os.makedirs(args.save, exist_ok=True)

    train_args = Seq2SeqTrainingArguments(
        output_dir=args.save,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        predict_with_generate=True,
        generation_num_beams=args.num_beams,
        generation_max_length=args.max_gen_len,
        report_to="none",
    )

    # Make sure trainer uses correct target language at generation time
    model.config.forced_bos_token_id = forced_bos_token_id

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        #tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Final save (so directory contains final weights + tokenizer)
    trainer.save_model(args.save)
    tokenizer.save_pretrained(args.save)

    # Optional: show examples
    if args.show_val_examples > 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        sample_and_show_examples(
            model=model,
            tokenizer=tokenizer,
            src_lines=val_src,
            tgt_lines=val_tgt,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            n=args.show_val_examples,
            max_src_len=args.max_src_len,
            num_beams=args.num_beams,
            max_gen_len=args.max_gen_len,
        )

    # Optional: write history json (eval logs, losses, metrics)
    if args.history_json:
        with open(args.history_json, "w", encoding="utf-8") as f:
            json.dump(trainer.state.log_history, f, indent=2, ensure_ascii=False)
        print(f"\nWrote training history to {args.history_json}")


if __name__ == "__main__":
    main()