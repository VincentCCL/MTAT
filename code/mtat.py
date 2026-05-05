#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mtat.py

Unified command-line interface for MTAT machine translation experiments.

Supported in this first integrated version:

  finetune:
    - t5
    - mbart
    - m2m
    - nllb
    - madlad

  translate:
    - t5
    - mbart
    - m2m
    - nllb
    - madlad

Examples
--------

Fine-tune M2M:

  python mtat.py finetune \
    --model-type m2m \
    --pretrained-model facebook/m2m100_418M \
    --src-file train.en --tgt-file train.nl \
    --src-val dev.en --tgt-val dev.nl \
    --src-lang en --tgt-lang nl \
    --save runs/m2m_en_nl \
    --epochs 3 --batch-size 8 --lr 5e-5 \
    --eval-metrics --history-json runs/m2m_en_nl/history.json

Translate with the fine-tuned M2M model:

  python mtat.py translate \
    --model-type m2m \
    --model-dir runs/m2m_en_nl \
    --src-file test.en \
    --out-file test.m2m.nl \
    --src-lang en --tgt-lang nl \
    --batch-size 32 --num-beams 4

Fine-tune mBART:

  python mtat.py finetune \
    --model-type mbart \
    --pretrained-model facebook/mbart-large-50-many-to-many-mmt \
    --src-file train.en --tgt-file train.nl \
    --src-val dev.en --tgt-val dev.nl \
    --src-lang en_XX --tgt-lang nl_XX \
    --save runs/mbart_en_nl

Fine-tune T5:

  python mtat.py finetune \
    --model-type t5 \
    --pretrained-model google-t5/t5-small \
    --src-file train.en --tgt-file train.nl \
    --src-val dev.en --tgt-val dev.nl \
    --prefix "translate English to Dutch: " \
    --save runs/t5_en_nl

Translate with T5:

  python mtat.py translate \
    --model-type t5 \
    --model-dir runs/t5_en_nl \
    --src-file test.en \
    --out-file test.t5.nl \
    --prefix "translate English to Dutch: "
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)

try:
    import sacrebleu
except Exception:
    sacrebleu = None


HF_SEQ2SEQ_TYPES = {"t5", "mbart", "m2m", "nllb", "madlad"}


# ---------------------------------------------------------------------
# Basic file helpers
# ---------------------------------------------------------------------

def read_lines(path: str, lower: bool = False) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    if lower:
        lines = [line.lower() for line in lines]
    return lines


def write_lines(path: str, lines: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def batched(xs: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(xs), batch_size):
        yield xs[i : i + batch_size]


def parse_metrics(metrics: str) -> Set[str]:
    return {
        metric.strip().lower()
        for metric in metrics.replace(",", " ").split()
        if metric.strip()
    }


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

@dataclass
class ParallelDataset(torch.utils.data.Dataset):
    src: List[str]
    tgt: List[str]
    tokenizer: object
    max_src_len: int
    max_tgt_len: int
    prefix: str = ""

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src_text = self.prefix + self.src[idx]
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


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def compute_sacrebleu_metrics(
    predictions: List[str],
    references: List[str],
    requested: Set[str],
) -> Dict[str, float]:
    if sacrebleu is None:
        raise RuntimeError("sacrebleu is not installed. Install it with: pip install sacrebleu")

    output: Dict[str, float] = {}
    ref_streams = [references]

    if "bleu" in requested:
        output["bleu"] = float(sacrebleu.corpus_bleu(predictions, ref_streams).score)

    if "chrf" in requested or "chrf++" in requested:
        output["chrf"] = float(sacrebleu.corpus_chrf(predictions, ref_streams).score)

    return output


def build_compute_metrics_fn(tokenizer, requested: Set[str]):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        if isinstance(preds, tuple):
            preds = preds[0]

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
        ref_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

        pred_str = [x.strip() for x in pred_str]
        ref_str = [x.strip() for x in ref_str]

        return compute_sacrebleu_metrics(pred_str, ref_str, requested)

    return compute_metrics


# ---------------------------------------------------------------------
# Model-specific language handling
# ---------------------------------------------------------------------

def resolve_forced_bos_token_id(tokenizer, model_type: str, tgt_lang: Optional[str]) -> Optional[int]:
    if not tgt_lang:
        return None

    if model_type == "m2m":
        if hasattr(tokenizer, "get_lang_id"):
            return tokenizer.get_lang_id(tgt_lang)
        raise ValueError("The selected tokenizer does not support get_lang_id(); is this really an M2M model?")

    if model_type == "mbart":
        if hasattr(tokenizer, "lang_code_to_id") and tgt_lang in tokenizer.lang_code_to_id:
            return tokenizer.lang_code_to_id[tgt_lang]
        raise ValueError(f"Unknown mBART target language code: {tgt_lang}")

    if model_type == "nllb":
        token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            return token_id

        if hasattr(tokenizer, "lang_code_to_id") and tgt_lang in tokenizer.lang_code_to_id:
            return tokenizer.lang_code_to_id[tgt_lang]

        raise ValueError(
            f"Could not resolve NLLB target language token id for {tgt_lang!r}. "
            "Use codes such as eng_Latn, nld_Latn, fra_Latn."
        )

    return None


def configure_tokenizer_and_model_for_languages(
    tokenizer,
    model,
    model_type: str,
    src_lang: Optional[str],
    tgt_lang: Optional[str],
) -> Optional[int]:
    forced_bos_token_id = resolve_forced_bos_token_id(tokenizer, model_type, tgt_lang)

    if model_type in {"m2m", "mbart", "nllb"}:
        if not src_lang or not tgt_lang:
            raise ValueError(f"--src-lang and --tgt-lang are required for --model-type {model_type}")

        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = src_lang

        if hasattr(tokenizer, "tgt_lang"):
            tokenizer.tgt_lang = tgt_lang

    if forced_bos_token_id is not None:
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.forced_bos_token_id = forced_bos_token_id
        model.config.forced_bos_token_id = forced_bos_token_id

    return forced_bos_token_id


# ---------------------------------------------------------------------
# Validation examples
# ---------------------------------------------------------------------

class ShowValExamplesCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        model_type: str,
        src_lang: Optional[str],
        tgt_lang: Optional[str],
        val_src: List[str],
        val_tgt: List[str],
        n: int,
        max_src_len: int,
        num_beams: int,
        max_gen_len: int,
        prefix: str = "",
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.val_src = val_src
        self.val_tgt = val_tgt
        self.n = n
        self.max_src_len = max_src_len
        self.num_beams = num_beams
        self.max_gen_len = max_gen_len
        self.prefix = prefix

        rng = random.Random(seed)
        idxs = list(range(len(val_src)))
        rng.shuffle(idxs)
        self.fixed_idxs = idxs[: min(n, len(val_src))]

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.n <= 0:
            return

        model = kwargs["model"]
        model.eval()
        device = model.device

        forced_bos_token_id = resolve_forced_bos_token_id(
            self.tokenizer,
            self.model_type,
            self.tgt_lang,
        )

        print(f"\n=== Validation examples @ epoch {state.epoch:.2f} ===")

        for i in self.fixed_idxs:
            src = self.val_src[i]
            ref = self.val_tgt[i]
            src_in = self.prefix + src

            enc = self.tokenizer(
                src_in,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_src_len,
            ).to(device)

            generation_kwargs = {
                "num_beams": self.num_beams,
                "max_new_tokens": self.max_gen_len,
            }
            if forced_bos_token_id is not None:
                generation_kwargs["forced_bos_token_id"] = forced_bos_token_id

            with torch.no_grad():
                gen = model.generate(**enc, **generation_kwargs)

            hyp = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
            print(f"\nSRC: {src}\nREF: {ref}\nHYP: {hyp}")


# ---------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------

def finetune_hf_seq2seq(args: argparse.Namespace) -> None:
    if args.model_type not in HF_SEQ2SEQ_TYPES:
        raise ValueError(f"Unsupported model type for finetuning: {args.model_type}")

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_src = read_lines(args.src_file, lower=args.lower)
    train_tgt = read_lines(args.tgt_file, lower=args.lower)
    val_src = read_lines(args.src_val, lower=args.lower)
    val_tgt = read_lines(args.tgt_val, lower=args.lower)

    if len(train_src) != len(train_tgt):
        raise ValueError(f"Train src/tgt line counts differ: {len(train_src)} vs {len(train_tgt)}")
    if len(val_src) != len(val_tgt):
        raise ValueError(f"Validation src/tgt line counts differ: {len(val_src)} vs {len(val_tgt)}")

    os.makedirs(args.save, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model)

    configure_tokenizer_and_model_for_languages(
        tokenizer=tokenizer,
        model=model,
        model_type=args.model_type,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    prefix = "" if args.no_prefix else args.prefix

    train_ds = ParallelDataset(
        train_src,
        train_tgt,
        tokenizer,
        args.max_src_len,
        args.max_tgt_len,
        prefix=prefix,
    )
    val_ds = ParallelDataset(
        val_src,
        val_tgt,
        tokenizer,
        args.max_src_len,
        args.max_tgt_len,
        prefix=prefix,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    requested_metrics = parse_metrics(args.metrics) if args.eval_metrics else set()
    compute_metrics = None
    if args.eval_metrics:
        if sacrebleu is None:
            print("WARNING: sacrebleu is not installed; BLEU/chrF will be skipped.")
        else:
            compute_metrics = build_compute_metrics_fn(tokenizer, requested_metrics)

    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.save,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        eval_strategy="no" if args.eval_disabled else args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        predict_with_generate=not args.no_generate,
        generation_num_beams=args.eval_num_beams,
        generation_max_length=args.eval_max_gen_len,
        report_to=[],
        load_best_model_at_end=False,
        save_only_model=args.save_only_model,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None if args.eval_disabled else val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if args.show_val_examples > 0:
        trainer.add_callback(
            ShowValExamplesCallback(
                tokenizer=tokenizer,
                model_type=args.model_type,
                src_lang=args.src_lang,
                tgt_lang=args.tgt_lang,
                val_src=val_src,
                val_tgt=val_tgt,
                n=args.show_val_examples,
                max_src_len=args.max_src_len,
                num_beams=args.num_beams,
                max_gen_len=args.max_gen_len,
                prefix=prefix,
                seed=args.seed,
            )
        )

    print("Model device before training:", next(model.parameters()).device)
    trainer.train()

    if not args.eval_disabled:
        print("\nFinal evaluation:")
        eval_out = trainer.evaluate()
        for key, value in eval_out.items():
            print(f"  {key}: {value}")

    trainer.save_model(args.save)
    tokenizer.save_pretrained(args.save)

    if args.history_json:
        with open(args.history_json, "w", encoding="utf-8") as f:
            json.dump(trainer.state.log_history, f, indent=2, ensure_ascii=False)
        print(f"\nWrote training history to {args.history_json}")


# ---------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------

def translate_hf_seq2seq(args: argparse.Namespace) -> None:
    if args.model_type not in HF_SEQ2SEQ_TYPES:
        raise ValueError(f"Unsupported model type for translation: {args.model_type}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)

    forced_bos_token_id = configure_tokenizer_and_model_for_languages(
        tokenizer=tokenizer,
        model=model,
        model_type=args.model_type,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
    )

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    model.to(device)
    model.eval()

    src_lines = read_lines(args.src_file, lower=args.lower)
    prefix = "" if args.no_prefix else args.prefix

    outputs: List[str] = []

    with torch.no_grad():
        for batch in batched(src_lines, args.batch_size):
            batch_in = [prefix + line for line in batch]

            enc = tokenizer(
                batch_in,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_src_len,
            ).to(device)

            generation_kwargs = {
                "num_beams": args.num_beams,
                "max_new_tokens": args.max_gen_len,
            }
            if forced_bos_token_id is not None:
                generation_kwargs["forced_bos_token_id"] = forced_bos_token_id

            gen = model.generate(**enc, **generation_kwargs)
            texts = tokenizer.batch_decode(
                gen,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            outputs.extend(texts)

    write_lines(args.out_file, outputs)
    print(f"Wrote {len(outputs)} translations to {args.out_file}")

    if args.ref_file:
        ref_lines = read_lines(args.ref_file)
        if len(ref_lines) != len(outputs):
            raise ValueError(
                f"Line count mismatch: {len(outputs)} system outputs vs {len(ref_lines)} references."
            )

        requested_metrics = parse_metrics(args.metrics)
        if requested_metrics:
            scores = compute_sacrebleu_metrics(outputs, ref_lines, requested_metrics)
            for name, score in scores.items():
                print(f"{name}: {score:.2f}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def add_common_data_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--src-file", required=True, help="Source file, one sentence per line")
    ap.add_argument("--lower", action="store_true", help="Lowercase input text")


def add_common_generation_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-src-len", type=int, default=128)
    ap.add_argument("--max-gen-len", type=int, default=128)
    ap.add_argument("--num-beams", type=int, default=4)


def add_language_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--src-lang", default=None, help="Source language code, required for m2m/mbart/nllb")
    ap.add_argument("--tgt-lang", default=None, help="Target language code, required for m2m/mbart/nllb")


def add_prefix_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--prefix",
        default="",
        help="Prefix prepended to each source sentence; useful for T5/MADLAD",
    )
    ap.add_argument("--no-prefix", action="store_true", help="Ignore --prefix")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Unified MTAT CLI for training, finetuning and translating MT models."
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # finetune
    ft = sub.add_parser("finetune", help="Fine-tune a pretrained seq2seq model")
    ft.add_argument("--model-type", required=True, choices=sorted(HF_SEQ2SEQ_TYPES))
    ft.add_argument("--pretrained-model", required=True)
    add_common_data_args(ft)
    ft.add_argument("--tgt-file", required=True, help="Target training file")
    ft.add_argument("--src-val", required=True, help="Source validation file")
    ft.add_argument("--tgt-val", required=True, help="Target validation file")
    add_language_args(ft)
    add_prefix_args(ft)

    ft.add_argument("--save", required=True, help="Output directory")
    ft.add_argument("--epochs", type=int, default=3)
    ft.add_argument("--lr", type=float, default=5e-5)
    ft.add_argument("--weight-decay", type=float, default=0.0)
    ft.add_argument("--warmup-steps", type=int, default=0)
    ft.add_argument("--grad-accum", type=int, default=1)
    ft.add_argument("--fp16", action="store_true")
    ft.add_argument("--seed", type=int, default=42)

    ft.add_argument("--batch-size", type=int, default=8)
    ft.add_argument("--eval-batch-size", type=int, default=None)
    ft.add_argument("--max-src-len", type=int, default=128)
    ft.add_argument("--max-tgt-len", type=int, default=128)

    ft.add_argument("--num-beams", type=int, default=4)
    ft.add_argument("--max-gen-len", type=int, default=128)
    ft.add_argument("--eval-num-beams", type=int, default=1)
    ft.add_argument("--eval-max-gen-len", type=int, default=64)

    ft.add_argument("--eval-metrics", action="store_true")
    ft.add_argument("--metrics", default="bleu,chrf")
    ft.add_argument("--no-generate", action="store_true")
    ft.add_argument("--eval-disabled", action="store_true")
    ft.add_argument("--eval-strategy", default="steps", choices=["no", "steps", "epoch"])
    ft.add_argument("--save-strategy", default="steps", choices=["no", "steps", "epoch"])
    ft.add_argument("--logging-steps", type=int, default=200)
    ft.add_argument("--eval-steps", type=int, default=500)
    ft.add_argument("--save-steps", type=int, default=500)
    ft.add_argument("--save-total-limit", type=int, default=2)
    ft.add_argument("--save-only-model", action="store_true")
    ft.add_argument("--history-json", default=None)
    ft.add_argument("--show-val-examples", type=int, default=0)
    ft.add_argument("--gradient-checkpointing", action="store_true")

    # translate
    tr = sub.add_parser("translate", help="Translate a source file with a model")
    tr.add_argument("--model-type", required=True, choices=sorted(HF_SEQ2SEQ_TYPES))
    tr.add_argument("--model-dir", required=True)
    add_common_data_args(tr)
    tr.add_argument("--out-file", required=True)
    add_language_args(tr)
    add_prefix_args(tr)
    add_common_generation_args(tr)
    tr.add_argument("--device", default=None, choices=["cpu", "cuda"])
    tr.add_argument("--ref-file", default=None)
    tr.add_argument("--metrics", default="bleu,chrf")

    return ap


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "finetune":
        finetune_hf_seq2seq(args)
    elif args.command == "translate":
        translate_hf_seq2seq(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
