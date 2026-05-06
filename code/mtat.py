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
import glob
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Set
import re
import subprocess
from tqdm.auto import tqdm

import numpy as np
import torch

try:
    import yaml
except Exception:
    yaml = None

import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
    EarlyStoppingCallback
)

try:
    import sacrebleu
    from sacrebleu.metrics import TER
except Exception:
    sacrebleu = None
    TER = None


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

    if "ter" in requested:
        if TER is None:
            raise RuntimeError("TER metric is unavailable. Please upgrade sacrebleu: pip install -U sacrebleu")
        ter = TER()
        output["ter"] = float(ter.corpus_score(predictions, ref_streams).score)

    return output

def build_compute_metrics_fn(
    tokenizer,
    requested: Set[str],
    save_predictions_dir: Optional[str] = None,
    val_src: Optional[List[str]] = None,
):
    eval_counter = {"n": 0}

    def compute_metrics(eval_pred):
        eval_counter["n"] += 1

        preds, labels = eval_pred

        if isinstance(preds, tuple):
            preds = preds[0]

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
        ref_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

        pred_str = [x.strip() for x in pred_str]
        ref_str = [x.strip() for x in ref_str]

        metrics = compute_sacrebleu_metrics(pred_str, ref_str, requested)

        if save_predictions_dir is not None:
            os.makedirs(save_predictions_dir, exist_ok=True)

            stem = os.path.join(
                save_predictions_dir,
                f"eval_{eval_counter['n']:03d}",
            )

            if val_src is not None:
                write_lines(stem + ".src", val_src)

            write_lines(stem + ".ref", ref_str)
            write_lines(stem + ".hyp", pred_str)

            with open(stem + ".scores.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

        return metrics

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



class SaveEvalTranslationsCallback(TrainerCallback):
    """
    During training, save validation translations and metric scores whenever
    Trainer evaluates.

    Files are written as:

      eval_translations/eval_step_500.src
      eval_translations/eval_step_500.ref
      eval_translations/eval_step_500.hyp
      eval_translations/eval_step_500.scores.json
    """

    def __init__(
        self,
        tokenizer,
        model_type: str,
        src_lang: Optional[str],
        tgt_lang: Optional[str],
        val_src: List[str],
        val_tgt: List[str],
        out_dir: str,
        metrics: Set[str],
        batch_size: int,
        max_src_len: int,
        max_gen_len: int,
        num_beams: int,
        prefix: str = "",
    ):
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.val_src = val_src
        self.val_tgt = val_tgt
        self.out_dir = out_dir
        self.metrics = metrics
        self.batch_size = batch_size
        self.max_src_len = max_src_len
        self.max_gen_len = max_gen_len
        self.num_beams = num_beams
        self.prefix = prefix

        os.makedirs(self.out_dir, exist_ok=True)

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()
        device = model.device

        forced_bos_token_id = resolve_forced_bos_token_id(
            self.tokenizer,
            self.model_type,
            self.tgt_lang,
        )

        hyps: List[str] = []

        with torch.no_grad():
            for batch in batched(self.val_src, self.batch_size):
                batch_in = [self.prefix + line for line in batch]
                enc = self.tokenizer(
                    batch_in,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_src_len,
                ).to(device)

                generation_kwargs = {
                    "num_beams": self.num_beams,
                    "max_new_tokens": self.max_gen_len,
                }
                if forced_bos_token_id is not None:
                    generation_kwargs["forced_bos_token_id"] = forced_bos_token_id

                gen = model.generate(**enc, **generation_kwargs)
                texts = self.tokenizer.batch_decode(
                    gen,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                hyps.extend([x.strip() for x in texts])

        step = int(state.global_step)
        stem = os.path.join(self.out_dir, f"eval_step_{step}")

        write_lines(stem + ".src", self.val_src)
        write_lines(stem + ".ref", self.val_tgt)
        write_lines(stem + ".hyp", hyps)

        scores = {
            "step": step,
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "note": "Scores are already available in Trainer eval logs.",
        }

        with open(stem + ".scores.json", "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)
        print(f"\nSaved validation translations to {stem}.hyp")
        print("Validation translation scores:")
        for key, value in scores.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


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
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
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
    if args.early_stopping_patience is not None:
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
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

    if args.save_eval_translations:
        if sacrebleu is None:
            raise RuntimeError("Saving eval translation scores requires sacrebleu: pip install sacrebleu")

        eval_translations_dir = args.eval_translations_dir
        if eval_translations_dir is None:
            eval_translations_dir = os.path.join(args.save, "eval_translations")

        trainer.add_callback(
            SaveEvalTranslationsCallback(
                tokenizer=tokenizer,
                model_type=args.model_type,
                src_lang=args.src_lang,
                tgt_lang=args.tgt_lang,
                val_src=val_src,
                val_tgt=val_tgt,
                out_dir=eval_translations_dir,
                metrics=requested_metrics or parse_metrics(args.metrics),
                batch_size=eval_batch_size,
                max_src_len=args.max_src_len,
                max_gen_len=args.eval_max_gen_len,
                num_beams=args.eval_num_beams,
                prefix=prefix,
            )
        )

    print("Model device before training:", next(model.parameters()).device)

    resume_checkpoint = resolve_resume_checkpoint(args)
    args.resolved_resume_from_checkpoint = resume_checkpoint
    save_run_config(args)

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    if not args.eval_disabled:
        print("\nFinal evaluation:")
        eval_out = trainer.evaluate()
        for key, value in eval_out.items():
            print(f"  {key}: {value}")

    trainer.save_model(args.save)
    tokenizer.save_pretrained(args.save)

    if args.history_json:
        # --------------------------------------------------------------
        # Load existing history if resuming training
        # --------------------------------------------------------------
        existing_history = []

        if os.path.exists(args.history_json):

            try:
                with open(args.history_json, "r", encoding="utf-8") as f:
                    existing_history = json.load(f)

                print(f"Loaded existing history from {args.history_json}")

            except Exception as e:
                print(f"WARNING: could not load existing history: {e}")
                existing_history = []

        # --------------------------------------------------------------
        # Merge old + new history
        # --------------------------------------------------------------

        combined_history = existing_history + trainer.state.log_history

        with open(args.history_json, "w", encoding="utf-8") as f:
            json.dump(combined_history, f, indent=2, ensure_ascii=False)

        print(f"\nWrote merged training history to {args.history_json}")
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
        
        for batch in tqdm(
            batched(src_lines, args.batch_size),
            total=(len(src_lines) + args.batch_size - 1) // args.batch_size,
            desc="Translating",
        ):
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
# Checkpoint / continuation helpers
# ---------------------------------------------------------------------

def checkpoint_step(path: str) -> int:
    """
    Extract numeric step from a Hugging Face Trainer checkpoint directory.

    Example:
      checkpoint-500 -> 500
    """
    base = os.path.basename(os.path.normpath(path))
    match = re.match(r"checkpoint-(\d+)$", base)
    if not match:
        return -1
    return int(match.group(1))


def find_latest_checkpoint(run_dir: str) -> Optional[str]:
    """
    Return the latest checkpoint-* directory inside run_dir, or None.
    """
    pattern = os.path.join(run_dir, "checkpoint-*")
    checkpoints = [
        path for path in glob.glob(pattern)
        if os.path.isdir(path) and checkpoint_step(path) >= 0
    ]

    if not checkpoints:
        return None

    return max(checkpoints, key=checkpoint_step)


def resolve_resume_checkpoint(args: argparse.Namespace) -> Optional[str]:
    """
    Decide whether Trainer should resume from a checkpoint.

    Supported modes:

      --resume-from-checkpoint PATH
          Resume exactly from PATH.

      --resume-from-checkpoint latest
          Resume from the latest checkpoint inside --save.

      --auto-resume
          If --save contains checkpoint-* directories, resume from the latest one.

      no option
          Start a new Trainer run.

    Note:
      This is different from using --pretrained-model <run_dir>, which loads
      model weights but does not restore optimizer/scheduler/trainer state.
    """
    resume = getattr(args, "resume_from_checkpoint", None)

    if resume:
        if str(resume).lower() == "latest":
            latest = find_latest_checkpoint(args.save)
            if latest is None:
                raise ValueError(
                    f"--resume-from-checkpoint latest was requested, "
                    f"but no checkpoint-* directories were found in {args.save}"
                )
            print(f"Resuming from latest checkpoint: {latest}")
            return latest

        if not os.path.isdir(resume):
            raise ValueError(f"Checkpoint directory does not exist: {resume}")

        print(f"Resuming from checkpoint: {resume}")
        return resume

    if getattr(args, "auto_resume", False):
        latest = find_latest_checkpoint(args.save)
        if latest is not None:
            print(f"Auto-resuming from latest checkpoint: {latest}")
            return latest

        print(f"--auto-resume was set, but no checkpoint-* directories were found in {args.save}")
        print("Starting a new training run instead.")
        return None

    return None



# ---------------------------------------------------------------------
# Config logging / YAML config loading
# ---------------------------------------------------------------------

def make_yaml_safe(value):
    """
    Recursively convert values to plain YAML-safe Python types.
    This avoids PyYAML crashes on version objects, paths, numpy scalars, etc.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): make_yaml_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [make_yaml_safe(v) for v in value]

    # numpy scalar support, without requiring numpy-specific logic elsewhere
    if hasattr(value, "item"):
        try:
            return make_yaml_safe(value.item())
        except Exception:
            pass

    return str(value)


def serializable_config(args: argparse.Namespace) -> Dict[str, object]:
    config = dict(vars(args))

    config["_metadata"] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.replace("\n", " "),
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
    }

    try:
        config["_metadata"]["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
    except Exception:
        config["_metadata"]["git_commit"] = None

    return make_yaml_safe(config)


def save_run_config(args: argparse.Namespace) -> None:
    """
    Save the fully resolved configuration, including argparse defaults,
    to the run directory.
    """
    if not hasattr(args, "save") or args.save is None:
        return

    os.makedirs(args.save, exist_ok=True)
    config = serializable_config(args)

    #json_path = os.path.join(args.save, "config.json")
    #txt_path = os.path.join(args.save, "config.txt")
    yaml_path = os.path.join(args.save, "config.yaml")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        for key in sorted(k for k in config.keys() if k != "_metadata"):
            f.write(f"{key}: {config[key]}\n")
        f.write("\n[_metadata]\n")
        for key in sorted(config["_metadata"].keys()):
            f.write(f"{key}: {config['_metadata'][key]}\n")

    if yaml is not None:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        print(f"Saved run config to {yaml_path}")
    else:
        print("WARNING: PyYAML is not installed; config.yaml was not written.")
        print("Install with: pip install pyyaml")

    print(f"Saved run config to {json_path}")
    print(f"Saved run config to {txt_path}")


def load_yaml_config(path: str) -> Dict[str, object]:
    if yaml is None:
        raise RuntimeError("Using --config requires PyYAML. Install with: pip install pyyaml")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file should contain a YAML mapping/dictionary: {path}")

    # Metadata is saved for reproducibility, but should not be parsed as CLI args.
    data.pop("_metadata", None)

    return data


def config_to_cli_args(config: Dict[str, object]) -> List[str]:
    """
    Convert YAML config values to argparse-style command-line arguments.

    This lets a saved config.yaml act as a rerunnable CLI command.
    Values explicitly supplied on the command line after --config override
    the YAML values.
    """
    cli: List[str] = []

    command = config.pop("command", None)

    # Boolean flags need special care. False means: omit flag.
    for key, value in config.items():
        if value is None:
            continue

        option = "--" + key.replace("_", "-")

        if isinstance(value, bool):
            if value:
                cli.append(option)
            continue

        if isinstance(value, list):
            for item in value:
                cli.extend([option, str(item)])
            continue

        cli.extend([option, str(value)])

    if command:
        return [str(command)] + cli

    return cli


def parse_args_with_optional_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    First parse only --config, then combine:
      1. YAML config values
      2. explicit CLI values, which override YAML values
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None, help="YAML config file from a previous run")
    known, remaining = pre.parse_known_args()

    if known.config is None:
        return parser.parse_args()

    config = load_yaml_config(known.config)
    yaml_cli = config_to_cli_args(config)

    # Explicit CLI options come last, so argparse lets them override earlier values.
    args = parser.parse_args(yaml_cli + remaining)
    args.config = known.config
    return args


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
    ap.add_argument(
        "--config",
        default=None,
        help="YAML config file. CLI arguments supplied after --config override the YAML values.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # finetune
    ft = sub.add_parser("finetune", help="Fine-tune a pretrained seq2seq model")
    ft.add_argument("--model-type", required=True, choices=sorted(HF_SEQ2SEQ_TYPES))
    ft.add_argument(
        "--pretrained-model",
        required=True,
        help=(
            "Base model or saved model directory. "
            "Use an original HF checkpoint for a new fine-tuning run, or a previous "
            "run directory to continue from saved weights without optimizer state."
        ),
    )
    ft.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help=(
            "Resume full Trainer state from a checkpoint directory, e.g. "
            "runs/exp/checkpoint-1000. Use 'latest' to pick the latest checkpoint "
            "inside --save."
        ),
    )
    ft.add_argument(
        "--auto-resume",
        action="store_true",
        help=(
            "Automatically resume from the latest checkpoint-* directory inside --save "
            "if one exists. If none exists, start a new run."
        ),
    )
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
    ft.add_argument("--metrics", default="bleu,chrf,ter")
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
    ft.add_argument(
        "--save-eval-translations",
        action="store_true",
        help="Save validation translations at every evaluation point",
    )
    ft.add_argument(
        "--eval-translations-dir",
        default=None,
        help="Directory for validation translations; default: <save>/eval_translations",
    )
    ft.add_argument("--gradient-checkpointing", action="store_true")
    ft.add_argument("--load-best-model-at-end", action="store_true")
    ft.add_argument("--metric-for-best-model", default="bleu")
    ft.add_argument("--greater-is-better", action="store_true")
    ft.add_argument("--early-stopping-patience", type=int, default=None)
    ft.add_argument("--early-stopping-threshold", type=float, default=0.0)
    
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
    tr.add_argument("--metrics", default="bleu,chrf,ter")

    return ap


def main() -> None:
    parser = build_arg_parser()
    args = parse_args_with_optional_config(parser)

    if args.command == "finetune":
        finetune_hf_seq2seq(args)
    elif args.command == "translate":
        translate_hf_seq2seq(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
