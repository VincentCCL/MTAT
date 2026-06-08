#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mtat.py

Unified command-line interface for MTAT machine translation experiments.

The script supports two workflows:

1. finetune
   Fine-tune Hugging Face encoder-decoder / seq2seq models:
   - t5
   - mbart
   - m2m
   - nllb
   - madlad

2. translate
   Translate a source file with either:
   - a local Hugging Face seq2seq model, or
   - an OpenAI-compatible chat-completion API.

The file is organised in the same order as the program runs:
configuration constants, file helpers, metrics, model/language helpers,
dataset preparation, training, translation, checkpoint/config handling,
CLI construction, and the main dispatcher.
"""

from __future__ import annotations

import argparse
import glob
import math
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.utils.data import DataLoader, Dataset
from openai import OpenAI
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)

# Optional dependencies are imported lazily. This keeps the core script usable
# even when optional features such as YAML configs, LoRA/QLoRA, or sacreBLEU
# metrics are not installed. The script raises a clear error only when the user
# actually requests the missing feature.
try:
    import yaml
except Exception:
    yaml = None

try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
except Exception:
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

try:
    import sacrebleu
    from sacrebleu.metrics import TER
except Exception:
    sacrebleu = None
    TER = None

try:
    import sentencepiece as spm
except Exception:
    spm = None


# -----------------------------------------------------------------------------
# Supported model types
# -----------------------------------------------------------------------------

# These constants centralise model-type validation. Fine-tuning only supports
# Hugging Face seq2seq models. Translation additionally supports an
# OpenAI-compatible API backend.
HF_SEQ2SEQ_TYPES = {"t5", "mbart", "m2m", "nllb", "madlad"}
RNN_SEQ2SEQ_TYPES = {"rnn"}
SCRATCH_TRANSFORMER_TYPES = {"transformer-scratch"}
FINETUNE_MODEL_TYPES = HF_SEQ2SEQ_TYPES | RNN_SEQ2SEQ_TYPES | SCRATCH_TRANSFORMER_TYPES
TRANSLATE_MODEL_TYPES = HF_SEQ2SEQ_TYPES | RNN_SEQ2SEQ_TYPES | SCRATCH_TRANSFORMER_TYPES | {"openai"}


# -----------------------------------------------------------------------------
# Basic file and batching helpers
# -----------------------------------------------------------------------------

def read_lines(path: str, lower: bool = False) -> List[str]:
    """Read a one-sentence-per-line text file."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    if lower:
        lines = [line.lower() for line in lines]

    return lines


def iter_lines(path: str, lower: bool = False) -> Iterable[str]:
    """Stream a text file line by line to avoid loading large test sets."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.rstrip("\n")
            yield text.lower() if lower else text


def count_lines(path: str) -> int:
    """Count lines efficiently in binary mode for progress-bar sizing."""
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def write_lines(path: str, lines: Sequence[str]) -> None:
    """Write one sentence per line, creating the output directory if needed."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def write_lines_if_missing(path: str, lines: Sequence[str]) -> None:
    """Write stable helper files, such as validation references, only once."""
    if not os.path.exists(path):
        write_lines(path, lines)


def batched_iter(xs: Iterable[str], batch_size: int) -> Iterable[List[str]]:
    """Yield fixed-size batches from an iterable, with a final shorter batch."""
    batch: List[str] = []
    for x in xs:
        batch.append(x)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def parse_metrics(metrics: str) -> Set[str]:
    """Parse metric names from comma- or whitespace-separated CLI input."""
    return {
        metric.strip().lower()
        for metric in metrics.replace(",", " ").split()
        if metric.strip()
    }


# -----------------------------------------------------------------------------
# API-key helper for OpenAI-compatible translation
# -----------------------------------------------------------------------------

def get_api_key(
    api_key: Optional[str] = None,
    api_env: Optional[str] = None,
    api_key_file: Optional[str] = None,
) -> str:
    """
    Resolve an API key from a direct value, environment variable, or file.

    This gives users a safer alternative to placing secrets directly on the
    command line. In practice, --api-env or --api-key-file is preferable.
    """
    if api_key:
        return api_key

    if api_env:
        value = os.environ.get(api_env)
        if value:
            return value
        raise ValueError(f"Environment variable '{api_env}' is not set or is empty")

    if api_key_file:
        with open(api_key_file, "r", encoding="utf-8") as f:
            value = f.read().strip()
        if value:
            return value
        raise ValueError(f"API key file '{api_key_file}' is empty")

    raise ValueError("Provide one of: --api-key, --api-env, or --api-key-file")


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def compute_sacrebleu_metrics(
    predictions: List[str],
    references: List[str],
    requested: Set[str],
) -> Dict[str, float]:
    """Compute standard MT metrics with sacreBLEU."""
    output: Dict[str, float] = {}
    if not requested:
        return output

    if sacrebleu is None:
        raise RuntimeError("sacrebleu is not installed. Install it with: pip install sacrebleu")

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
    show_example_indices: Optional[List[int]] = None,
):
    """
    Build the metric function expected by Hugging Face Seq2SeqTrainer.

    The Trainer passes generated token IDs and label IDs. This function decodes
    them, protects the tokenizer from invalid IDs, computes MT metrics, and can
    also save validation hypotheses after each evaluation.
    """
    eval_counter = {"n": 0}
    show_example_indices = show_example_indices or []

    def compute_metrics(eval_pred):
        eval_counter["n"] += 1

        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        # Some Transformers/model combinations return logits instead of token
        # IDs. In that case, use argmax to obtain token IDs before decoding.
        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)

        # Replace ignored label IDs and invalid prediction IDs before decoding.
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        vocab_size = len(tokenizer)
        preds = np.where((preds >= 0) & (preds < vocab_size), preds, tokenizer.pad_token_id)
        labels = np.where((labels >= 0) & (labels < vocab_size), labels, tokenizer.pad_token_id)

        pred_str = tokenizer.batch_decode(preds.astype(np.int64), skip_special_tokens=True)
        ref_str = tokenizer.batch_decode(labels.astype(np.int64), skip_special_tokens=True)

        metrics = compute_sacrebleu_metrics(pred_str, ref_str, requested)

        if show_example_indices:
            print(f"\n=== Validation examples @ evaluation {eval_counter['n']:03d} ===")
            for i in show_example_indices:
                if i >= len(pred_str):
                    continue
                src = val_src[i] if val_src is not None else f"<source {i}>"
                print(f"\nSRC: {src}\nREF: {ref_str[i]}\nHYP: {pred_str[i]}")

        if save_predictions_dir is not None:
            os.makedirs(save_predictions_dir, exist_ok=True)

            # The validation source/reference files do not change across
            # evaluations, so they are written only once. Hypotheses and scores
            # are written per evaluation point.
            if val_src is not None:
                write_lines_if_missing(os.path.join(save_predictions_dir, "validation.src"), val_src)
            write_lines_if_missing(os.path.join(save_predictions_dir, "validation.ref"), ref_str)

            stem = os.path.join(save_predictions_dir, f"eval_{eval_counter['n']:03d}")
            write_lines(stem + ".hyp", pred_str)

            with open(stem + ".scores.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

        return metrics

    return compute_metrics


# -----------------------------------------------------------------------------
# Model and tokenizer language handling
# -----------------------------------------------------------------------------

def resolve_forced_bos_token_id(tokenizer, model_type: str, tgt_lang: Optional[str]) -> Optional[int]:
    """
    Resolve the target-language token required by multilingual MT models.

    M2M, mBART, and NLLB use different conventions for target-language control.
    This helper hides those differences from the rest of the training and
    translation code.
    """
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
    """Set source/target language fields and generation config when needed."""
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


def disable_generation_cache(model) -> None:
    """
    Disable generation cache during training.

    Training does not need the generation cache. Disabling it avoids cache-related
    warnings and is required when gradient checkpointing is active.
    """
    if hasattr(model, "config"):
        model.config.use_cache = False

    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = False


def enable_gradient_checkpointing_safely(model) -> None:
    """
    Enable gradient checkpointing in a way that works with PEFT/LoRA.

    LoRA/PEFT often freezes the base model. The non-reentrant checkpointing mode
    avoids detached-loss issues when no ordinary input tensors require gradients.
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


def build_training_arguments(**kwargs) -> Seq2SeqTrainingArguments:
    """
    Create Seq2SeqTrainingArguments across Transformers versions.

    Newer Transformers versions use eval_strategy; older versions use
    evaluation_strategy. This compatibility layer keeps saved configs and CLI
    commands usable across installations.
    """
    import inspect

    params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters

    if "eval_strategy" not in params and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    if "save_only_model" not in params:
        kwargs.pop("save_only_model", None)

    return Seq2SeqTrainingArguments(**kwargs)


# -----------------------------------------------------------------------------
# Dataset preparation
# -----------------------------------------------------------------------------

class EncodedParallelDataset(torch.utils.data.Dataset):
    """
    Parallel MT dataset that tokenizes source and target text once up front.

    Pre-tokenising is usually faster than tokenising inside __getitem__, because
    the Trainer will revisit the same examples across epochs. This is a good
    trade-off when the corpus fits in CPU memory.
    """

    def __init__(
        self,
        src: Sequence[str],
        tgt: Sequence[str],
        tokenizer,
        max_src_len: int,
        max_tgt_len: int,
        prefix: str = "",
    ) -> None:
        if len(src) != len(tgt):
            raise ValueError(f"Source/target line counts differ: {len(src)} vs {len(tgt)}")

        src_texts = [prefix + line for line in src]

        self.inputs = tokenizer(
            src_texts,
            max_length=max_src_len,
            truncation=True,
            padding=False,
        )
        self.labels = tokenizer(
            text_target=list(tgt),
            max_length=max_tgt_len,
            truncation=True,
            padding=False,
        )["input_ids"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        item = {key: value[idx] for key, value in self.inputs.items()}
        item["labels"] = self.labels[idx]
        return item


# -----------------------------------------------------------------------------
# Validation-example callback
# -----------------------------------------------------------------------------

def latest_hyp_file(directory: Optional[str]) -> Optional[str]:
    """Return the most recently updated validation hypothesis file, if any."""
    if not directory or not os.path.isdir(directory):
        return None

    candidates = glob.glob(os.path.join(directory, "*.hyp"))
    if not candidates:
        return None

    return max(candidates, key=lambda path: (os.path.getmtime(path), path))


def load_existing_hypotheses(directory: Optional[str], expected_len: int) -> Tuple[Optional[str], Optional[List[str]]]:
    """Load the latest complete validation hypothesis file for display reuse."""
    hyp_path = latest_hyp_file(directory)
    if hyp_path is None:
        return None, None

    hyps = read_lines(hyp_path)
    if len(hyps) != expected_len:
        return None, None

    return hyp_path, hyps


class ShowValExamplesCallback(TrainerCallback):
    """
    Print a fixed random sample of validation translations at epoch end.

    This provides a qualitative sanity check alongside numeric MT metrics. When
    validation translations were already saved, the callback can reuse them
    instead of generating the same examples again.
    """

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
        reuse_translations_dir: Optional[str] = None,
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
        self.reuse_translations_dir = reuse_translations_dir

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

        hyp_path, existing_hyps = load_existing_hypotheses(
            self.reuse_translations_dir,
            expected_len=len(self.val_src),
        )
        if existing_hyps is not None:
            print(f"Reusing validation translations from {hyp_path}")

        for i in self.fixed_idxs:
            src = self.val_src[i]
            ref = self.val_tgt[i]

            if existing_hyps is not None:
                hyp = existing_hyps[i].strip()
                print(f"\nSRC: {src}\nREF: {ref}\nHYP: {hyp}")
                continue

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

            with torch.inference_mode():
                gen = model.generate(**enc, **generation_kwargs)

            hyp = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
            print(f"\nSRC: {src}\nREF: {ref}\nHYP: {hyp}")


# -----------------------------------------------------------------------------
# Fine-tuning
# -----------------------------------------------------------------------------

def finetune_hf_seq2seq(args: argparse.Namespace) -> None:
    """Run the complete Hugging Face fine-tuning workflow."""
    if args.model_type not in HF_SEQ2SEQ_TYPES:
        raise ValueError(f"Unsupported model type for finetuning: {args.model_type}")

    # Set all common random seeds to make sampling and training as reproducible
    # as possible.
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # MT training data is represented as aligned source and target text files:
    # line n in the source file corresponds to line n in the target file.
    train_src = read_lines(args.src_file, lower=args.lower)
    train_tgt = read_lines(args.tgt_file, lower=args.lower)
    val_src = read_lines(args.src_val, lower=args.lower)
    val_tgt = read_lines(args.tgt_val, lower=args.lower)

    if len(train_src) != len(train_tgt):
        raise ValueError(f"Train src/tgt line counts differ: {len(train_src)} vs {len(train_tgt)}")
    if len(val_src) != len(val_tgt):
        raise ValueError(f"Validation src/tgt line counts differ: {len(val_src)} vs {len(val_tgt)}")

    os.makedirs(args.save, exist_ok=True)

    # Avoid accidental overwrites. To continue a previous run with optimiser and
    # scheduler state, use --resume-from-checkpoint or --auto-resume.
    if (
        os.path.exists(args.save)
        and os.listdir(args.save)
        and not args.overwrite_output_dir
        and args.resume_from_checkpoint is None
        and not args.auto_resume
    ):
        raise ValueError(
            f"Output directory already exists and is not empty: {args.save}\n"
            "Use --overwrite-output-dir or --resume-from-checkpoint."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, use_fast=True)

    quantization_config = None
    if args.qlora:
        if BitsAndBytesConfig is None:
            raise RuntimeError("QLoRA requires bitsandbytes/transformers BitsAndBytesConfig.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.pretrained_model,
        torch_dtype=torch.float16 if args.fp16 or args.qlora else None,
        quantization_config=quantization_config,
        device_map="auto" if args.qlora else None,
    )

    configure_tokenizer_and_model_for_languages(
        tokenizer=tokenizer,
        model=model,
        model_type=args.model_type,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
    )

    # LoRA/QLoRA fine-tunes a small set of adapter weights instead of all model
    # weights, making large-model experiments cheaper in memory and storage.
    if args.lora or args.qlora:
        if LoraConfig is None or get_peft_model is None:
            raise RuntimeError("LoRA/QLoRA requires PEFT. Install with: pip install peft")

        if args.qlora:
            if prepare_model_for_kbit_training is None:
                raise RuntimeError("QLoRA requires prepare_model_for_kbit_training from PEFT.")
            model = prepare_model_for_kbit_training(model)

        target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=target_modules,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable == 0:
            raise RuntimeError(
                "LoRA/QLoRA created zero trainable parameters. "
                "Check --lora-target-modules."
            )

    disable_generation_cache(model)

    if args.gradient_checkpointing:
        enable_gradient_checkpointing_safely(model)

    prefix = "" if args.no_prefix else args.prefix

    train_ds = EncodedParallelDataset(
        train_src,
        train_tgt,
        tokenizer,
        args.max_src_len,
        args.max_tgt_len,
        prefix=prefix,
    )
    val_ds = EncodedParallelDataset(
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
    eval_translations_dir = None

    examples_from_eval = (
        args.show_val_examples > 0
        and not args.eval_disabled
        and not args.no_generate
    )
    should_decode_eval_predictions = (
        bool(requested_metrics)
        or args.save_eval_translations
        or examples_from_eval
    )

    # Decoding validation predictions is relatively expensive, so it is only
    # enabled when metrics, saved hypotheses, or printed examples require it.
    if should_decode_eval_predictions:
        if args.no_generate:
            raise ValueError("Validation metrics/translations/examples require generation; remove --no-generate.")
        if requested_metrics and sacrebleu is None:
            raise RuntimeError("sacrebleu is required for validation metrics: pip install sacrebleu")

        if args.save_eval_translations:
            eval_translations_dir = args.eval_translations_dir or os.path.join(args.save, "eval_translations")

        example_indices = None
        if examples_from_eval:
            rng = random.Random(args.seed)
            idxs = list(range(len(val_src)))
            rng.shuffle(idxs)
            example_indices = idxs[: min(args.show_val_examples, len(val_src))]

        compute_metrics = build_compute_metrics_fn(
            tokenizer,
            requested_metrics,
            save_predictions_dir=eval_translations_dir,
            val_src=val_src,
            show_example_indices=example_indices,
        )

    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size

    if args.metric_for_best_model != "eval_loss" and not requested_metrics:
        raise ValueError(
            "metric_for_best_model requires evaluation metrics. "
            "Enable metrics with --eval-metrics or use --metric-for-best-model eval_loss"
        )

    training_args = build_training_arguments(
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
        gradient_checkpointing_kwargs={"use_reentrant": False},
        label_names=["labels"],
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

    # If Trainer already generated validation predictions for metrics/examples,
    # examples are printed inside compute_metrics to avoid duplicate generation.
    if args.show_val_examples > 0 and not examples_from_eval:
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
                reuse_translations_dir=(
                    args.eval_translations_dir or os.path.join(args.save, "eval_translations")
                    if args.save_eval_translations else None
                ),
            )
        )

    print("Model device before training:", next(model.parameters()).device)

    resume_checkpoint = resolve_resume_checkpoint(args)
    args.resolved_resume_from_checkpoint = resume_checkpoint
    save_run_config(args)

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    if not args.eval_disabled and args.eval_strategy == "no":
        print("\nFinal evaluation:")
        eval_out = trainer.evaluate()
        for key, value in eval_out.items():
            print(f"  {key}: {value}")

    trainer.save_model(args.save)
    tokenizer.save_pretrained(args.save)

    if args.history_json:
        existing_history = []

        if os.path.exists(args.history_json):
            try:
                with open(args.history_json, "r", encoding="utf-8") as f:
                    existing_history = json.load(f)
                print(f"Loaded existing history from {args.history_json}")
            except Exception as e:
                print(f"WARNING: could not load existing history: {e}")
                existing_history = []

        combined_history = existing_history + trainer.state.log_history

        with open(args.history_json, "w", encoding="utf-8") as f:
            json.dump(combined_history, f, indent=2, ensure_ascii=False)

        print(f"\nWrote merged training history to {args.history_json}")


# -----------------------------------------------------------------------------
# Hugging Face translation
# -----------------------------------------------------------------------------

def translate_hf_seq2seq(args: argparse.Namespace) -> None:
    """Translate a source file with a local Hugging Face seq2seq model."""
    if args.model_type not in HF_SEQ2SEQ_TYPES:
        raise ValueError(f"Unsupported model type for translation: {args.model_type}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16 if args.fp16 else None,
    )

    if args.no_cache:
        disable_generation_cache(model)

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

    prefix = "" if args.no_prefix else args.prefix
    total_lines = count_lines(args.src_file)
    total_batches = (total_lines + args.batch_size - 1) // args.batch_size

    collect_outputs = args.ref_file is not None
    outputs: List[str] = []
    n_written = 0

    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with torch.inference_mode(), open(args.out_file, "w", encoding="utf-8") as out_f:
        for batch in tqdm(
            batched_iter(iter_lines(args.src_file, lower=args.lower), args.batch_size),
            total=total_batches,
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
            texts = [text.strip() for text in texts]

            for text in texts:
                out_f.write(text + "\n")
            n_written += len(texts)

            if collect_outputs:
                outputs.extend(texts)

    print(f"Wrote {n_written} translations to {args.out_file}")

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


# -----------------------------------------------------------------------------
# OpenAI-compatible translation
# -----------------------------------------------------------------------------

def build_openai_prompt(batch: Sequence[str], source_lang: Optional[str], target_lang: str) -> str:
    """
    Build a strict JSON translation prompt for an OpenAI-compatible chat model.

    The JSON format is used so that batch outputs can be parsed reliably and
    matched back to the input sentences in the original order.
    """
    payload = {"sentences": list(batch)}

    if source_lang:
        instruction = f"Translate the following sentences from {source_lang} to {target_lang}. "
    else:
        instruction = f"Translate the following sentences to {target_lang}. "

    instruction += (
        "Return ONLY valid JSON with exactly one key: 'translations'. "
        "Its value must be a list of translated strings in exactly the same order "
        "and with exactly the same length as the input. "
        "Do not add explanations, comments, markdown, or extra text."
    )

    return instruction + "\n\nInput JSON:\n" + json.dumps(payload, ensure_ascii=False)


def parse_openai_translation_output(text: str, expected_n: int) -> List[str]:
    raw = text.strip()

    # Remove markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    # Try direct JSON first
    candidates = [raw]

    # Try extracting the outermost JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw[start:end + 1])

    last_error = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            translations = data.get("translations")

            if isinstance(translations, list) and len(translations) == expected_n:
                return [str(item).strip() for item in translations]
        except Exception as e:
            last_error = e

    # Fallback: accept plain line-per-translation output
    lines = [
        line.strip().lstrip("-0123456789. )")
        for line in raw.splitlines()
        if line.strip()
    ]
    if len(lines) == expected_n:
        return lines

    raise ValueError(
        f"Could not parse model output as {expected_n} translations. "
        f"Last JSON error: {last_error}. Raw output preview: {raw[:500]!r}"
    )


def translate_openai_batch(
    client: OpenAI,
    batch: Sequence[str],
    model: str,
    source_lang: Optional[str],
    target_lang: str,
    temperature: float,
    timeout: float,
    max_retries: int,
    retry_wait: float,
    debug: bool = False,
) -> List[str]:
    """Translate one batch with retry logic for temporary API failures."""
    prompt = build_openai_prompt(batch, source_lang, target_lang)
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a machine translation system. "
                            "Follow the requested output format exactly."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                timeout=timeout,
            )

            text = response.choices[0].message.content
            if text is None:
                raise ValueError("Empty response content")

            if debug:
                preview = text[:500].replace("\n", "\\n")
                print(f"[debug] raw response: {preview}", file=sys.stderr, flush=True)

            return parse_openai_translation_output(text, len(batch))

        except Exception as e:
            last_error = e
            if attempt == max_retries - 1:
                break

            wait = retry_wait * (2 ** attempt)
            print(
                f"[warn] batch failed (attempt {attempt + 1}/{max_retries}): {e}",
                file=sys.stderr,
                flush=True,
            )
            print(f"[warn] retrying in {wait:.1f}s", file=sys.stderr, flush=True)
            time.sleep(wait)

    raise RuntimeError(f"Batch failed after {max_retries} attempts: {last_error}")


def translate_openai(args: argparse.Namespace) -> None:
    """Translate a source file with an OpenAI-compatible chat-completion API."""
    api_key = get_api_key(
        api_key=args.api_key,
        api_env=args.api_env,
        api_key_file=args.api_key_file,
    )

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    total_lines = count_lines(args.src_file)
    total_batches = (total_lines + args.batch_size - 1) // args.batch_size

    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    outputs: List[str] = []
    n_written = 0

    with open(args.out_file, "w", encoding="utf-8") as out_f:
        for batch in tqdm(
            batched_iter(iter_lines(args.src_file, lower=args.lower), args.batch_size),
            total=total_batches,
            desc="Translating",
        ):
            translations = translate_openai_batch(
                client=client,
                batch=batch,
                model=args.model,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                temperature=args.temperature,
                timeout=args.timeout,
                max_retries=args.max_retries,
                retry_wait=args.retry_wait,
                debug=args.debug,
            )

            for hyp in translations:
                out_f.write(hyp + "\n")
                outputs.append(hyp)

            out_f.flush()
            n_written += len(translations)

    print(f"Wrote {n_written} translations to {args.out_file}")

    if args.ref_file:
        refs = read_lines(args.ref_file)

        if len(refs) != len(outputs):
            raise ValueError(
                f"Line count mismatch: {len(outputs)} system outputs vs {len(refs)} references."
            )

        requested_metrics = parse_metrics(args.metrics)
        if requested_metrics:
            scores = compute_sacrebleu_metrics(outputs, refs, requested_metrics)
            for name, score in scores.items():
                print(f"{name}: {score:.2f}")



# -----------------------------------------------------------------------------
# Pre-Transformer RNN encoder-decoder models
# -----------------------------------------------------------------------------

# These classes and helpers integrate the earlier rnn_seq2seq.py workflow into
# the unified MTAT CLI. They deliberately do not use Hugging Face Trainer:
# classical RNN/GRU/LSTM encoder-decoder models are plain PyTorch modules, so a
# compact explicit training loop is clearer and easier to teach.

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


@dataclass
class Vocab:
    """Minimal word/subword vocabulary for the RNN models."""

    word2idx: Dict[str, int]
    idx2word: List[str]

    @property
    def pad_idx(self) -> int:
        return self.word2idx[PAD_TOKEN]

    @property
    def sos_idx(self) -> int:
        return self.word2idx[SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.word2idx[EOS_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.word2idx[UNK_TOKEN]

    def encode(self, sentence: str, add_eos: bool = True) -> List[int]:
        ids = [self.word2idx.get(token, self.unk_idx) for token in sentence.split()]
        if add_eos:
            ids.append(self.eos_idx)
        return ids

    def decode(self, ids: List[int]) -> str:
        words: List[str] = []
        for idx in ids:
            if idx == self.eos_idx:
                break
            if 0 <= idx < len(self.idx2word):
                word = self.idx2word[idx]
                if word not in {SOS_TOKEN, PAD_TOKEN}:
                    words.append(word)
        return " ".join(words)


def build_vocab(sentences: Sequence[str], max_size: Optional[int] = None) -> Vocab:
    """
    Build a deterministic vocabulary from whitespace-tokenised sentences.

    max_size counts real tokens only; the four special tokens are added on top.
    This preserves the behaviour of the original RNN script and makes vocabulary
    limits easier to interpret in teaching settings.
    """
    freq: Dict[str, int] = {}
    for sentence in sentences:
        for token in sentence.split():
            freq[token] = freq.get(token, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda item: (-item[1], item[0]))
    if max_size is not None:
        sorted_words = sorted_words[:max_size]

    idx2word = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    idx2word.extend(word for word, _ in sorted_words)
    word2idx = {word: idx for idx, word in enumerate(idx2word)}
    return Vocab(word2idx=word2idx, idx2word=idx2word)


def read_parallel(
    src_path: str,
    tgt_path: str,
    lower: bool = False,
    limit: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """Read non-empty source/target sentence pairs from parallel text files."""
    pairs: List[Tuple[str, str]] = []
    with open(src_path, encoding="utf-8") as src_f, open(tgt_path, encoding="utf-8") as tgt_f:
        for src, tgt in zip(src_f, tgt_f):
            src = src.strip()
            tgt = tgt.strip()
            if lower:
                src = src.lower()
                tgt = tgt.lower()
            if not src or not tgt:
                continue
            pairs.append((src, tgt))
            if limit is not None and len(pairs) >= limit:
                break
    return pairs


class RNNParallelDataset(Dataset):
    """Encode sentence pairs for the custom PyTorch RNN training loop."""

    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        max_len: Optional[int] = None,
    ) -> None:
        self.data: List[Tuple[List[int], List[int]]] = []
        for src, tgt in pairs:
            src_ids = src_vocab.encode(src, add_eos=True)
            tgt_ids = tgt_vocab.encode(tgt, add_eos=True)
            if max_len is not None and (len(src_ids) > max_len or len(tgt_ids) > max_len):
                continue
            self.data.append((src_ids, tgt_ids))

        if not self.data:
            raise ValueError("No sentence pairs left after max_len filtering.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.data[idx]


def pad_token_ids(seq: List[int], max_len: int, pad_idx: int) -> List[int]:
    """Pad or truncate one integer sequence."""
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_idx] * (max_len - len(seq))


def rnn_collate_fn(
    batch: List[Tuple[List[int], List[int]]],
    src_pad_idx: int,
    tgt_pad_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create padded source and target tensors for one mini-batch."""
    src_seqs, tgt_seqs = zip(*batch)
    max_src = max(len(seq) for seq in src_seqs)
    max_tgt = max(len(seq) for seq in tgt_seqs)

    src_tensor = torch.tensor(
        [pad_token_ids(seq, max_src, src_pad_idx) for seq in src_seqs],
        dtype=torch.long,
    )
    tgt_tensor = torch.tensor(
        [pad_token_ids(seq, max_tgt, tgt_pad_idx) for seq in tgt_seqs],
        dtype=torch.long,
    )
    return src_tensor, tgt_tensor


class EncoderRNN(nn.Module):
    """RNN/GRU/LSTM encoder with optional bidirectionality."""

    def __init__(
        self,
        vocab_size: int,
        emb_size: int,
        hidden_size: int,
        pad_idx: int,
        rnn_type: str = "rnn",
        num_layers: int = 1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.is_lstm = self.rnn_type == "lstm"
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.enc_hidden_size = hidden_size
        self.output_hidden_size = hidden_size * (2 if bidirectional else 1)

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        rnn_cls, rnn_kwargs = get_rnn_class_and_kwargs(self.rnn_type)
        self.rnn = rnn_cls(
            emb_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            **rnn_kwargs,
        )

    def forward(self, src: torch.Tensor):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)

        if self.bidirectional:
            batch_size, src_len, _ = outputs.size()
            outputs = outputs.view(batch_size, src_len, self.output_hidden_size)

            if self.is_lstm:
                h, c = hidden
                h = h.view(self.num_layers, 2, batch_size, self.enc_hidden_size)
                c = c.view(self.num_layers, 2, batch_size, self.enc_hidden_size)
                hidden = (
                    torch.cat([h[:, 0], h[:, 1]], dim=-1),
                    torch.cat([c[:, 0], c[:, 1]], dim=-1),
                )
            else:
                h = hidden.view(self.num_layers, 2, batch_size, self.enc_hidden_size)
                hidden = torch.cat([h[:, 0], h[:, 1]], dim=-1)

        return outputs, hidden


def get_rnn_class_and_kwargs(rnn_type: str):
    """Map a CLI RNN type string to the corresponding PyTorch class."""
    if rnn_type == "gru":
        return nn.GRU, {}
    if rnn_type == "lstm":
        return nn.LSTM, {}
    if rnn_type == "rnn":
        return nn.RNN, {"nonlinearity": "tanh"}
    raise ValueError(f"Unknown rnn_type: {rnn_type}")


class LuongAttention(nn.Module):
    """Luong-style multiplicative attention over encoder states."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        decoder_state: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dec_proj = self.linear(decoder_state).unsqueeze(2)
        scores = torch.bmm(encoder_outputs, dec_proj).squeeze(2)
        if src_mask is not None:
            scores = scores.masked_fill(~src_mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class DecoderRNN(nn.Module):
    """RNN/GRU/LSTM decoder with optional Luong attention."""

    def __init__(
        self,
        vocab_size: int,
        emb_size: int,
        hidden_size: int,
        pad_idx: int,
        rnn_type: str = "rnn",
        num_layers: int = 1,
        attention: str = "none",
    ) -> None:
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.attention_type = attention.lower()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)

        rnn_cls, rnn_kwargs = get_rnn_class_and_kwargs(self.rnn_type)
        self.rnn = rnn_cls(
            emb_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            **rnn_kwargs,
        )

        if self.attention_type == "luong":
            self.attn = LuongAttention(hidden_size)
            self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.attn = None
            self.attn_combine = None

        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_step: torch.Tensor,
        hidden,
        encoder_outputs: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        embedded = self.embedding(input_step.unsqueeze(1))
        output, hidden = self.rnn(embedded, hidden)
        dec_state = output.squeeze(1)

        attn_weights = None
        if self.attn is not None and encoder_outputs is not None:
            context, attn_weights = self.attn(dec_state, encoder_outputs, src_mask)
            dec_state = torch.tanh(self.attn_combine(torch.cat([dec_state, context], dim=-1)))

        logits = self.out(dec_state)
        if return_attn:
            return logits, hidden, attn_weights
        return logits, hidden


class Seq2SeqRNN(nn.Module):
    """Full encoder-decoder model used for classical pre-Transformer MT."""

    def __init__(
        self,
        encoder: EncoderRNN,
        decoder: DecoderRNN,
        tgt_sos_idx: int,
        tgt_eos_idx: int,
        max_len: int = 50,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_sos_idx = tgt_sos_idx
        self.tgt_eos_idx = tgt_eos_idx
        self.max_len = max_len

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        pad_idx = self.encoder.embedding.padding_idx
        if pad_idx is None:
            return torch.ones_like(src, dtype=torch.bool)
        return src != pad_idx

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        batch_size = src.size(0)
        encoder_outputs, enc_hidden = self.encoder(src)
        src_mask = self.make_src_mask(src)

        dec_input = torch.full(
            (batch_size,),
            self.tgt_sos_idx,
            dtype=torch.long,
            device=src.device,
        )
        dec_hidden = enc_hidden
        outputs: List[torch.Tensor] = []

        for t in range(tgt.size(1)):
            logits, dec_hidden = self.decoder(
                dec_input,
                dec_hidden,
                encoder_outputs=encoder_outputs,
                src_mask=src_mask,
            )
            outputs.append(logits.unsqueeze(1))
            teacher_force = random.random() < teacher_forcing_ratio
            dec_input = tgt[:, t] if teacher_force else logits.argmax(-1)

        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        max_len: Optional[int] = None,
    ) -> Tuple[List[int], Optional[torch.Tensor]]:
        """Greedy decoding for one sentence; also returns attention weights."""
        self.eval()
        if src.size(0) != 1:
            raise ValueError("greedy_decode currently supports batch size 1 only.")

        max_len = max_len or self.max_len
        encoder_outputs, enc_hidden = self.encoder(src)
        src_mask = self.make_src_mask(src)
        dec_input = torch.full((1,), self.tgt_sos_idx, dtype=torch.long, device=src.device)
        dec_hidden = enc_hidden
        hyp_ids: List[int] = []
        attn_list: List[torch.Tensor] = []

        for _ in range(max_len):
            logits, dec_hidden, attn_weights = self.decoder(
                dec_input,
                dec_hidden,
                encoder_outputs=encoder_outputs,
                src_mask=src_mask,
                return_attn=True,
            )
            top1 = logits.argmax(-1)
            token_id = top1.item()
            if token_id == self.tgt_eos_idx:
                break
            hyp_ids.append(token_id)
            if attn_weights is not None:
                attn_list.append(attn_weights.squeeze(0).cpu())
            dec_input = top1

        attn_matrix = torch.stack(attn_list, dim=0) if attn_list else None
        return hyp_ids, attn_matrix

    @torch.no_grad()
    def greedy_decode_batch(
        self,
        src: torch.Tensor,
        max_len: Optional[int] = None,
    ) -> Tuple[List[List[int]], List[Optional[torch.Tensor]]]:
        """
        Greedy decoding for a padded mini-batch.

        The original RNN implementation decoded validation/test sentences one by
        one.  That made BLEU validation very slow, because each sentence ran a
        separate encoder pass.  This method encodes a whole batch once, then
        performs the autoregressive decoder loop for all active sentences in
        parallel.

        Returns one list of hypothesis ids per batch item, plus one optional
        attention matrix per item for --replace-unk.
        """
        self.eval()
        max_len = max_len or self.max_len
        batch_size = src.size(0)
        encoder_outputs, enc_hidden = self.encoder(src)
        src_mask = self.make_src_mask(src)

        dec_input = torch.full(
            (batch_size,),
            self.tgt_sos_idx,
            dtype=torch.long,
            device=src.device,
        )
        dec_hidden = enc_hidden
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
        hyp_ids: List[List[int]] = [[] for _ in range(batch_size)]
        attn_steps: List[torch.Tensor] = []

        for _ in range(max_len):
            logits, dec_hidden, attn_weights = self.decoder(
                dec_input,
                dec_hidden,
                encoder_outputs=encoder_outputs,
                src_mask=src_mask,
                return_attn=True,
            )
            next_token = logits.argmax(-1)

            if attn_weights is not None:
                attn_steps.append(attn_weights.detach().cpu())

            for i, token in enumerate(next_token.tolist()):
                if finished[i]:
                    continue
                if token == self.tgt_eos_idx:
                    finished[i] = True
                else:
                    hyp_ids[i].append(token)

            dec_input = next_token
            if bool(finished.all()):
                break

        attn_matrices: List[Optional[torch.Tensor]] = [None for _ in range(batch_size)]
        if attn_steps:
            # [steps, batch, src_len] -> one [tgt_len, src_len] matrix per item.
            stacked = torch.stack(attn_steps, dim=0)
            for i in range(batch_size):
                attn_matrices[i] = stacked[: len(hyp_ids[i]), i, :]

        return hyp_ids, attn_matrices


def set_rnn_seed(seed: int) -> None:
    """Set random seeds for the custom PyTorch RNN path."""
    if seed <= 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

def compute_rnn_loss(model, batch, criterion, device, teacher_forcing):
    src, tgt = batch
    src = src.to(device)
    tgt = tgt.to(device)

    logits = model(src, tgt, teacher_forcing_ratio=teacher_forcing)
    return criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))


def evaluate_rnn_nll(model, data_loader, criterion, device, pad_idx):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in data_loader:
            src = src.to(device)
            tgt = tgt.to(device)

            logits = model(src, tgt, teacher_forcing_ratio=1.0)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

            num_tokens = (tgt != pad_idx).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / max(total_tokens, 1)

def simple_detok(text: str) -> str:
    """Tiny detokeniser used by the original RNN script before sacreBLEU."""
    return re.sub(r"\s+([.,!?;:])", r"\1", text)


def compute_rnn_sacrebleu_metrics(
    predictions: List[str],
    references: List[str],
    requested: Set[str],
) -> Dict[str, float]:
    """Compute MT metrics for RNN outputs, preserving the old detok behaviour."""
    if not requested:
        return {}
    if sacrebleu is None:
        raise RuntimeError("sacrebleu is not installed. Install it with: pip install sacrebleu")

    preds = [simple_detok(pred) for pred in predictions]
    refs = [simple_detok(ref) for ref in references]
    output: Dict[str, float] = {}

    if "bleu" in requested:
        output["bleu"] = float(sacrebleu.corpus_bleu(preds, [refs], force=True).score)
    if "chrf" in requested or "chrf++" in requested:
        output["chrf"] = float(sacrebleu.corpus_chrf(preds, [refs]).score)
    if "ter" in requested:
        if TER is not None:
            output["ter"] = float(TER().corpus_score(preds, [refs]).score)
        else:
            output["ter"] = float(sacrebleu.corpus_ter(preds, [refs]).score)
    return output


def apply_sentencepiece_to_pairs(
    pairs: Optional[Sequence[Tuple[str, str]]],
    src_sp: Optional["spm.SentencePieceProcessor"],
    tgt_sp: Optional["spm.SentencePieceProcessor"],
) -> Optional[List[Tuple[str, str]]]:
    """Convert raw text pairs into space-separated SentencePiece token strings."""
    if pairs is None or src_sp is None or tgt_sp is None:
        return list(pairs) if pairs is not None else None
    return [
        (" ".join(src_sp.encode(src, out_type=str)), " ".join(tgt_sp.encode(tgt, out_type=str)))
        for src, tgt in pairs
    ]


def load_sentencepiece_processors(args_or_model_args) -> Tuple[str, Optional[object], Optional[object]]:
    """Load SentencePiece processors when the RNN model uses subword tokens."""
    if isinstance(args_or_model_args, dict):
        subword_type = args_or_model_args.get("subword_type", "none")
        src_sp_model = args_or_model_args.get("src_sp_model")
        tgt_sp_model = args_or_model_args.get("tgt_sp_model")
    else:
        subword_type = args_or_model_args.subword_type
        src_sp_model = args_or_model_args.src_sp_model
        tgt_sp_model = args_or_model_args.tgt_sp_model

    if subword_type == "none":
        return subword_type, None, None
    if spm is None:
        raise ImportError("sentencepiece is required when --subword-type is not 'none'.")
    if not src_sp_model or not tgt_sp_model:
        raise ValueError("Subword mode requires --src-sp-model and --tgt-sp-model.")

    src_sp = spm.SentencePieceProcessor()
    src_sp.load(src_sp_model)
    tgt_sp = spm.SentencePieceProcessor()
    tgt_sp.load(tgt_sp_model)
    return subword_type, src_sp, tgt_sp


def build_rnn_model(
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    model_args: Dict[str, object],
    device: torch.device,
) -> Seq2SeqRNN:
    """Construct an RNN seq2seq model from stored or CLI architecture args."""
    bidirectional = bool(model_args.get("bidirectional", False))
    enc_hidden_size = int(model_args["hidden_size"])
    encoder = EncoderRNN(
        vocab_size=len(src_vocab.idx2word),
        emb_size=int(model_args["emb_size"]),
        hidden_size=enc_hidden_size,
        pad_idx=src_vocab.pad_idx,
        rnn_type=str(model_args.get("rnn_type", "rnn")),
        num_layers=int(model_args.get("enc_layers", 1)),
        bidirectional=bidirectional,
    )
    decoder_hidden_size = enc_hidden_size * (2 if bidirectional else 1)
    decoder = DecoderRNN(
        vocab_size=len(tgt_vocab.idx2word),
        emb_size=int(model_args["emb_size"]),
        hidden_size=decoder_hidden_size,
        pad_idx=tgt_vocab.pad_idx,
        rnn_type=str(model_args.get("rnn_type", "rnn")),
        num_layers=int(model_args.get("dec_layers", 1)),
        attention=str(model_args.get("attention", "none")),
    )
    return Seq2SeqRNN(
        encoder,
        decoder,
        tgt_sos_idx=tgt_vocab.sos_idx,
        tgt_eos_idx=tgt_vocab.eos_idx,
        max_len=int(model_args.get("max_len", 50)),
    ).to(device)


def save_rnn_checkpoint(
    path: str,
    model: Seq2SeqRNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    model_args: Dict[str, object],
    best_metric: Optional[float] = None,
) -> None:
    """Save model state, optimiser state, vocabularies, and architecture args."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "src_vocab": src_vocab,
            "tgt_vocab": tgt_vocab,
            "model_args": model_args,
            "best_metric": best_metric,
        },
        path,
    )


def load_rnn_checkpoint(path: str, device: torch.device):
    """Load an RNN checkpoint and rebuild the corresponding model."""
    try:
        from torch.serialization import safe_globals

        with safe_globals([Vocab]):
            ckpt = torch.load(path, map_location=device)
    except Exception:
        ckpt = torch.load(path, map_location=device)

    src_vocab: Vocab = ckpt["src_vocab"]
    tgt_vocab: Vocab = ckpt["tgt_vocab"]
    model_args: Dict[str, object] = ckpt["model_args"]
    model = build_rnn_model(src_vocab, tgt_vocab, model_args, device)
    model.load_state_dict(ckpt["model_state"])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return model, optimizer, src_vocab, tgt_vocab, model_args, int(ckpt.get("epoch", 0))


def rnn_epoch_checkpoint_path(save_path: str, epoch: int) -> str:
    if save_path.endswith(".pt"):
        return re.sub(r"\.pt$", f".epoch{epoch:03d}.pt", save_path)
    return f"{save_path}.epoch{epoch:03d}.pt"


def cleanup_rnn_checkpoints(save_path: str, keep_last: int) -> None:
    """Keep only the last N epoch checkpoints; the best checkpoint is separate."""
    prefix = save_path[:-3] if save_path.endswith(".pt") else save_path
    checkpoints = sorted(glob.glob(f"{prefix}.epoch*.pt"))
    if keep_last <= 0:
        for checkpoint in checkpoints:
            os.remove(checkpoint)
        return
    for checkpoint in checkpoints[:-keep_last]:
        os.remove(checkpoint)


def replace_unk_with_attention(
    hyp_ids: List[int],
    attn_matrix: Optional[torch.Tensor],
    src_sentence: str,
    tgt_vocab: Vocab,
) -> str:
    """Replace generated <unk> tokens by copying attended source tokens."""
    src_tokens = src_sentence.split()
    out_words: List[str] = []
    for t, idx in enumerate(hyp_ids):
        if idx == tgt_vocab.eos_idx:
            break
        if idx == tgt_vocab.unk_idx:
            src_pos = int(attn_matrix[t].argmax().item()) if attn_matrix is not None and t < attn_matrix.size(0) else t
            if 0 <= src_pos < len(src_tokens):
                out_words.append(src_tokens[src_pos])
                continue
        if 0 <= idx < len(tgt_vocab.idx2word):
            word = tgt_vocab.idx2word[idx]
            if word not in {SOS_TOKEN, PAD_TOKEN}:
                out_words.append(word)
    return " ".join(out_words)


def translate_rnn_lines_from_model(
    model: Seq2SeqRNN,
    src_lines: Sequence[str],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device,
    max_len: int,
    replace_unk: bool = False,
    subword_type: str = "none",
    src_sp: Optional["spm.SentencePieceProcessor"] = None,
    tgt_sp: Optional["spm.SentencePieceProcessor"] = None,
    batch_size: int = 32,
) -> List[str]:
    """Translate source strings with a trained RNN checkpoint in batches."""
    hyps: List[str] = []
    model.eval()
    batch_size = max(1, int(batch_size))

    for start in tqdm(
            range(0, len(src_lines), batch_size),
            desc="Translating",
            leave=False,
    ):
        batch = src_lines[start:start + batch_size]

        model_src_batch = [
            " ".join(src_sp.encode(src_sentence, out_type=str))
            if src_sp is not None and subword_type != "none"
            else src_sentence
            for src_sentence in batch
        ]
        encoded = [src_vocab.encode(model_src, add_eos=True) for model_src in model_src_batch]
        max_src_len = max(len(ids) for ids in encoded)
        src_tensor = torch.tensor(
            [pad_token_ids(ids, max_src_len, src_vocab.pad_idx) for ids in encoded],
            dtype=torch.long,
            device=device,
        )

        batch_hyp_ids, batch_attn = model.greedy_decode_batch(src_tensor, max_len=max_len)

        for hyp_ids, attn_matrix, model_src in zip(batch_hyp_ids, batch_attn, model_src_batch):
            if replace_unk:
                tokenised_hyp = replace_unk_with_attention(hyp_ids, attn_matrix, model_src, tgt_vocab)
            else:
                tokenised_hyp = tgt_vocab.decode(hyp_ids)
            hyp = tgt_sp.decode(tokenised_hyp.split()) if tgt_sp is not None and subword_type != "none" else tokenised_hyp
            hyps.append(hyp)

    return hyps


def translate_rnn_pairs(
    model: Seq2SeqRNN,
    pairs: Sequence[Tuple[str, str]],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device,
    max_len: int,
    replace_unk: bool,
    subword_type: str,
    src_sp: Optional["spm.SentencePieceProcessor"],
    tgt_sp: Optional["spm.SentencePieceProcessor"],
    batch_size: int = 32,
) -> List[str]:
    """Translate just the source side of validation/test sentence pairs."""
    return translate_rnn_lines_from_model(
        model,
        [src for src, _ in pairs],
        src_vocab,
        tgt_vocab,
        device,
        max_len,
        replace_unk=replace_unk,
        subword_type=subword_type,
        src_sp=src_sp,
        tgt_sp=tgt_sp,
        batch_size=batch_size,
    )


def show_rnn_val_examples(
    model: Seq2SeqRNN,
    val_pairs: Sequence[Tuple[str, str]],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device,
    args: argparse.Namespace,
    subword_type: str,
    src_sp: Optional["spm.SentencePieceProcessor"],
    tgt_sp: Optional["spm.SentencePieceProcessor"],
) -> None:
    """Print a few qualitative validation examples after an epoch."""
    if args.show_val_examples <= 0 or not val_pairs:
        return
    subset = list(val_pairs[: args.show_val_examples])
    hyps = translate_rnn_pairs(
        model,
        subset,
        src_vocab,
        tgt_vocab,
        device,
        args.max_len,
        args.replace_unk,
        subword_type,
        src_sp,
        tgt_sp,
        batch_size=args.eval_batch_size or args.batch_size,
    )
    print("\n--- RNN validation examples ---")
    for i, ((src, ref), hyp) in enumerate(zip(subset, hyps), start=1):
        print(f"[{i}] SRC: {src}")
        print(f"    REF: {ref}")
        print(f"    HYP: {hyp}")
    print("-------------------------------")


def rnn_metric_is_better(curr: float, best: Optional[float], metric_name: str) -> bool:
    if best is None:
        return True
    if metric_name in {"loss", "ter"}:
        return curr < best
    return curr > best


def finetune_rnn_seq2seq(args: argparse.Namespace) -> None:
    """Train or resume a pre-Transformer RNN/GRU/LSTM encoder-decoder model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_rnn_seed(args.seed)

    subword_type, src_sp, tgt_sp = load_sentencepiece_processors(args)

    train_pairs = read_parallel(args.src_file, args.tgt_file, lower=args.lower, limit=args.limit)
    if not train_pairs:
        raise ValueError("No training sentence pairs were loaded.")
    val_pairs = read_parallel(args.src_val, args.tgt_val, lower=args.lower) if args.src_val and args.tgt_val else None

    if args.rnn_load:
        model, optimizer, src_vocab, tgt_vocab, model_args, stored_epoch = load_rnn_checkpoint(args.rnn_load, device)
        subword_type, src_sp, tgt_sp = load_sentencepiece_processors(model_args)
        start_epoch = stored_epoch + 1
        print(f"Loaded RNN checkpoint from {args.rnn_load}; resuming at epoch {start_epoch}.")
    else:
        train_pairs_tok = apply_sentencepiece_to_pairs(train_pairs, src_sp, tgt_sp)
        all_src = [src for src, _ in train_pairs_tok]
        all_tgt = [tgt for _, tgt in train_pairs_tok]
        src_vocab = build_vocab(all_src, max_size=args.max_src_vocab)
        tgt_vocab = build_vocab(all_tgt, max_size=args.max_tgt_vocab)
        model_args = {
            "emb_size": args.emb_size,
            "hidden_size": args.hidden_size,
            "max_len": args.max_len,
            "rnn_type": args.rnn_type,
            "enc_layers": args.enc_layers,
            "dec_layers": args.dec_layers,
            "attention": args.attention,
            "bidirectional": args.bidirectional,
            "subword_type": args.subword_type,
            "src_sp_model": args.src_sp_model,
            "tgt_sp_model": args.tgt_sp_model,
            "lower": args.lower,
        }
        model = build_rnn_model(src_vocab, tgt_vocab, model_args, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        start_epoch = 1
        print(f"Source vocab size: {len(src_vocab.idx2word)}")
        print(f"Target vocab size: {len(tgt_vocab.idx2word)}")

    train_pairs_tok = apply_sentencepiece_to_pairs(train_pairs, src_sp, tgt_sp)
    val_pairs_tok = apply_sentencepiece_to_pairs(val_pairs, src_sp, tgt_sp)

    train_dataset = RNNParallelDataset(train_pairs_tok, src_vocab, tgt_vocab, max_len=args.max_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: rnn_collate_fn(batch, src_vocab.pad_idx, tgt_vocab.pad_idx),
    )
    val_loader = None
    if val_pairs_tok is not None:
        val_dataset = RNNParallelDataset(val_pairs_tok, src_vocab, tgt_vocab, max_len=None)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size or args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: rnn_collate_fn(batch, src_vocab.pad_idx, tgt_vocab.pad_idx),
        )

    print(
        f"Loaded {len(train_pairs)} training pairs; {len(train_dataset)} used after max_len={args.max_len} filtering."
    )
    print(f"Total trainable parameters: {count_trainable_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)
    requested_metrics = parse_metrics(args.metrics) if args.eval_metrics else set()
    if requested_metrics and sacrebleu is None:
        raise RuntimeError("sacrebleu is required for RNN validation metrics: pip install sacrebleu")

    save_best = args.rnn_save_best
    if save_best is None:
        base, ext = os.path.splitext(args.save)
        save_best = f"{base}.best{ext or '.pt'}"

    # The unified CLI exposes Hugging Face-style early-stopping flags
    # (--early-stopping-patience, --metric-for-best-model).  The original RNN
    # script used --early-stopping and --early-metric.  Map both styles onto one
    # internal monitor so users can use the same command style for HF and RNN
    # models.
    monitor_metric = args.early_metric
    metric_from_hf_style = getattr(args, "metric_for_best_model", None)
    if metric_from_hf_style:
        metric_from_hf_style = str(metric_from_hf_style).lower()
        if metric_from_hf_style == "eval_loss":
            metric_from_hf_style = "loss"
        elif metric_from_hf_style.startswith("eval_"):
            metric_from_hf_style = metric_from_hf_style[5:]
        if metric_from_hf_style in {"loss", "bleu", "chrf", "ter"}:
            monitor_metric = metric_from_hf_style

    patience_limit = args.early_stopping
    if patience_limit <= 0 and getattr(args, "early_stopping_patience", None) is not None:
        patience_limit = args.early_stopping_patience

    early_stopping_threshold = float(getattr(args, "early_stopping_threshold", 0.0) or 0.0)

    def is_better_for_monitor(curr: float, best: Optional[float]) -> bool:
        if best is None:
            return True
        if monitor_metric in {"loss", "ter"}:
            return curr < (best - early_stopping_threshold)
        return curr > (best + early_stopping_threshold)

    print(f"RNN early-stopping monitor: {monitor_metric}; patience={patience_limit}")

    history: List[Dict[str, object]] = []
    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for src, tgt in tqdm(train_loader, desc=f"RNN epoch {epoch}", leave=False):
            optimizer.zero_grad()
            loss = compute_rnn_loss(model, (src, tgt), criterion, device, args.teacher_forcing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                gold = tgt[:, 1:].contiguous()
                num_tokens = (gold != tgt_vocab.pad_idx).sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        train_nll = total_loss / max(total_tokens, 1)
        train_ppl = math.exp(train_nll) if train_nll < 20 else float("inf")
        record: Dict[str, object] = {"epoch": epoch, "train_nll": train_nll, "train_ppl": None if train_ppl == float("inf") else train_ppl}
        log_msg = f"Epoch {epoch:03d}: train NLL={train_nll:.4f} (ppl={train_ppl:.2f})"

        metric_value: Optional[float] = None
        if val_loader is not None and val_pairs is not None:
            val_nll = evaluate_rnn_nll(model, val_loader, criterion, device, tgt_vocab.pad_idx)
            val_ppl = math.exp(val_nll) if val_nll < 20 else float("inf")
            record.update({"val_nll": val_nll, "val_ppl": None if val_ppl == float("inf") else val_ppl})
            log_msg += f"  val NLL={val_nll:.4f} (ppl={val_ppl:.2f})"

            if monitor_metric == "loss":
                metric_value = val_nll

            need_translations = bool(requested_metrics or args.save_eval_translations or args.show_val_examples > 0)
            hyps: Optional[List[str]] = None
            refs: Optional[List[str]] = None
            if need_translations:
                refs = [tgt for _, tgt in val_pairs]
                hyps = translate_rnn_pairs(
                    model,
                    val_pairs,
                    src_vocab,
                    tgt_vocab,
                    device,
                    args.max_len,
                    args.replace_unk,
                    subword_type,
                    src_sp,
                    tgt_sp,
                    batch_size=args.eval_batch_size or args.batch_size,
                )

            if requested_metrics and hyps is not None and refs is not None:
                metrics = compute_rnn_sacrebleu_metrics(hyps, refs, requested_metrics)
                record.update(metrics)
                for name, value in metrics.items():
                    log_msg += f"  {name.upper()}={value:.2f}"
                if monitor_metric in metrics:
                    metric_value = metrics[monitor_metric]

            if args.save_eval_translations and hyps is not None and refs is not None:
                out_dir = args.eval_translations_dir or os.path.join(args.save, "eval_translations")
                os.makedirs(out_dir, exist_ok=True)
                stem = os.path.join(out_dir, f"rnn_eval_epoch_{epoch:03d}")
                write_lines_if_missing(os.path.join(out_dir, "validation.src"), [src for src, _ in val_pairs])
                write_lines_if_missing(os.path.join(out_dir, "validation.ref"), refs)
                write_lines(stem + ".hyp", hyps)
                with open(stem + ".scores.json", "w", encoding="utf-8") as f:
                    json.dump({k: v for k, v in record.items() if k in {"bleu", "chrf", "ter", "val_nll"}}, f, indent=2, ensure_ascii=False)

            if args.show_val_examples > 0 and hyps is not None:
                print("\n--- RNN validation examples ---")
                for i, ((src, ref), hyp) in enumerate(zip(val_pairs[: args.show_val_examples], hyps[: args.show_val_examples]), start=1):
                    print(f"[{i}] SRC: {src}")
                    print(f"    REF: {ref}")
                    print(f"    HYP: {hyp}")
                print("-------------------------------")

        print(log_msg)

        if metric_value is not None and is_better_for_monitor(metric_value, best_metric):
            best_metric = metric_value
            best_epoch = epoch
            patience_counter = 0
            save_rnn_checkpoint(save_best, model, optimizer, epoch, src_vocab, tgt_vocab, model_args, best_metric)
            print(f"  -> New best {monitor_metric}={metric_value:.4f}; saved {save_best}")
        elif metric_value is not None and patience_limit > 0:
            patience_counter += 1
            print(f"  -> No improvement on {monitor_metric}; patience={patience_counter}/{patience_limit}")
            if patience_counter >= patience_limit:
                print("Early stopping triggered.")
                break

        record["best_metric"] = best_metric
        record["best_epoch"] = best_epoch
        history.append(record)
        if args.history_json:
            out_dir = os.path.dirname(args.history_json)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.history_json, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

        save_path = rnn_epoch_checkpoint_path(args.save, epoch)
        save_rnn_checkpoint(save_path, model, optimizer, epoch, src_vocab, tgt_vocab, model_args, best_metric)
        cleanup_rnn_checkpoints(args.save, args.rnn_keep_last)

    if best_epoch is not None:
        print(f"Training finished. Best {monitor_metric}={best_metric:.4f} at epoch {best_epoch}.")
    else:
        print("Training finished.")


def translate_rnn_seq2seq(args: argparse.Namespace) -> None:
    """Translate a file with a trained RNN/GRU/LSTM checkpoint."""
    if not args.model_dir:
        raise ValueError("--model-dir must point to an RNN .pt checkpoint when --model-type rnn")

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(device_name)
    set_rnn_seed(getattr(args, "seed", 42))

    model, _optimizer, src_vocab, tgt_vocab, model_args, epoch = load_rnn_checkpoint(args.model_dir, device)
    subword_type, src_sp, tgt_sp = load_sentencepiece_processors(model_args)
    src_lines = list(iter_lines(args.src_file, lower=args.lower))
    hyps = translate_rnn_lines_from_model(
        model,
        src_lines,
        src_vocab,
        tgt_vocab,
        device,
        max_len=int(model_args.get("max_len", args.max_gen_len)),
        replace_unk=args.replace_unk,
        subword_type=subword_type,
        src_sp=src_sp,
        tgt_sp=tgt_sp,
        batch_size=args.batch_size,
    )
    write_lines(args.out_file, hyps)
    print(f"Loaded RNN checkpoint from {args.model_dir} (epoch {epoch}).")
    print(f"Wrote {len(hyps)} translations to {args.out_file}")

    if args.ref_file:
        refs = read_lines(args.ref_file, lower=args.lower)
        if len(refs) != len(hyps):
            raise ValueError(f"Line count mismatch: {len(hyps)} system outputs vs {len(refs)} references.")
        scores = compute_rnn_sacrebleu_metrics(hyps, refs, parse_metrics(args.metrics))
        for name, score in scores.items():
            print(f"{name}: {score:.2f}")




# -----------------------------------------------------------------------------
# Transformer encoder-decoder from scratch
# -----------------------------------------------------------------------------

# This model type is intentionally separate from pretrained T5/mBART/M2M/NLLB/
# MADLAD.  Those pretrained models must keep their original tokenizers.  The
# scratch Transformer below is randomly initialised and therefore can use the
# same educational tokenisation choices as the RNN baseline: word-level tokens
# (`--subword-type none`) or externally trained SentencePiece models
# (`--subword-type bpe|unigram`).

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for batch-first Transformer inputs."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ScratchTransformerSeq2Seq(nn.Module):
    """Small PyTorch Transformer encoder-decoder for MT experiments from scratch."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_idx: int,
        tgt_pad_idx: int,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.d_model = d_model
        self.max_len = max_len

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=max_len + 5)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def make_tgt_mask(self, tgt_len: int, device: torch.device) -> torch.Tensor:
        # True entries are masked for PyTorch Transformer boolean masks.
        return torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, src: torch.Tensor, tgt_input: torch.Tensor) -> torch.Tensor:
        src_key_padding_mask = src == self.src_pad_idx
        tgt_key_padding_mask = tgt_input == self.tgt_pad_idx
        tgt_mask = self.make_tgt_mask(tgt_input.size(1), tgt_input.device)

        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt_input) * math.sqrt(self.d_model))
        hidden = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.output_projection(hidden)

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, sos_idx: int, eos_idx: int, max_len: Optional[int] = None) -> List[int]:
        self.eval()
        max_len = max_len or self.max_len
        generated = torch.full((src.size(0), 1), sos_idx, dtype=torch.long, device=src.device)
        for _ in range(max_len):
            logits = self.forward(src, generated)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if bool((next_token == eos_idx).all()):
                break
        return generated[0, 1:].tolist()

    @torch.no_grad()
    def greedy_decode_batch(
        self,
        src: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: Optional[int] = None,
    ) -> List[List[int]]:
        """Greedy decode a padded mini-batch."""
        self.eval()
        max_len = max_len or self.max_len
        batch_size = src.size(0)
        generated = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=src.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
        hyp_ids: List[List[int]] = [[] for _ in range(batch_size)]

        for _ in range(max_len):
            logits = self.forward(src, generated)
            next_token = logits[:, -1].argmax(dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            for i, token in enumerate(next_token.tolist()):
                if finished[i]:
                    continue
                if token == eos_idx:
                    finished[i] = True
                else:
                    hyp_ids[i].append(token)

            if bool(finished.all()):
                break

        return hyp_ids


def build_scratch_transformer_model(
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    model_args: Dict[str, object],
    device: torch.device,
) -> ScratchTransformerSeq2Seq:
    """Construct a randomly initialised Transformer from stored/CLI arguments."""
    return ScratchTransformerSeq2Seq(
        src_vocab_size=len(src_vocab.idx2word),
        tgt_vocab_size=len(tgt_vocab.idx2word),
        src_pad_idx=src_vocab.pad_idx,
        tgt_pad_idx=tgt_vocab.pad_idx,
        d_model=int(model_args.get("d_model", 256)),
        nhead=int(model_args.get("nhead", 4)),
        num_encoder_layers=int(model_args.get("num_encoder_layers", 3)),
        num_decoder_layers=int(model_args.get("num_decoder_layers", 3)),
        dim_feedforward=int(model_args.get("dim_feedforward", 1024)),
        dropout=float(model_args.get("dropout", 0.1)),
        max_len=int(model_args.get("max_len", 256)),
    ).to(device)


def compute_scratch_transformer_loss(
    model: ScratchTransformerSeq2Seq,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    src, tgt = batch
    src = src.to(device)
    tgt = tgt.to(device)
    tgt_input = tgt[:, :-1]
    gold = tgt[:, 1:].contiguous()
    logits = model(src, tgt_input)
    return criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))


def evaluate_scratch_transformer_nll(
    model: ScratchTransformerSeq2Seq,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    pad_idx: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for src, tgt in data_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            gold = tgt[:, 1:].contiguous()
            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
            num_tokens = (gold != pad_idx).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    return total_loss / max(total_tokens, 1)


def save_scratch_transformer_checkpoint(
    path: str,
    model: ScratchTransformerSeq2Seq,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    model_args: Dict[str, object],
    best_metric: Optional[float] = None,
) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "src_vocab": src_vocab,
            "tgt_vocab": tgt_vocab,
            "model_args": model_args,
            "best_metric": best_metric,
        },
        path,
    )


def load_scratch_transformer_checkpoint(path: str, device: torch.device):
    try:
        from torch.serialization import safe_globals

        with safe_globals([Vocab]):
            ckpt = torch.load(path, map_location=device)
    except Exception:
        ckpt = torch.load(path, map_location=device)
    src_vocab: Vocab = ckpt["src_vocab"]
    tgt_vocab: Vocab = ckpt["tgt_vocab"]
    model_args: Dict[str, object] = ckpt["model_args"]
    model = build_scratch_transformer_model(src_vocab, tgt_vocab, model_args, device)
    model.load_state_dict(ckpt["model_state"])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return model, optimizer, src_vocab, tgt_vocab, model_args, int(ckpt.get("epoch", 0))


def scratch_epoch_checkpoint_path(save_path: str, epoch: int) -> str:
    if save_path.endswith(".pt"):
        return re.sub(r"\.pt$", f".epoch{epoch:03d}.pt", save_path)
    return f"{save_path}.epoch{epoch:03d}.pt"


def cleanup_scratch_checkpoints(save_path: str, keep_last: int) -> None:
    prefix = save_path[:-3] if save_path.endswith(".pt") else save_path
    checkpoints = sorted(glob.glob(f"{prefix}.epoch*.pt"))
    if keep_last <= 0:
        for checkpoint in checkpoints:
            os.remove(checkpoint)
        return
    for checkpoint in checkpoints[:-keep_last]:
        os.remove(checkpoint)


def translate_scratch_transformer_lines(
    model: ScratchTransformerSeq2Seq,
    src_lines: Sequence[str],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device,
    max_len: int,
    subword_type: str = "none",
    src_sp: Optional[object] = None,
    tgt_sp: Optional[object] = None,
    batch_size: int = 32,
) -> List[str]:
    """Translate source strings with the scratch Transformer in batches."""
    hyps: List[str] = []
    model.eval()
    batch_size = max(1, int(batch_size))

    for batch in tqdm(list(batched(src_lines, batch_size)), desc="Translating", leave=False):
        model_src_batch = [
            " ".join(src_sp.encode(src_sentence, out_type=str))
            if src_sp is not None and subword_type != "none"
            else src_sentence
            for src_sentence in batch
        ]
        encoded = [src_vocab.encode(model_src, add_eos=True) for model_src in model_src_batch]
        max_src_len = max(len(ids) for ids in encoded)
        src_tensor = torch.tensor(
            [pad_token_ids(ids, max_src_len, src_vocab.pad_idx) for ids in encoded],
            dtype=torch.long,
            device=device,
        )

        batch_hyp_ids = model.greedy_decode_batch(
            src_tensor,
            tgt_vocab.sos_idx,
            tgt_vocab.eos_idx,
            max_len=max_len,
        )

        for hyp_ids in batch_hyp_ids:
            tokenised_hyp = tgt_vocab.decode(hyp_ids)
            hyp = tgt_sp.decode(tokenised_hyp.split()) if tgt_sp is not None and subword_type != "none" else tokenised_hyp
            hyps.append(hyp)

    return hyps


def translate_scratch_transformer_pairs(
    model: ScratchTransformerSeq2Seq,
    pairs: Sequence[Tuple[str, str]],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device,
    max_len: int,
    subword_type: str,
    src_sp: Optional[object],
    tgt_sp: Optional[object],
    batch_size: int = 32,
) -> List[str]:
    return translate_scratch_transformer_lines(
        model,
        [src for src, _ in pairs],
        src_vocab,
        tgt_vocab,
        device,
        max_len=max_len,
        subword_type=subword_type,
        src_sp=src_sp,
        tgt_sp=tgt_sp,
        batch_size=batch_size,
    )


def finetune_scratch_transformer(args: argparse.Namespace) -> None:
    """Train or resume a Transformer encoder-decoder from random initialisation."""
    device_name = args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(device_name)
    set_rnn_seed(args.seed)

    subword_type, src_sp, tgt_sp = load_sentencepiece_processors(args)
    train_pairs = read_parallel(args.src_file, args.tgt_file, lower=args.lower, limit=args.limit)
    if not train_pairs:
        raise ValueError("No training sentence pairs were loaded.")
    val_pairs = read_parallel(args.src_val, args.tgt_val, lower=args.lower) if args.src_val and args.tgt_val else None

    if args.scratch_load:
        model, optimizer, src_vocab, tgt_vocab, model_args, stored_epoch = load_scratch_transformer_checkpoint(args.scratch_load, device)
        subword_type, src_sp, tgt_sp = load_sentencepiece_processors(model_args)
        start_epoch = stored_epoch + 1
        print(f"Loaded scratch Transformer checkpoint from {args.scratch_load}; resuming at epoch {start_epoch}.")
    else:
        train_pairs_tok = apply_sentencepiece_to_pairs(train_pairs, src_sp, tgt_sp)
        src_vocab = build_vocab([src for src, _ in train_pairs_tok], max_size=args.max_src_vocab)
        tgt_vocab = build_vocab([tgt for _, tgt in train_pairs_tok], max_size=args.max_tgt_vocab)
        model_args = {
            "model_type": "transformer-scratch",
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_encoder_layers": args.transformer_enc_layers,
            "num_decoder_layers": args.transformer_dec_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
            "max_len": args.max_len,
            "subword_type": args.subword_type,
            "src_sp_model": args.src_sp_model,
            "tgt_sp_model": args.tgt_sp_model,
            "lower": args.lower,
        }
        model = build_scratch_transformer_model(src_vocab, tgt_vocab, model_args, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        start_epoch = 1
        print(f"Source vocab size: {len(src_vocab.idx2word)}")
        print(f"Target vocab size: {len(tgt_vocab.idx2word)}")

    train_pairs_tok = apply_sentencepiece_to_pairs(train_pairs, src_sp, tgt_sp)
    val_pairs_tok = apply_sentencepiece_to_pairs(val_pairs, src_sp, tgt_sp)
    train_dataset = RNNParallelDataset(train_pairs_tok, src_vocab, tgt_vocab, max_len=args.max_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: rnn_collate_fn(batch, src_vocab.pad_idx, tgt_vocab.pad_idx),
    )
    val_loader = None
    if val_pairs_tok is not None:
        val_dataset = RNNParallelDataset(val_pairs_tok, src_vocab, tgt_vocab, max_len=None)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size or args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: rnn_collate_fn(batch, src_vocab.pad_idx, tgt_vocab.pad_idx),
        )

    print(f"Loaded {len(train_pairs)} training pairs; {len(train_dataset)} used after max_len={args.max_len} filtering.")
    print(f"Total trainable parameters: {count_trainable_parameters(model):,}")
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)
    requested_metrics = parse_metrics(args.metrics) if args.eval_metrics else set()
    if requested_metrics and sacrebleu is None:
        raise RuntimeError("sacrebleu is required for scratch Transformer validation metrics: pip install sacrebleu")

    save_best = args.scratch_save_best
    if save_best is None:
        base, ext = os.path.splitext(args.save)
        save_best = f"{base}.best{ext or '.pt'}"

    monitor_metric = str(getattr(args, "metric_for_best_model", None) or args.early_metric).lower()
    if monitor_metric == "eval_loss":
        monitor_metric = "loss"
    elif monitor_metric.startswith("eval_"):
        monitor_metric = monitor_metric[5:]
    if monitor_metric not in {"loss", "bleu", "chrf", "ter"}:
        monitor_metric = "loss"
    patience_limit = args.early_stopping
    if patience_limit <= 0 and getattr(args, "early_stopping_patience", None) is not None:
        patience_limit = args.early_stopping_patience
    threshold = float(getattr(args, "early_stopping_threshold", 0.0) or 0.0)

    def is_better(curr: float, best: Optional[float]) -> bool:
        if best is None:
            return True
        if monitor_metric in {"loss", "ter"}:
            return curr < best - threshold
        return curr > best + threshold

    print(f"Scratch Transformer early-stopping monitor: {monitor_metric}; patience={patience_limit}")
    history: List[Dict[str, object]] = []
    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for src, tgt in tqdm(train_loader, desc=f"Transformer epoch {epoch}", leave=False):
            optimizer.zero_grad()
            loss = compute_scratch_transformer_loss(model, (src, tgt), criterion, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                gold = tgt[:, 1:].contiguous()
                n_tokens = (gold != tgt_vocab.pad_idx).sum().item()
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens

        train_nll = total_loss / max(total_tokens, 1)
        train_ppl = math.exp(train_nll) if train_nll < 20 else float("inf")
        record: Dict[str, object] = {"epoch": epoch, "train_nll": train_nll, "train_ppl": None if train_ppl == float("inf") else train_ppl}
        log_msg = f"Epoch {epoch:03d}: train NLL={train_nll:.4f} (ppl={train_ppl:.2f})"
        metric_value: Optional[float] = None

        if val_loader is not None and val_pairs is not None:
            val_nll = evaluate_scratch_transformer_nll(model, val_loader, criterion, device, tgt_vocab.pad_idx)
            val_ppl = math.exp(val_nll) if val_nll < 20 else float("inf")
            record.update({"val_nll": val_nll, "val_ppl": None if val_ppl == float("inf") else val_ppl})
            log_msg += f"  val NLL={val_nll:.4f} (ppl={val_ppl:.2f})"
            if monitor_metric == "loss":
                metric_value = val_nll

            need_translations = bool(requested_metrics or args.save_eval_translations or args.show_val_examples > 0)
            hyps: Optional[List[str]] = None
            refs: Optional[List[str]] = None
            if need_translations:
                refs = [tgt for _, tgt in val_pairs]
                hyps = translate_scratch_transformer_pairs(
                    model, val_pairs, src_vocab, tgt_vocab, device, args.max_len, subword_type, src_sp, tgt_sp,
                    batch_size=args.eval_batch_size or args.batch_size,
                )
            if requested_metrics and hyps is not None and refs is not None:
                metrics = compute_rnn_sacrebleu_metrics(hyps, refs, requested_metrics)
                record.update(metrics)
                for name, value in metrics.items():
                    log_msg += f"  {name.upper()}={value:.2f}"
                if monitor_metric in metrics:
                    metric_value = metrics[monitor_metric]
            if args.save_eval_translations and hyps is not None and refs is not None:
                out_dir = args.eval_translations_dir or os.path.join(args.save, "eval_translations")
                os.makedirs(out_dir, exist_ok=True)
                stem = os.path.join(out_dir, f"transformer_scratch_eval_epoch_{epoch:03d}")
                write_lines_if_missing(os.path.join(out_dir, "validation.src"), [src for src, _ in val_pairs])
                write_lines_if_missing(os.path.join(out_dir, "validation.ref"), refs)
                write_lines(stem + ".hyp", hyps)
                with open(stem + ".scores.json", "w", encoding="utf-8") as f:
                    json.dump({k: v for k, v in record.items() if k in {"bleu", "chrf", "ter", "val_nll"}}, f, indent=2, ensure_ascii=False)
            if args.show_val_examples > 0 and hyps is not None and refs is not None:
                print("\n--- Scratch Transformer validation examples ---")
                for i, ((src, ref), hyp) in enumerate(zip(val_pairs[: args.show_val_examples], hyps[: args.show_val_examples]), start=1):
                    print(f"[{i}] SRC: {src}")
                    print(f"    REF: {ref}")
                    print(f"    HYP: {hyp}")
                print("----------------------------------------------")

        print(log_msg)
        if metric_value is not None and is_better(metric_value, best_metric):
            best_metric = metric_value
            best_epoch = epoch
            patience_counter = 0
            save_scratch_transformer_checkpoint(save_best, model, optimizer, epoch, src_vocab, tgt_vocab, model_args, best_metric)
            print(f"  -> New best {monitor_metric}={metric_value:.4f}; saved {save_best}")
        elif metric_value is not None and patience_limit > 0:
            patience_counter += 1
            print(f"  -> No improvement on {monitor_metric}; patience={patience_counter}/{patience_limit}")
            if patience_counter >= patience_limit:
                print("Early stopping triggered.")
                break

        record["best_metric"] = best_metric
        record["best_epoch"] = best_epoch
        history.append(record)
        if args.history_json:
            out_dir = os.path.dirname(args.history_json)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.history_json, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

        save_path = scratch_epoch_checkpoint_path(args.save, epoch)
        save_scratch_transformer_checkpoint(save_path, model, optimizer, epoch, src_vocab, tgt_vocab, model_args, best_metric)
        cleanup_scratch_checkpoints(args.save, args.scratch_keep_last)

    if best_epoch is not None:
        print(f"Training finished. Best {monitor_metric}={best_metric:.4f} at epoch {best_epoch}.")
    else:
        print("Training finished.")


def translate_scratch_transformer(args: argparse.Namespace) -> None:
    """Translate a file with a scratch Transformer checkpoint."""
    if not args.model_dir:
        raise ValueError("--model-dir must point to a scratch Transformer .pt checkpoint when --model-type transformer-scratch")
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(device_name)
    set_rnn_seed(getattr(args, "seed", 42))
    model, _optimizer, src_vocab, tgt_vocab, model_args, epoch = load_scratch_transformer_checkpoint(args.model_dir, device)
    subword_type, src_sp, tgt_sp = load_sentencepiece_processors(model_args)
    src_lines = list(iter_lines(args.src_file, lower=bool(model_args.get("lower", args.lower))))
    hyps = translate_scratch_transformer_lines(
        model,
        src_lines,
        src_vocab,
        tgt_vocab,
        device,
        max_len=int(model_args.get("max_len", args.max_gen_len)),
        subword_type=subword_type,
        src_sp=src_sp,
        tgt_sp=tgt_sp,
        batch_size=args.batch_size,
    )
    write_lines(args.out_file, hyps)
    print(f"Loaded scratch Transformer checkpoint from {args.model_dir} (epoch {epoch}).")
    print(f"Wrote {len(hyps)} translations to {args.out_file}")
    if args.ref_file:
        refs = read_lines(args.ref_file, lower=bool(model_args.get("lower", args.lower)))
        if len(refs) != len(hyps):
            raise ValueError(f"Line count mismatch: {len(hyps)} system outputs vs {len(refs)} references.")
        scores = compute_rnn_sacrebleu_metrics(hyps, refs, parse_metrics(args.metrics))
        for name, score in scores.items():
            print(f"{name}: {score:.2f}")


# -----------------------------------------------------------------------------
# Checkpoint and continuation helpers
# -----------------------------------------------------------------------------

def checkpoint_step(path: str) -> int:
    """Extract the numeric step from a Hugging Face checkpoint directory."""
    base = os.path.basename(os.path.normpath(path))
    match = re.match(r"checkpoint-(\d+)$", base)
    if not match:
        return -1
    return int(match.group(1))


def find_latest_checkpoint(run_dir: str) -> Optional[str]:
    """Return the latest checkpoint-* directory inside a run directory."""
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
    Decide whether Hugging Face Trainer should resume from a checkpoint.

    Loading --pretrained-model from a previous run restores only model weights.
    Resuming from a checkpoint restores the full Trainer state, including
    optimiser and scheduler state.
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


# -----------------------------------------------------------------------------
# Config logging and YAML config loading
# -----------------------------------------------------------------------------

def make_yaml_safe(value):
    """Convert arbitrary values to YAML-safe plain Python objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): make_yaml_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [make_yaml_safe(v) for v in value]

    # Handles numpy scalar types without hard-coding every numpy dtype.
    if hasattr(value, "item"):
        try:
            return make_yaml_safe(value.item())
        except Exception:
            pass

    return str(value)


def serializable_config(args: argparse.Namespace) -> Dict[str, object]:
    """Create a reproducible run configuration from parsed CLI arguments."""
    config = dict(vars(args))
    runtime_only_keys = {
        "config",
        "resolved_resume_from_checkpoint",
    }

    for key in list(config.keys()):
        if key in runtime_only_keys:
            del config[key]

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
    Save the fully resolved run configuration to config.yaml.

    This makes experiments easier to reproduce: the saved YAML contains argparse
    defaults, explicit user options, package versions, Python version, and a Git
    commit when available.
    """
    if not hasattr(args, "save") or args.save is None:
        return

    os.makedirs(args.save, exist_ok=True)
    yaml_path = os.path.join(args.save, "config.yaml")
    config = serializable_config(args)

    if yaml is None:
        print("WARNING: PyYAML is not installed; config.yaml was not written.")
        print("Install with: pip install pyyaml")
        return

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    print(f"Saved run config to {yaml_path}")


def load_yaml_config(path: str) -> Dict[str, object]:
    """Load a YAML config from a previous run."""
    if yaml is None:
        raise RuntimeError("Using --config requires PyYAML. Install with: pip install pyyaml")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file should contain a YAML mapping/dictionary: {path}")

    # Metadata is useful for humans but should not be parsed as CLI arguments.
    data.pop("_metadata", None)
    return data


def config_to_cli_args(config: Dict[str, object]) -> List[str]:
    """
    Convert YAML config values to argparse-style command-line arguments.

    Values explicitly supplied on the command line after --config are appended
    later and therefore override the YAML values.
    """
    cli: List[str] = []
    command = config.pop("command", None)

    for key, value in config.items():
        if value is None:
            continue

        option = "--" + key.replace("_", "-")

        # Boolean flags are represented by their presence. False means omit.
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
    """Parse --config first, then combine YAML values with explicit CLI args."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None, help="YAML config file from a previous run")
    known, remaining = pre.parse_known_args()

    if known.config is None:
        return parser.parse_args()

    config = load_yaml_config(known.config)
    yaml_cli = config_to_cli_args(config)

    args = parser.parse_args(yaml_cli + remaining)
    args.config = known.config
    return args


# -----------------------------------------------------------------------------
# CLI construction
# -----------------------------------------------------------------------------

def add_common_data_args(ap: argparse.ArgumentParser) -> None:
    """Arguments shared by training and translation data inputs."""
    ap.add_argument("--src-file", required=True, help="Source file, one sentence per line")
    ap.add_argument("--lower", action="store_true", help="Lowercase input text")


def add_common_generation_args(ap: argparse.ArgumentParser) -> None:
    """Generation arguments shared by local-model translation workflows."""
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-src-len", type=int, default=128)
    ap.add_argument("--max-gen-len", type=int, default=128)
    ap.add_argument("--num-beams", type=int, default=4)


def add_language_args(ap: argparse.ArgumentParser) -> None:
    """Language-code arguments used by multilingual Hugging Face models."""
    ap.add_argument("--src-lang", default=None, help="Source language code, required for m2m/mbart/nllb")
    ap.add_argument("--tgt-lang", default=None, help="Target language code, required for m2m/mbart/nllb")


def add_prefix_args(ap: argparse.ArgumentParser) -> None:
    """Optional source prefix, useful for models such as T5 and MADLAD."""
    ap.add_argument(
        "--prefix",
        default="",
        help="Prefix prepended to each source sentence; useful for T5/MADLAD",
    )
    ap.add_argument("--no-prefix", action="store_true", help="Ignore --prefix")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the full command-line interface."""
    ap = argparse.ArgumentParser(
        description="Unified MTAT CLI for training, finetuning and translating MT models."
    )
    ap.add_argument(
        "--config",
        default=None,
        help="YAML config file. CLI arguments supplied after --config override the YAML values.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # Fine-tuning subcommand ---------------------------------------------------
    ft = sub.add_parser("finetune", help="Fine-tune a pretrained seq2seq model")
    ft.add_argument("--model-type", required=True, choices=sorted(FINETUNE_MODEL_TYPES))
    ft.add_argument(
        "--pretrained-model",
        default=None,
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
    ft.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="Allow training in an existing non-empty output directory.",
    )
    ft.add_argument("--lora", action="store_true")
    ft.add_argument("--qlora", action="store_true")
    ft.add_argument("--lora-r", type=int, default=16)
    ft.add_argument("--lora-alpha", type=int, default=32)
    ft.add_argument("--lora-dropout", type=float, default=0.05)
    ft.add_argument(
        "--lora-target-modules",
        default="q,v",
        help="Comma-separated module names, e.g. q,v or q,k,v,o",
    )

    # Custom-from-scratch architecture/training options. These are ignored by pretrained
    # HF model types. RNN and transformer-scratch can use word tokens or SentencePiece.
    ft.add_argument("--emb-size", type=int, default=64)
    ft.add_argument("--hidden-size", type=int, default=128)
    ft.add_argument("--max-len", type=int, default=50, help="Max sentence/generation length for rnn and transformer-scratch")
    ft.add_argument("--teacher-forcing", type=float, default=0.7)
    ft.add_argument("--enc-layers", type=int, default=1)
    ft.add_argument("--dec-layers", type=int, default=1)
    ft.add_argument("--rnn-type", choices=["rnn", "gru", "lstm"], default="rnn")
    ft.add_argument("--attention", choices=["none", "luong"], default="none")
    ft.add_argument("--bidirectional", action="store_true")
    ft.add_argument("--limit", type=int, default=None, help="Limit training pairs for debugging")
    ft.add_argument("--max-src-vocab", type=int, default=None)
    ft.add_argument("--max-tgt-vocab", type=int, default=None)
    ft.add_argument("--subword-type", choices=["none", "bpe", "unigram"], default="none")
    ft.add_argument("--src-sp-model", default=None)
    ft.add_argument("--tgt-sp-model", default=None)
    ft.add_argument("--rnn-load", default=None, help="RNN checkpoint to resume from")
    ft.add_argument("--rnn-save-best", default=None, help="Best RNN checkpoint path; default: <save>.best.pt")
    ft.add_argument("--rnn-keep-last", type=int, default=1, help="Number of RNN epoch checkpoints to keep")
    ft.add_argument("--early-stopping", type=int, default=0, help="RNN patience in epochs; 0 disables")
    ft.add_argument("--early-metric", choices=["loss", "bleu", "chrf", "ter"], default="loss")
    ft.add_argument("--replace-unk", action="store_true", help="RNN attention-based <unk> replacement")
    ft.add_argument("--d-model", type=int, default=256, help="Scratch Transformer embedding/hidden size")
    ft.add_argument("--nhead", type=int, default=4, help="Scratch Transformer attention heads")
    ft.add_argument("--transformer-enc-layers", type=int, default=3, help="Scratch Transformer encoder layers")
    ft.add_argument("--transformer-dec-layers", type=int, default=3, help="Scratch Transformer decoder layers")
    ft.add_argument("--dim-feedforward", type=int, default=1024, help="Scratch Transformer feed-forward size")
    ft.add_argument("--dropout", type=float, default=0.1, help="Scratch Transformer dropout")
    ft.add_argument("--scratch-load", default=None, help="Scratch Transformer checkpoint to resume from")
    ft.add_argument("--scratch-save-best", default=None, help="Best scratch Transformer checkpoint path; default: <save>.best.pt")
    ft.add_argument("--scratch-keep-last", type=int, default=1, help="Number of scratch Transformer epoch checkpoints to keep")

    # Translation subcommand ---------------------------------------------------
    tr = sub.add_parser("translate", help="Translate a source file with a model")
    tr.add_argument("--model-type", required=True, choices=sorted(TRANSLATE_MODEL_TYPES))
    tr.add_argument(
        "--model-dir",
        default=None,
        help="HF model directory (required for local models, not for --model-type openai)",
    )
    add_common_data_args(tr)
    tr.add_argument("--out-file", required=True)
    add_language_args(tr)
    add_prefix_args(tr)
    add_common_generation_args(tr)
    tr.add_argument("--device", default=None, choices=["cpu", "cuda"])
    tr.add_argument("--ref-file", default=None)
    tr.add_argument("--metrics", default="bleu,chrf,ter")
    tr.add_argument("--fp16", action="store_true")
    tr.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable generation cache; slower, but avoids cache-related warnings.",
    )
    tr.add_argument("--replace-unk", action="store_true", help="RNN attention-based <unk> replacement")
    tr.add_argument("--seed", type=int, default=42, help="Random seed for RNN translation")

    # OpenAI-compatible API options. These are ignored for local HF models.
    tr.add_argument("--model", default=None, help="OpenAI-compatible model name")
    tr.add_argument("--base-url", default="https://api.helmholtz-blablador.fz-juelich.de/v1/")
    tr.add_argument("--api-key", default=None)
    tr.add_argument("--api-env", default=None)
    tr.add_argument("--api-key-file", default=None)
    tr.add_argument("--source-lang", default=None, help="Source language name, e.g. French")
    tr.add_argument("--target-lang", default=None, help="Target language name, e.g. Dutch")
    tr.add_argument("--temperature", type=float, default=0.0)
    tr.add_argument("--timeout", type=float, default=60.0)
    tr.add_argument("--max-retries", type=int, default=5)
    tr.add_argument("--retry-wait", type=float, default=2.0)
    tr.add_argument("--debug", action="store_true")

    return ap


# -----------------------------------------------------------------------------
# Main dispatcher
# -----------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and dispatch to the selected workflow."""
    parser = build_arg_parser()
    args = parse_args_with_optional_config(parser)

    if args.command == "finetune":
        if args.model_type == "rnn":
            finetune_rnn_seq2seq(args)
            return
        if args.model_type == "transformer-scratch":
            finetune_scratch_transformer(args)
            return
        if not args.pretrained_model:
            raise ValueError("--pretrained-model is required for Hugging Face pretrained model types")
        finetune_hf_seq2seq(args)
        return

    if args.command == "translate":
        if args.model_type == "openai":
            if not args.model:
                raise ValueError("--model is required when --model-type openai")
            if not args.target_lang:
                raise ValueError("--target-lang is required when --model-type openai")
            translate_openai(args)
            return

        if args.model_type == "rnn":
            translate_rnn_seq2seq(args)
            return

        if args.model_type == "transformer-scratch":
            translate_scratch_transformer(args)
            return

        if not args.model_dir:
            raise ValueError(
                "--model-dir is required for model types t5, mbart, m2m, nllb and madlad"
            )
        translate_hf_seq2seq(args)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
