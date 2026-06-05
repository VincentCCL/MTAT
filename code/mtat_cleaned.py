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
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import transformers
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


# -----------------------------------------------------------------------------
# Supported model types
# -----------------------------------------------------------------------------

# These constants centralise model-type validation. Fine-tuning only supports
# Hugging Face seq2seq models. Translation additionally supports an
# OpenAI-compatible API backend.
HF_SEQ2SEQ_TYPES = {"t5", "mbart", "m2m", "nllb", "madlad"}
TRANSLATE_MODEL_TYPES = HF_SEQ2SEQ_TYPES | {"openai"}


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
    """Validate and extract the translations list from the model response."""
    data = json.loads(text)

    if not isinstance(data, dict) or "translations" not in data:
        raise ValueError("Model output does not contain a 'translations' key")

    translations = data["translations"]

    if not isinstance(translations, list):
        raise ValueError("'translations' is not a list")

    if len(translations) != expected_n:
        raise ValueError(
            f"Translation length mismatch: expected {expected_n}, got {len(translations)}"
        )

    return [str(item).strip() for item in translations]


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

        if not args.model_dir:
            raise ValueError(
                "--model-dir is required for model types t5, mbart, m2m, nllb and madlad"
            )
        translate_hf_seq2seq(args)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
