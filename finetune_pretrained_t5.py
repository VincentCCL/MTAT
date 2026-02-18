#!/usr/bin/env python3
"""
finetune_pretrained_t5.py — Fine-tune pretrained encoder-decoder models (T5/mT5 etc.)
on parallel data, using (mostly) the same CLI options as transformer.py (from scratch).

Compatibility notes:
- --spm-src-model and --spm-tgt-model are accepted but ignored (pretrained models use their own tokenizer).
- Keeps: --src-file --tgt-file --src-val --tgt-val --epochs --save --batch-size --lr --weight-decay
         --warmup-ratio --num-beams --max-gen-len --logging-steps --save-steps --eval-metrics
         --history-json --show-val-examples --seed --max-src-len --max-tgt-len
Adds:
- --pretrained-model (default: t5-small)
- --src-lang / --tgt-lang (for prefix text)
- --no-prefix (disable "translate X to Y:" style)
"""

import argparse
import json
import os

# Avoid TF/XLA noise in Kaggle/Colab environments
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    TrainerCallback,
)

try:
    import evaluate
except Exception:
    evaluate = None


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


@dataclass
class SimpleDataset(torch.utils.data.Dataset):
    features: List[Dict[str, torch.Tensor]]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.features[idx]


def build_prefix(src_lang: str, tgt_lang: str) -> str:
    # Match the classic T5 paper-style prompting
    return f"translate {src_lang} to {tgt_lang}: "


def tokenize_parallel_one_tokenizer(
    tokenizer,
    src_lines: List[str],
    tgt_lines: List[str],
    max_src_len: int,
    max_tgt_len: int,
    prefix: str = "",
) -> SimpleDataset:
    if len(src_lines) != len(tgt_lines):
        raise ValueError("Parallel files must have the same number of lines.")

    feats: List[Dict[str, torch.Tensor]] = []
    for s, t in zip(src_lines, tgt_lines):
        # Encoder input
        enc = tokenizer(
            prefix + s,
            truncation=True,
            max_length=max_src_len,
            add_special_tokens=True,
        )

        # Labels (decoder targets)
        # Prefer modern HF API when available:
        # tokenizer(text_target=...) sets up target-side tokenization cleanly.
        try:
            dec = tokenizer(
                text_target=t,
                truncation=True,
                max_length=max_tgt_len,
                add_special_tokens=True,
            )
            label_ids = dec["input_ids"]
        except TypeError:
            # Fallback for older tokenizers
            dec = tokenizer(
                t,
                truncation=True,
                max_length=max_tgt_len,
                add_special_tokens=True,
            )
            label_ids = dec["input_ids"]

        feats.append(
            {
                "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(label_ids, dtype=torch.long),
            }
        )

    return SimpleDataset(feats)


def decode_batch(tokenizer, arr):
    # arr: np.ndarray or torch tensor
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()

    arr = np.asarray(arr)

    # If we got logits: [B, T, V] -> ids [B, T]
    if arr.ndim == 3:
        arr = arr.argmax(axis=-1)

    # Replace masked labels (-100) to avoid decoding negatives
    arr = np.where(arr == -100, tokenizer.pad_token_id, arr).astype(np.int64)

    return tokenizer.batch_decode(
        arr,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )


def compute_metrics_builder(tokenizer):
    if evaluate is None:
        return None

    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        decoded_preds = decode_batch(tokenizer, preds)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = decode_batch(tokenizer, labels)

        bleu_out = sacrebleu.compute(
            predictions=decoded_preds,
            references=[[x] for x in decoded_labels],
        )
        chrf_out = chrf.compute(
            predictions=decoded_preds,
            references=decoded_labels,
        )

        return {
            "bleu": float(bleu_out["score"]),
            "chrf": float(chrf_out["score"]),
        }

    return compute_metrics


class ShowValidationExamplesCallback(TrainerCallback):
    def __init__(
        self,
        model,
        tokenizer,
        val_src_lines,
        val_tgt_lines=None,
        prefix: str = "",
        num_examples: int = 5,
        max_len: int = 128,
        num_beams: int = 4,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.val_src_lines = val_src_lines
        self.val_tgt_lines = val_tgt_lines
        self.prefix = prefix
        self.num_examples = num_examples
        self.max_len = max_len
        self.num_beams = num_beams
        self.device = device

        k = min(self.num_examples, len(self.val_src_lines))
        rng = random.Random(seed)
        self.idxs = rng.sample(range(len(self.val_src_lines)), k)

    def on_evaluate(self, args, state, control, **kwargs):
        if self.num_examples <= 0:
            return

        print("\n=== Validation examples (step {}) ===".format(state.global_step))

        self.model.eval()

        idxs = self.idxs
        src_lines = [self.val_src_lines[i] for i in idxs]
        ref_lines = [self.val_tgt_lines[i] for i in idxs] if self.val_tgt_lines is not None else None

        enc = self.tokenizer(
            [self.prefix + s for s in src_lines],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.generation_max_length,
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **enc,
                num_beams=self.num_beams,
                max_length=self.max_len,
            )

        hyps = self.tokenizer.batch_decode(out, skip_special_tokens=True)

        for j, i in enumerate(idxs):
            print(f"[{j+1}] (dev idx {i}) SRC: {src_lines[j]}")
            if ref_lines is not None:
                print(f"             REF: {ref_lines[j]}")
            print(f"             HYP: {hyps[j]}")
        print()


def write_history(trainer: Seq2SeqTrainer, path: str) -> None:
    write_json(trainer.state.log_history, path)


def main():
    p = argparse.ArgumentParser()

    # Keep the "from scratch transformer.py" core options
    p.add_argument("--src-file", required=True)
    p.add_argument("--tgt-file", required=True)
    p.add_argument("--src-val", required=True)
    p.add_argument("--tgt-val", required=True)

    p.add_argument("--enc-layers", type=int, default=6)   # kept for CLI compatibility (ignored)
    p.add_argument("--dec-layers", type=int, default=6)   # kept for CLI compatibility (ignored)
    p.add_argument("--emb-size", type=int, default=512)   # kept for CLI compatibility (ignored)
    p.add_argument("--hidden-size", type=int, default=2048)  # kept for CLI compatibility (ignored)

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--save", required=True)

    p.add_argument("--save-val-json", default=None)  # kept for compatibility (unused here)
    p.add_argument("--show-val-examples", type=int, default=0)
    p.add_argument("--eval-metrics", action="store_true")
    p.add_argument("--history-json", default=None)

    # Keep SPM args but do not require them (pretrained uses its own tokenizer)
    p.add_argument("--spm-src-model", default=None)
    p.add_argument("--spm-tgt-model", default=None)

    # Extras (same names as your scratch script)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-src-len", type=int, default=128)
    p.add_argument("--max-tgt-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--num-beams", type=int, default=4)
    p.add_argument("--max-gen-len", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=8)  # kept for CLI compatibility (ignored)
    p.add_argument("--dropout", type=float, default=0.1)  # kept for CLI compatibility (ignored)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=0)

    # New: pretrained model selection + prefix control
    p.add_argument("--pretrained-model", default="t5-small",
                   help="HF model id, e.g. t5-small, t5-base, google/mt5-small")
    p.add_argument("--src-lang", default="English")
    p.add_argument("--tgt-lang", default="Dutch")
    p.add_argument("--no-prefix", action="store_true",
                   help="Disable 'translate X to Y:' prefix (not recommended for T5-style training).")

    args = p.parse_args()

    ensure_dir(args.save)
    set_seed(args.seed)

    if args.spm_src_model or args.spm_tgt_model:
        print("NOTE: --spm-src-model/--spm-tgt-model are ignored for pretrained T5/mT5 fine-tuning.")

    # Build prefix (unless disabled)
    prefix = "" if args.no_prefix else build_prefix(args.src_lang, args.tgt_lang)

    # Data
    src_train = read_lines(args.src_file)
    tgt_train = read_lines(args.tgt_file)
    src_val = read_lines(args.src_val)
    tgt_val = read_lines(args.tgt_val)

    if len(src_train) != len(tgt_train):
        raise ValueError("Training src/tgt have different number of lines.")
    if len(src_val) != len(tgt_val):
        raise ValueError("Validation src/tgt have different number of lines.")

    # Tokenizer + pretrained model
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model)

    # Ensure pad token exists (rarely needed for T5, but safe for some models)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Save tokenizer for reproducibility
    tokenizer.save_pretrained(os.path.join(args.save, "tokenizer"))

    # Datasets (we pre-tokenize to keep close to your original script structure)
    train_ds = tokenize_parallel_one_tokenizer(
        tokenizer, src_train, tgt_train, args.max_src_len, args.max_tgt_len, prefix=prefix
    )
    val_ds = tokenize_parallel_one_tokenizer(
        tokenizer, src_val, tgt_val, args.max_src_len, args.max_tgt_len, prefix=prefix
    )

    # Collator: dynamic padding + label masking (-100)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    compute_metrics = compute_metrics_builder(tokenizer) if args.eval_metrics else None

    save_strategy = "epoch" if args.save_steps <= 0 else "steps"

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.save,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        predict_with_generate=True,
        generation_max_length=args.max_gen_len,
        generation_num_beams=args.num_beams,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy=save_strategy,
        save_steps=(args.save_steps if save_strategy == "steps" else None),
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",   # avoids wandb/comet/mlflow surprises
        label_smoothing_factor=args.label_smoothing,
    )

    callbacks = []
    if args.show_val_examples > 0:
        callbacks.append(
            ShowValidationExamplesCallback(
                model=model,
                tokenizer=tokenizer,
                val_src_lines=src_val,
                val_tgt_lines=tgt_val,
                prefix=prefix,
                num_examples=args.show_val_examples,
                max_len=args.max_gen_len,
                num_beams=args.num_beams,
                device=training_args.device,
                seed=args.seed,
            )
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(args.save)

    if args.history_json:
        write_history(trainer, args.history_json)

    print(f"\nDone. Model saved to: {args.save}")
    if prefix:
        print(f"Used prefix: {prefix!r}")


if __name__ == "__main__":
    main()
