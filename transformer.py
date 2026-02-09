#!/usr/bin/env python3
"""
transformer.py — Train a Transformer from scratch using existing SentencePiece models
(spm.en.model and spm.nl.model). SPM-only.

Fixes:
- If SPM has no PAD id, we add <pad> as an added token.
- IMPORTANT: use len(tokenizer) (base + added tokens) for model vocab sizes,
  because tokenizer.vocab_size excludes added tokens.
"""

import argparse
import json
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import sentencepiece as spm

from transformers import (
    BertConfig,
    BertModel,
    BertLMHeadModel,
    EncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    TrainerCallback
)

from transformers import T5TokenizerFast, T5Tokenizer

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

class ShowValidationExamplesCallback(TrainerCallback):
    def __init__(
        self,
        model,
        tokenizer_src,
        tokenizer_tgt,
        val_src_lines,
        val_tgt_lines=None,          
        num_examples=5,
        max_len=128,
        num_beams=4,
        device="cpu",
        seed=42,
    ):
        self.model = model
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.val_src_lines = val_src_lines
        self.val_tgt_lines = val_tgt_lines
        self.num_examples = num_examples
        self.max_len = max_len
        self.num_beams = num_beams
        self.device = device
        
        # NEW: choose a fixed random subset ONCE (reused every epoch)
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

        enc = self.tokenizer_src(
            src_lines,
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

        hyps = self.tokenizer_tgt.batch_decode(out, skip_special_tokens=True)

        for j, i in enumerate(idxs):
            print(f"[{j+1}] (dev idx {i}) SRC: {src_lines[j]}")
            if ref_lines is not None:
                print(f"             REF: {ref_lines[j]}")
            print(f"             HYP: {hyps[j]}")
        print()
        
@dataclass
class SimpleDataset(torch.utils.data.Dataset):
    features: List[Dict[str, torch.Tensor]]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.features[idx]


def _sp_ids(sp_model_path: str) -> Dict[str, int]:
    proc = spm.SentencePieceProcessor()
    proc.Load(sp_model_path)
    return {
        "unk_id": proc.unk_id(),
        "bos_id": proc.bos_id(),
        "eos_id": proc.eos_id(),
        "pad_id": proc.pad_id(),
        "sp_vocab_size": proc.GetPieceSize(),
    }


def load_spm_tokenizer(sp_model_path: str):
    """
    Load SPM tokenizer via HF wrapper, align ids with SentencePiece where possible.
    If pad is not defined in SPM, add <pad> as an added token.
    """
    try:
        tok = T5TokenizerFast(vocab_file=sp_model_path, extra_ids=0)
    except Exception:
        tok = T5Tokenizer(vocab_file=sp_model_path, extra_ids=0)

    ids = _sp_ids(sp_model_path)

    # UNK always exists in SPM
    tok.unk_token = "<unk>"
    # eos/bos may or may not exist
    if ids["eos_id"] != -1:
        tok.eos_token = "</s>"
    if ids["bos_id"] != -1:
        tok.bos_token = "<s>"

    # PAD: if not in the model, add it
    if ids["pad_id"] != -1:
        tok.pad_token = "<pad>"
    else:
        if tok.pad_token is None:
            tok.add_special_tokens({"pad_token": "<pad>"})

    # If BOS absent, use PAD as BOS (for decoder_start_token_id fallback)
    if tok.bos_token_id is None:
        tok.bos_token = tok.pad_token

    return tok


def tokenize_parallel_two_tokenizers(
    src_tok,
    tgt_tok,
    src_lines: List[str],
    tgt_lines: List[str],
    max_src_len: int,
    max_tgt_len: int,
) -> SimpleDataset:
    if len(src_lines) != len(tgt_lines):
        raise ValueError("Parallel files must have the same number of lines.")

    feats: List[Dict[str, torch.Tensor]] = []
    for s, t in zip(src_lines, tgt_lines):
        enc = src_tok(s, truncation=True, max_length=max_src_len, add_special_tokens=True)
        dec = tgt_tok(t, truncation=True, max_length=max_tgt_len, add_special_tokens=True)
        feats.append(
            {
                "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(dec["input_ids"], dtype=torch.long),
            }
        )
    return SimpleDataset(feats)


class TwoTokenizerSeq2SeqCollator:
    def __init__(self, src_tok, tgt_tok):
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        batch_enc = self.src_tok.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )

        batch_lab = self.tgt_tok.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt",
        )
        lab = batch_lab["input_ids"]
        lab[lab == self.tgt_tok.pad_token_id] = -100
        batch_enc["labels"] = lab
        return batch_enc


def decode_batch(tokenizer, arr):
    # arr: np.ndarray or torch tensor
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()

    arr = np.asarray(arr)

    # If we got logits: [B, T, V] -> take argmax to get ids [B, T]
    if arr.ndim == 3:
        arr = arr.argmax(axis=-1)

    # Replace any masked labels (-100) to avoid decoding negatives
    arr = np.where(arr == -100, tokenizer.pad_token_id, arr)

    # Ensure integer dtype expected by fast tokenizer
    arr = arr.astype(np.int64)

    return tokenizer.batch_decode(
        arr,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )


def compute_metrics_builder(tgt_tok):
    if evaluate is None:
        return None
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")  # <-- add

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        decoded_preds = decode_batch(tgt_tok, preds)

        labels = np.where(labels != -100, labels, tgt_tok.pad_token_id)
        decoded_labels = decode_batch(tgt_tok, labels)

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


def write_history(trainer: Seq2SeqTrainer, path: str) -> None:
    write_json(trainer.state.log_history, path)


def main():
    p = argparse.ArgumentParser()

    # RNN-compatible core options
    p.add_argument("--src-file", required=True)
    p.add_argument("--tgt-file", required=True)
    p.add_argument("--src-val", required=True)
    p.add_argument("--tgt-val", required=True)

    #p.add_argument("--rnn-type", default="rnn", choices=["rnn", "lstm", "gru"])
    p.add_argument("--enc-layers", type=int, default=6)
    p.add_argument("--dec-layers", type=int, default=6)
    p.add_argument("--emb-size", type=int, default=512)
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--save", required=True)

    p.add_argument("--save-val-json", default=None)
    p.add_argument("--show-val-examples", type=int, default=0)
    p.add_argument("--eval-metrics", action="store_true")
    p.add_argument("--history-json", default=None)

    # SPM-only
    p.add_argument("--spm-src-model", required=True)
    p.add_argument("--spm-tgt-model", required=True)

    # Extras
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-src-len", type=int, default=128)
    p.add_argument("--max-tgt-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--num-beams", type=int, default=4)
    p.add_argument("--max-gen-len", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=0)

    args = p.parse_args()

    ensure_dir(args.save)
    set_seed(args.seed)

    # Data
    src_train = read_lines(args.src_file)
    tgt_train = read_lines(args.tgt_file)
    src_val = read_lines(args.src_val)
    tgt_val = read_lines(args.tgt_val)

    if len(src_train) != len(tgt_train):
        raise ValueError("Training src/tgt have different number of lines.")
    if len(src_val) != len(tgt_val):
        raise ValueError("Validation src/tgt have different number of lines.")

    # Tokenizers
    src_tok = load_spm_tokenizer(args.spm_src_model)
    tgt_tok = load_spm_tokenizer(args.spm_tgt_model)

    # IMPORTANT: total vocab sizes (incl. added tokens like <pad>)
    src_vocab = len(src_tok)
    tgt_vocab = len(tgt_tok)

    if not (0 <= src_tok.pad_token_id < src_vocab):
        raise ValueError(f"Bad src pad_token_id={src_tok.pad_token_id} for len(src_tok)={src_vocab}")
    if not (0 <= tgt_tok.pad_token_id < tgt_vocab):
        raise ValueError(f"Bad tgt pad_token_id={tgt_tok.pad_token_id} for len(tgt_tok)={tgt_vocab}")

    src_tok.save_pretrained(os.path.join(args.save, "tokenizer_src"))
    tgt_tok.save_pretrained(os.path.join(args.save, "tokenizer_tgt"))

    # Datasets
    train_ds = tokenize_parallel_two_tokenizers(src_tok, tgt_tok, src_train, tgt_train, args.max_src_len, args.max_tgt_len)
    val_ds = tokenize_parallel_two_tokenizers(src_tok, tgt_tok, src_val, tgt_val, args.max_src_len, args.max_tgt_len)

    # Model from scratch
    max_pos = max(args.max_src_len, args.max_tgt_len) + 32

    enc_config = BertConfig(
        vocab_size=src_vocab,
        hidden_size=args.emb_size,
        num_hidden_layers=args.enc_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.hidden_size,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        max_position_embeddings=max_pos,
        pad_token_id=src_tok.pad_token_id,
    )

    dec_config = BertConfig(
        vocab_size=tgt_vocab,
        hidden_size=args.emb_size,
        num_hidden_layers=args.dec_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.hidden_size,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        max_position_embeddings=max_pos,
        pad_token_id=tgt_tok.pad_token_id,
        is_decoder=True,
        add_cross_attention=True,
    )

    encoder = BertModel(enc_config)
    decoder = BertLMHeadModel(dec_config)
    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

    # --- generation-critical special tokens (must be defined for encoder-decoder generate) ---
    # Choose a decoder start token:
    # 1) BOS if available, else
    # 2) PAD, else
    # 3) EOS (last resort)
    start_id = tgt_tok.bos_token_id
    if start_id is None:
        start_id = tgt_tok.pad_token_id
    if start_id is None:
        start_id = tgt_tok.eos_token_id

    if start_id is None:
        raise ValueError("Cannot determine decoder start token id (no BOS/PAD/EOS in target tokenizer).")

    model.config.decoder_start_token_id = start_id
    model.config.bos_token_id = start_id  # some versions require bos_token_id explicitly
    model.config.eos_token_id = tgt_tok.eos_token_id
    model.config.pad_token_id = tgt_tok.pad_token_id

    # Some transformers versions use generation_config if present
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.decoder_start_token_id = start_id
        model.generation_config.bos_token_id = start_id
        model.generation_config.eos_token_id = tgt_tok.eos_token_id
        model.generation_config.pad_token_id = tgt_tok.pad_token_id

    # generation/loss config
    #model.config.decoder_start_token_id = (
    #    tgt_tok.bos_token_id if tgt_tok.bos_token_id is not None else tgt_tok.pad_token_id
    #)
    #model.config.eos_token_id = tgt_tok.eos_token_id
    #model.config.pad_token_id = tgt_tok.pad_token_id
    model.config.vocab_size = tgt_vocab

    data_collator = TwoTokenizerSeq2SeqCollator(src_tok, tgt_tok)
    compute_metrics = compute_metrics_builder(tgt_tok) if args.eval_metrics else None

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
        report_to=[],
        label_smoothing_factor=args.label_smoothing,
    )
    callbacks = []

    if args.show_val_examples > 0:
        callbacks.append(
            ShowValidationExamplesCallback(
                model=model,
                tokenizer_src=src_tok,
                tokenizer_tgt=tgt_tok,
                val_src_lines=src_val,          # your validation SRC lines
                val_tgt_lines=tgt_val,
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
        tokenizer=src_tok,                    # Trainer tokenizer is typically the SRC side
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )


    trainer.train()
    trainer.save_model(args.save)

    if args.history_json:
        write_history(trainer, args.history_json)

    if args.show_val_examples and args.show_val_examples > 0:
        k = min(args.show_val_examples, len(src_val))
        idxs = random.sample(range(len(src_val)), k)
        device = trainer.args.device
        model.eval()
        
    print(f"\nDone. Model saved to: {args.save}")


if __name__ == "__main__":
    main()
