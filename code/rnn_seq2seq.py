#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal RNN encoder-decoder with optional bidirectional encoder, Luong attention,
and optional SentencePiece subwording (BPE / unigram), implemented in PyTorch.

----------------------------------------------------------------------
USAGE OVERVIEW
----------------------------------------------------------------------

The script supports two modes:

1) TRAINING MODE (default)
   Trains a neural MT model from parallel data.

2) TRANSLATION MODE
   Loads a trained model and performs interactive translation.

----------------------------------------------------------------------
MODES
----------------------------------------------------------------------

--mode {train,translate}
    Select training or interactive translation mode.
    Default: train

----------------------------------------------------------------------
DATA OPTIONS (training mode)
----------------------------------------------------------------------

--src-file PATH
    Source-language training file (one sentence per line).

--tgt-file PATH
    Target-language training file (one sentence per line).

--src-val PATH
--tgt-val PATH
    Validation files (optional but recommended).

----------------------------------------------------------------------
TRAINING OPTIONS (training mode only)
----------------------------------------------------------------------

--epochs
--batch-size
--lr
--teacher-forcing
--early-stopping
--early-metric {loss,bleu,chrf,ter}
--eval-metrics
--limit                 Limit number of training pairs (debugging)

----------------------------------------------------------------------
MODEL ARCHITECTURE OPTIONS (shared)
----------------------------------------------------------------------

--emb-size
--hidden-size
--enc-layers
--dec-layers
--rnn-type {rnn,gru,lstm}
--attention {none,luong}
--bidirectional
--max-len

----------------------------------------------------------------------
VOCABULARY & PREPROCESSING OPTIONS (shared)
----------------------------------------------------------------------

--max-src-vocab
--max-tgt-vocab
--lower
--seed

----------------------------------------------------------------------
TRANSLATION OPTIONS (translation mode only)
----------------------------------------------------------------------

--load PATH
    Path to trained checkpoint (required).

--replace-unk
    Copy source tokens for <unk> using attention.

----------------------------------------------------------------------
SUBWORD OPTIONS (SentencePiece)
----------------------------------------------------------------------

--subword-type {none,bpe,unigram}
--src-sp-model PATH
--tgt-sp-model PATH

SentencePiece models must be trained externally.

----------------------------------------------------------------------
EXAMPLES
----------------------------------------------------------------------

1) Word-level training with attention and bidirectional encoder:

python rnn_seq2seq_attention_subwording.py \
    --src-file train.en \
    --tgt-file train.nl \
    --src-val dev.en \
    --tgt-val dev.nl \
    --attention luong \
    --bidirectional \
    --epochs 30 \
    --save rnn_att.pt


2) Training with SentencePiece subwords (BPE):

python rnn_seq2seq_attention_subwording.py \
    --src-file train.en \
    --tgt-file train.nl \
    --src-val dev.en \
    --tgt-val dev.nl \
    --subword-type bpe \
    --src-sp-model spm.en.model \
    --tgt-sp-model spm.nl.model \
    --attention luong \
    --bidirectional \
    --epochs 30 \
    --save rnn_sp.pt


3) Interactive translation (no training data needed):

python rnn_seq2seq_attention_subwording.py \
    --mode translate \
    --load rnn_sp.best.pt


4) Interactive translation with UNK replacement:

python rnn_seq2seq_attention_subwording.py \
    --mode translate \
    --load rnn_sp.best.pt \
    --replace-unk

----------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import math
import glob
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import sacrebleu
except ImportError:  # sacrebleu is optional; metrics disabled if missing
    sacrebleu = None
try:
    import sentencepiece as spm
except ImportError:
    spm = None

# ----------------------------------------------------------------------
# 1. Special tokens
# ----------------------------------------------------------------------

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


# ----------------------------------------------------------------------
# 2. Vocab
# ----------------------------------------------------------------------


@dataclass
class Vocab:
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

    def encode(self, s: str, add_eos: bool = True) -> List[int]:
        ids = []
        for w in s.split():
            ids.append(self.word2idx.get(w, self.unk_idx))
        if add_eos:
            ids.append(self.eos_idx)
        return ids

    def decode(self, ids: List[int]) -> str:
        """Plain decode without replace_unk heuristic."""
        words = []
        for i in ids:
            if i == self.eos_idx:
                break
            if 0 <= i < len(self.idx2word):
                w = self.idx2word[i]
                if w not in [SOS_TOKEN, PAD_TOKEN]:
                    words.append(w)
        return " ".join(words)


def build_vocab(sentences: List[str], max_size: Optional[int] = None) -> Vocab:
    """
    Build a vocabulary from sentences.

    OLD INTERPRETATION (like previous script):
    - max_size = maximum number of *non-special* word types.
    - We then add 4 specials (PAD, SOS, EOS, UNK) on top.
      So with max_size=1000 you can get a vocab of size 1004.
    """
    freq: Dict[str, int] = {}
    for s in sentences:
        for w in s.split():
            freq[w] = freq.get(w, 0) + 1

    # Sort by frequency (descending), then alphabetically for determinism
    sorted_words = sorted(freq.items(), key=lambda x: (-x[1], x[0]))

    if max_size is not None:
        # Keep at most 'max_size' real word types (specials added later)
        sorted_words = sorted_words[: max_size]

    # Specials are added on top of those 'max_size' words
    idx2word = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    idx2word += [w for w, _ in sorted_words]

    word2idx = {w: i for i, w in enumerate(idx2word)}
    return Vocab(word2idx=word2idx, idx2word=idx2word)


def render_source_with_unk(sentence: str, vocab: Vocab) -> str:
    """Return source string as seen by model (tokenised and with <unk>)."""
    tokens = sentence.split()
    ids = vocab.encode(sentence, add_eos=False)
    out_tokens = []
    for w, idx in zip(tokens, ids):
        if idx == vocab.unk_idx:
            out_tokens.append("<unk>")
        else:
            out_tokens.append(w)
    return " ".join(out_tokens)


# ----------------------------------------------------------------------
# 3. Reading parallel data
# ----------------------------------------------------------------------


def read_parallel(
    src_path: str,
    tgt_path: str,
    lower: bool = False,
    limit: Optional[int] = None,
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(src_path, encoding="utf-8") as fs, open(
        tgt_path, encoding="utf-8"
    ) as ft:
        for s, t in zip(fs, ft):
            s = s.strip()
            t = t.strip()
            if lower:
                s = s.lower()
                t = t.lower()
            if not s or not t:
                continue
            pairs.append((s, t))
            if limit is not None and len(pairs) >= limit:
                break
    return pairs


# ----------------------------------------------------------------------
# 4. Dataset / batching
# ----------------------------------------------------------------------


class ParallelDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        max_len: Optional[int] = None,
    ):
        self.data: List[Tuple[List[int], List[int]]] = []
        for src, tgt in pairs:
            src_ids = src_vocab.encode(src, add_eos=True)
            tgt_ids = tgt_vocab.encode(tgt, add_eos=True)
            if max_len is not None:
                if len(src_ids) > max_len or len(tgt_ids) > max_len:
                    continue
            self.data.append((src_ids, tgt_ids))

        if not self.data:
            raise ValueError("No sentence pairs left after max_len filtering!")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.data[idx]


def pad_sequence(seq: List[int], max_len: int, pad_idx: int) -> List[int]:
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_idx] * (max_len - len(seq))


def collate_fn(
    batch: List[Tuple[List[int], List[int]]],
    src_pad_idx: int,
    tgt_pad_idx: int,
):
    src_seqs, tgt_seqs = zip(*batch)
    max_src = max(len(s) for s in src_seqs)
    max_tgt = max(len(t) for t in tgt_seqs)

    src_tensor = torch.tensor(
        [pad_sequence(s, max_src, src_pad_idx) for s in src_seqs],
        dtype=torch.long,
    )
    tgt_tensor = torch.tensor(
        [pad_sequence(t, max_tgt, tgt_pad_idx) for t in tgt_seqs],
        dtype=torch.long,
    )

    return src_tensor, tgt_tensor


# ----------------------------------------------------------------------
# 5. Encoder / Decoder / Seq2Seq (+ optional Luong attention + bidi encoder)
# ----------------------------------------------------------------------


class EncoderRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_size: int,
        hidden_size: int,
        pad_idx: int,
        rnn_type: str = "rnn",
        num_layers: int = 1,
        bidirectional: bool = False,
    ):
        """
        hidden_size: encoder hidden size *per direction*.
        If bidirectional=True, the encoder outputs and final hidden state
        have dimensionality hidden_size * 2. The decoder and attention
        will then use that combined size.
        """
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.is_lstm = self.rnn_type == "lstm"
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Per-direction size
        self.enc_hidden_size = hidden_size
        # Size seen by decoder / attention
        self.output_hidden_size = hidden_size * (2 if bidirectional else 1)

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)

        if self.rnn_type == "gru":
            rnn_cls = nn.GRU
            rnn_kwargs = {}
        elif self.rnn_type == "lstm":
            rnn_cls = nn.LSTM
            rnn_kwargs = {}
        elif self.rnn_type == "rnn":
            rnn_cls = nn.RNN
            rnn_kwargs = {"nonlinearity": "tanh"}
        else:
            raise ValueError(f"Unknown rnn_type: {self.rnn_type}")

        self.rnn = rnn_cls(
            emb_size,
            self.enc_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            **rnn_kwargs,
        )

        # No projection layer needed; decoder will work directly
        # in 'output_hidden_size' space.
        self.fw_bw_combine = None

    def forward(self, src: torch.Tensor):
        # src: [batch, src_len]
        embedded = self.embedding(src)  # [batch, src_len, emb]
        outputs, hidden = self.rnn(embedded)  # outputs: [batch, src_len, enc_hidden*dir]

        if self.bidirectional:
            # outputs is already [batch, src_len, 2 * enc_hidden_size]
            batch_size, src_len, _ = outputs.size()
            outputs = outputs.view(batch_size, src_len, self.output_hidden_size)

            if self.is_lstm:
                # hidden: (h, c) with shapes [num_layers*2, batch, enc_hidden]
                h, c = hidden
                h = h.view(self.num_layers, 2, batch_size, self.enc_hidden_size)
                c = c.view(self.num_layers, 2, batch_size, self.enc_hidden_size)
                # Concatenate forward & backward along hidden dimension
                h = torch.cat([h[:, 0], h[:, 1]], dim=-1)  # [num_layers, batch, 2*enc_hidden]
                c = torch.cat([c[:, 0], c[:, 1]], dim=-1)
                hidden = (h, c)
            else:
                # hidden: [num_layers*2, batch, enc_hidden]
                h = hidden.view(self.num_layers, 2, batch_size, self.enc_hidden_size)
                h = torch.cat([h[:, 0], h[:, 1]], dim=-1)  # [num_layers, batch, 2*enc_hidden]
                hidden = h

        # For unidirectional: outputs: [batch, src_len, enc_hidden_size],
        # hidden: [num_layers, batch, enc_hidden_size] (or (h,c))
        return outputs, hidden


class LuongAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        decoder_state: torch.Tensor,     # [batch, hidden]
        encoder_outputs: torch.Tensor,   # [batch, src_len, hidden]
        src_mask: Optional[torch.Tensor] = None,  # [batch, src_len] (bool)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Transform decoder state
        dec_proj = self.linear(decoder_state).unsqueeze(2)  # [batch, hidden, 1]
        # scores: [batch, src_len]
        scores = torch.bmm(encoder_outputs, dec_proj).squeeze(2)

        if src_mask is not None:
            scores = scores.masked_fill(~src_mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)  # [batch, src_len]
        # context: [batch, hidden]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

class DecoderRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_size: int,
        hidden_size: int,
        pad_idx: int,
        rnn_type: str = "rnn",
        num_layers: int = 1,
        attention: str = "none",
    ):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.is_lstm = self.rnn_type == "lstm"
        self.attention_type = attention.lower()

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)

        if self.rnn_type == "gru":
            rnn_cls = nn.GRU
            rnn_kwargs = {}
        elif self.rnn_type == "lstm":
            rnn_cls = nn.LSTM
            rnn_kwargs = {}
        elif self.rnn_type == "rnn":
            rnn_cls = nn.RNN
            rnn_kwargs = {"nonlinearity": "tanh"}
        else:
            raise ValueError(f"Unknown rnn_type: {self.rnn_type}")

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
        hidden: torch.Tensor,
        encoder_outputs: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        """
        input_step: [batch]
        hidden: RNN hidden state or (h,c) for LSTM
        encoder_outputs: [batch, src_len, hidden] (for attention)
        src_mask: [batch, src_len] (1 for tokens, 0 for pads)
        """
        embedded = self.embedding(input_step.unsqueeze(1))  # [batch, 1, emb]
        output, hidden = self.rnn(embedded, hidden)  # output: [batch, 1, hidden]
        dec_state = output.squeeze(1)  # [batch, hidden]

        attn_weights = None
        if self.attn is not None and encoder_outputs is not None:
            context, attn_weights = self.attn(dec_state, encoder_outputs, src_mask)
            combined = torch.cat([dec_state, context], dim=-1)
            dec_state = torch.tanh(self.attn_combine(combined))

        logits = self.out(dec_state)  # [batch, vocab]

        if return_attn:
            return logits, hidden, attn_weights
        else:
            return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder: EncoderRNN,
        decoder: DecoderRNN,
        tgt_sos_idx: int,
        tgt_eos_idx: int,
        max_len: int = 20,
    ):
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
        tgt: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Training/validation forward.
        If tgt is provided: returns logits [batch, tgt_len, vocab].
        """
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

        if tgt is not None:
            tgt_len = tgt.size(1)
            for t in range(tgt_len):
                logits, dec_hidden = self.decoder(
                    dec_input,
                    dec_hidden,
                    encoder_outputs=encoder_outputs,
                    src_mask=src_mask,
                    return_attn=False,
                )
                outputs.append(logits.unsqueeze(1))
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = logits.argmax(-1)
                dec_input = tgt[:, t] if teacher_force else top1
            return torch.cat(outputs, dim=1)
        else:
            raise ValueError("Forward without tgt is reserved for greedy_decode().")

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        max_len: Optional[int] = None,
    ) -> Tuple[List[int], Optional[torch.Tensor]]:
        """
        Greedy decoding for a single example (batch size 1).
        Returns:
            hyp_ids: list of token ids (without initial <sos>)
            attn_matrix: [tgt_len, src_len] or None
        """
        self.eval()
        if src.size(0) != 1:
            raise ValueError("greedy_decode currently supports batch size 1 only.")

        max_len = max_len or self.max_len
        encoder_outputs, enc_hidden = self.encoder(src)
        src_mask = self.make_src_mask(src)

        dec_input = torch.full(
            (1,),
            self.tgt_sos_idx,
            dtype=torch.long,
            device=src.device,
        )
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
            top1 = logits.argmax(-1)  # [1]
            token_id = top1.item()
            if token_id == self.tgt_eos_idx:
                break
            hyp_ids.append(token_id)
            if attn_weights is not None:
                attn_list.append(attn_weights.squeeze(0).cpu())

            dec_input = top1

        attn_matrix = None
        if attn_list:
            attn_matrix = torch.stack(attn_list, dim=0)  # [tgt_len, src_len]
        else:
            attn_matrix = None
        return hyp_ids, attn_matrix


# ----------------------------------------------------------------------
# 6. Training utilities
# ----------------------------------------------------------------------


def compute_loss(
    model: Seq2Seq,
    batch,
    criterion: nn.Module,
    device: torch.device,
    teacher_forcing: float,
) -> torch.Tensor:
    src, tgt = batch
    src = src.to(device)
    tgt = tgt.to(device)
    inputs = tgt[:, :-1]
    gold = tgt[:, 1:].contiguous()
    logits = model(src, inputs, teacher_forcing_ratio=teacher_forcing)
    vocab_size = logits.size(-1)
    loss = criterion(
        logits.view(-1, vocab_size),
        gold.view(-1),
    )
    return loss


def evaluate_nll(
    model: Seq2Seq,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for src, tgt in data_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            inputs = tgt[:, :-1]
            gold = tgt[:, 1:].contiguous()
            logits = model(src, inputs, teacher_forcing_ratio=0.0)
            vocab_size = logits.size(-1)
            loss = criterion(
                logits.view(-1, vocab_size),
                gold.view(-1),
            )
            num_tokens = (gold != 0).sum().item()  # assume pad_idx==0
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    return total_loss / max(total_tokens, 1)


def translate_dataset(
    model: Seq2Seq,
    pairs: List[Tuple[str, str]],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device,
    max_len: int,
    replace_unk: bool = False,
    subword_type: str = "none",
    src_sp: Optional["spm.SentencePieceProcessor"] = None,
    tgt_sp: Optional["spm.SentencePieceProcessor"] = None,
) -> List[str]:
    """
    Translate just the sources from pairs and return list of hypotheses.
    """
    hyps: List[str] = []
    for src_sentence, _ in tqdm(pairs, desc="Translating dev set", leave=False):
        # Model-side source string (word or SP subwords)
        if src_sp is not None and subword_type != "none":
            model_src = " ".join(src_sp.encode(src_sentence, out_type=str))
        else:
            model_src = src_sentence

        src_ids = src_vocab.encode(model_src, add_eos=True)
        src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
        hyp_ids, attn_matrix = model.greedy_decode(src_tensor, max_len=max_len)

        if replace_unk:
            subword_hyp = replace_unk_with_attention(
                hyp_ids, attn_matrix, model_src, tgt_vocab
            )
        else:
            subword_hyp = tgt_vocab.decode(hyp_ids)

        if tgt_sp is not None and subword_type != "none":
            hyp = tgt_sp.decode(subword_hyp.split())
        else:
            hyp = subword_hyp

        hyps.append(hyp)
    return hyps

def show_val_examples(
    model: Seq2Seq,
    val_pairs: List[Tuple[str, str]],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device,
    max_len: int,
    num_examples: int = 5,
    replace_unk: bool = False,
    subword_type: str = "none",
    src_sp: Optional["spm.SentencePieceProcessor"] = None,
    tgt_sp: Optional["spm.SentencePieceProcessor"] = None,
):
    """
    Print a few example translations from the validation set.
    """
    if not val_pairs:
        return

    model.eval()
    # take first num_examples sentences (or fewer if dataset is small)
    subset = val_pairs[:num_examples]
    hyps = translate_dataset(
        model,
        subset,
        src_vocab,
        tgt_vocab,
        device,
        max_len,
        replace_unk=replace_unk,
        subword_type=subword_type,
        src_sp=src_sp,
        tgt_sp=tgt_sp,
    )
    print("\n--- Example validation translations ---")
    for i, ((src, ref), hyp) in enumerate(zip(subset, hyps), start=1):
        print(f"[{i}] SRC: {src}")
        print(f"    REF: {ref}")
        print(f"    HYP: {hyp}")
        print()
    print("---------------------------------------")

def compute_sacrebleu_metrics(hyps: List[str], refs: List[str]) -> Dict[str, float]:
    """
    Compute BLEU / ChrF / TER using the old script's detokenisation behaviour.
    Matching the calling convention of the new script:
        compute_sacrebleu_metrics(hyps, refs)
    where hyps and refs are *lists of strings*.
    """
    import re
    try:
        import sacrebleu
    except ImportError:
        print("sacrebleu not installed; cannot compute metrics.")
        return {}

    # Old script's simple detokeniser
    def detok(s: str) -> str:
        return re.sub(r"\s+([.,!?;:])", r"\1", s)

    hyps_detok = [detok(h) for h in hyps]
    refs_detok = [detok(r) for r in refs]

    # sacrebleu expects refs as List[List[str]]
    bleu = sacrebleu.corpus_bleu(hyps_detok, [refs_detok], force=True).score
    chrf = sacrebleu.corpus_chrf(hyps_detok, [refs_detok]).score

    try:
        ter = sacrebleu.corpus_ter(hyps_detok, [refs_detok]).score
    except Exception:
        ter = float("nan")

    return {"bleu": bleu, "chrf": chrf, "ter": ter}


# ----------------------------------------------------------------------
# 7. Checkpointing
# ----------------------------------------------------------------------


def save_checkpoint(
    path: str,
    model: Seq2Seq,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    model_args: Dict,
    best_metric: Optional[float] = None,
):
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "model_args": model_args,
        "best_metric": best_metric,
    }
    torch.save(ckpt, path)


def infer_epoch_from_filename(path: str) -> Optional[int]:
    m = re.search(r"epoch(\d+)", os.path.basename(path))
    if m:
        return int(m.group(1))
    return None


def load_checkpoint(path: str, device: torch.device):
    from torch.serialization import safe_globals
    with safe_globals([Vocab]):
        ckpt = torch.load(path, map_location=device)
    src_vocab: Vocab = ckpt["src_vocab"]
    tgt_vocab: Vocab = ckpt["tgt_vocab"]
    model_args = ckpt["model_args"]

    bidirectional = model_args.get("bidirectional", False)
    enc_hidden_size = model_args["hidden_size"]              # per-direction size
    dec_hidden_size = enc_hidden_size * (2 if bidirectional else 1)

    encoder = EncoderRNN(
        len(src_vocab.idx2word),
        model_args["emb_size"],
        enc_hidden_size,
        src_vocab.pad_idx,
        rnn_type=model_args.get("rnn_type", "rnn"),
        num_layers=model_args.get("enc_layers", 1),
        bidirectional=bidirectional,
    )
    decoder = DecoderRNN(
        len(tgt_vocab.idx2word),
        model_args["emb_size"],
        dec_hidden_size,
        tgt_vocab.pad_idx,
        rnn_type=model_args.get("rnn_type", "rnn"),
        num_layers=model_args.get("dec_layers", 1),
        attention=model_args.get("attention", "none"),
    )
    model = Seq2Seq(
        encoder,
        decoder,
        tgt_sos_idx=tgt_vocab.sos_idx,
        tgt_eos_idx=tgt_vocab.eos_idx,
        max_len=model_args.get("max_len", 50),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(ckpt["optimizer_state"])
    epoch = ckpt.get("epoch", 0)

    return model, optimizer, src_vocab, tgt_vocab, model_args, epoch

def cleanup_checkpoints(save_prefix, keep_last):
    prefix = save_prefix[:-3] if save_prefix.endswith(".pt") else save_prefix
    checkpoints = sorted(glob.glob(f"{prefix}.epoch*.pt"))

    if keep_last <= 0:
        for p in checkpoints:
            os.remove(p)
        return

    if len(checkpoints) <= keep_last:
        return

    for ckpt in checkpoints[:-keep_last]:
        os.remove(ckpt)

# ----------------------------------------------------------------------
# 8. replace-unk with attention
# ----------------------------------------------------------------------


def replace_unk_with_attention(
    hyp_ids: List[int],
    attn_matrix: Optional[torch.Tensor],
    src_sentence: str,
    tgt_vocab: Vocab,
) -> str:
    """
    Replace <unk> tokens by copying from the source.

    - If attn_matrix is provided: copy from the source position with
      highest attention weight at that timestep.
    - If attn_matrix is None (no attention model): fallback heuristic
      copies the source token at the same timestep index, if it exists.
    """
    src_tokens = src_sentence.split()
    out_words: List[str] = []

    tgt_eos = tgt_vocab.eos_idx
    tgt_unk = tgt_vocab.unk_idx

    for t, idx in enumerate(hyp_ids):
        if idx == tgt_eos:
            break

        if idx == tgt_unk:
            # Decide which source position to copy from
            if attn_matrix is not None and t < attn_matrix.size(0):
                # Attention-based: pick max-attended source position
                src_pos = int(attn_matrix[t].argmax().item())
            else:
                # No attention: simple same-position heuristic
                src_pos = t

            if 0 <= src_pos < len(src_tokens):
                out_words.append(src_tokens[src_pos])
                continue  # skip normal decoding for this token

        # Normal vocab-based decoding for non-UNK tokens
        if 0 <= idx < len(tgt_vocab.idx2word):
            w = tgt_vocab.idx2word[idx]
            if w not in [SOS_TOKEN, PAD_TOKEN]:
                out_words.append(w)

    return " ".join(out_words)


# ----------------------------------------------------------------------
# 9. Utils: parameter counting
# ----------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------
# 10. CLI + training / translation entry points
# ----------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "RNN encoder-decoder with optional bidirectional encoder and "
            "Luong attention (options aligned with rnn_seq2seq.py)"
        )
    )

    p.add_argument("--src-file", type=str, default=None,
               help="Training source file (one sentence per line). Required in --mode train.")
    p.add_argument("--tgt-file", type=str, default=None,
               help="Training target file (one sentence per line). Required in --mode train.")

    p.add_argument("--src-val", type=str, help="Validation source file")
    p.add_argument("--tgt-val", type=str, help="Validation target file")

    p.add_argument(
        "--mode",
        choices=["train", "translate"],
        default="train",
        help="Train a model or translate using a checkpoint",
    )

    # Core hyperparameters (aligned with rnn_seq2seq.py)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--emb-size", type=int, default=64)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument(
        "--max-len",
        type=int,
        default=20,
        help="Max sentence length (tokens incl. <eos>)",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--teacher-forcing", type=float, default=0.7)

    p.add_argument(
        "--enc-layers",
        type=int,
        default=1,
        help="Number of layers in the encoder RNN/GRU/LSTM",
    )
    p.add_argument(
        "--dec-layers",
        type=int,
        default=1,
        help="Number of layers in the decoder RNN/GRU/LSTM",
    )
    p.add_argument(
        "--rnn-type",
        type=str,
        choices=["rnn", "gru", "lstm"],
        default="rnn",
        help="Type of recurrent cell to use",
    )

    # Attention + bidirectional encoder (extra compared to base script)
    p.add_argument(
        "--attention",
        type=str,
        choices=["none", "luong"],
        default="none",
        help="Type of cross-lingual attention to use",
    )
    p.add_argument(
        "--bidirectional",
        action="store_true",
        help="Use a bidirectional encoder",
    )

    # Data options, vocab cutoffs
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of training sentence pairs (for debugging)",
    )
    p.add_argument("--max-src-vocab", type=int, default=None)
    p.add_argument("--max-tgt-vocab", type=int, default=None)

    p.add_argument(
        "--lower",
        action="store_true",
        help="Lowercase all training/validation data",
    )
    # Subword options (SentencePiece)
    p.add_argument(
        "--subword-type",
        type=str,
        choices=["none", "bpe", "unigram"],
        default="none",
        help=(
            "Use SentencePiece subwords instead of word tokens. "
            "'none' = word-level; 'bpe' = BPE; 'unigram' = unigram LM. "
            "This script expects you to train the SentencePiece models externally."
        ),
    )
    p.add_argument(
        "--src-sp-model",
        type=str,
        help="Path to SentencePiece model for the source language (required if subword-type != none).",
    )
    p.add_argument(
        "--tgt-sp-model",
        type=str,
        help="Path to SentencePiece model for the target language (required if subword-type != none).",
    )

    # Checkpointing
    p.add_argument(
        "--save",
        type=str,
        default="model_att.pt",
        help="Base path for checkpoints (will add .epochNNN.pt)",
    )
    p.add_argument(
        "--save-best",
        type=str,
        default=None,
        help="Path for best checkpoint (default: <save>.best.pt)",
    )
    p.add_argument(
        "--load",
        type=str,
        default=None,
        help=(
            "Path to checkpoint to load. "
            "In train mode: resume training; in translate mode: required."
        ),
    )
    p.add_argument(
        "--keep-last",
        type=int,
        default=1,
        help="Keep only the last N epoch checkpoints (default: 1). "
         "The best checkpoint is always kept separately."
    )


    # Early stopping & evaluation metrics (aligned naming)
    p.add_argument(
        "--early-stopping",
        type=int,
        default=0,
        help="Stop if no improvement after N epochs (0 disables early stopping)",
    )
    p.add_argument(
        "--early-metric",
        type=str,
        choices=["loss", "bleu", "chrf", "ter"],
        default="loss",
        help="Metric to monitor for early stopping / best checkpoint",
    )
    p.add_argument(
        "--eval-metrics",
        action="store_true",
        help="Compute sacreBLEU metrics on validation data each epoch",
    )

    # Translate-time options
    p.add_argument(
        "--replace-unk",
        action="store_true",
        help="At translation time, use attention to copy source tokens for <unk>",
    )
    p.add_argument(
        "--show-src-unk",
        action="store_true",
        help="In translate mode, also print the source as seen by the model with <unk> tokens.",
    )
    # Batch translation (translate mode)
    p.add_argument("--src-test", type=str, default=None,
                   help="Source test file to translate (one sentence per line). If set, translate mode runs in batch mode.")
    p.add_argument("--tgt-test", type=str, default=None,
                   help="Optional reference translations for --src-test (one sentence per line). Used for metrics.")
    p.add_argument("--output", type=str, default=None,
                   help="Output file path for batch translation hypotheses (one sentence per line). Default: <src-test>.hyp")

    # Validation translations saving
    p.add_argument(
        "--save-val-json",
        type=str,
        default=None,
        help=(
            "If set, save validation translations as JSON each epoch. "
            "Treat this as a base path; '.epochXXX.json' will be appended."
        ),
    )
    p.add_argument(
        "--save-val-trans",
        type=str,
        default=None,
        help=(
            "If set, save validation translations as TSV each epoch. "
            "Treat this as a base path; '.epochXXX.tsv' will be appended."
        ),
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (<=0 to disable seeding)",
    )
    p.add_argument("--show-val-examples", type=int, default=0, help="Show this many example validations after each epoch")
    p.add_argument("--history-json", type=str, default=None, help="If set, write per-epoch training/validation metrics to this JSON file (overwritten each epoch).",
)

    return p.parse_args()



def set_seed(seed: int):
    """Set random seeds for Python and PyTorch (same behaviour as rnn_seq2seq.py)."""
    if seed <= 0:
        return
    print(f"Setting random seed to {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------
# 11. Train / translate
# ----------------------------------------------------------------------

def apply_sentencepiece_to_pairs(
    pairs: Optional[List[Tuple[str, str]]],
    src_sp: Optional["spm.SentencePieceProcessor"],
    tgt_sp: Optional["spm.SentencePieceProcessor"],
) -> Optional[List[Tuple[str, str]]]:
    """
    Given raw (src, tgt) text pairs, return pairs where each side is
    tokenised into SentencePiece subword *strings* (space-separated).
    If src_sp/tgt_sp is None or pairs is None, returns pairs unchanged.
    """
    if pairs is None or src_sp is None or tgt_sp is None:
        return pairs
    new_pairs: List[Tuple[str, str]] = []
    for s, t in pairs:
        s_pieces = src_sp.encode(s, out_type=str)
        t_pieces = tgt_sp.encode(t, out_type=str)
        new_pairs.append((" ".join(s_pieces), " ".join(t_pieces)))
    return new_pairs

def train_main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # 0) SentencePiece setup (optional)
    subword_type = args.subword_type
    src_sp = None
    tgt_sp = None
    if subword_type != "none":
        if spm is None:
            raise ImportError(
                "sentencepiece is not installed, but --subword-type was set. "
                "Install it with 'pip install sentencepiece'."
            )
        if not args.src_sp_model or not args.tgt_sp_model:
            raise ValueError(
                "In subword mode, you must provide --src-sp-model and --tgt-sp-model."
            )
        print(
            f"Using SentencePiece subwords ({subword_type}) from "
            f"{args.src_sp_model} / {args.tgt_sp_model}"
        )
        src_sp = spm.SentencePieceProcessor()
        src_sp.load(args.src_sp_model)
        tgt_sp = spm.SentencePieceProcessor()
        tgt_sp.load(args.tgt_sp_model)

    # 1) Read train data
    train_pairs = read_parallel(
        args.src_file,
        args.tgt_file,
        lower=args.lower,
        limit=args.limit,
    )
    raw_train_count = len(train_pairs)

    # 2) Read validation data (optional)
    val_pairs: Optional[List[Tuple[str, str]]] = None
    if args.src_val and args.tgt_val:
        val_pairs = read_parallel(
            args.src_val,
            args.tgt_val,
            lower=args.lower,
            limit=None,  # do not limit; we also won't filter by max_len in dataset
        )
        print(f"Loaded {len(val_pairs)} validation sentence pairs.")

    # Create SentencePiece-tokenised versions for training/vocab, if enabled.
    train_pairs_tok = apply_sentencepiece_to_pairs(train_pairs, src_sp, tgt_sp)
    val_pairs_tok = apply_sentencepiece_to_pairs(val_pairs, src_sp, tgt_sp)
        
    # 3) Build vocab + model
    if args.load:
        # Resume from checkpoint
        model, optimizer, src_vocab, tgt_vocab, model_args, ckpt_epoch = load_checkpoint(
            args.load, device
        )
        print(f"Loaded checkpoint from {args.load} (stored epoch {ckpt_epoch}).")

        ckpt_lower = bool(model_args.get("lower", False))
        if args.lower != ckpt_lower:
            print(f"[warn] Overriding --lower={args.lower} to match checkpoint lower={ckpt_lower}")
            args.lower = ckpt_lower

        # IMPORTANT: re-read train/val with the (possibly) corrected lower flag
        train_pairs = read_parallel(args.src_file, args.tgt_file, lower=args.lower, limit=args.limit)
        val_pairs = None
        if args.src_val and args.tgt_val:
            val_pairs = read_parallel(args.src_val, args.tgt_val, lower=args.lower, limit=None)

        # and re-tokenise after SentencePiece has been loaded:
        train_pairs_tok = apply_sentencepiece_to_pairs(train_pairs, src_sp, tgt_sp)
        val_pairs_tok   = apply_sentencepiece_to_pairs(val_pairs,   src_sp, tgt_sp)

        # Override subword config from checkpoint when resuming
        subword_type = model_args.get("subword_type", "none")
        src_sp = None
        tgt_sp = None
        if subword_type != "none":
            if spm is None:
                raise ImportError("sentencepiece is required to resume a subword-trained checkpoint.")
        src_sp = spm.SentencePieceProcessor()
        src_sp.load(model_args["src_sp_model"])
        tgt_sp = spm.SentencePieceProcessor()
        tgt_sp.load(model_args["tgt_sp_model"])
        print(f"Resuming with SentencePiece subwords ({subword_type}) from "
              f"{model_args['src_sp_model']} / {model_args['tgt_sp_model']}")
        # IMPORTANT: we may have created *word-level* pairs before we knew this was a subword checkpoint.
        # Re-tokenise train/val pairs in the same token space as the checkpoint vocab.
        train_pairs_tok = apply_sentencepiece_to_pairs(train_pairs, src_sp, tgt_sp)
        val_pairs_tok   = apply_sentencepiece_to_pairs(val_pairs,   src_sp, tgt_sp)

        # Determine starting epoch from filename if possible
        start_epoch = 1
        inferred = infer_epoch_from_filename(args.load)
        if inferred is not None:
            start_epoch = inferred + 1
            print(
                f"Inferred last completed epoch = {inferred} from checkpoint name; "
                f"resuming training at epoch {start_epoch}."
            )
        else:
            start_epoch = ckpt_epoch + 1
            print(
                f"Could not infer epoch from filename; "
                f"using stored epoch={ckpt_epoch} -> starting at {start_epoch}."
            )

    else:
        # Fresh training: build vocabs from SP-tokenised or word-level train data
        effective_train_pairs = train_pairs_tok if train_pairs_tok is not None else train_pairs
        all_src = [s for s, _ in effective_train_pairs]
        all_tgt = [t for _, t in effective_train_pairs]

        src_vocab = build_vocab(all_src, max_size=args.max_src_vocab)
        tgt_vocab = build_vocab(all_tgt, max_size=args.max_tgt_vocab)

        print(f"Source vocab size: {len(src_vocab.idx2word)}")
        print(f"Target vocab size: {len(tgt_vocab.idx2word)}")

        encoder = EncoderRNN(
            vocab_size=len(src_vocab.idx2word),
            emb_size=args.emb_size,
            hidden_size=args.hidden_size,  # per-direction encoder size
            pad_idx=src_vocab.pad_idx,
            rnn_type=args.rnn_type,
            num_layers=args.enc_layers,
            bidirectional=args.bidirectional,
        )
        decoder_hidden_size = encoder.output_hidden_size  #  = hidden_size*(2 if bidi else 1)

        decoder = DecoderRNN(
            vocab_size=len(tgt_vocab.idx2word),
            emb_size=args.emb_size,
            hidden_size=decoder_hidden_size,
            pad_idx=tgt_vocab.pad_idx,
            rnn_type=args.rnn_type,
            num_layers=args.dec_layers,
            attention=args.attention,
        )
        model = Seq2Seq(
            encoder,
            decoder,
            tgt_sos_idx=tgt_vocab.sos_idx,
            tgt_eos_idx=tgt_vocab.eos_idx,
            max_len=args.max_len,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
        print(f"[info] lower = {args.lower}")

        start_epoch = 1

    # 4) Datasets and loaders
    effective_train_pairs = train_pairs_tok if train_pairs_tok is not None else train_pairs
    train_dataset = ParallelDataset(
        effective_train_pairs,
        src_vocab,
        tgt_vocab,
        max_len=args.max_len,
    )

    used_train_count = len(train_dataset)

    print(f"Loaded {raw_train_count} training pairs. "
          f"{used_train_count} used after max_len={args.max_len} filtering. "
         ) 
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, src_vocab.pad_idx, tgt_vocab.pad_idx),
    )

    if val_pairs is not None:
        effective_val_pairs = val_pairs_tok if val_pairs_tok is not None else val_pairs
        val_dataset = ParallelDataset(
            effective_val_pairs,
            src_vocab,
            tgt_vocab,
            max_len=None,  # IMPORTANT: do not filter by max_len here
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, src_vocab.pad_idx, tgt_vocab.pad_idx),
        )
    else:
        val_loader = None

    # 5) Loss
    pad_idx = tgt_vocab.pad_idx
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # 6) Parameter count
    num_params = count_parameters(model)
    print(f"Total trainable parameters: {num_params:,}")

    # 7) Early stopping / best checkpoint tracking
    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    patience_counter = 0

    if args.save_best is None:
        base, ext = os.path.splitext(args.save)
        args.save_best = f"{base}.best{ext or '.pt'}"

    def is_better(curr, best, metric_name: str):
        if best is None:
            return True
        if metric_name in ("loss", "ter"):
            return curr < best
        if metric_name in ("bleu", "chrf"):
            return curr > best
        return False

    # 8) Training loop
    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        # tqdm progress bar over training batches
        for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            optimizer.zero_grad()
            loss = compute_loss(
                model,
                (src, tgt),
                criterion,
                device,
                teacher_forcing=args.teacher_forcing,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                tgt_gold = tgt[:, 1:].contiguous()
                num_tokens = (tgt_gold != pad_idx).sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        train_nll = total_loss / max(total_tokens, 1)
        train_ppl = math.exp(train_nll) if train_nll < 20 else float("inf")

        log_msg = f"Epoch {epoch:02d}: train NLL={train_nll:.4f} (ppl={train_ppl:.2f})"

        # Validation
        val_nll = None
        metric_value = None

        if val_loader is not None:
            val_nll = evaluate_nll(model, val_loader, criterion, device)
            val_ppl = math.exp(val_nll) if val_nll < 20 else float("inf")
            log_msg += f"  val NLL={val_nll:.4f} (ppl={val_ppl:.2f})"

            metrics = {}
            bleu = chrf = ter = None

            # Do we need actual translations? (for metrics and/or saving)
            need_translations = (
                (args.eval_metrics and sacrebleu is not None)
                or args.save_val_json
                or args.save_val_trans
            )

            refs = None
            hyps = None

            if need_translations:
                refs = [t for _, t in val_pairs]
                hyps = translate_dataset(
                    model,
                    val_pairs,
                    src_vocab,
                    tgt_vocab,
                    device,
                    max_len=args.max_len,
                    replace_unk=args.replace_unk,
                    subword_type=subword_type,
                    src_sp=src_sp,
                    tgt_sp=tgt_sp,
                )
                
            # Metrics (if requested and sacrebleu available)
            if args.eval_metrics and sacrebleu is not None and hyps is not None and refs is not None:
                metrics = compute_sacrebleu_metrics(hyps, refs)
                bleu = metrics.get("bleu")
                chrf = metrics.get("chrf")
                ter = metrics.get("ter")
                log_msg += f"  BLEU={bleu:.2f}  ChrF={chrf:.2f}  TER={ter:.2f}"

            # Save validation translations if requested
            if hyps is not None and refs is not None:
                if args.save_val_json:
                    base = args.save_val_json
                    json_path = re.sub(
                        r"\.json$",
                        f".epoch{epoch:03d}.json",
                        base,
                    )
                    with open(json_path, "w", encoding="utf-8") as f_json:
                        json.dump(
                            [
                                {"src": s, "ref": r, "hyp": h}
                                for (s, r), h in zip(val_pairs, hyps)
                            ],
                            f_json,
                            ensure_ascii=False,
                            indent=2,
                        )
                    print(f"  [info] Saved validation translations to {json_path}")

                if args.save_val_trans:
                    base = args.save_val_trans
                    tsv_path = re.sub(
                        r"\.tsv$",
                        f".epoch{epoch:03d}.tsv",
                        base,
                    )
                    with open(tsv_path, "w", encoding="utf-8") as f_tsv:
                        for (s, r), h in zip(val_pairs, hyps):
                            f_tsv.write(f"{s}\t{r}\t{h}\n")
                    print(f"  [info] Saved validation translations to {tsv_path}")

            # Early-stopping metric selection
            if args.early_metric == "loss" and val_nll is not None:
                metric_value = val_nll
            elif args.early_metric == "bleu" and bleu is not None:
                metric_value = bleu
            elif args.early_metric == "chrf" and chrf is not None:
                metric_value = chrf
            elif args.early_metric == "ter" and ter is not None:
                metric_value = ter
        print(log_msg)
        # ---- Save best checkpoint whenever we have a monitored metric ----
        if metric_value is not None:
            if is_better(metric_value, best_metric, args.early_metric):
                best_metric = metric_value
                best_epoch = epoch
                save_checkpoint(
                    args.save_best,
                    model,
                    optimizer,
                    epoch,
                    src_vocab,
                    tgt_vocab,
                    model_args,
                    best_metric,
                )

        # ---- History tracking (optional) ----
        epoch_record = {
            "epoch": epoch,
            "train_nll": float(train_nll),
            "train_ppl": float(train_ppl) if train_ppl != float("inf") else None,
        }

        if val_nll is not None:
            epoch_record["val_nll"] = float(val_nll)
            epoch_record["val_ppl"] = float(math.exp(val_nll)) if val_nll < 20 else None

        # If metrics were computed this epoch, store them too
        if args.eval_metrics and sacrebleu is not None:
            if bleu is not None:
                epoch_record["bleu"] = float(bleu)
            if chrf is not None:
                epoch_record["chrf"] = float(chrf)
            if ter is not None:
                epoch_record["ter"] = float(ter)

        # Track early-stopping selection (optional, but useful for plotting)
        if best_metric is not None:
            epoch_record["best_metric"] = float(best_metric)
        if best_epoch is not None:
            epoch_record["best_epoch"] = int(best_epoch)

        history.append(epoch_record)

        if args.history_json:
            with open(args.history_json, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

        # NEW: show a few validation translations every epoch
        if args.show_val_examples > 0 and val_pairs: 
            show_val_examples(
                model,
                val_pairs,
                src_vocab,
                tgt_vocab,
                device,
                max_len=args.max_len,
                num_examples=args.show_val_examples,          # tweak if you like
                replace_unk=args.replace_unk,       # or True if you want to see the copying effect
                subword_type = subword_type,
                src_sp = src_sp,
                tgt_sp = tgt_sp
            )
        # Save last checkpoint with epoch in name
        if args.save.endswith(".pt"):
            save_path = re.sub(r"\.pt$", f".epoch{epoch:03d}.pt", args.save)
        else:
            save_path = f"{args.save}.epoch{epoch:03d}.pt"

        save_checkpoint(
            save_path,
            model,
            optimizer,
            epoch,
            src_vocab,
            tgt_vocab,
            model_args,
            best_metric,
        )
        cleanup_checkpoints(args.save, args.keep_last)


        # Early stopping / best checkpoint
        if args.early_stopping > 0 and metric_value is not None:
            if is_better(metric_value, best_metric, args.early_metric):
                print(
                    f"  -> New best {args.early_metric}={metric_value:.4f} "
                    f"(was {best_metric})"
                )
                best_metric = metric_value
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(
                    args.save_best,
                    model,
                    optimizer,
                    epoch,
                    src_vocab,
                    tgt_vocab,
                    model_args,
                    best_metric,
                )
                
            else:
                patience_counter += 1
                print(
                    f"  -> No improvement on {args.early_metric} "
                    f"(best={best_metric:.4f}), patience={patience_counter}"
                )
                if patience_counter >= args.early_stopping:
                    print("Early stopping triggered.")
                    break

    if best_epoch is not None:
        print(
            f"Training finished. Best {args.early_metric}={best_metric:.4f} "
            f"at epoch {best_epoch}."
        )
    else:
        print("Training finished.")


def translate_main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    if not args.load:
        raise ValueError("In translate mode, --load CHECKPOINT is required.")

    model, optimizer, src_vocab, tgt_vocab, model_args, epoch = load_checkpoint(
        args.load, device
    )
    subword_type = model_args.get("subword_type", "none")
    src_sp = None
    tgt_sp = None
    if subword_type != "none":
        if spm is None:
            raise ImportError(
                "sentencepiece is not installed, but the loaded model was trained "
                "with subwords (subword_type != 'none')."
            )
        src_sp_model = model_args.get("src_sp_model")
        tgt_sp_model = model_args.get("tgt_sp_model")
        if not src_sp_model or not tgt_sp_model:
            raise ValueError(
                "Loaded model_args indicate subword training, but no SP model paths "
                "were stored."
            )
        src_sp = spm.SentencePieceProcessor()
        src_sp.load(src_sp_model)
        tgt_sp = spm.SentencePieceProcessor()
        tgt_sp.load(tgt_sp_model)
        print(f"Subword mode: {subword_type} (SP models: {src_sp_model}, {tgt_sp_model})")
    else:
        print("Subword mode: none (word-level).")
    print(f"Loaded checkpoint from {args.load} (epoch {epoch})")
    print(f"Total trainable parameters: {count_parameters(model):,}")

    print("Interactive translation mode. Type a sentence and press Enter (Ctrl+C to exit).")
    # Batch translation mode
    if args.src_test:
        src_lines = read_lines(args.src_test, lower=args.lower)

        hyps = translate_lines(
            model,
            src_lines,
            src_vocab,
            tgt_vocab,
            device,
            max_len=model_args.get("max_len", args.max_len),
            replace_unk=args.replace_unk,
            subword_type=subword_type,
            src_sp=src_sp,
            tgt_sp=tgt_sp,
        )

        out_path = args.output or (args.src_test + ".hyp")
        with open(out_path, "w", encoding="utf-8") as f:
            for h in hyps:
                f.write(h + "\n")
        print(f"[info] Wrote hypotheses to {out_path}")

        # Optional: compute metrics if references provided
        if args.tgt_test:
            refs = read_lines(args.tgt_test, lower=args.lower)
            if len(refs) != len(hyps):
                raise ValueError(f"Reference file has {len(refs)} lines but hypotheses have {len(hyps)} lines.")
            if args.eval_metrics and sacrebleu is not None:
                metrics = compute_sacrebleu_metrics(hyps, refs)
                print(f"[metrics] BLEU={metrics.get('bleu'):.2f}  ChrF={metrics.get('chrf'):.2f}  TER={metrics.get('ter'):.2f}")

        return

    while True:
        try:
            s = input("> ").strip()
        except KeyboardInterrupt:
            print("\nBye!")
            break

        if not s:
            continue

        user_src = s.lower() if args.lower else s

        # Sentence as seen by the model (word or SP subwords)
        if src_sp is not None:
            model_src = " ".join(src_sp.encode(user_src, out_type=str))
        else:
            model_src = user_src

        # Optionally show how the model sees the source (with <unk>)
        if args.show_src_unk:
            print("[SRC]", render_source_with_unk(model_src, src_vocab))

        src_ids = src_vocab.encode(model_src, add_eos=True)
        src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

        hyp_ids, attn_matrix = model.greedy_decode(
            src_tensor,
            max_len=model_args.get("max_len", args.max_len),
        )

        # Decode to subword string first
        if args.replace_unk:
            # Work in model token space (word or subword)
            subword_hyp = replace_unk_with_attention(
                hyp_ids, attn_matrix,
                model_src,  # important: SP-tokenised if applicable
                tgt_vocab,
            )
        else:
            subword_hyp = tgt_vocab.decode(hyp_ids)

        # If using subwords, detokenise with SentencePiece
        if tgt_sp is not None:
            hyp = tgt_sp.decode(subword_hyp.split())
        else:
            hyp = subword_hyp

        print(hyp)

def read_lines(path: str, lower: bool = False) -> List[str]:
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if lower:
                s = s.lower()
            lines.append(s)
    return lines

def translate_lines(
    model: Seq2Seq,
    src_lines: List[str],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device,
    max_len: int,
    replace_unk: bool = False,
    subword_type: str = "none",
    src_sp: Optional["spm.SentencePieceProcessor"] = None,
    tgt_sp: Optional["spm.SentencePieceProcessor"] = None,
) -> List[str]:
    hyps: List[str] = []
    for src_sentence in tqdm(src_lines, desc="Translating", leave=False):
        # Model-side source string (word or SP subwords)
        if src_sp is not None and subword_type != "none":
            model_src = " ".join(src_sp.encode(src_sentence, out_type=str))
        else:
            model_src = src_sentence

        src_ids = src_vocab.encode(model_src, add_eos=True)
        src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

        hyp_ids, attn_matrix = model.greedy_decode(src_tensor, max_len=max_len)

        # Decode in model token space first
        if replace_unk:
            subword_hyp = replace_unk_with_attention(
                hyp_ids, attn_matrix,
                model_src,   # IMPORTANT: SP-tokenised if applicable
                tgt_vocab,
            )
        else:
            subword_hyp = tgt_vocab.decode(hyp_ids)

        # If using subwords, detokenise with SentencePiece
        if tgt_sp is not None and subword_type != "none":
            hyp = tgt_sp.decode(subword_hyp.split())
        else:
            hyp = subword_hyp

        hyps.append(hyp)

    return hyps

def main():
    args = parse_args()
    # Note: seed is set in train_main/translate_main for clearer control.
    if args.mode == "train":
        if not args.src_file or not args.tgt_file:
            raise SystemExit("Error: src_file and tgt_file are required in --mode train.")
        train_main(args)
    elif args.mode == "translate":
        translate_main(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
