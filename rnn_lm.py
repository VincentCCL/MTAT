"""
rnn_lm.py

A tiny word-level RNN language model in Keras with two main functions:

- train(corpus_source, ...)
    * corpus_source: URL (http/https) or path to a local .txt file.
    * returns the trained Keras model.

- sample(seed_text, ...)
    * seed_text: string prefix like "she likes"
    * returns a generated continuation as a string.

This is intended for teaching and small toy corpora, not for serious LM work.
"""

from __future__ import annotations

import io
import os
import urllib.request
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------------------------
# Module-level globals (set by train(), used by sample()).
# ---------------------------------------------------------------------
_model: Optional[keras.Model] = None
_tokenizer: Optional[Tokenizer] = None
_context_len: Optional[int] = None
_eos_id: Optional[int] = None
_word_index: Optional[dict] = None  # word -> id
_index_word: Optional[dict] = None  # id -> word


# ---------------------------------------------------------------------
# Utility: load text from URL or local file
# ---------------------------------------------------------------------
def _load_corpus(corpus_source: str) -> str:
    """
    Load raw text from either a URL (http/https) or a local file path.
    Returns the content as a single string.
    """
    if corpus_source.startswith("http://") or corpus_source.startswith("https://"):
        with urllib.request.urlopen(corpus_source) as f:
            # Assume UTF-8; adjust if needed
            text = f.read().decode("utf-8", errors="replace")
    else:
        if not os.path.exists(corpus_source):
            raise FileNotFoundError(f"File not found: {corpus_source}")
        with io.open(corpus_source, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    return text


# ---------------------------------------------------------------------
# Utility: simple sentence splitting
# ---------------------------------------------------------------------
def _text_to_sentences(raw_text: str) -> list[str]:
    """
    Sentence splitting using NLTK's Punkt tokenizer.
    Much more robust for prose such as Project Gutenberg texts.
    """
    import nltk
    nltk.download("punkt", quiet=True)

    from nltk.tokenize import sent_tokenize

    # Remove Gutenberg header/footer markers
    start = raw_text.find("*** START")
    end = raw_text.find("*** END")
    if start != -1 and end != -1:
        raw_text = raw_text[start:end]

    # Actual sentence splitting
    sentences = sent_tokenize(raw_text)

    # Clean whitespace
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

    return sentences


# ---------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------
def train(
    corpus_source: str,
    embed_dim: int = 64,
    hidden_size: int = 128,
    batch_size: int = 32,
    epochs: int = 30,
    max_sentences: Optional[int] = None,
    verbose: int = 1,
) -> keras.Model:
    """
    Train a small word-level RNN language model.

    Parameters
    ----------
    corpus_source : str
        Either a URL (http/https) or a path to a local .txt file.
        The file should contain one sentence per line for best results.
    embed_dim : int
        Dimension of the word embeddings.
    hidden_size : int
        Size of the RNN hidden state.
    batch_size : int
        Batch size for training.
    epochs : int
        Number of training epochs.
    max_sentences : int or None
        If not None, only use the first max_sentences for training
        (useful for quick experiments).
    verbose : int
        Verbosity flag passed to model.fit().

    Returns
    -------
    model : keras.Model
        The trained language model.

    Side effects
    ------------
    Sets module-level globals so that sample() can be called afterwards:
        _model, _tokenizer, _context_len, _eos_id, _word_index, _index_word
    """
    global _model, _tokenizer, _context_len, _eos_id, _word_index, _index_word

    # 1. Load raw text
    raw_text = _load_corpus(corpus_source)

    # 2. Convert to sentence list
    sentences = _text_to_sentences(raw_text)
    if max_sentences is not None:
        sentences = sentences[:max_sentences]

    if not sentences:
        raise ValueError("No sentences found in the corpus.")

    # 3. Add <sos> and <eos> markers
    src_raw = sentences
    src = [f"<sos> {s.lower()} <eos>" for s in src_raw]

    # 4. Tokenizer: word-level, keep all characters, add <unk> for OOV
    tokenizer = Tokenizer(
        filters="",      # do not strip punctuation/special tokens
        lower=True,
        oov_token="<unk>",
    )
    tokenizer.fit_on_texts(src)

    # Build vocabulary mappings
    word_index = tokenizer.word_index  # word -> id
    index_word = {i: w for w, i in word_index.items()}
    num_tokens = len(word_index) + 1   # +1 for padding index 0

    if "<eos>" not in word_index:
        raise ValueError("The token <eos> was not found in the vocabulary.")
    eos_id = word_index["<eos>"]

    # 5. Turn sentences into sequences of integer IDs
    sequences = tokenizer.texts_to_sequences(src)

    # 6. Pad to fixed length
    max_len = max(len(seq) for seq in sequences)
    sequences_padded = pad_sequences(
        sequences,
        maxlen=max_len,
        padding="post",  # pad on the right with 0s
    )

    # 7. Build input/target sequences
    context_len = max_len - 1
    X_lm = sequences_padded[:, :-1]  # all but last token
    y_lm = sequences_padded[:, 1:]   # all but first token

    # 8. Define the RNN LM model
    inputs = keras.Input(shape=(context_len,), dtype="int32")

    x = layers.Embedding(
        input_dim=num_tokens,
        output_dim=embed_dim,
        mask_zero=True,          # ignore padding index 0
    )(inputs)

    x = layers.SimpleRNN(
        hidden_size,
        return_sequences=True,
    )(x)

    outputs = layers.Dense(
        num_tokens,
        activation="softmax",
    )(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 9. Train the model
    model.fit(
        X_lm,
        y_lm,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
    )

    # 10. Store state in module-level globals for sampling
    _model = model
    _tokenizer = tokenizer
    _context_len = context_len
    _eos_id = eos_id
    _word_index = word_index
    _index_word = index_word

    return model


# ---------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------
def _sample_next(probs: np.ndarray, temperature: float = 1.0) -> int:
    """
    Sample a token ID from a probability distribution, using temperature.
    """
    probs = np.asarray(probs, dtype="float64")
    probs = np.maximum(probs, 1e-8)  # avoid log(0)
    probs = np.log(probs) / temperature
    probs = np.exp(probs)
    probs /= np.sum(probs)
    return np.random.choice(len(probs), p=probs)


def sample(
    seed_text: str,
    num_steps: int = 10,
    temperature: float = 1.0,
) -> str:
    """
    Generate text from the trained language model.

    Parameters
    ----------
    seed_text : str
        A prefix to condition on, e.g. "she likes".
    num_steps : int
        Maximum number of additional tokens to generate (excluding the seed).
    temperature : float
        Sampling temperature:
            < 1.0 -> more conservative, deterministic.
            > 1.0 -> more random.

    Returns
    -------
    text : str
        The generated text, without the initial <sos> and any trailing <eos>.
    """
    if _model is None or _tokenizer is None or _context_len is None:
        raise RuntimeError(
            "Model has not been trained yet. Call train(...) first."
        )

    # Prepend <sos> so the model sees a proper start symbol
    full_seed = f"<sos> {seed_text.lower()}"
    seed_ids = _tokenizer.texts_to_sequences([full_seed])[0]

    if not seed_ids:
        raise ValueError(
            "The seed_text could not be tokenized into known tokens. "
            "Try a different seed or make sure your corpus covers similar text."
        )

    # Ensure we have exactly _context_len tokens (pad on the left if needed)
    if len(seed_ids) < _context_len:
        seed_ids = [0] * (_context_len - len(seed_ids)) + seed_ids
    else:
        seed_ids = seed_ids[-_context_len:]

    generated_ids = list(seed_ids)

    for _ in range(num_steps):
        context = np.array(generated_ids[-_context_len:], dtype="int32")[None, :]
        preds = _model.predict(context, verbose=0)[0]  # shape: (context_len, vocab)
        next_probs = preds[-1]                         # last time step

        next_id = _sample_next(next_probs, temperature=temperature)
        generated_ids.append(next_id)

        if next_id == _eos_id:
            break

    # Decode IDs back to words, skipping padding (0)
    id_to_word = dict(_index_word)
    id_to_word[0] = "<pad>"

    words = [id_to_word.get(i, "<unk>") for i in generated_ids if i != 0]

    # Drop initial <sos> in the human-readable output
    if words and words[0] == "<sos>":
        words = words[1:]

    # Cut at first <eos> if present
    if "<eos>" in words:
        eos_pos = words.index("<eos>")
        words = words[:eos_pos]

    return " ".join(words)
