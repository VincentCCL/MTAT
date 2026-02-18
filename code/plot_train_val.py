#!/usr/bin/env python3
"""
plot_train_val.py

Plot training vs validation loss (NLL) from a single history JSON file
produced by rnn_seq2seq.py.

Usage:
  python plot_train_val.py rnn.hist
  python plot_train_val.py rnn.hist --save rnn_train_val.png
  python plot_train_val.py rnn.hist --train-key train_nll --val-key val_nll

Notes for Colab:
  If you run this via `!python ...`, inline display may not work.
  Use --save to write a PNG and then display it in the notebook.
"""

import argparse
import json
import matplotlib.pyplot as plt


def load_history(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_train_val(
    history_file: str,
    train_key: str = "train_nll",
    val_key: str = "val_nll",
    save_path: str | None = None,
):
    history = load_history(history_file)

    if not isinstance(history, list) or len(history) == 0:
        raise ValueError(f"{history_file} does not look like a non-empty history list.")

    epochs = [h.get("epoch") for h in history]
    train = [h.get(train_key) for h in history]
    val = [h.get(val_key) for h in history]

    if all(v is None for v in train):
        raise KeyError(f"Training key '{train_key}' not found in {history_file}.")
    if all(v is None for v in val):
        raise KeyError(f"Validation key '{val_key}' not found in {history_file}.")

    plt.figure()
    plt.plot(epochs, train, label=f"{train_key}")
    plt.plot(epochs, val, label=f"{val_key}")
    plt.xlabel("epoch")
    plt.ylabel("loss (NLL)")
    plt.title("Training vs validation loss")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot training vs validation loss from a single history JSON file."
    )
    parser.add_argument("history_file", help="History JSON file (e.g. rnn.hist)")
    parser.add_argument(
        "--train-key",
        default="train_nll",
        help="Key name for training loss in the history file (default: train_nll)",
    )
    parser.add_argument(
        "--val-key",
        default="val_nll",
        help="Key name for validation loss in the history file (default: val_nll)",
    )
    parser.add_argument(
        "--save",
        help="Save plot to this PNG instead of displaying it (recommended in Colab)",
    )

    args = parser.parse_args()
    plot_train_val(
        args.history_file,
        train_key=args.train_key,
        val_key=args.val_key,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
