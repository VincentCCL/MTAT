#!/usr/bin/env python3
"""
plot_train_val_new.py

Robust plotting for:
- RNN histories (train_nll / val_nll, x=epoch)
- HF Trainer histories (loss / eval_loss / eval_bleu / eval_chrf, x=step/epoch)

Features:
- Auto-detect default train/val keys when not provided
- Allow plotting only one metric (e.g., just eval_bleu)
- Optional dual y-axis (recommended when mixing loss with BLEU/chrF)
"""

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def load_history(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"{path} does not look like a non-empty history list.")
    return data


def union_keys(history: List[Dict[str, Any]]) -> List[str]:
    keys = set()
    for h in history:
        keys.update(h.keys())
    return sorted(keys)


def choose_x_mode(history: List[Dict[str, Any]]) -> str:
    # Prefer step if present (HF), else epoch (RNN)
    for h in history:
        if "step" in h and h.get("step") is not None:
            return "step"
    return "epoch"


def get_x(h: Dict[str, Any], x_mode: str) -> Optional[float]:
    if x_mode == "step":
        return h.get("step", h.get("epoch"))
    return h.get("epoch", h.get("step"))


def extract_series(history: List[Dict[str, Any]], key: str, x_mode: str) -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    for h in history:
        if key not in h:
            continue
        x = get_x(h, x_mode)
        y = h.get(key)
        if x is None or y is None:
            continue
        if not isinstance(y, (int, float)):
            continue
        xs.append(float(x))
        ys.append(float(y))
    return xs, ys


def autodetect_default_keys(history: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    keys = set(union_keys(history))

    # RNN style
    if "train_nll" in keys and "val_nll" in keys:
        return "train_nll", "val_nll"

    # HF style (loss)
    if "loss" in keys and "eval_loss" in keys:
        return "loss", "eval_loss"

    # If only eval_loss exists (rare)
    if "eval_loss" in keys:
        return None, "eval_loss"

    return None, None


def is_eval_metric(key: str) -> bool:
    return key.startswith("eval_")


def plot_metrics(
    history_file: str,
    train_key: Optional[str],
    val_key: Optional[str],
    save_path: Optional[str],
    x_mode: str = "auto",
    dual_y: str = "auto",  # auto|on|off
):
    history = load_history(history_file)
    keys = union_keys(history)

    # Choose x-axis mode
    x_mode_final = choose_x_mode(history) if x_mode == "auto" else x_mode

    # Autodetect if needed
    auto_train, auto_val = autodetect_default_keys(history)

    # If user didn't specify keys, use auto defaults
    if train_key is None and val_key is None:
        train_key, val_key = auto_train, auto_val

    # If user specified only val_key, decide default train_key:
    # - If val_key is eval_bleu/eval_chrf/etc, default to "no train curve" to avoid nonsense plots
    if val_key is not None and train_key is None:
        if is_eval_metric(val_key) and val_key not in ("eval_loss",):
            train_key = None
        else:
            train_key = auto_train

    # Allow explicit disabling via --train-key none / --val-key none
    if isinstance(train_key, str) and train_key.lower() == "none":
        train_key = None
    if isinstance(val_key, str) and val_key.lower() == "none":
        val_key = None

    if train_key is None and val_key is None:
        raise KeyError(
            f"No metrics selected. Available keys include: {', '.join(keys[:40])}"
            + (" ..." if len(keys) > 40 else "")
        )

    # Extract series
    x_train, y_train = ([], [])
    x_val, y_val = ([], [])

    if train_key is not None:
        if train_key not in keys:
            raise KeyError(
                f"Training key '{train_key}' not found in {history_file}.\n"
                f"Available keys include: {', '.join(keys[:40])}"
                + (" ..." if len(keys) > 40 else "")
            )
        x_train, y_train = extract_series(history, train_key, x_mode_final)
        if len(y_train) == 0:
            raise KeyError(f"Training key '{train_key}' found but has no numeric values in {history_file}.")

    if val_key is not None:
        if val_key not in keys:
            raise KeyError(
                f"Validation key '{val_key}' not found in {history_file}.\n"
                f"Available keys include: {', '.join(keys[:40])}"
                + (" ..." if len(keys) > 40 else "")
            )
        x_val, y_val = extract_series(history, val_key, x_mode_final)
        if len(y_val) == 0:
            raise KeyError(f"Validation key '{val_key}' found but has no numeric values in {history_file}.")

    # Decide whether to use dual y-axis
    dual = False
    if dual_y == "on":
        dual = True
    elif dual_y == "off":
        dual = False
    else:
        # auto: if mixing loss-ish with BLEU/chrF-ish, use dual axis
        if train_key is not None and val_key is not None:
            loss_like = {"loss", "eval_loss", "train_loss", "train_nll", "val_nll"}
            metric_like = {"eval_bleu", "eval_chrf"}
            if (train_key in loss_like and val_key in metric_like) or (val_key in loss_like and train_key in metric_like):
                dual = True

    # Plot
    fig, ax1 = plt.subplots()

    ax1.set_xlabel(x_mode_final)

    # If dual axis, plot train on left, val on right (or vice versa)
    if dual and train_key is not None and val_key is not None:
        ax1.plot(x_train, y_train, label=train_key)
        ax1.set_ylabel(train_key)

        ax2 = ax1.twinx()
        ax2.plot(x_val, y_val, label=val_key, marker="o")
        ax2.set_ylabel(val_key)

        # Merge legends
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best")
        title = f"{train_key} (left) vs {val_key} (right)"
    else:
        # Single axis
        if train_key is not None:
            ax1.plot(x_train, y_train, label=train_key)
        if val_key is not None:
            ax1.plot(x_val, y_val, label=val_key, marker="o")
        ax1.set_ylabel("value")
        ax1.legend(loc="best")
        title = " / ".join([k for k in [train_key, val_key] if k is not None])

    ax1.set_title(title)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Robust train/val plotting for RNN and HF Trainer histories.")
    parser.add_argument("history_file", help="History JSON file (e.g. rnn.hist or transformer.hist)")

    parser.add_argument("--train-key", default=None,
                        help="Training key (e.g. loss or train_nll). Use 'none' to disable. If omitted, auto-detect.")
    parser.add_argument("--val-key", default=None,
                        help="Validation key (e.g. eval_loss, eval_bleu, val_nll). Use 'none' to disable. If omitted, auto-detect.")

    parser.add_argument("--x", choices=["auto", "step", "epoch"], default="auto",
                        help="X axis: auto prefers step if present, else epoch.")
    parser.add_argument("--dual-y", choices=["auto", "on", "off"], default="auto",
                        help="Use two y-axes when metrics have different scales (auto/on/off).")
    parser.add_argument("--save", default=None, help="Save plot to PNG instead of showing it")

    args = parser.parse_args()

    plot_metrics(
        history_file=args.history_file,
        train_key=args.train_key,
        val_key=args.val_key,
        save_path=args.save,
        x_mode=args.x,
        dual_y=args.dual_y,
    )


if __name__ == "__main__":
    main()
