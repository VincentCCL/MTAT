#!/usr/bin/env python3
"""
plot_history_transformers.py

Plot Hugging Face Trainer history JSON (trainer.state.log_history):
- training curve: loss
- validation curve: eval_loss
Optionally plot eval metrics like eval_bleu / eval_chrf.

Works with JSON produced by dumping `trainer.state.log_history`.
"""

import argparse
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def load_history(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not look like a Trainer log_history JSON list.")
    return data


def available_keys(history: List[Dict[str, Any]]) -> List[str]:
    keys = set()
    for h in history:
        keys.update(h.keys())
    return sorted(keys)


def pick_x(h: Dict[str, Any], x_mode: str) -> Optional[float]:
    """
    Choose x-axis value.
    HF log_history often includes:
      - "step" (global step)
      - "epoch" (float)
    """
    if x_mode == "step":
        return h.get("step", None)
    if x_mode == "epoch":
        return h.get("epoch", None)

    # auto: prefer step if present, else epoch
    return h.get("step", h.get("epoch", None))


def extract_series(
    history: List[Dict[str, Any]],
    metric: str,
    x_mode: str,
) -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    for h in history:
        if metric not in h:
            continue
        x = pick_x(h, x_mode)
        y = h.get(metric)

        if x is None or y is None:
            continue
        if isinstance(y, (int, float)) and (not (isinstance(y, float) and math.isnan(y))):
            xs.append(float(x))
            ys.append(float(y))
    return xs, ys


def plot_runs(
    files: List[str],
    labels: List[str],
    train_metric: str,
    val_metric: str,
    x_mode: str,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    plt.figure()

    for path, label in zip(files, labels):
        history = load_history(path)

        x_tr, y_tr = extract_series(history, train_metric, x_mode)
        x_va, y_va = extract_series(history, val_metric, x_mode)

        if len(y_tr) == 0 and len(y_va) == 0:
            keys = ", ".join(available_keys(history)[:40])
            more = "" if len(available_keys(history)) <= 40 else " ..."
            raise KeyError(
                f"Neither '{train_metric}' nor '{val_metric}' found in {path}.\n"
                f"Available keys include: {keys}{more}"
            )

        if len(y_tr) > 0:
            plt.plot(x_tr, y_tr, label=f"{label} (train: {train_metric})")
        else:
            print(f"Warning: training metric '{train_metric}' not found in {path}")

        if len(y_va) > 0:
            plt.plot(x_va, y_va, label=f"{label} (val: {val_metric})")
        else:
            print(f"Warning: validation metric '{val_metric}' not found in {path}")

    xlabel = "step" if (x_mode == "step") else ("epoch" if x_mode == "epoch" else "step/epoch")
    plt.xlabel(xlabel)
    plt.ylabel("value")
    plt.title(title or f"Train vs validation ({train_metric} vs {val_metric})")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("history_files", nargs="+", help="One or more Trainer log_history JSON files")
    parser.add_argument("--labels", nargs="+", help="Labels for runs (same length as history_files)")
    parser.add_argument("--train-metric", default="loss", help="Training metric key (default: loss)")
    parser.add_argument("--val-metric", default="eval_loss", help="Validation metric key (default: eval_loss)")
    parser.add_argument(
        "--x",
        choices=["auto", "step", "epoch"],
        default="auto",
        help="X-axis: auto (prefer step), step, or epoch",
    )
    parser.add_argument("--title", default=None, help="Custom plot title")
    parser.add_argument("--save", default=None, help="Save plot to PNG instead of showing it")

    args = parser.parse_args()

    labels = args.labels if args.labels else args.history_files
    if len(labels) != len(args.history_files):
        raise ValueError("Number of labels must match number of history files")

    plot_runs(
        files=args.history_files,
        labels=labels,
        train_metric=args.train_metric,
        val_metric=args.val_metric,
        x_mode=args.x,
        save_path=args.save,
        title=args.title,
    )


if __name__ == "__main__":
    main()
