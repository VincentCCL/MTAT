#!/usr/bin/env python3
import argparse
import json
import matplotlib.pyplot as plt


def load_history(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_histories(history_files, labels, metric, save_path=None):
    plt.figure()

    for path, label in zip(history_files, labels):
        history = load_history(path)
        epochs = [h["epoch"] for h in history]
        values = [h.get(metric) for h in history]

        if all(v is None for v in values):
            print(f"Warning: metric '{metric}' not found in {path}")
            continue

        plt.plot(epochs, values, label=label)

    plt.xlabel("epoch")
    plt.ylabel(metric.replace("_", " "))
    plt.title(f"Comparison of {metric.replace('_', ' ')}")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("history_files", nargs="+")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--metric", default="val_nll")
    parser.add_argument("--save", help="Save plot to PNG instead of showing it")

    args = parser.parse_args()

    labels = args.labels if args.labels else args.history_files
    if len(labels) != len(args.history_files):
        raise ValueError("Number of labels must match number of history files")

    plot_histories(
        args.history_files,
        labels,
        args.metric,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()

