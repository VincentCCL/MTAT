#!/usr/bin/env python3

import argparse
import itertools
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


OOM_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "out of memory",
    "oom",
    "killed",
    "sigkill",
    "cannot allocate memory",
)


def normalize_key(key):
    return key.strip().lstrip("-").replace("-", "_")


def cli_key(key):
    return key.replace("_", "-")


def parse_kv(text):
    if "=" not in text:
        raise ValueError(f"--set expects key=value, got: {text}")
    key, value = text.split("=", 1)
    return normalize_key(key), value.strip()


def parse_values(text):
    if "=" not in text:
        raise ValueError(f"--values expects key=v1,v2,v3, got: {text}")
    key, values = text.split("=", 1)
    return normalize_key(key), [v.strip() for v in values.split(",")]


def parse_range(text):
    if "=" not in text:
        raise ValueError(f"--range expects key=start:stop:step, got: {text}")

    key, spec = text.split("=", 1)
    start, stop, step = map(float, spec.split(":"))

    values = []
    x = start
    while x <= stop + 1e-12:
        values.append(f"{x:g}")
        x += step

    return normalize_key(key), values


def add_arg(cmd, key, value):
    option = f"--{cli_key(key)}"

    if isinstance(value, bool):
        if value:
            cmd.append(option)
        return

    value_str = str(value)

    if value_str.lower() == "true":
        cmd.append(option)
        return

    if value_str.lower() == "false":
        return

    cmd.extend([option, value_str])


def timestamp():
    return datetime.now().isoformat(timespec="seconds")


def append_log(path, message):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp()}] {message}\n")


def model_exists(params, run_base):
    """Return True only if the expected model/checkpoint output exists."""
    model_type = params.get("model_type")

    # RNN runs save explicit files in the current wrapper convention.
    if model_type == "rnn":
        candidates = []
        for key in ("save", "rnn_save_best"):
            if key in params:
                candidates.append(Path(params[key]))
        candidates.extend([run_base / "model.pt", run_base / "best.pt"])
        return any(p.is_file() and p.stat().st_size > 0 for p in candidates)

    # Transformer / HF-style directories can contain several possible model files.
    model_files = [
        "pytorch_model.bin",
        "model.safetensors",
        "tf_model.h5",
        "flax_model.msgpack",
        "checkpoint_best.pt",
        "model.pt",
        "best.pt",
    ]
    if any((run_base / name).is_file() and (run_base / name).stat().st_size > 0 for name in model_files):
        return True

    # HF Trainer may save numbered checkpoint dirs.
    for child in run_base.glob("checkpoint-*"):
        if child.is_dir() and any((child / name).is_file() for name in model_files):
            return True

    return False


def classify_failure(returncode, stdout_file):
    reason = f"return code {returncode}"

    if returncode < 0:
        reason += f"; terminated by signal {-returncode}"
    elif returncode == 137:
        reason += "; likely killed by SIGKILL, often OOM"

    try:
        text = stdout_file.read_text(encoding="utf-8", errors="ignore").lower()
    except FileNotFoundError:
        text = ""

    if any(marker in text for marker in OOM_MARKERS):
        reason += "; log contains OOM/killed marker"

    return reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtat", default="mtat.py")
    ap.add_argument("--python", default="python")
    ap.add_argument("--command", required=True, choices=["finetune", "translate"])

    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--values", action="append", default=[])
    ap.add_argument("--range", action="append", default=[])

    ap.add_argument("--save-template", default=None)
    ap.add_argument("--out-template", default=None)

    ap.add_argument("--execute", action="store_true")
    ap.add_argument(
        "--force",
        action="store_true",
        help="rerun even if an expected model/checkpoint already exists",
    )

    args = ap.parse_args()

    fixed = dict(parse_kv(x) for x in args.set)

    grid = {}

    for item in args.values:
        key, vals = parse_values(item)
        grid[key] = vals

    for item in args.range:
        key, vals = parse_range(item)
        grid[key] = vals

    keys = list(grid.keys())
    combinations = itertools.product(*(grid[k] for k in keys))

    failed = []
    successful = 0
    skipped = 0
    planned = 0

    for combo in combinations:
        params = dict(fixed)
        params.update(dict(zip(keys, combo)))
        if args.save_template:
            run_dir = Path(args.save_template.format(**params))
            params["run_dir"] = str(run_dir)

            if params.get("model_type") == "rnn":
                params["save"] = str(run_dir / "model.pt")
                params["rnn_save_best"] = str(run_dir / "best.pt")
            else:
                params["save"] = str(run_dir)

        if "run_dir" in params:
            run_dir = Path(params["run_dir"])
            params["history_json"] = str(run_dir / "history.json")
            params["log_file"] = str(run_dir / "stdout.log")
        elif "save" in params:
            save_dir = Path(params["save"])
            params["history_json"] = str(save_dir / "history.json")
            params["log_file"] = str(save_dir / "stdout.log")

        if args.out_template:
            params["out_file"] = args.out_template.format(**params)

        run_base = Path(params["run_dir"]) if "run_dir" in params else Path(params["save"])
        run_base.mkdir(parents=True, exist_ok=True)

        wrapper_log = run_base / "wrapper.log"
        stdout_file = Path(params.get("log_file", run_base / "stdout.log"))

        cmd = [args.python, args.mtat, args.command]

        for key, value in params.items():
            if key in {"log_file", "run_dir"}:
                continue
            add_arg(cmd, key, value)

        printable = " ".join(shlex.quote(x) for x in cmd)

        if not args.force and model_exists(params, run_base):
            skipped += 1
            msg = f"SKIP: model/checkpoint exists: {run_base}"
            print(msg)
            append_log(wrapper_log, msg)
            continue

        print(printable)
        append_log(wrapper_log, f"START: {printable}")

        if args.execute:
            # Store both stdout and stderr in stdout.log so CUDA/Python errors are captured.
            env = os.environ.copy()
            with open(stdout_file, "w", encoding="utf-8") as log:
                proc = subprocess.run(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )

            if proc.returncode != 0:
                reason = classify_failure(proc.returncode, stdout_file)
                append_log(wrapper_log, f"FAILED: {reason}")
                failed.append((printable, str(stdout_file), str(wrapper_log), reason))
                print(f"\nFAILED: {printable}")
                print(f"Reason: {reason}")
                print(f"Log: {stdout_file}")
                print(f"Wrapper log: {wrapper_log}")
                continue

            if model_exists(params, run_base):
                successful += 1
                append_log(wrapper_log, "SUCCESS: command returned 0 and model/checkpoint exists")
            else:
                reason = "command returned 0 but no expected model/checkpoint was found"
                append_log(wrapper_log, f"FAILED: {reason}")
                failed.append((printable, str(stdout_file), str(wrapper_log), reason))
                print(f"\nFAILED: {printable}")
                print(f"Reason: {reason}")
                print(f"Log: {stdout_file}")
                print(f"Wrapper log: {wrapper_log}")
        else:
            planned += 1

    print(f"\nFinished. Successful runs: {successful}. Failed runs: {len(failed)}. Skipped runs: {skipped}. Planned-only runs: {planned}.")

    for cmd_text, log_file, wrapper_log, reason in failed:
        print(f"\nFAILED: {cmd_text}")
        print(f"Reason: {reason}")
        print(f"Log: {log_file}")
        print(f"Wrapper log: {wrapper_log}")


if __name__ == "__main__":
    main()
