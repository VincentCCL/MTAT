#!/usr/bin/env python3
"""
wrapper_mtat_dynamic.py

Dynamic hyperparameter search wrapper for mtat.py.

Unlike wrapper_mtat_resumable_v2.py, this script does not enumerate the full
Cartesian product of --values/--range. Instead it repeatedly proposes one run,
executes it, reads its history.json, and uses the observed score to bias the next
proposal.

Search strategy:
  1. random exploration for the first --random-starts trials
  2. local mutation around the best completed runs afterwards

This keeps the script dependency-free and easy to explain in class. It is not a
full Bayesian optimiser, but it behaves like a practical dynamic search: later
trials depend on earlier results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ARGPARSE_MARKERS = (
    "usage:",
    "error: unrecognized arguments",
    "error: argument",
    "the following arguments are required",
)

OOM_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "out of memory",
    "oom",
    "killed",
    "sigkill",
    "cannot allocate memory",
)

LOWER_IS_BETTER = {"loss", "eval_loss", "val_loss", "validation_loss", "nll", "val_nll", "eval_nll", "ter"}
HIGHER_IS_BETTER = {"bleu", "eval_bleu", "val_bleu", "chrf", "eval_chrf", "accuracy"}

WRAPPER_ONLY_KEYS = {"log_file", "run_dir"}


def normalize_key(key: str) -> str:
    return key.strip().lstrip("-").replace("-", "_")


def cli_key(key: str) -> str:
    return key.replace("_", "-")


def parse_kv(text: str) -> Tuple[str, str]:
    if "=" not in text:
        raise ValueError(f"--set expects key=value, got: {text}")
    key, value = text.split("=", 1)
    key = normalize_key(key)
    # Compatibility alias: MTAT RNN uses --early-stopping, not --early-stopping-patience.
    if key == "early_stopping_patience":
        key = "early_stopping"
    return key, value.strip()


def parse_values(text: str) -> Tuple[str, List[str]]:
    if "=" not in text:
        raise ValueError(f"--values expects key=v1,v2,v3, got: {text}")
    key, values = text.split("=", 1)
    key = normalize_key(key)
    if key == "early_stopping_patience":
        key = "early_stopping"
    return key, [v.strip() for v in values.split(",") if v.strip()]


def parse_range(text: str) -> Tuple[str, List[str]]:
    if "=" not in text:
        raise ValueError(f"--range expects key=start:stop:step, got: {text}")

    key, spec = text.split("=", 1)
    start, stop, step = map(float, spec.split(":"))
    if step <= 0:
        raise ValueError("--range step must be > 0")

    values: List[str] = []
    x = start
    while x <= stop + 1e-12:
        values.append(f"{x:g}")
        x += step
    return normalize_key(key), values


def as_number(value: str) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def add_arg(cmd: List[str], key: str, value: Any) -> None:
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


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp()}] {message}\n")


def model_exists(params: Dict[str, Any], run_base: Path) -> bool:
    model_type = params.get("model_type")

    if model_type == "rnn":
        candidates = []
        for key in ("save", "rnn_save_best"):
            if key in params:
                candidates.append(Path(str(params[key])))
        candidates.extend([run_base / "model.pt", run_base / "best.pt"])
        return any(p.is_file() and p.stat().st_size > 0 for p in candidates)

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

    for child in run_base.glob("checkpoint-*"):
        if child.is_dir() and any((child / name).is_file() for name in model_files):
            return True

    return False


def classify_failure(returncode: int, stdout_file: Path) -> str:
    reason = f"return code {returncode}"

    if returncode < 0:
        reason += f"; terminated by signal {-returncode}"
    elif returncode == 137:
        reason += "; likely killed by SIGKILL, often OOM"

    try:
        text = stdout_file.read_text(encoding="utf-8", errors="ignore").lower()
    except FileNotFoundError:
        text = ""

    if returncode == 2 and any(marker in text for marker in ARGPARSE_MARKERS):
        reason += "; likely mtat.py argument/CLI error"
    if any(marker in text for marker in OOM_MARKERS):
        reason += "; log contains OOM/killed marker"

    return reason


def metric_direction(metric: str, explicit: Optional[str]) -> str:
    if explicit in {"min", "max"}:
        return explicit
    m = metric.lower().removeprefix("eval_").removeprefix("val_")
    if metric.lower() in LOWER_IS_BETTER or m in LOWER_IS_BETTER:
        return "min"
    if metric.lower() in HIGHER_IS_BETTER or m in HIGHER_IS_BETTER:
        return "max"
    return "max"


def score_is_better(a: float, b: float, direction: str) -> bool:
    return a < b if direction == "min" else a > b


def metric_from_record(record: Dict[str, Any], metric: str) -> Optional[float]:
    candidates = [metric]
    if not metric.startswith("eval_"):
        candidates.append("eval_" + metric)
    if not metric.startswith("val_"):
        candidates.append("val_" + metric)

    for key in candidates:
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def read_best_score(history_json: Path, metric: str, direction: str) -> Optional[float]:
    if not history_json.exists():
        return None
    try:
        with open(history_json, "r", encoding="utf-8") as f:
            history = json.load(f)
    except Exception:
        return None

    values: List[float] = []
    if isinstance(history, list):
        for record in history:
            if isinstance(record, dict):
                value = metric_from_record(record, metric)
                if value is not None:
                    values.append(value)
    elif isinstance(history, dict):
        value = metric_from_record(history, metric)
        if value is not None:
            values.append(value)

    if not values:
        return None
    return min(values) if direction == "min" else max(values)


def params_id(params: Dict[str, Any], search_keys: Iterable[str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((key, str(params[key])) for key in search_keys))


def choose_random(space: Dict[str, List[str]], rng: random.Random) -> Dict[str, str]:
    return {key: rng.choice(values) for key, values in space.items()}


def mutate_from_parent(parent: Dict[str, Any], space: Dict[str, List[str]], rng: random.Random, mutation_rate: float) -> Dict[str, str]:
    child: Dict[str, str] = {}
    for key, values in space.items():
        current = str(parent[key])
        if rng.random() > mutation_rate and current in values:
            child[key] = current
            continue

        # Numeric values are mutated locally; categorical values are resampled.
        current_idx = values.index(current) if current in values else None
        numeric_values = [as_number(v) for v in values]
        if current_idx is not None and all(v is not None for v in numeric_values) and len(values) > 1:
            step = rng.choice([-1, 1])
            new_idx = max(0, min(len(values) - 1, current_idx + step))
            if new_idx == current_idx and len(values) > 2:
                new_idx = rng.randrange(len(values))
            child[key] = values[new_idx]
        else:
            alternatives = [v for v in values if v != current]
            child[key] = rng.choice(alternatives or values)
    return child


def propose_params(
    fixed: Dict[str, str],
    space: Dict[str, List[str]],
    completed: List[Dict[str, Any]],
    tried: set,
    random_starts: int,
    top_k: int,
    mutation_rate: float,
    rng: random.Random,
    max_attempts: int = 500,
) -> Optional[Dict[str, Any]]:
    search_keys = list(space.keys())

    for attempt in range(max_attempts):
        if len(completed) < random_starts:
            candidate_dynamic = choose_random(space, rng)
        else:
            ranked = completed[: max(1, min(top_k, len(completed)))]
            parent = rng.choice(ranked)["params"]
            candidate_dynamic = mutate_from_parent(parent, space, rng, mutation_rate)

            # Keep some exploration after random starts.
            if rng.random() < 0.20:
                candidate_dynamic = choose_random(space, rng)

        candidate = dict(fixed)
        candidate.update(candidate_dynamic)
        pid = params_id(candidate, search_keys)
        if pid not in tried:
            return candidate

    return None


def apply_templates(params: Dict[str, Any], save_template: Optional[str], out_template: Optional[str]) -> None:
    if save_template:
        run_dir = Path(save_template.format(**params))
        params["run_dir"] = str(run_dir)
        if params.get("model_type") == "rnn":
            params["save"] = str(run_dir / "model.pt")
            params["rnn_save_best"] = str(run_dir / "best.pt")
        else:
            params["save"] = str(run_dir)

    if "run_dir" in params:
        run_dir = Path(str(params["run_dir"]))
        params["history_json"] = str(run_dir / "history.json")
        params["log_file"] = str(run_dir / "stdout.log")
    elif "save" in params:
        save_dir = Path(str(params["save"]))
        params["history_json"] = str(save_dir / "history.json")
        params["log_file"] = str(save_dir / "stdout.log")

    if out_template:
        params["out_file"] = out_template.format(**params)


def build_command(py: str, mtat: str, command: str, params: Dict[str, Any]) -> List[str]:
    cmd = [py, mtat, command]
    for key, value in params.items():
        if key in WRAPPER_ONLY_KEYS:
            continue
        add_arg(cmd, key, value)
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="Dynamic hyperparameter search wrapper for mtat.py")
    ap.add_argument("--mtat", default="mtat.py")
    ap.add_argument("--python", default="python")
    ap.add_argument("--command", required=True, choices=["finetune", "translate"])

    ap.add_argument("--set", action="append", default=[], help="Fixed mtat.py argument, key=value")
    ap.add_argument("--values", action="append", default=[], help="Search values, key=v1,v2,v3")
    ap.add_argument("--range", action="append", default=[], help="Search range, key=start:stop:step")

    ap.add_argument("--save-template", required=True, help="Template for run directories, e.g. runs/dyn_bs{batch_size}_lr{lr}")
    ap.add_argument("--out-template", default=None)
    ap.add_argument("--study-dir", default=None, help="Directory for dynamic_search.jsonl; default is parent of save-template")

    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--random-starts", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--mutation-rate", type=float, default=0.5)
    ap.add_argument("--metric", default="bleu", help="Metric to optimise; reads this key or eval_/val_ variants from history.json")
    ap.add_argument("--direction", choices=["auto", "min", "max"], default="auto")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--force", action="store_true", help="rerun even if an expected model/checkpoint already exists")

    args = ap.parse_args()
    rng = random.Random(args.seed)
    direction = metric_direction(args.metric, None if args.direction == "auto" else args.direction)

    fixed = dict(parse_kv(x) for x in args.set)
    space: Dict[str, List[str]] = {}
    for item in args.values:
        key, vals = parse_values(item)
        space[key] = vals
    for item in args.range:
        key, vals = parse_range(item)
        space[key] = vals

    if not space:
        raise ValueError("Dynamic search needs at least one --values or --range argument.")

    study_dir = Path(args.study_dir) if args.study_dir else Path(args.save_template.split("{")[0] or ".").parent
    study_dir.mkdir(parents=True, exist_ok=True)
    study_log = study_dir / "dynamic_search.jsonl"

    completed: List[Dict[str, Any]] = []
    tried = set()
    successful = 0
    failed: List[Tuple[str, str, str, str]] = []
    skipped = 0
    planned = 0

    for trial in range(1, args.trials + 1):
        candidate = propose_params(
            fixed=fixed,
            space=space,
            completed=completed,
            tried=tried,
            random_starts=args.random_starts,
            top_k=args.top_k,
            mutation_rate=args.mutation_rate,
            rng=rng,
        )
        if candidate is None:
            print("No untried candidate left in the supplied search space.")
            break

        tried.add(params_id(candidate, space.keys()))
        params = dict(candidate)
        apply_templates(params, args.save_template, args.out_template)

        run_base = Path(str(params.get("run_dir", params.get("save"))))
        run_base.mkdir(parents=True, exist_ok=True)
        wrapper_log = run_base / "wrapper.log"
        stdout_file = Path(str(params.get("log_file", run_base / "stdout.log")))
        history_file = Path(str(params.get("history_json", run_base / "history.json")))

        cmd = build_command(args.python, args.mtat, args.command, params)
        printable = " ".join(shlex.quote(x) for x in cmd)

        print(f"\n=== Trial {trial}/{args.trials} ===")
        print(printable)
        append_log(wrapper_log, f"START trial={trial}: {printable}")

        event: Dict[str, Any] = {
            "timestamp": timestamp(),
            "trial": trial,
            "params": params,
            "command": printable,
            "metric": args.metric,
            "direction": direction,
        }

        if not args.force and model_exists(params, run_base):
            skipped += 1
            score = read_best_score(history_file, args.metric, direction)
            event.update({"status": "skipped_existing", "score": score})
            print(f"SKIP: model/checkpoint exists: {run_base}")
            if score is not None:
                completed.append({"score": score, "params": candidate, "run_base": str(run_base)})
                completed.sort(key=lambda x: x["score"], reverse=(direction == "max"))
        elif args.execute:
            with open(stdout_file, "w", encoding="utf-8") as log:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=os.environ.copy())

            if proc.returncode != 0:
                reason = classify_failure(proc.returncode, stdout_file)
                event.update({"status": "failed", "reason": reason})
                append_log(wrapper_log, f"FAILED: {reason}")
                failed.append((printable, str(stdout_file), str(wrapper_log), reason))
                print(f"FAILED: {reason}")
            elif not model_exists(params, run_base):
                reason = "command returned 0 but no expected model/checkpoint was found"
                event.update({"status": "failed", "reason": reason})
                append_log(wrapper_log, f"FAILED: {reason}")
                failed.append((printable, str(stdout_file), str(wrapper_log), reason))
                print(f"FAILED: {reason}")
            else:
                score = read_best_score(history_file, args.metric, direction)
                successful += 1
                event.update({"status": "success", "score": score})
                append_log(wrapper_log, f"SUCCESS: score={score}")
                print(f"SUCCESS: {args.metric}={score}")
                if score is not None:
                    completed.append({"score": score, "params": candidate, "run_base": str(run_base)})
                    completed.sort(key=lambda x: x["score"], reverse=(direction == "max"))
        else:
            planned += 1
            event.update({"status": "planned"})

        with open(study_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        if completed:
            best = completed[0]
            print(f"Current best: {args.metric}={best['score']} at {best['run_base']}")

    print(f"\nFinished. Successful runs: {successful}. Failed runs: {len(failed)}. Skipped runs: {skipped}. Planned-only runs: {planned}.")
    print(f"Study log: {study_log}")
    if completed:
        best = completed[0]
        print(f"Best observed {args.metric}: {best['score']} at {best['run_base']}")

    for cmd_text, log_file, wrapper_log, reason in failed:
        print(f"\nFAILED: {cmd_text}")
        print(f"Reason: {reason}")
        print(f"Log: {log_file}")
        print(f"Wrapper log: {wrapper_log}")


if __name__ == "__main__":
    main()
