#!/usr/bin/env python3
"""
wrapper_mtat_dynamic.py

Dynamic beam-search hyperparameter wrapper for mtat.py.

This script searches over hyperparameters without enumerating the full Cartesian
product. It keeps a beam of the best completed configurations and expands each
beam item into nearby candidate configurations for the next generation.

Typical use:

  python wrapper_mtat_dynamic.py \
    --mtat mtat.py \
    --command finetune \
    --execute \
    --generations 5 \
    --beam-width 5 \
    --expand-per-parent 3 \
    --metric bleu \
    --direction max \
    --set model-type=rnn \
    --values rnn-type=rnn,gru,lstm \
    --values hidden-size=32,64,128 \
    --save-template 'runs/beam_{rnn_type}_{hidden_size}'

Notes:
- --set supplies fixed mtat.py arguments.
- --values and --range define the hyperparameter search space.
- generation 1 starts with random candidates.
- later generations expand/mutate the current best beam.
- --model-type rnn is passed through correctly.
- --early-stopping-patience is accepted as an alias for --early-stopping.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import csv
import sys

csv.field_size_limit(sys.maxsize)

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

LOWER_IS_BETTER = {
    "loss", "eval_loss", "val_loss", "validation_loss", "nll", "val_nll",
    "eval_nll", "ter", "eval_ter", "val_ter"
}
HIGHER_IS_BETTER = {
    "bleu", "eval_bleu", "val_bleu", "chrf", "eval_chrf", "val_chrf",
    "accuracy", "eval_accuracy", "val_accuracy"
}

# These are generated/used by the wrapper and should not be passed to mtat.py.
WRAPPER_ONLY_KEYS = {"log_file", "run_dir"}


def normalize_key(key: str) -> str:
    return key.strip().lstrip("-").replace("-", "_")


def cli_key(key: str) -> str:
    return key.replace("_", "-")


def canonical_key(key: str) -> str:
    key = normalize_key(key)
    # Compatibility alias: MTAT RNN uses --early-stopping.
    if key == "early_stopping_patience":
        return "early_stopping"
    return key


def parse_kv(text: str) -> Tuple[str, str]:
    if "=" not in text:
        raise ValueError(f"--set expects key=value, got: {text}")
    key, value = text.split("=", 1)
    return canonical_key(key), value.strip()


def parse_values(text: str) -> Tuple[str, List[str]]:
    if "=" not in text:
        raise ValueError(f"--values expects key=v1,v2,v3, got: {text}")
    key, values = text.split("=", 1)
    vals = [v.strip() for v in values.split(",") if v.strip()]
    if not vals:
        raise ValueError(f"--values for {key} is empty")
    return canonical_key(key), vals


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
    return canonical_key(key), values


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


def as_number(value: str) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def metric_direction(metric: str, explicit: str) -> str:
    if explicit in {"min", "max"}:
        return explicit
    m = metric.lower().removeprefix("eval_").removeprefix("val_")
    if metric.lower() in LOWER_IS_BETTER or m in LOWER_IS_BETTER:
        return "min"
    if metric.lower() in HIGHER_IS_BETTER or m in HIGHER_IS_BETTER:
        return "max"
    return "max"


def metric_from_record(record: Dict[str, Any], metric: str) -> Optional[float]:
    candidates = [metric]
    if not metric.startswith("eval_"):
        candidates.append("eval_" + metric)
    if not metric.startswith("val_"):
        candidates.append("val_" + metric)
    # common MTAT/RNN names
    if metric == "bleu":
        candidates.extend(["valid_bleu", "validation_bleu", "dev_bleu"])
    if metric in {"loss", "nll"}:
        candidates.extend(["val_loss", "valid_loss", "validation_loss", "eval_loss"])

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
        # Some histories store a list under common keys.
        for maybe_key in ("history", "log_history", "records", "epochs"):
            seq = history.get(maybe_key)
            if isinstance(seq, list):
                for record in seq:
                    if isinstance(record, dict):
                        value = metric_from_record(record, metric)
                        if value is not None:
                            values.append(value)
        value = metric_from_record(history, metric)
        if value is not None:
            values.append(value)

    if not values:
        return None
    return min(values) if direction == "min" else max(values)

def numeric_records_from_history(history_json: Path) -> List[Dict[str, Any]]:
    if not history_json.exists():
        return []
    try:
        history = json.loads(history_json.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(history, dict):
        for key in ("history", "log_history", "records", "epochs"):
            if isinstance(history.get(key), list):
                history = history[key]
                break

    if not isinstance(history, list):
        return []

    rows = []
    for rec in history:
        if isinstance(rec, dict):
            rows.append({
                k: v for k, v in rec.items()
                if isinstance(v, (int, float, str, bool)) or v is None
            })
    return rows


def best_record_from_history(history_json: Path, metric: str, direction: str) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    rows = numeric_records_from_history(history_json)
    best_row = None
    best_score = None

    for row in rows:
        score = metric_from_record(row, metric)
        if score is None:
            continue
        if best_score is None or (direction == "max" and score > best_score) or (direction == "min" and score < best_score):
            best_score = score
            best_row = row

    return best_row, best_score

TSV_FIELDS = [
    "timestamp",
    "generation",
    "trial_in_generation",
    "global_trial",
    "status",
    "reason",
    "optim_metric",
    "optim_direction",
    "score",
    "best_epoch",
    "run_dir",
    "stdout_log",
    "wrapper_log",
    "history_json",
    "param_rnn_type",
    "param_emb_size",
    "param_enc_layers",
    "param_dec_layers",
    "param_hidden_size",
    "param_bidirectional",
    "param_attention",
    "param_batch_size",
    "param_subword_type",
    "param_max_len",
    "param_max_src_vocab",
    "param_max_tgt_vocab",
    "metric_loss",
    "metric_nll",
    "metric_val_nll",
    "metric_bleu",
    "metric_chrf",
    "metric_ter",
]

def append_tsv_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists() and path.stat().st_size > 0

    clean_row = {
        field: "" if row.get(field) is None else str(row.get(field, ""))
        for field in TSV_FIELDS
    }

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=TSV_FIELDS,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
            quoting=csv.QUOTE_MINIMAL,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(clean_row)
        
def score_sort_key(item: Dict[str, Any], direction: str) -> float:
    score = item.get("score")
    if score is None:
        return math.inf if direction == "min" else -math.inf
    return float(score)


def rank_completed(completed: List[Dict[str, Any]], direction: str) -> List[Dict[str, Any]]:
    return sorted(completed, key=lambda x: score_sort_key(x, direction), reverse=(direction == "max"))


def params_id(params: Dict[str, Any], search_keys: Iterable[str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((key, str(params[key])) for key in search_keys if key in params))


def choose_random(space: Dict[str, List[str]], rng: random.Random) -> Dict[str, str]:
    return {key: rng.choice(values) for key, values in space.items()}


def neighbor_values(values: List[str], current: str, rng: random.Random) -> List[str]:
    if current not in values:
        return [rng.choice(values)]
    idx = values.index(current)
    numeric_values = [as_number(v) for v in values]
    if all(v is not None for v in numeric_values):
        candidates = []
        if idx > 0:
            candidates.append(values[idx - 1])
        if idx + 1 < len(values):
            candidates.append(values[idx + 1])
        if not candidates:
            candidates = [v for v in values if v != current]
        return candidates or values
    return [v for v in values if v != current] or values


def mutate_one_or_more(parent: Dict[str, Any], space: Dict[str, List[str]], rng: random.Random, max_changes: int = 2) -> Dict[str, str]:
    child = {key: str(parent.get(key, rng.choice(values))) for key, values in space.items()}
    keys = list(space.keys())
    rng.shuffle(keys)
    n_changes = rng.randint(1, max(1, min(max_changes, len(keys))))
    for key in keys[:n_changes]:
        current = child[key]
        options = neighbor_values(space[key], current, rng)
        child[key] = rng.choice(options)
    return child


def initial_candidates(
    fixed: Dict[str, str],
    space: Dict[str, List[str]],
    n: int,
    rng: random.Random,
    tried: Set[Tuple[Tuple[str, str], ...]],
    max_attempts: int = 10000,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    search_keys = list(space.keys())
    for _ in range(max_attempts):
        if len(out) >= n:
            break
        candidate = dict(fixed)
        candidate.update(choose_random(space, rng))
        pid = params_id(candidate, search_keys)
        if pid not in tried:
            tried.add(pid)
            out.append(candidate)
    return out


def expand_beam(
    beam: List[Dict[str, Any]],
    fixed: Dict[str, str],
    space: Dict[str, List[str]],
    expand_per_parent: int,
    rng: random.Random,
    tried: Set[Tuple[Tuple[str, str], ...]],
    random_fraction: float = 0.20,
    max_attempts: int = 10000,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    search_keys = list(space.keys())
    target = max(1, len(beam)) * max(1, expand_per_parent)
    attempts = 0

    while len(out) < target and attempts < max_attempts:
        attempts += 1
        if not beam or rng.random() < random_fraction:
            dynamic = choose_random(space, rng)
        else:
            parent_record = rng.choice(beam)
            parent = parent_record["params"]
            dynamic = mutate_one_or_more(parent, space, rng)

        candidate = dict(fixed)
        candidate.update(dynamic)
        pid = params_id(candidate, search_keys)
        if pid in tried:
            continue
        tried.add(pid)
        out.append(candidate)
    return out


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
        "pytorch_model.bin", "model.safetensors", "tf_model.h5",
        "flax_model.msgpack", "checkpoint_best.pt", "model.pt", "best.pt",
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


def run_candidate(
    params_in: Dict[str, Any],
    args: argparse.Namespace,
    generation: int,
    trial_in_generation: int,
    global_trial: int,
    direction: str,
    study_log: Path,
) -> Optional[Dict[str, Any]]:
    params = dict(params_in)
    apply_templates(params, args.save_template, args.out_template)

    run_base = Path(str(params.get("run_dir", params.get("save"))))
    run_base.mkdir(parents=True, exist_ok=True)
    wrapper_log = run_base / "wrapper.log"
    stdout_file = Path(str(params.get("log_file", run_base / "stdout.log")))
    history_file = Path(str(params.get("history_json", run_base / "history.json")))
    results_tsv = study_log.parent / "beam_results.tsv"
    cmd = build_command(args.python, args.mtat, args.command, params)
    printable = " ".join(shlex.quote(x) for x in cmd)

    print(f"\n=== Generation {generation}, trial {trial_in_generation} ===")
    print(printable)
    append_log(wrapper_log, f"START generation={generation} trial={trial_in_generation} global_trial={global_trial}: {printable}")

    event: Dict[str, Any] = {
        "timestamp": timestamp(),
        "generation": generation,
        "trial_in_generation": trial_in_generation,
        "global_trial": global_trial,
        "params": params,
        "search_params": params_in,
        "command": printable,
        "metric": args.metric,
        "direction": direction,
    }

    completed_record: Optional[Dict[str, Any]] = None

    if not args.force and model_exists(params, run_base):
        score = read_best_score(history_file, args.metric, direction)
        event.update({"status": "skipped_existing", "score": score})
        print(f"SKIP: model/checkpoint exists: {run_base}")
        if score is not None:
            completed_record = {"score": score, "params": params_in, "run_base": str(run_base)}
    elif args.execute:
        cmd_str = " ".join(shlex.quote(x) for x in cmd)

        tee_cmd = (
            f"set -o pipefail; "
            f"{cmd_str} 2>&1 | tee {shlex.quote(str(stdout_file))}"
        )

        proc = subprocess.run(
            ["bash", "-c", tee_cmd],
            env=os.environ.copy(),
        )

        returncode = proc.returncode

        if returncode != 0:
            reason = classify_failure(returncode, stdout_file)
            event.update({"status": "failed", "reason": reason})
            append_log(wrapper_log, f"FAILED: {reason}")
            print(f"FAILED: {reason}")
        elif not model_exists(params, run_base):
            reason = "command returned 0 but no expected model/checkpoint was found"
            event.update({"status": "failed", "reason": reason})
            append_log(wrapper_log, f"FAILED: {reason}")
            print(f"FAILED: {reason}")
        else:
            score = read_best_score(history_file, args.metric, direction)
            event.update({"status": "success", "score": score})
            append_log(wrapper_log, f"SUCCESS: score={score}")
            print(f"SUCCESS: {args.metric}={score}")
            if score is not None:
                completed_record = {
                    "score": score,
                    "params": params_in,
                    "run_base": str(run_base),
                }
    else:
        event.update({"status": "planned"})

    with open(study_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

    best_row, best_score = best_record_from_history(history_file, args.metric, direction)

    tsv_row = {
        "timestamp": event["timestamp"],
        "generation": generation,
        "trial_in_generation": trial_in_generation,
        "global_trial": global_trial,
        "status": event.get("status"),
        "reason": event.get("reason"),
        "optim_metric": args.metric,
        "optim_direction": direction,
        "score": event.get("score", best_score),
        "run_dir": str(run_base),
        "stdout_log": str(stdout_file),
        "wrapper_log": str(wrapper_log),
        "history_json": str(history_file),
        "command": printable,
    }

    tsv_row.update({f"param_{k}": v for k, v in params_in.items()})

    ESSENTIAL_METRICS = {
        "epoch",
        "step",
        "loss",
        "train_loss",
        "eval_loss",
        "val_loss",
        "nll",
        "val_nll",
        "bleu",
        "eval_bleu",
        "chrf",
        "eval_chrf",
        "ter",
        "eval_ter",
    }

    if best_row:
        tsv_row["best_epoch"] = best_row.get("epoch")
        for k in ESSENTIAL_METRICS:
            if k in best_row:
                tsv_row[f"metric_{k}"] = best_row[k]

    append_tsv_row(results_tsv, tsv_row)
    return completed_record


def main() -> None:
    ap = argparse.ArgumentParser(description="Dynamic beam-search hyperparameter wrapper for mtat.py")
    ap.add_argument("--mtat", default="mtat.py")
    ap.add_argument("--python", default="python")
    ap.add_argument("--command", required=True, choices=["finetune", "translate"])

    ap.add_argument("--set", action="append", default=[], help="Fixed mtat.py argument, key=value")
    ap.add_argument("--values", action="append", default=[], help="Search values, key=v1,v2,v3")
    ap.add_argument("--range", action="append", default=[], help="Search range, key=start:stop:step")

    ap.add_argument("--save-template", required=True, help="Template for run directories")
    ap.add_argument("--out-template", default=None)
    ap.add_argument("--study-dir", default=None, help="Directory for dynamic_beam_search.jsonl; default is parent of save-template")

    # Beam-search controls.
    ap.add_argument("--generations", type=int, default=5, help="Number of beam-search generations")
    ap.add_argument("--beam-width", type=int, default=5, help="Number of best configs kept after each generation")
    ap.add_argument("--expand-per-parent", type=int, default=3, help="Candidates generated per beam item")
    ap.add_argument("--initial-candidates", type=int, default=None, help="Generation-1 candidates; default beam_width * expand_per_parent")
    ap.add_argument("--random-fraction", type=float, default=0.20, help="Fraction of candidates sampled fully randomly after generation 1")

    # Backwards-compatible dynamic-wrapper aliases. They are accepted, but beam args are preferred.
    ap.add_argument("--trials", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--random-starts", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--top-k", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--mutation-rate", type=float, default=None, help=argparse.SUPPRESS)

    ap.add_argument("--metric", default="bleu", help="Metric to optimise; reads this key or eval_/val_ variants from history.json")
    ap.add_argument("--direction", choices=["auto", "min", "max"], default="auto")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--force", action="store_true", help="rerun even if an expected model/checkpoint already exists")

    args = ap.parse_args()

    if args.top_k is not None:
        args.beam_width = args.top_k
    if args.random_starts is not None and args.initial_candidates is None:
        args.initial_candidates = args.random_starts

    rng = random.Random(args.seed)
    direction = metric_direction(args.metric, "auto" if args.direction == "auto" else args.direction)

    fixed = dict(parse_kv(x) for x in args.set)
    space: Dict[str, List[str]] = {}
    for item in args.values:
        key, vals = parse_values(item)
        space[key] = vals
    for item in args.range:
        key, vals = parse_range(item)
        space[key] = vals

    if not space:
        raise ValueError("Beam search needs at least one --values or --range argument.")
    if args.beam_width <= 0:
        raise ValueError("--beam-width must be > 0")
    if args.expand_per_parent <= 0:
        raise ValueError("--expand-per-parent must be > 0")
    if args.generations <= 0:
        raise ValueError("--generations must be > 0")

    initial_n = args.initial_candidates or args.beam_width * args.expand_per_parent

    study_dir = Path(args.study_dir) if args.study_dir else Path(args.save_template.split("{")[0] or ".").parent
    study_dir.mkdir(parents=True, exist_ok=True)
    study_log = study_dir / "dynamic_beam_search.jsonl"

    completed: List[Dict[str, Any]] = []
    beam: List[Dict[str, Any]] = []
    tried: Set[Tuple[Tuple[str, str], ...]] = set()
    global_trial = 0

    for generation in range(1, args.generations + 1):
        if generation == 1 or not beam:
            candidates = initial_candidates(
                fixed=fixed,
                space=space,
                n=initial_n,
                rng=rng,
                tried=tried,
            )
        else:
            candidates = expand_beam(
                beam=beam,
                fixed=fixed,
                space=space,
                expand_per_parent=args.expand_per_parent,
                rng=rng,
                tried=tried,
                random_fraction=args.random_fraction,
            )

        if not candidates:
            print("No untried candidates left in the supplied search space.")
            break

        print(f"\n### Generation {generation}/{args.generations}: {len(candidates)} candidate(s) ###")

        for trial_in_generation, candidate in enumerate(candidates, start=1):
            global_trial += 1
            record = run_candidate(
                params_in=candidate,
                args=args,
                generation=generation,
                trial_in_generation=trial_in_generation,
                global_trial=global_trial,
                direction=direction,
                study_log=study_log,
            )
            if record is not None:
                completed.append(record)
                completed = rank_completed(completed, direction)
                best = completed[0]
                print(f"Current best: {args.metric}={best['score']} at {best['run_base']}")

        completed = rank_completed(completed, direction)
        beam = completed[: args.beam_width]

        print(f"\n--- Beam after generation {generation} ---")
        if not beam:
            print("No scored successful/skipped runs yet; next generation will sample randomly.")
        else:
            for i, item in enumerate(beam, start=1):
                print(f"{i}. {args.metric}={item['score']} at {item['run_base']}")

    print(f"\nFinished. Total attempted/planned trials: {global_trial}.")
    print(f"Study log: {study_log}")
    if completed:
        best = rank_completed(completed, direction)[0]
        print(f"Best observed {args.metric}: {best['score']} at {best['run_base']}")
    else:
        print("No successful scored run was observed.")


if __name__ == "__main__":
    main()
