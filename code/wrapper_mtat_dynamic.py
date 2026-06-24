#!/usr/bin/env python3
"""
wrapper_mtat_beam.py

Beam-search hyperparameter wrapper for mtat.py.

This script does not enumerate the full Cartesian product of --values/--range.
It performs beam search in hyperparameter space:

  1. evaluate an initial random population
  2. keep the best --beam-width configurations
  3. expand each beam item into local neighbours
  4. evaluate those children and keep the best beam again
  5. repeat for --generations

This is dependency-free and intentionally simple enough to explain in class. The
"beam" here is not the MT decoding beam; it is a beam over hyperparameter
configurations.
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

WRAPPER_ONLY_KEYS = {"log_file", "run_dir", "model_type"}


def normalize_key(key: str) -> str:
    return key.strip().lstrip("-").replace("-", "_")


def cli_key(key: str) -> str:
    return key.replace("_", "-")


def parse_kv(text: str) -> Tuple[str, str]:
    if "=" not in text:
        raise ValueError(f"--set expects key=value, got: {text}")
    key, value = text.split("=", 1)
    return normalize_key(key), value.strip()


def parse_values(text: str) -> Tuple[str, List[str]]:
    if "=" not in text:
        raise ValueError(f"--values expects key=v1,v2,v3, got: {text}")
    key, values = text.split("=", 1)
    return normalize_key(key), [v.strip() for v in values.split(",") if v.strip()]


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


def neighbour_values(values: List[str], current: str) -> List[str]:
    """Return local neighbours for numeric/discrete ordered values, else all alternatives."""
    if current not in values:
        return list(values)
    idx = values.index(current)
    numeric_values = [as_number(v) for v in values]
    if all(v is not None for v in numeric_values):
        neighbours: List[str] = []
        if idx > 0:
            neighbours.append(values[idx - 1])
        if idx + 1 < len(values):
            neighbours.append(values[idx + 1])
        return neighbours or [v for v in values if v != current]
    return [v for v in values if v != current]


def mutate_one_key(parent: Dict[str, Any], space: Dict[str, List[str]], key: str, rng: random.Random) -> Dict[str, str]:
    child = {k: str(parent[k]) for k in space.keys()}
    values = space[key]
    current = child.get(key, values[0])
    alternatives = neighbour_values(values, current)
    if alternatives:
        child[key] = rng.choice(alternatives)
    return child


def mutate_multiple_keys(
    parent: Dict[str, Any],
    space: Dict[str, List[str]],
    rng: random.Random,
    max_changes: int,
) -> Dict[str, str]:
    child = {k: str(parent[k]) for k in space.keys()}
    keys = list(space.keys())
    rng.shuffle(keys)
    n_changes = rng.randint(1, max(1, min(max_changes, len(keys))))
    for key in keys[:n_changes]:
        alternatives = neighbour_values(space[key], child[key])
        if alternatives:
            child[key] = rng.choice(alternatives)
    return child


def expand_parent(
    parent: Dict[str, Any],
    space: Dict[str, List[str]],
    rng: random.Random,
    expand_per_parent: int,
    max_changes: int,
    exploration_rate: float,
) -> List[Dict[str, str]]:
    """Generate neighbouring hyperparameter candidates around one parent."""
    children: List[Dict[str, str]] = []
    keys = list(space.keys())

    # Deterministic-style local moves: change one hyperparameter at a time.
    shuffled = keys[:]
    rng.shuffle(shuffled)
    for key in shuffled:
        children.append(mutate_one_key(parent, space, key, rng))
        if len(children) >= expand_per_parent:
            return children

    # If more children are requested, try multi-key local moves.
    while len(children) < expand_per_parent:
        if rng.random() < exploration_rate:
            children.append(choose_random(space, rng))
        else:
            children.append(mutate_multiple_keys(parent, space, rng, max_changes=max_changes))
    return children


def candidate_id(params: Dict[str, Any], search_keys: Iterable[str]) -> Tuple[Tuple[str, str], ...]:
    return params_id(params, search_keys)


def propose_initial_population(
    fixed: Dict[str, str],
    space: Dict[str, List[str]],
    population_size: int,
    rng: random.Random,
    tried: set,
    max_attempts: int = 2000,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    search_keys = list(space.keys())
    attempts = 0
    while len(candidates) < population_size and attempts < max_attempts:
        attempts += 1
        dyn = choose_random(space, rng)
        candidate = dict(fixed)
        candidate.update(dyn)
        cid = candidate_id(candidate, search_keys)
        if cid in tried:
            continue
        tried.add(cid)
        candidates.append(candidate)
    return candidates


def propose_beam_expansion(
    fixed: Dict[str, str],
    space: Dict[str, List[str]],
    beam: List[Dict[str, Any]],
    tried: set,
    rng: random.Random,
    expand_per_parent: int,
    max_changes: int,
    exploration_rate: float,
    max_candidates: int,
    max_attempts: int = 5000,
) -> List[Dict[str, Any]]:
    """Expand current beam into untried candidate configurations."""
    candidates: List[Dict[str, Any]] = []
    search_keys = list(space.keys())
    attempts = 0

    parents = beam[:]
    rng.shuffle(parents)

    while len(candidates) < max_candidates and attempts < max_attempts:
        attempts += 1
        if not parents:
            dyn = choose_random(space, rng)
        else:
            parent = parents[(attempts - 1) % len(parents)]["params"]
            child_options = expand_parent(
                parent=parent,
                space=space,
                rng=rng,
                expand_per_parent=expand_per_parent,
                max_changes=max_changes,
                exploration_rate=exploration_rate,
            )
            dyn = rng.choice(child_options)

        candidate = dict(fixed)
        candidate.update(dyn)
        cid = candidate_id(candidate, search_keys)
        if cid in tried:
            continue
        tried.add(cid)
        candidates.append(candidate)

    return candidates


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


def evaluate_candidate(
    args: argparse.Namespace,
    params: Dict[str, Any],
    candidate_core: Dict[str, Any],
    trial: int,
    generation: int,
    direction: str,
    study_log: Path,
) -> Tuple[str, Optional[float], Optional[str]]:
    """Run or plan one candidate. Returns status, score, run_base string."""
    apply_templates(params, args.save_template, args.out_template)

    run_base = Path(str(params.get("run_dir", params.get("save"))))
    run_base.mkdir(parents=True, exist_ok=True)
    wrapper_log = run_base / "wrapper.log"
    stdout_file = Path(str(params.get("log_file", run_base / "stdout.log")))
    history_file = Path(str(params.get("history_json", run_base / "history.json")))

    cmd = build_command(args.python, args.mtat, args.command, params)
    printable = " ".join(shlex.quote(x) for x in cmd)

    print(f"\n=== Generation {generation}, trial {trial} ===")
    print(printable)
    append_log(wrapper_log, f"START generation={generation} trial={trial}: {printable}")

    event: Dict[str, Any] = {
        "timestamp": timestamp(),
        "generation": generation,
        "trial": trial,
        "params": params,
        "command": printable,
        "metric": args.metric,
        "direction": direction,
    }

    status = "planned"
    score: Optional[float] = None

    if not args.force and model_exists(params, run_base):
        status = "skipped_existing"
        score = read_best_score(history_file, args.metric, direction)
        print(f"SKIP: model/checkpoint exists: {run_base}")
    elif args.execute:
        with open(stdout_file, "w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=os.environ.copy())

        if proc.returncode != 0:
            status = "failed"
            reason = classify_failure(proc.returncode, stdout_file)
            event.update({"status": status, "reason": reason})
            append_log(wrapper_log, f"FAILED: {reason}")
            print(f"FAILED: {reason}")
        elif not model_exists(params, run_base):
            status = "failed"
            reason = "command returned 0 but no expected model/checkpoint was found"
            event.update({"status": status, "reason": reason})
            append_log(wrapper_log, f"FAILED: {reason}")
            print(f"FAILED: {reason}")
        else:
            status = "success"
            score = read_best_score(history_file, args.metric, direction)
            append_log(wrapper_log, f"SUCCESS: score={score}")
            print(f"SUCCESS: {args.metric}={score}")
    else:
        status = "planned"

    event.update({"status": status, "score": score})
    with open(study_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

    return status, score, str(run_base)


def main() -> None:
    ap = argparse.ArgumentParser(description="Beam search over mtat.py hyperparameters")
    ap.add_argument("--mtat", default="mtat.py")
    ap.add_argument("--python", default="python")
    ap.add_argument("--command", required=True, choices=["finetune", "translate"])

    ap.add_argument("--set", action="append", default=[], help="Fixed mtat.py argument, key=value")
    ap.add_argument("--values", action="append", default=[], help="Search values, key=v1,v2,v3")
    ap.add_argument("--range", action="append", default=[], help="Search range, key=start:stop:step")

    ap.add_argument("--save-template", required=True, help="Template for run directories, e.g. runs/beam_bs{batch_size}_lr{lr}")
    ap.add_argument("--out-template", default=None)
    ap.add_argument("--study-dir", default=None, help="Directory for beam_search.jsonl; default is parent of save-template")

    ap.add_argument("--generations", type=int, default=5, help="Number of beam-search generations")
    ap.add_argument("--beam-width", type=int, default=5, help="Number of best configurations kept after each generation")
    ap.add_argument("--initial-population", type=int, default=None, help="Initial random candidates; default beam_width * expand_per_parent")
    ap.add_argument("--expand-per-parent", type=int, default=3, help="Number of children proposed per beam item")
    ap.add_argument("--max-changes", type=int, default=2, help="Maximum hyperparameters changed in stochastic mutations")
    ap.add_argument("--exploration-rate", type=float, default=0.15, help="Probability of random exploration during expansion")
    ap.add_argument("--max-trials", type=int, default=None, help="Optional global cap on evaluated/planned candidates")

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
        raise ValueError("Beam search needs at least one --values or --range argument.")
    if args.beam_width <= 0:
        raise ValueError("--beam-width must be > 0")
    if args.expand_per_parent <= 0:
        raise ValueError("--expand-per-parent must be > 0")

    initial_population = args.initial_population or (args.beam_width * args.expand_per_parent)
    study_dir = Path(args.study_dir) if args.study_dir else Path(args.save_template.split("{")[0] or ".").parent
    study_dir.mkdir(parents=True, exist_ok=True)
    study_log = study_dir / "beam_search.jsonl"

    tried = set()
    all_scored: List[Dict[str, Any]] = []
    beam: List[Dict[str, Any]] = []
    total_trials = 0
    successful = 0
    failed = 0
    skipped = 0
    planned = 0

    for generation in range(1, args.generations + 1):
        if generation == 1:
            candidates = propose_initial_population(
                fixed=fixed,
                space=space,
                population_size=initial_population,
                rng=rng,
                tried=tried,
            )
        else:
            candidates = propose_beam_expansion(
                fixed=fixed,
                space=space,
                beam=beam,
                tried=tried,
                rng=rng,
                expand_per_parent=args.expand_per_parent,
                max_changes=args.max_changes,
                exploration_rate=args.exploration_rate,
                max_candidates=args.beam_width * args.expand_per_parent,
            )

        if not candidates:
            print("No untried candidates left in the supplied search space.")
            break

        print(f"\n### Generation {generation}/{args.generations}: {len(candidates)} candidate(s) ###")

        for candidate in candidates:
            if args.max_trials is not None and total_trials >= args.max_trials:
                break
            total_trials += 1
            params = dict(candidate)
            status, score, run_base = evaluate_candidate(
                args=args,
                params=params,
                candidate_core=candidate,
                trial=total_trials,
                generation=generation,
                direction=direction,
                study_log=study_log,
            )
            if status == "success":
                successful += 1
            elif status == "failed":
                failed += 1
            elif status == "skipped_existing":
                skipped += 1
            elif status == "planned":
                planned += 1

            if score is not None:
                all_scored.append({"score": score, "params": candidate, "run_base": run_base, "generation": generation})
                all_scored.sort(key=lambda x: x["score"], reverse=(direction == "max"))
                beam = all_scored[: args.beam_width]
                best = beam[0]
                print(f"Current best: {args.metric}={best['score']} at {best['run_base']}")

        if args.max_trials is not None and total_trials >= args.max_trials:
            break

        if beam:
            print(f"\nBeam after generation {generation}:")
            for rank, item in enumerate(beam, start=1):
                compact = ", ".join(f"{k}={v}" for k, v in item["params"].items() if k in space)
                print(f"  {rank}. {args.metric}={item['score']} | {compact}")
        elif args.execute:
            print("No scored runs yet; next generation will continue random exploration.")

    print(f"\nFinished. Successful runs: {successful}. Failed runs: {failed}. Skipped runs: {skipped}. Planned-only runs: {planned}.")
    print(f"Study log: {study_log}")
    if beam:
        best = beam[0]
        print(f"Best observed {args.metric}: {best['score']} at {best['run_base']}")


if __name__ == "__main__":
    main()
