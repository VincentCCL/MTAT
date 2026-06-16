#!/usr/bin/env python3

import argparse
import itertools
import shlex
import subprocess
from pathlib import Path


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

        config_file = run_base / "config.yaml"
        history_file = run_base / "history.json"

        if config_file.exists() and history_file.exists():
            print(f"SKIPPING completed run: {run_base}")
            continue


        cmd = [args.python, args.mtat, args.command]

        for key, value in params.items():
            if key in {"log_file","run_dir"}:
                continue
            add_arg(cmd, key, value)

        printable = " ".join(shlex.quote(x) for x in cmd)
        print(printable)

        if args.execute:
            if "save" in params:
                Path(params.get("run_dir", params["save"])).mkdir(parents=True, exist_ok=True)

            if "log_file" in params:
                with open(params["log_file"], "w", encoding="utf-8") as log:
                    shell_cmd = (
                        " ".join(shlex.quote(x) for x in cmd)
                        + " 2>&1 | tee "
                        + shlex.quote(params["log_file"])
                    )

                    proc = subprocess.run(
                        shell_cmd,
                        shell=True,
                        executable="/bin/bash",
                    )

                    if proc.returncode != 0:
                        failed.append((printable, params.get("log_file")))
                        print(f"\nFAILED: {printable}")
                        print(f"See log: {params.get('log_file')}")
                        continue
                    
            else:
                result = subprocess.run(cmd)
                if result.returncode != 0:
                    failed.append((printable, None))
                    print(f"\nFAILED: {printable}")
                    continue

            successful += 1

        print(f"\nFinished. Successful runs: {successful}. Failed runs: {len(failed)}.")

        for cmd_text, log_file in failed:
            print(f"\nFAILED: {cmd_text}")
            if log_file:
                print(f"Log: {log_file}")
        

if __name__ == "__main__":
    main()
