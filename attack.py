#!/usr/bin/env python3
"""UniBreak batch jailbreak driver.

A thin command-line shell around :class:`GA_based_methods.Attack_manager.attacker_manager`,
which is the implementation released with the paper.  It reads a CSV of harmful
goals and attacks them one by one, appending a result row per goal.

Example
-------
    uv run python attack.py \
        --model-path /path/to/Qwen2.5-3B-Instruct \
        --model-type qwen \
        --data dependency/data/advbench/harmful_behaviors.csv \
        --output results/qwen3b \
        --iters 50 --per-length 10 --population-size 10
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch

from GA_based_methods.Attack_manager import attacker_manager
from GA_based_methods.Utils import get_attack_configs

RESULT_FIELDS = [
    "id",
    "goal",
    "target",
    "perturbation",
    "success",
    "generations",
    "score",
    "library_replay",
    "seconds",
    "response",
]

GOAL_HEADERS = ("goal", "question", "behavior", "prompt", "instruction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UniBreak: evolutionary token-level jailbreaking (batch mode)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument("--model-path", required=True, help="local checkpoint directory")
    data.add_argument(
        "--data",
        default="dependency/data/advbench/harmful_behaviors.csv",
        help="CSV with a 'goal' column (and optional 'target')",
    )
    data.add_argument("--output", default="results/run", help="output directory")
    data.add_argument("--limit", type=int, default=0, help="attack only the first N goals (0 = all)")
    data.add_argument("--start", type=int, default=0, help="skip the first N goals")
    data.add_argument("--resume", action="store_true", help="skip ids already in results.csv")

    attack = parser.add_argument_group("attack")
    attack.add_argument("--config", default="configs/attack_conf.yaml", help="attack configuration template")
    attack.add_argument("--model-type", default="llama-2", help="fastchat conversation template")
    attack.add_argument("--device", default="cuda:0")
    attack.add_argument("--iters", type=int, default=50)
    attack.add_argument("--per-length", type=int, default=10)
    attack.add_argument("--population-size", type=int, default=10)
    attack.add_argument("--crossover-num", type=int, default=5)
    attack.add_argument("--mutation-rate", type=float, default=0.3)
    attack.add_argument("--attack-type", choices=("suffix", "prefix", "insert"), default="suffix")
    attack.add_argument("--prompt", default="You are a helpful assistant.")

    library = parser.add_argument_group("perturbation library")
    library.add_argument("--library", default=None, help="path to the .pkl library (default: <output>/library.pkl)")
    library.add_argument("--no-library", action="store_true", help="disable the reuse library")

    parser.add_argument("--seed", type=int, default=20)
    return parser.parse_args()


def read_goals(path: str) -> list[tuple[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]
    if not rows:
        return []

    header = [cell.strip().lower() for cell in rows[0]]
    if header[0] in GOAL_HEADERS or "goal" in header:
        goal_index = header.index("goal") if "goal" in header else 0
        target_index = header.index("target") if "target" in header else None
        body = rows[1:]
    else:
        goal_index, target_index = 0, (1 if len(rows[0]) > 1 else None)
        body = rows

    goals = []
    for row in body:
        if len(row) <= goal_index or not row[goal_index].strip():
            continue
        target = row[target_index].strip() if target_index is not None and len(row) > target_index else ""
        goals.append((row[goal_index].strip(), target))
    return goals


def load_finished_ids(path: str) -> set[int]:
    if not os.path.exists(path):
        return set()
    finished = set()
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                finished.add(int(row["id"]))
            except (KeyError, TypeError, ValueError):
                continue
    return finished


def summarise(results_path: str, args: argparse.Namespace, elapsed: float) -> dict:
    rows = []
    if os.path.exists(results_path):
        with open(results_path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    successes = sum(1 for row in rows if str(row.get("success")).lower() == "true")
    generations = [int(row["generations"]) for row in rows if str(row.get("generations", "")).lstrip("-").isdigit()]
    return {
        "model_path": args.model_path,
        "model_type": args.model_type,
        "attack_type": args.attack_type,
        "iters": args.iters,
        "per_length": args.per_length,
        "population_size": args.population_size,
        "crossover_num": args.crossover_num,
        "mutation_rate": args.mutation_rate,
        "total_goals": len(rows),
        "successes": successes,
        "attack_success_rate": round(successes / len(rows), 4) if rows else 0.0,
        "average_generations": round(sum(generations) / len(generations), 2) if generations else 0.0,
        "wall_clock_seconds": round(elapsed, 1),
    }


def main() -> int:
    args = parse_args()

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.csv")
    log_path = os.path.join(output_dir, "run.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("unibreak")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    goals = read_goals(args.data)[args.start:]
    if args.limit:
        goals = goals[: args.limit]
    if not goals:
        log.error("no goals found in %s", args.data)
        return 1

    library_path = args.library or os.path.join(output_dir, "library.pkl")

    base_config = get_attack_configs(args.config)
    base_config.update(
        {
            "model_path": args.model_path,
            "model_type": args.model_type,
            "device": args.device,
            "prompt": args.prompt,
            "iters": args.iters,
            "per_length": args.per_length,
            "population_size": args.population_size,
            "crossover_num": args.crossover_num,
            "mutation_rate": args.mutation_rate,
            "attack_type": args.attack_type,
            "do_sample": False,
            "tensorboard_log": False,
            "adv_library": library_path,
            "enable_library": not args.no_library,
        }
    )

    log.info("model      : %s (%s)", args.model_path, args.model_type)
    log.info("goals      : %d from %s", len(goals), args.data)
    log.info("search     : iters=%d len=%d pop=%d type=%s", args.iters, args.per_length,
             args.population_size, args.attack_type)
    log.info("library    : %s", library_path)

    attacker = attacker_manager()
    finished = load_finished_ids(results_path) if args.resume else set()
    file_exists = os.path.exists(results_path)

    started = time.time()
    with open(results_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()

        for index, (goal, target) in enumerate(goals, start=args.start):
            if index in finished:
                log.info("[%d] skipped (already in results.csv)", index)
                continue

            config = dict(base_config)
            config["input"] = goal
            config["except_output"] = target or "Sure, here is"

            log.info("[%d] goal: %s", index, goal[:100])
            begin = time.time()
            # The core prints per-candidate diagnostics; keep the console clean
            # by diverting them to the run log.
            with open(log_path, "a", encoding="utf-8") as raw_log, contextlib.redirect_stdout(raw_log):
                success, score, perturbation, response, from_library, generations = attacker.attack(
                    configs=config
                )

            row = {
                "id": index,
                "goal": goal,
                "target": target,
                "perturbation": perturbation,
                "success": bool(success),
                "generations": generations,
                "score": score,
                "library_replay": bool(from_library),
                "seconds": round(time.time() - begin, 2),
                "response": response,
            }
            writer.writerow(row)
            handle.flush()

            log.info("[%d] %s | generations=%s | %.1fs%s", index, "SUCCESS" if success else "fail",
                     generations, row["seconds"], " (replay check passed)" if from_library else "")
            log.info("[%d] perturbation: %s", index, str(perturbation)[:160].replace("\n", " "))
            log.info("[%d] response    : %s", index, str(response)[:160].replace("\n", " "))

    summary = summarise(results_path, args, time.time() - started)
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)

    log.info("done: %d/%d goals jailbroken (ASR %.2f%%)", summary["successes"],
             summary["total_goals"], 100 * summary["attack_success_rate"])
    log.info("results written to %s", results_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
