#!/usr/bin/env python3
"""Simulated graph-retrieval failure study on *real* Python function symbols.

This is a transparent deterministic proxy test, NOT a Gemma model evaluation.
"""
from __future__ import annotations
import argparse
import ast
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from datetime import datetime, timezone

RATES = (0.0, 0.10, 0.25, 0.50, 0.75)
POLICIES = ("graph_only", "unconditional_fallback", "confidence_gated")
SOURCE_EXCLUDE = {".git", ".venv", "venv", "__pycache__", "node_modules", ".isolated_vendor"}


def get_repo_commit(repo: Path) -> str:
    p = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                       text=True, capture_output=True, check=False)
    return p.stdout.strip() if p.returncode == 0 else "UNKNOWN"


def symbols_in_repository(repo: Path) -> tuple[list[dict], dict]:
    functions = []
    scanned = 0
    unparsable = 0
    for file in sorted(repo.rglob("*.py")):
        if any(part in SOURCE_EXCLUDE for part in file.relative_to(repo).parts):
            continue
        if file.stat().st_size > 2_000_000:
            continue
        scanned += 1
        try:
            text = file.read_text(encoding="utf-8")
            root = ast.parse(text, filename=str(file))
        except (SyntaxError, UnicodeError, OSError):
            unparsable += 1
            continue
        # A function name must be unique in the whole corpus for unambiguous
        # lexical symbol recovery. Method names may repeat; those are excluded.
        for node in ast.walk(root):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_") and len(node.name) >= 5:
                    functions.append({
                        "symbol": node.name,
                        "file": file.relative_to(repo).as_posix(),
                        "line": node.lineno,
                    })
    counts = collections.Counter(f["symbol"] for f in functions)
    unique = [v for v in functions if counts[v["symbol"]] == 1]
    return unique, {"files_parsed_or_attempted": scanned,
                    "unparsable_files": unparsable,
                    "function_definitions": len(functions),
                    "unique_symbols": len(unique)}


def wilson(success: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p = success / total
    d = 1 + z * z / total
    center = (p + z*z/(2*total))/d
    half = z/d * math.sqrt(p*(1-p)/total + z*z/(4*total*total))
    return [round(max(0.0, center-half), 4),
            round(min(1.0, center+half), 4)]


def run(repo: Path, seed: int, limit: int, repeats: int) -> dict:
    symbols, audit = symbols_in_repository(repo)
    if len(symbols) < 16:
        raise ValueError("At least 16 unique public function symbols are required")
    sample_size = min(len(symbols), limit)
    sampled = random.Random(seed).sample(symbols, sample_size)
    selected = sorted(sampled, key=lambda x: (x["file"], x["line"]))
    totals = []
    for rate in RATES:
        per_policy = {name: {"correct": 0, "calls": 0} for name in POLICIES}
        injected = 0
        misleading = 0
        for repeat in range(repeats):
            rng = random.Random(seed * 10_000 + int(rate*1000)*100 + repeat)
            for ix, task in enumerate(selected):
                failure = rng.random() < rate
                injected += int(failure)
                # Inject sometimes falsely confident but wrong graph replies.
                false_high = failure and rng.random() < 0.25
                misleading += int(false_high)
                graph_hit = not failure
                confidence = (0.85 if false_high else 0.15) if failure else 0.92
                # One graph lookup per task. Textual fallback resolves
                # uniquely indexed symbols (ground truth by AST construction).
                per_policy["graph_only"]["correct"] += int(graph_hit)
                per_policy["graph_only"]["calls"] += 1
                per_policy["unconditional_fallback"]["correct"] += 1
                per_policy["unconditional_fallback"]["calls"] += 2
                use_fallback = confidence < 0.70
                per_policy["confidence_gated"]["correct"] += int(
                    graph_hit or use_fallback)
                per_policy["confidence_gated"]["calls"] += 1 + int(use_fallback)
        n = len(selected) * repeats
        row = {"injected_graph_failure_rate": rate,
               "tasks": n,
               "injected_failures": injected,
               "misleading_high_confidence": misleading,
               "policies": {}}
        for name, values in per_policy.items():
            row["policies"][name] = {
                "correct": values["correct"],
                "localization_accuracy": round(values["correct"]/n, 4),
                "wilson_95pct": wilson(values["correct"], n),
                "mean_retrieval_calls": round(values["calls"]/n, 4),
            }
        totals.append(row)
    return {
        "study": "Confidence-gated code localization under injected graph failures",
        "experiment_type": "SIMULATED retrieval faults over real AST symbols",
        "not": ["Gemma model inference", "real Kaggle graph retrieval",
                "patch acceptance", "SWE-bench score"],
        "repository": str(repo.resolve()),
        "source_commit": get_repo_commit(repo),
        "seed": seed, "symbols_per_replicate": len(selected),
        "repeats_per_rate": repeats,
        "total_decisions": sum(x["tasks"] for x in totals),
        "methods": {
            "graph_only": "single graph result, no fallback",
            "unconditional_fallback": "graph then exact unique symbol lookup",
            "confidence_gated": "graph then exact symbol lookup if confidence < 0.70",
            "faults": "wrong symbol returned with 25% misleading high confidence",
        },
        "corpus_audit": audit,
        "selected_ground_truth": selected,
        "results_by_fault_rate": totals,
        "limitations": [
            "Retrieval failures and confidence are simulated, not a trained graph model.",
            "Exact identifier string is supplied in each issue in this proxy test.",
            "Unique symbol filter makes lexical recovery easier than realistic issues.",
            "Repeated scenarios share tasks and are not statistically independent.",
            "No model runs, human patch review, hidden tests, or official score.",
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261101)
    p.add_argument("--limit", type=int, default=48)
    p.add_argument("--repeats", type=int, default=10)
    args = p.parse_args()
    if not args.repo.is_dir() or args.repeats < 1 or args.limit < 16:
        p.error("Provide a real repository directory; limit >=16, repeats >=1")
    result = run(args.repo, args.seed, args.limit, args.repeats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for row in result["results_by_fault_rate"]:
        print("rate", row["injected_graph_failure_rate"],
              "graph", row["policies"]["graph_only"]["localization_accuracy"],
              "gated", row["policies"]["confidence_gated"]["localization_accuracy"],
              "gated_calls", row["policies"]["confidence_gated"]["mean_retrieval_calls"],
              "unconditional", row["policies"]["unconditional_fallback"]["localization_accuracy"])
    print("RESULT", str(args.out), "TASKS", result["total_decisions"],
          "GROUND_TRUTH", len(result["selected_ground_truth"]))


if __name__ == "__main__":
    main()
