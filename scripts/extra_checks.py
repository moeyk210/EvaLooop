"""Secondary numbers quoted in the report that compute_metrics.py does not produce.

Usage (from the repo root; reads JSON/CSV only, executes no generated code):
    python scripts/extra_checks.py [--results results]

Prints:
  1. Output volume per run: number of cycles executed and characters of raw model output
     (basis for the cost estimate).
  2. Rename evidence: pseudocode outputs in which none of the FUNCTION names occurs in the
     original task prompt (the prompt contains the expected function name via the assert line).
  3. Determinism at temperature 0: cycle-1 code is produced by the identical prompt in
     every configuration, so for each model we compare it across runs (identical outputs,
     and tasks whose cycle-1 pass/fail differs).
  4. Failure-label tally from results/inspection/labels.csv (manual labels).
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

SUFFIX = "_code_generation_summarization_results.json"
RUNS = ["std_31lite", "pseudo_31lite", "pseudonamed_31lite",
        "std_35lite", "pseudo_35lite", "pseudonamed_35lite"]
FUNC = re.compile(r"FUNCTION\s+(\w+)", re.IGNORECASE)


def load(results, prefix):
    data = json.loads((results / f"{prefix}{SUFFIX}").read_text(encoding="utf-8"))[0]
    return {p["task_id"]: p for p in data["prompt_results"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    results = parser.parse_args().results
    runs = {prefix: load(results, prefix) for prefix in RUNS}

    print("1. Output volume")
    for prefix, tasks in runs.items():
        cycles = [c for t in tasks.values() for c in t["cycles"]]
        chars = sum(len(c.get(k) or "") for c in cycles for k in ("task_1_raw_output", "task_2_raw_output"))
        errors = sum(1 for t in tasks.values() if t.get("api_error"))
        print(f"   {prefix:<20} cycles={len(cycles):>4} output_chars={chars:>8,} api_error_loops={errors}")

    print("\n2. Pseudocode outputs in which no FUNCTION name occurs in the original prompt")
    for prefix in (p for p in RUNS if p.startswith("pseudo")):
        total = renamed = 0
        for task in runs[prefix].values():
            for cycle in task["cycles"]:
                if "task_2_output" not in cycle:
                    continue
                total += 1
                names = FUNC.findall(cycle["task_2_output"])
                renamed += bool(names) and not any(name in task["initial_prompt"] for name in names)
        print(f"   {prefix:<20} {renamed}/{total}")

    print("\n3. Cycle-1 code identical across runs (same prompt in every configuration)")
    for tag in ("31lite", "35lite"):
        base = runs[f"std_{tag}"]
        for other in (f"pseudo_{tag}", f"pseudonamed_{tag}"):
            same = flips = 0
            for task_id, task in base.items():
                a, b = task["cycles"][0], runs[other][task_id]["cycles"][0]
                same += a["task_1_output"] == b["task_1_output"]
                flips += (task["successful_cycles"] > 0) != (runs[other][task_id]["successful_cycles"] > 0)
            print(f"   std_{tag} vs {other:<20} identical={same}/{len(base)} pass/fail flips={flips}")

    labels = results / "inspection" / "labels.csv"
    if labels.exists():
        print("\n4. Failure labels (results/inspection/labels.csv)")
        with labels.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for (model, loop, label), n in sorted(Counter((r["model"], r["loop"], r["label"]) for r in rows).items()):
            print(f"   {model:<22} {loop:<17} {label:<24} {n}")
        print(f"   total cases: {len(rows)}")


if __name__ == "__main__":
    main()
