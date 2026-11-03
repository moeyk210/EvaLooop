"""Compute robustness metrics, summary.csv and the survival plot for the assignment runs.

Usage (from the repo root; reads JSON only, executes no generated code):
    python scripts/compute_metrics.py [--results results] [--package <data package dir>]

Inputs: results/<prefix>_code_generation_summarization_results.json for every run in RUNS.
Outputs (written to the --results directory):
    summary.csv            required format: model, configuration, n_tasks, success_rate, asl_score
    summary_detailed.csv   same rows plus ASL 95% CI, robustness, pass@10, pass drop, rank
    loop_comparison.csv    per model: each pseudocode configuration minus the standard loop,
                           with paired bootstrap CIs
    survival.png / .pdf    fraction of tasks whose code still passes at each cycle
    subset_check.csv       (with --package) package models scored on all 378 tasks vs our subset

Metric definitions follow the reference data package (README "leaderboard.json"),
generalised from T=378 to the number of tasks T in each run, with M = max_cycles = 10:
    success rate  = pass@1 / T
    ASL           = sum_i i^2 * n_i / (T * M)          (package "aslScore"; evaloop ASL_pow)
    robustness    = sum_i i^2 * n_i / (pass@1 * M)     (package "robustnessScore"; ASL_bias)
    pass drop     = pass@M - pass@1                     (package "passDrop", a task count)
where n_i is the number of tasks that sustained exactly i cycles. The judge-weighted
semantic ASL is not computed (no LLM judge was used).
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaloop.analysis.metrics import asl_bias, asl_pow, pass_at_k  # noqa: E402

MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]
MODEL_TAGS = {"gemini-3.1-flash-lite": "31lite", "gemini-3.5-flash-lite": "35lite"}

# configuration -> (results file prefix, --loop value recorded in the file, plot label)
STANDARD = "standard-mbpp-plus"
CONFIGS = {
    STANDARD: ("std", "summarization_detailed", "standard (summary) loop"),
    "pseudocode-loop": ("pseudo", "pseudocode", "pseudocode loop"),
    "pseudocode-loop-keep-names": ("pseudonamed", "pseudocode_named", "pseudocode loop, names kept"),
}
RESULT_SUFFIX = "_code_generation_summarization_results.json"
N_BOOT = 10_000
BOOT_SEED = 12345
DEFAULT_PACKAGE = Path("tai4se-HW4/assignment4-data/assignment2-data")

# Validated categorical palette (identity = configuration); line style + marker repeat it.
CONFIG_STYLES = {
    STANDARD: ("#2a78d6", "-", "o"),
    "pseudocode-loop": ("#eb6834", "--", "s"),
    "pseudocode-loop-keep-names": ("#1baf7a", ":", "^"),
}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_run(path: Path, expected_loop: str):
    """Return (max_cycles, {task_id: successful_cycles}, api_error_loops) for one results file."""
    result = json.loads(path.read_text())[0]
    if result.get("loop") not in (None, expected_loop):
        raise SystemExit(f"{path.name}: file says loop={result.get('loop')}, expected {expected_loop}")
    tasks = {p["task_id"]: p["successful_cycles"] for p in result["prompt_results"]}
    api_errors = sum(1 for p in result["prompt_results"] if p.get("api_error"))
    return result["max_cycles"], tasks, api_errors


def metrics(cycles, max_cycles):
    n = len(cycles)
    p1, pm = pass_at_k(cycles, 1), pass_at_k(cycles, max_cycles)
    return {
        "n_tasks": n,
        "success_rate": p1 / n,
        "asl_score": asl_pow(cycles, max_cycles),
        "pass1_count": p1,
        "robustness": asl_bias(cycles, max_cycles),
        "mean_sustained_cycles": sum(cycles) / n,
        f"pass{max_cycles}_count": pm,
        "pass_drop_count": pm - p1,
        "pass_drop_pct": 100 * (pm - p1) / n,
    }


def percentile_ci(values, alpha=0.05):
    values = sorted(values)
    lo = values[int(alpha / 2 * len(values))]
    hi = values[min(int((1 - alpha / 2) * len(values)), len(values) - 1)]
    return lo, hi


def bootstrap(task_ids, stat, rng):
    """Resample tasks with replacement; stat(sample_ids) -> float."""
    return [stat([rng.choice(task_ids) for _ in task_ids]) for _ in range(N_BOOT)]


def survival_plot(runs, max_cycles, out_base: Path):
    """One panel per model; color/line style/marker = configuration."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.2, 3.6), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    xs = list(range(0, max_cycles + 1))
    for ax, model in zip(axes, MODELS):
        ax.set_facecolor(SURFACE)
        end_values = set()
        for config, (color, style, marker) in CONFIG_STYLES.items():
            cycles = list(runs[(model, config)].values())
            # x = 0 is the start (all tasks alive); x = k is pass@k / T.
            ys = [100.0] + [100 * pass_at_k(cycles, k) / len(cycles) for k in xs[1:]]
            ax.plot(xs, ys, linestyle=style, marker=marker, markersize=4, linewidth=2, color=color,
                    label=CONFIGS[config][2], markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3)
            end_values.add(round(ys[-1]))
        for value in end_values:  # one label per distinct end value (coinciding lines share it)
            ax.annotate(f"{value}%", (xs[-1], value), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=INK_2)
        ax.set_title(model, loc="left", color=INK, fontsize=9)
        ax.set_xlim(0, max_cycles + 1.2)
        ax.set_ylim(50, 100)  # line chart: axis need not start at 0; all values lie in 60-100
        ax.set_xticks(xs[::2])
        ax.set_xlabel("Cycle k", color=INK_2)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, length=0)
    axes[0].set_ylabel("Tasks still passing (%)", color=INK_2)
    fig.suptitle("Loop survival on 100 MBPP+ tasks", x=0.01, ha="left", color=INK, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8,
               labelcolor=INK, handlelength=3.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out_base.with_suffix(f".{ext}"), facecolor=SURFACE)
    plt.close(fig)


def subset_check(package: Path, subset_ids, max_cycles, out_path: Path):
    """Score every package model on all its tasks and on our subset; check formula + representativeness."""
    leaderboard = {m["model_id"]: m for m in json.loads((package / "leaderboard.json").read_text())["models"]}
    rows = []
    for f in sorted((package / "loops").glob("*.json")):
        data = json.loads(f.read_text())
        cycles = {t["task_id"]: t["successful_cycles"] for t in data["tasks"]}
        full = list(cycles.values())
        sub = [cycles[t] for t in subset_ids if t in cycles]
        ref = leaderboard.get(data["model_id"], {})
        rows.append({
            "model_id": data["model_id"],
            "asl_full_ours": round(asl_pow(full, max_cycles), 3),
            "asl_full_package": ref.get("aslScore"),
            "success_full_ours": round(pass_at_k(full, 1) / len(full), 3),
            "success_full_package": ref.get("successRate"),
            "asl_subset": round(asl_pow(sub, max_cycles), 3),
            "success_subset": round(pass_at_k(sub, 1) / len(sub), 3),
        })
    write_csv(out_path, rows)

    mismatches = [r["model_id"] for r in rows if r["asl_full_package"] is not None
                  and abs(r["asl_full_ours"] - r["asl_full_package"]) > 0.0015]

    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        r = [0.0] * len(values)
        for pos, i in enumerate(order):
            r[i] = pos
        return r

    def spearman(a, b):
        ra, rb = ranks(a), ranks(b)
        ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
        cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
        return cov / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5

    full_asl = [r["asl_full_ours"] for r in rows]
    sub_asl = [r["asl_subset"] for r in rows]
    mean_diff = sum(s - f for s, f in zip(sub_asl, full_asl)) / len(rows)
    return {"models": len(rows), "formula_mismatches": mismatches,
            "spearman_full_vs_subset": spearman(full_asl, sub_asl), "mean_asl_subset_minus_full": mean_diff}


def write_csv(path, rows, digits=4):
    def fmt(v):
        return round(v, digits) if isinstance(v, float) else v
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: fmt(v) for k, v in r.items()} for r in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()

    runs, max_cycles, detailed = {}, None, []
    for model in MODELS:
        for config, (prefix, loop, _) in CONFIGS.items():
            name = f"{prefix}_{MODEL_TAGS[model]}{RESULT_SUFFIX}"
            m, tasks, api_errors = load_run(args.results / name, loop)
            if max_cycles not in (None, m):
                raise SystemExit("Runs use different max_cycles")
            max_cycles = m
            runs[(model, config)] = tasks
            detailed.append({"model": model, "configuration": config, **metrics(list(tasks.values()), m),
                             "max_cycles": m, "api_error_loops": api_errors, "results_file": name})

    task_sets = {frozenset(t) for t in runs.values()}
    if len(task_sets) != 1:
        raise SystemExit("Runs do not cover the same task ids; paired comparison impossible")
    task_ids = sorted(next(iter(task_sets)))

    # Bootstrap CIs (tasks resampled with a fixed seed).
    rng = random.Random(BOOT_SEED)
    for row in detailed:
        tasks = runs[(row["model"], row["configuration"])]
        boot = bootstrap(task_ids, lambda ids: asl_pow([tasks[t] for t in ids], max_cycles), rng)
        row["asl_ci95_low"], row["asl_ci95_high"] = percentile_ci(boot)

    # Rank of each model within a configuration (1 = most robust by ASL).
    for config in CONFIGS:
        rows = sorted((r for r in detailed if r["configuration"] == config), key=lambda r: -r["asl_score"])
        for rank, row in enumerate(rows, 1):
            row["rank_within_configuration"] = rank

    comparison = []
    for model in MODELS:
        std = runs[(model, STANDARD)]
        for config in CONFIGS:
            if config == STANDARD:
                continue
            ext = runs[(model, config)]

            def delta(ids, key, std=std, ext=ext):
                a, b = [std[t] for t in ids], [ext[t] for t in ids]
                if key == "asl":
                    return asl_pow(b, max_cycles) - asl_pow(a, max_cycles)
                return (pass_at_k(b, 1) - pass_at_k(a, 1)) / len(ids)

            row = {"model": model, "configuration": config, "baseline": STANDARD}
            for key in ("asl", "success_rate"):
                lo, hi = percentile_ci(bootstrap(task_ids, lambda ids: delta(ids, key), rng))
                row.update({f"delta_{key}": delta(task_ids, key),
                            f"delta_{key}_ci95_low": lo, f"delta_{key}_ci95_high": hi})
            row["tasks_worse"] = sum(ext[t] < std[t] for t in task_ids)
            row["tasks_better"] = sum(ext[t] > std[t] for t in task_ids)
            row["tasks_equal"] = sum(ext[t] == std[t] for t in task_ids)
            comparison.append(row)

    # Required format, fixed decimals as in the assignment's example (0.70, 3.940).
    summary = [{"model": r["model"], "configuration": r["configuration"], "n_tasks": r["n_tasks"],
                "success_rate": f"{r['success_rate']:.2f}", "asl_score": f"{r['asl_score']:.3f}"}
               for r in detailed]
    write_csv(args.results / "summary.csv", summary)
    write_csv(args.results / "summary_detailed.csv", detailed)
    write_csv(args.results / "loop_comparison.csv", comparison)
    survival_plot(runs, max_cycles, args.results / "survival")

    print("summary_detailed.csv")
    for r in detailed:
        print(f"  {r['model']:<22} {r['configuration']:<27} success={r['success_rate']:.2f} "
              f"ASL={r['asl_score']:.3f} [{r['asl_ci95_low']:.2f}, {r['asl_ci95_high']:.2f}] "
              f"robustness={r['robustness']:.3f} passDrop={r['pass_drop_count']} "
              f"rank={r['rank_within_configuration']} api_errors={r['api_error_loops']}")
    print("loop_comparison.csv (configuration - standard)")
    for r in comparison:
        print(f"  {r['model']:<22} {r['configuration']:<27} dASL={r['delta_asl']:+.3f} "
              f"[{r['delta_asl_ci95_low']:+.2f}, {r['delta_asl_ci95_high']:+.2f}]  "
              f"dSuccess={r['delta_success_rate']:+.2f}  "
              f"worse/better/equal={r['tasks_worse']}/{r['tasks_better']}/{r['tasks_equal']}")

    if (args.package / "leaderboard.json").exists():
        check = subset_check(args.package, task_ids, max_cycles, args.results / "subset_check.csv")
        print(f"subset_check.csv: {check['models']} package models; formula mismatches vs leaderboard: "
              f"{check['formula_mismatches'] or 'none'}; Spearman(full 378, our 100) = "
              f"{check['spearman_full_vs_subset']:.3f}; mean ASL(subset) - ASL(full) = "
              f"{check['mean_asl_subset_minus_full']:+.3f}")
    else:
        print(f"(package not found at {args.package}; skipped subset_check.csv)")


if __name__ == "__main__":
    main()
