"""Extract and diagnose loops that failed after cycle 1, for manual inspection.

Executes model-generated code, so run it ONLY inside the offline sandbox:
    ./run_sandbox.sh --python scripts/diagnose_failures.py

Selection: every loop (in every results file) whose code passed cycle 1 and then
failed at some cycle c >= 2. Cycle-1 failures are excluded because cycle 1 is
the same prompt in both loop types and says nothing about the intermediate
representation.

For each case it records the original task, the last passing code (cycle c-1),
the summary/pseudocode that seeded cycle c, the failing code, and a diagnosis
obtained by re-running the failing code on the EvalPlus inputs (base first,
then plus): the official evalplus status plus the first failing input with
expected vs. actual output, the exception, or a timeout.

Outputs (in results/inspection/):
    cases.json    full machine-readable records
    failures.md   side-by-side view for reading, one section per case
    labels.csv    one row per case with empty label/notes columns to fill in
"""

import copy
import csv
import json
import math
import multiprocessing as mp
import signal
import sys
from pathlib import Path

sys.path.insert(0, "/opt/evaloop")
from evaloop.evaluation.py_testing import load_mbpp_dataset  # noqa: E402
from evalplus.eval import untrusted_check  # noqa: E402

RESULTS = Path("results")
OUT = RESULTS / "inspection"
RUNS = [  # (file prefix, model, loop)
    ("pseudo_31lite", "gemini-3.1-flash-lite", "pseudocode"),
    ("pseudo_35lite", "gemini-3.5-flash-lite", "pseudocode"),
    ("std_31lite", "gemini-3.1-flash-lite", "summary"),
    ("std_35lite", "gemini-3.5-flash-lite", "summary"),
    ("pseudonamed_31lite", "gemini-3.1-flash-lite", "pseudocode_named"),
    ("pseudonamed_35lite", "gemini-3.5-flash-lite", "pseudocode_named"),
]
CASE_PREFIX = {"pseudocode": "P", "pseudocode_named": "N", "summary": "S"}
SUFFIX = "_code_generation_summarization_results.json"
PER_INPUT_TIMEOUT = 5
CASE_TIMEOUT = 120
REPR_LIMIT = 300


def short(value):
    text = repr(value)
    return text if len(text) <= REPR_LIMIT else text[:REPR_LIMIT] + f"... ({len(text)} chars)"


def close_enough(out, exp, atol):
    if isinstance(exp, float) or isinstance(out, float):
        try:
            return math.isclose(out, exp, rel_tol=1e-6, abs_tol=atol) or (math.isnan(out) and math.isnan(exp))
        except TypeError:
            return False
    if isinstance(exp, (list, tuple)) and isinstance(out, (list, tuple)) and type(exp) is type(out):
        return len(out) == len(exp) and all(close_enough(o, e, atol) for o, e in zip(out, exp))
    return out == exp


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


class _Reporter:
    """Sends exactly one result over a pipe (no background thread, unlike mp.Queue)."""

    def __init__(self, conn):
        self.conn = conn

    def put(self, item):
        self.conn.send(item)


def _worker(code, entry_point, suites, atol, conn):
    """Child process: exec the code, run inputs in order, report the first failure.

    Memory is bounded by the sandbox container's limit; the child is killed on timeout.
    """
    queue = _Reporter(conn)
    signal.signal(signal.SIGALRM, _alarm)
    namespace = {"__name__": "__candidate__"}
    try:
        signal.alarm(PER_INPUT_TIMEOUT)
        exec(compile(code, "<candidate>", "exec"), namespace)
        signal.alarm(0)
    except _Timeout:
        queue.put({"kind": "timeout_at_import"})
        return
    except BaseException as e:  # noqa: BLE001 - report anything the module raises
        queue.put({"kind": "error_at_import", "error": short(e)})
        return
    fn = namespace.get(entry_point)
    if callable(fn):
        queue.put(_run_suites(fn, suites, atol) or {"kind": "all_inputs_pass"})
        return

    # Entry point missing: was the function merely renamed? Try every function the
    # candidate itself defines (not imports) in place of the expected name.
    defined = sorted(k for k, v in namespace.items()
                     if callable(v) and getattr(v, "__module__", None) == "__candidate__")
    result = {"kind": "missing_function", "defined": defined, "renamed_to": None,
              "renamed_logic": None}
    for name in defined:
        failure = _run_suites(namespace[name], suites, atol)
        if failure is None:
            result.update(renamed_to=name, renamed_logic="passes_all_tests")
            break
        if result["renamed_logic"] is None:
            result.update(renamed_to=name, renamed_logic="fails", renamed_failure=failure)
    queue.put(result)


def _run_suites(fn, suites, atol):
    """Run base then plus inputs; return a dict describing the first failure, or None."""
    for suite, inputs, expected in suites:
        for index, (inp, exp) in enumerate(zip(inputs, expected)):
            try:
                signal.alarm(PER_INPUT_TIMEOUT)
                out = fn(*copy.deepcopy(inp))
                signal.alarm(0)
            except _Timeout:
                return {"kind": "timeout", "suite": suite, "index": index, "input": short(inp)}
            except BaseException as e:  # noqa: BLE001
                signal.alarm(0)
                return {"kind": "exception", "suite": suite, "index": index, "input": short(inp),
                        "expected": short(exp), "error": f"{type(e).__name__}: {short(str(e))}"}
            if not close_enough(out, exp, atol):
                return {"kind": "wrong_output", "suite": suite, "index": index, "input": short(inp),
                        "expected": short(exp), "actual": short(out)}
    return None


def first_failure(code, problem, truth):
    suites = [("base", problem["base_input"], truth["base"]), ("plus", problem["plus_input"], truth["plus"])]
    receiver, sender = mp.Pipe(duplex=False)
    proc = mp.Process(target=_worker, args=(code, problem["entry_point"], suites,
                                            problem.get("atol", 1e-6) or 1e-6, sender))
    proc.start()
    sender.close()
    try:
        result = receiver.recv() if receiver.poll(CASE_TIMEOUT) else {"kind": "case_timeout"}
    except EOFError:  # child died without reporting (e.g. killed for memory)
        result = {"kind": "crashed"}
    proc.join(5)
    if result["kind"] == "crashed":
        result["exitcode"] = proc.exitcode
    if proc.is_alive():
        proc.kill()
    return result


def official_status(code, problem, truth):
    """Re-run the exact evalplus check the evaluator used (status per suite)."""
    status = {}
    for suite in ("base", "plus"):
        result = untrusted_check("mbpp", code, problem[f"{suite}_input"], problem["entry_point"],
                                 expected=truth[suite], atol=problem.get("atol", 1e-6),
                                 ref_time=truth[f"{suite}_time"], fast_check=False,
                                 min_time_limit=1.0, gt_time_limit_factor=2.0)
        status[suite] = result[0]
    return status


def main():
    problems, ground_truth = load_mbpp_dataset()
    runs = {}
    for prefix, model, loop in RUNS:
        result = json.loads((RESULTS / f"{prefix}{SUFFIX}").read_text())[0]
        runs[(model, loop)] = {p["task_id"]: p for p in result["prompt_results"]}

    cases = []
    for (model, loop), tasks in runs.items():
        # Pseudocode variants are compared with the summary loop; the summary loop with v1 pseudocode.
        other = runs[(model, "pseudocode" if loop == "summary" else "summary")]
        for task_id, record in tasks.items():
            sustained = record["successful_cycles"]
            if not 1 <= sustained < len(record["cycles"]):  # passed cycle 1, failed later
                continue
            prev, failing = record["cycles"][sustained - 1], record["cycles"][sustained]
            problem, truth = problems[task_id], ground_truth[task_id]
            diagnosis = first_failure(failing["task_1_output"], problem, truth)
            status = official_status(failing["task_1_output"], problem, truth)
            case = {
                "case_id": f"{CASE_PREFIX[loop]}{model[7:10].replace('.', '')}-{task_id.split('/')[1]}",
                "model": model, "loop": loop, "task_id": task_id,
                "fail_cycle": sustained + 1, "sustained": sustained,
                "other_loop_sustained": other[task_id]["successful_cycles"],
                "entry_point": problem["entry_point"],
                "original_prompt": record["initial_prompt"],
                "last_passing_code": prev["task_1_output"],
                "intermediate_text": prev["task_2_output"],
                "failing_code": failing["task_1_output"],
                "official_status": status,
                "diagnosis": diagnosis,
            }
            cases.append(case)
            print(f"{case['case_id']:<14} fail@{case['fail_cycle']:<2} official={status} -> {diagnosis['kind']}"
                  + (f" (renamed to {diagnosis['renamed_to']}: {diagnosis['renamed_logic']})"
                     if diagnosis["kind"] == "missing_function" else ""),
                  flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cases.json").write_text(json.dumps(cases, indent=2))
    write_markdown(cases, OUT / "failures.md")
    # Keep manual labels already entered for known cases; new cases get empty ones.
    labels_path = OUT / "labels.csv"
    previous = {}
    if labels_path.exists():
        with labels_path.open(newline="") as fh:
            previous = {r["case_id"]: r for r in csv.DictReader(fh)}
    with labels_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["case_id", "model", "loop", "task_id", "fail_cycle", "other_loop_sustained",
                         "failed_suite", "diagnosis", "renamed_to", "renamed_logic", "label", "notes",
                         "label_status"])
        for c in cases:
            failed = ",".join(s for s, v in c["official_status"].items() if v != "pass") or "none"
            old = previous.get(c["case_id"], {})
            writer.writerow([c["case_id"], c["model"], c["loop"], c["task_id"], c["fail_cycle"],
                             c["other_loop_sustained"], failed, c["diagnosis"]["kind"],
                             c["diagnosis"].get("renamed_to") or "", c["diagnosis"].get("renamed_logic") or "",
                             old.get("label", ""), old.get("notes", ""), old.get("label_status", "")])
    print(f"\n{len(cases)} cases written to {OUT}/")


def write_markdown(cases, path):
    lines = ["# Failed loops (failure at cycle >= 2)", "",
             "Generated by `scripts/diagnose_failures.py`. Each case: the code that last passed, the "
             "intermediate text (summary or pseudocode) produced from it, and the code generated from "
             "that text, which failed. Diagnosis = first failing EvalPlus input when re-run in the sandbox.", ""]
    for c in sorted(cases, key=lambda c: (list(CASE_PREFIX).index(c["loop"]), c["model"], c["task_id"])):
        d = c["diagnosis"]
        lines += [f"## {c['case_id']}: {c['task_id']} ({c['model']}, {c['loop']} loop)", "",
                  f"- Failed at cycle **{c['fail_cycle']}** (sustained {c['sustained']}); "
                  f"{'pseudocode' if c['loop'] == 'summary' else 'summary'} loop sustained "
                  f"**{c['other_loop_sustained']}**",
                  f"- Official evalplus status: base={c['official_status']['base']}, "
                  f"plus={c['official_status']['plus']}",
                  "- Diagnosis: **" + d["kind"] + "**" + "".join(
                      f"; {k}: `{v}`" for k, v in d.items() if k != "kind"), "",
                  "**Original task**", "```", c["original_prompt"].strip(), "```", "",
                  f"**Last passing code (cycle {c['fail_cycle'] - 1})**", "```python",
                  c["last_passing_code"].strip(), "```", "",
                  f"**{'Summary' if c['loop'] == 'summary' else 'Pseudocode'} fed into cycle "
                  f"{c['fail_cycle']}**", "```", c["intermediate_text"].strip(), "```", "",
                  f"**Failing code (cycle {c['fail_cycle']})**", "```python", c["failing_code"].strip(), "```",
                  "", "---", ""]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
