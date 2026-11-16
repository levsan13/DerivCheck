"""``derivcheck bench``: run the agent on derivations whose verdict is known.

Each line of the cases file is a JSON object::

    {"id": "...", "expected": "consistent" | "error_found",
     "flag": "<part of the step that should be flagged>" | null, "tex": "<LaTeX>"}

A case passes when the verdict matches and, for derivations with an error, the first step
flagged contains ``flag``. The same cases run offline in CI and against any language
model (``--llm lmstudio:<model>``), which makes it easy to compare models on the task.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from importlib import resources
from pathlib import Path


def load_cases(path: str | None) -> list[dict]:
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        text = resources.files("derivcheck").joinpath("data/bench_cases.jsonl").read_text("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _norm(text: str) -> str:
    return "".join(text.split())


def run_bench(argv: list[str] | None = None) -> int:
    from .agent import check_derivation  # imported here: the CLI imports this module

    parser = argparse.ArgumentParser(
        prog="derivcheck bench", description="Run the agent on derivations with known verdicts."
    )
    parser.add_argument("cases", nargs="?", help="JSONL file (default: the cases shipped with derivcheck)")
    parser.add_argument("--llm", default=None, help="backend, as in 'derivcheck --llm'")
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--json", dest="json_out", help="also write the results to this JSON file")
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    rows, node_seconds = [], defaultdict(float)
    llm_calls = 0
    backend_name = None
    for case in cases:
        t0 = time.perf_counter()
        report = check_derivation(case["tex"], args.llm, source=case["id"], max_repairs=args.max_repairs)
        seconds = time.perf_counter() - t0
        backend_name = report.backend
        first = report.summary.first_failure
        flagged = report.steps[first - 1].latex if first else None
        located = case.get("flag") is None or (flagged is not None and _norm(case["flag"]) in _norm(flagged))
        passed = report.summary.verdict == case["expected"] and located
        rows.append(
            {
                "id": case["id"],
                "expected": case["expected"],
                "verdict": report.summary.verdict,
                "flagged_step": first,
                "passed": passed,
                "seconds": round(seconds, 3),
                "llm_calls": report.summary.llm_calls,
                "repairs": report.summary.repairs,
            }
        )
        llm_calls += report.summary.llm_calls
        for event in report.trace:
            node_seconds[event.node] += event.seconds or 0.0

    width = max(len(r["id"]) for r in rows)
    print(f"derivcheck bench | backend {backend_name} | {len(rows)} cases")
    print(f"  {'case':<{width}}  {'expected':<12} {'verdict':<12} {'step':>4}  {'s':>6}  result")
    for r in rows:
        step = r["flagged_step"] or "-"
        print(
            f"  {r['id']:<{width}}  {r['expected']:<12} {r['verdict']:<12} {step:>4}  "
            f"{r['seconds']:>6.2f}  {'pass' if r['passed'] else 'FAIL'}"
        )
    passed = sum(r["passed"] for r in rows)
    print(
        f"passed {passed}/{len(rows)}; LLM calls {llm_calls}; time per node: "
        + ", ".join(f"{node} {s:.2f}s" for node, s in node_seconds.items())
    )
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"backend": backend_name, "cases": rows, "node_seconds": node_seconds}, indent=2)
            + "\n",
            encoding="utf-8",
        )
    return 0 if passed == len(rows) else 1
