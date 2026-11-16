"""Command line: ``derivcheck FILE`` and ``derivcheck bench``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._version import __version__
from .agent import check_derivation
from .bench import run_bench
from .memory import Memory
from .schema import VERIFIED, DerivationReport

EXIT_CODES = {"consistent": 0, "error_found": 1, "incomplete": 3}
_SYMBOLS = {
    "=": "=",
    "approx": "~",
    "definition": ":=",
    "given": "=",
    "equals": "=",
    "solve": "=>",
    "equivalent": "<=>",
}


def _tag(status: str) -> str:
    if status in VERIFIED:
        return "ok"
    return {"failed": "FAIL", "given": "--", "definition": "def"}.get(status, "??")


def print_summary(report: DerivationReport, stream=None) -> None:
    stream = stream or sys.stderr
    print(
        f"DerivCheck {report.derivcheck_version} | {report.backend} | {report.source or 'stdin'}", file=stream
    )
    for step in report.steps:
        rel = _SYMBOLS.get(step.relation, "=")
        expr = f"{step.lhs} {rel} {step.rhs}" if step.lhs is not None else step.latex
        how = f"  [{step.check.method}]" if step.check.method else ""
        print(
            f"  {step.index:>2} {_tag(step.check.status):>4}  {step.check.status:<20} {expr}{how}",
            file=stream,
        )
        if step.check.status not in ("given", "definition", "verified") and step.check.detail:
            print(f"{'':10}{step.check.detail}", file=stream)
        if step.check.counterexample:
            point = ", ".join(f"{k} = {v:g}" for k, v in step.check.counterexample.items())
            print(f"{'':10}counterexample: {point}", file=stream)
    s = report.summary
    first = f", first failure at step {s.first_failure}" if s.first_failure else ""
    print(f"verdict: {s.verdict} ({s.verified}/{s.checked} checked steps verified{first})", file=stream)


def _names(value: str | None) -> list[str]:
    return [n.strip() for n in (value or "").split(",") if n.strip()]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "bench":
        return run_bench(argv[1:])

    parser = argparse.ArgumentParser(
        prog="derivcheck",
        description="Split a physics derivation into steps, check each step with SymPy and "
        "write a JSON report. 'derivcheck bench' runs the benchmark.",
    )
    parser.add_argument("input", help="LaTeX or text file with the derivation ('-' for stdin)")
    parser.add_argument("-o", "--output", help="write the JSON report here (default: stdout)")
    parser.add_argument(
        "--llm",
        default=None,
        help="offline (default), lmstudio[:model], ollama:model, openai:model or "
        "claude:model; DERIVCHECK_LLM sets the default",
    )
    parser.add_argument("--positive", help="comma-separated symbols known to be positive")
    parser.add_argument("--real", help="comma-separated symbols known to be real")
    parser.add_argument(
        "--memory",
        help="JSON notation memory to load and update "
        "(assumptions and definitions shared by derivations of one paper)",
    )
    parser.add_argument(
        "--max-repairs", type=int, default=2, help="times the model may revise a step that fails (default 2)"
    )
    parser.add_argument(
        "--timeout", type=float, default=5.0, help="seconds allowed for each symbolic proof (default 5)"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="no summary on stderr")
    parser.add_argument("--version", action="version", version=f"derivcheck {__version__}")
    args = parser.parse_args(argv)

    if args.input == "-":
        text, source = sys.stdin.read(), None
    else:
        path = Path(args.input)
        text, source = path.read_text(encoding="utf-8"), args.input
    memory = Memory.load(args.memory) if args.memory and Path(args.memory).exists() else Memory()
    for name in _names(args.positive):
        memory.assume(name, "positive")
    for name in _names(args.real):
        memory.assume(name, "real")

    try:
        report = check_derivation(
            text, args.llm, source=source, memory=memory, max_repairs=args.max_repairs, timeout=args.timeout
        )
    except ValueError as exc:  # bad backend specification
        parser.error(str(exc))

    payload = report.model_dump(mode="json")
    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    else:
        print(json.dumps(payload, indent=2))
    if args.memory:
        Memory(report.assumptions, report.definitions).save(args.memory)
    if not args.quiet:
        print_summary(report)
    return EXIT_CODES[report.summary.verdict]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
