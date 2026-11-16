"""The agent: a LangGraph workflow around the segmenter, the translator and the verifier.

::

    segment -> formalize -> verify -> report
                               |  ^
                               v  |
                              repair      (only with a language model, for steps that
                                           failed or did not parse, at most N times each)

* ``segment`` splits the text into steps: the language model in LLM mode, LaTeX rules
  offline (and as a fallback when the model cannot be reached).
* ``formalize`` turns every step into SymPy: the LaTeX parser offline, a whitelisted parse
  of the model's SymPy strings otherwise.
* ``verify`` is deterministic: SymPy checks each step against the one it follows from,
  using the notation memory (assumptions and definitions).
* ``repair`` shows the model each step that failed, with the checker's counterexample, and
  lets it fix a transcription mistake or confirm that the step as written is wrong. The
  model never sees a way to change the paper, and every replaced translation is kept in
  the step's history.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, TypedDict, get_args

from langgraph.graph import END, START, StateGraph

from ._version import __version__
from .formalize import FormalizationError, latex_to_sympy, parse_sympy, to_sympy_str
from .llm import Backend, LLMError, get_backend
from .memory import Memory
from .prompts import FORMALIZE_SYSTEM, REPAIR_SYSTEM, formalize_user, repair_user
from .schema import (
    PREMISES,
    VERIFIED,
    Attempt,
    Check,
    DerivationReport,
    Expansion,
    Relation,
    Step,
    StepReport,
    Summary,
    TraceEvent,
)
from .segment import segment_latex
from .verify import verify_steps

RELATIONS = set(get_args(Relation))
NEEDS_REPAIR = frozenset({"failed", "unparsed"})


class AgentState(TypedDict, total=False):
    text: str
    source: str | None
    backend: Backend
    memory: Memory  # assumptions given by the user and found in the text
    working: Memory  # memory after the last verification, with the derivation's definitions
    steps: list[Step]
    errors: dict[int, str]
    checks: dict[int, Check]
    attempts: dict[int, int]
    settled: list[int]  # steps whose translation the model confirmed or could not improve
    history: dict[int, list[Attempt]]
    trace: list[TraceEvent]
    max_repairs: int
    timeout: float
    report: DerivationReport


def _since(t0: float) -> float:
    return round(time.perf_counter() - t0, 4)


def _text(value: Any) -> str | None:
    return None if value is None else str(value)


def _expansion(value: Any) -> Expansion | None:
    if not isinstance(value, dict):
        return None
    try:
        order = value.get("order")
        return Expansion(
            variable=_text(value.get("variable")),
            point=str(value.get("point", "0")),
            order=int(order) if order is not None else None,
        )
    except (TypeError, ValueError):
        return None


def _steps_from_model(data: dict[str, Any]) -> list[Step]:
    steps: list[Step] = []
    for item in data.get("steps") or []:
        if not isinstance(item, dict):
            continue
        relation = item.get("relation") if item.get("relation") in RELATIONS else "equals"
        steps.append(
            Step(
                index=len(steps) + 1,
                latex=str(item.get("latex") or ""),
                lhs=_text(item.get("lhs")),
                rhs=_text(item.get("rhs")),
                relation=relation,
                justification=_text(item.get("justification")),
                expansion=_expansion(item.get("expansion")) if relation == "approx" else None,
            )
        )
    return steps


# ------------------------------------------------------------------------------ nodes


def _segment(state: AgentState) -> dict:
    t0 = time.perf_counter()
    backend, memory = state["backend"], state["memory"]
    trace = list(state.get("trace", []))
    found: dict[str, str] = {}
    steps: list[Step] = []
    if backend.is_llm:
        try:
            data = backend.complete_json(FORMALIZE_SYSTEM, formalize_user(state["text"]))
            steps = _steps_from_model(data)
            found = {str(k): str(v) for k, v in (data.get("assumptions") or {}).items()}
            how = f"{backend.name}"
        except LLMError as exc:
            trace.append(TraceEvent(node="segment", message=f"model failed ({exc}); using LaTeX rules"))
    if not steps:
        seg = segment_latex(state["text"])
        steps, found = seg.steps, seg.assumptions
        how = "LaTeX rules"
    for name, kind in found.items():
        if name not in memory.assumptions:  # assumptions stated by the user take precedence
            try:
                memory.assume(name, kind)
            except ValueError:
                continue
    trace.append(
        TraceEvent(
            node="segment",
            seconds=_since(t0),
            message=f"{len(steps)} steps and {len(found)} assumptions ({how})",
        )
    )
    return {"steps": steps, "trace": trace}


def _formalize(state: AgentState) -> dict:
    t0 = time.perf_counter()
    errors: dict[int, str] = {}
    for step in state["steps"]:
        try:
            if step.lhs is None and step.lhs_latex is not None:
                step.lhs = to_sympy_str(latex_to_sympy(step.lhs_latex))
                step.rhs = to_sympy_str(latex_to_sympy(step.rhs_latex or ""))
            else:
                step.lhs = to_sympy_str(parse_sympy(step.lhs or ""))
                step.rhs = to_sympy_str(parse_sympy(step.rhs or ""))
        except FormalizationError as exc:
            errors[step.index] = str(exc)
    trace = list(state.get("trace", []))
    done = len(state["steps"]) - len(errors)
    trace.append(
        TraceEvent(
            node="formalize",
            seconds=_since(t0),
            message=f"{done}/{len(state['steps'])} steps translated to SymPy",
        )
    )
    return {"errors": errors, "trace": trace}


def _verify(state: AgentState) -> dict:
    t0 = time.perf_counter()
    working = state["memory"].copy()
    checks = verify_steps(state["steps"], working, state.get("errors"), state.get("timeout", 5.0))
    counts = Counter(check.status for check in checks.values())
    trace = list(state.get("trace", []))
    trace.append(
        TraceEvent(
            node="verify",
            seconds=_since(t0),
            message=", ".join(f"{n} {status}" for status, n in sorted(counts.items())),
        )
    )
    return {"checks": checks, "working": working, "trace": trace}


def _pending(state: AgentState) -> list[int]:
    if not state["backend"].is_llm:
        return []
    attempts, settled = state.get("attempts", {}), set(state.get("settled", []))
    limit = state.get("max_repairs", 2)
    return [
        i
        for i, check in sorted(state["checks"].items())
        if check.status in NEEDS_REPAIR and attempts.get(i, 0) < limit and i not in settled
    ]


def _route(state: AgentState) -> str:
    return "repair" if _pending(state) else "report"


def _repair(state: AgentState) -> dict:
    backend, working = state["backend"], state["working"]
    by_index = {step.index: step for step in state["steps"]}
    checks = state["checks"]
    attempts = dict(state.get("attempts", {}))
    settled = list(state.get("settled", []))
    history = {k: list(v) for k, v in state.get("history", {}).items()}
    errors = dict(state.get("errors", {}))
    trace = list(state.get("trace", []))
    for i in _pending(state):
        t0 = time.perf_counter()
        step, check = by_index[i], checks[i]
        previous = by_index.get(check.against or i - 1)
        attempts[i] = attempts.get(i, 0) + 1
        try:
            data = backend.complete_json(
                REPAIR_SYSTEM,
                repair_user(step, check, previous, working.assumptions, working.definitions),
            )
        except LLMError as exc:
            settled.append(i)
            trace.append(
                TraceEvent(node="repair", step=i, seconds=_since(t0), message=f"model failed: {exc}")
            )
            continue
        if data.get("faithful") is True:
            settled.append(i)
            trace.append(
                TraceEvent(
                    node="repair",
                    step=i,
                    seconds=_since(t0),
                    message="model confirms its translation: the step as written does not follow",
                )
            )
            continue
        relation = data.get("relation") if data.get("relation") in RELATIONS else step.relation
        new = (
            _text(data.get("lhs")) or step.lhs,
            _text(data.get("rhs")) or step.rhs,
            relation,
            _expansion(data.get("expansion")) if relation == "approx" else None,
        )
        if new == (step.lhs, step.rhs, step.relation, step.expansion):
            settled.append(i)
            trace.append(
                TraceEvent(node="repair", step=i, seconds=_since(t0), message="model proposed no change")
            )
            continue
        history.setdefault(i, []).append(
            Attempt(
                lhs=step.lhs, rhs=step.rhs, relation=step.relation, status=check.status, detail=check.detail
            )
        )
        step.lhs, step.rhs, step.relation, step.expansion = new
        try:
            step.lhs = to_sympy_str(parse_sympy(step.lhs or ""))
            step.rhs = to_sympy_str(parse_sympy(step.rhs or ""))
            errors.pop(i, None)
        except FormalizationError as exc:
            errors[i] = str(exc)
        note = _text(data.get("note")) or "no note"
        trace.append(
            TraceEvent(
                node="repair",
                step=i,
                seconds=_since(t0),
                message=f"new translation {step.lhs} [{step.relation}] {step.rhs}: {note}",
            )
        )
    return {"attempts": attempts, "settled": settled, "history": history, "errors": errors, "trace": trace}


def _report(state: AgentState) -> dict:
    checks, history = state["checks"], state.get("history", {})
    attempts = state.get("attempts", {})
    reports = [
        StepReport(
            **step.model_dump(),
            check=checks.get(step.index) or Check(status="unparsed", detail="not checked"),
            attempts=attempts.get(step.index, 0),
            repaired=bool(history.get(step.index)),
            history=history.get(step.index, []),
        )
        for step in state["steps"]
    ]
    statuses = [r.check.status for r in reports]
    checked = [s for s in statuses if s not in PREMISES]
    failed = [r.index for r in reports if r.check.status == "failed"]
    unparsed, unverified = statuses.count("unparsed"), statuses.count("unverified")
    if failed:
        verdict = "error_found"
    elif checked and not unparsed and not unverified:
        verdict = "consistent"
    else:
        verdict = "incomplete"
    working = state.get("working") or state["memory"]
    report = DerivationReport(
        derivcheck_version=__version__,
        source=state.get("source"),
        backend=state["backend"].name,
        assumptions=dict(working.assumptions),
        definitions=dict(working.definitions),
        steps=reports,
        summary=Summary(
            steps=len(reports),
            checked=len(checked),
            verified=sum(s in VERIFIED for s in statuses),
            failed=len(failed),
            unparsed=unparsed,
            unverified=unverified,
            first_failure=failed[0] if failed else None,
            verdict=verdict,
            llm_calls=state["backend"].calls,
            repairs=sum(r.repaired for r in reports),
        ),
        trace=state.get("trace", []),
    )
    return {"report": report}


# ------------------------------------------------------------------------------ graph


def build_graph():
    """Compile the LangGraph state machine (see the module docstring)."""
    graph = StateGraph(AgentState)
    graph.add_node("segment", _segment)
    graph.add_node("formalize", _formalize)
    graph.add_node("verify", _verify)
    graph.add_node("repair", _repair)
    graph.add_node("report", _report)
    graph.add_edge(START, "segment")
    graph.add_edge("segment", "formalize")
    graph.add_edge("formalize", "verify")
    graph.add_conditional_edges("verify", _route, {"repair": "repair", "report": "report"})
    graph.add_edge("repair", "verify")
    graph.add_edge("report", END)
    return graph.compile()


_GRAPH = None


def check_derivation(
    text: str,
    llm: str | Backend | None = None,
    *,
    source: str | None = None,
    memory: Memory | None = None,
    max_repairs: int = 2,
    timeout: float = 5.0,
) -> DerivationReport:
    """Check a derivation and return the report.

    ``llm`` is a backend spec (``"offline"``, ``"lmstudio"``, ``"ollama:qwen3:32b"``, ...)
    or a :class:`~derivcheck.llm.Backend`; ``memory`` carries assumptions and definitions
    known beforehand (from the command line or from earlier derivations of the same paper).
    """
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    backend = llm if isinstance(llm, Backend) else get_backend(llm)
    state: AgentState = {
        "text": text,
        "source": source,
        "backend": backend,
        "memory": memory or Memory(),
        "max_repairs": max_repairs,
        "timeout": timeout,
        "trace": [],
        "attempts": {},
        "settled": [],
        "history": {},
        "errors": {},
    }
    final = _GRAPH.invoke(state, config={"recursion_limit": 25 + 4 * max_repairs})
    return final["report"]
