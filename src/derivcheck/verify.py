"""Check each step of a derivation with SymPy.

Every comparison runs in the same order:

1. cheap exact tests (``a - b`` expands to zero);
2. a numeric test at random points, which finds wrong steps quickly and gives a
   counterexample;
3. a symbolic proof (``simplify`` and friends) under a time limit.

A step that agrees numerically but resists ``simplify`` is reported as
``numerically_verified`` rather than being held back. Approximations are checked by
expanding the difference of the two sides in a series and asking that every term up to
the order kept vanishes.
"""

from __future__ import annotations

import math
import random
import threading
from collections.abc import Callable
from typing import Any

import sympy as sp
from sympy.core.function import AppliedUndef

from .formalize import FormalizationError, to_sympy_str
from .memory import Memory
from .schema import PREMISES, Check, Expansion, Step

TOLERANCE = 1e-9
TRIALS = 8


# ---------------------------------------------------------------------------- helpers


def _with_timeout(fn: Callable[[], Any], timeout: float) -> Any:
    """Run ``fn`` in a daemon thread; None if it raises or does not finish in time."""
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except Exception as exc:  # noqa: BLE001
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive() or "error" in box:
        return None
    return box.get("value")


def _is_zero(d: sp.Basic) -> bool:
    if d == 0:
        return True
    attempts: tuple[Callable[[sp.Basic], sp.Basic], ...] = (
        sp.expand,
        sp.cancel,
        sp.simplify,
        lambda x: sp.simplify(sp.expand_trig(x)),
        lambda x: sp.simplify(sp.radsimp(x)),
    )
    for transform in attempts:
        try:
            if transform(d) == 0:
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _sample(sym: sp.Symbol, rng: random.Random, force_positive: bool) -> sp.Basic:
    if sym.is_integer:
        return sp.Integer(rng.randint(1, 5))
    magnitude = sp.Float(rng.uniform(0.4, 2.5), 30)
    if force_positive or sym.is_positive or sym.is_nonnegative:
        return magnitude
    if sym.is_negative:
        return -magnitude
    return magnitude if rng.random() < 0.5 else -magnitude


def _evaluable(expr: sp.Basic) -> bool:
    return not expr.atoms(AppliedUndef) and not expr.has(sp.Derivative, sp.Integral)


def _numeric(a: sp.Basic, b: sp.Basic, force_positive: bool = False, seed: int = 7):
    """Compare ``a`` and ``b`` at random points.

    Returns ``(True, n_points)``, ``(False, (point, a_value, b_value))`` on a mismatch, or
    ``(None, reason)`` when the expressions cannot be evaluated.
    """
    if not (_evaluable(a) and _evaluable(b)):
        return None, "the expressions contain undefined functions or unevaluated operators"
    free = sorted(a.free_symbols | b.free_symbols, key=lambda s: s.name)
    rng = random.Random(seed)
    agree = 0
    for _ in range(TRIALS * 4):
        if agree >= TRIALS:
            break
        point = {s: _sample(s, rng, force_positive) for s in free}
        try:
            va = complex(sp.N(a, 30, subs=point))
            vb = complex(sp.N(b, 30, subs=point))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            continue
        values = (va.real, va.imag, vb.real, vb.imag)
        if not all(math.isfinite(x) for x in values):
            continue
        if abs(va - vb) > TOLERANCE * max(1.0, abs(va), abs(vb)):
            return False, (point, va, vb)
        agree += 1
    if agree == 0:
        return None, "could not evaluate the expressions at any sample point"
    return True, agree


def _fmt(z: complex) -> str:
    if abs(z.imag) < 1e-12 * max(1.0, abs(z.real)):
        return f"{z.real:.6g}"
    return f"{z.real:.6g}{z.imag:+.6g}i"


def _counterexample(point: dict) -> dict[str, float]:
    return {s.name: round(float(v), 6) for s, v in point.items()}


# ------------------------------------------------------------------------ comparisons


def check_equal(a: sp.Basic, b: sp.Basic, timeout: float = 5.0) -> Check:
    """Is ``a == b`` an identity?"""
    d = a - b
    try:
        if d == 0 or sp.expand(d) == 0:
            return Check(status="verified", method="symbolic")
    except Exception:  # noqa: BLE001
        pass
    ok, info = _numeric(a, b)
    if ok is False:
        point, va, vb = info
        if not all(s.is_positive for s in point):
            positive_ok, _ = _numeric(a, b, force_positive=True)
            if positive_ok:
                return Check(
                    status="unverified",
                    method="numeric",
                    detail="holds when every symbol is positive but not for every sign: "
                    "state the assumptions (for example m, c > 0)",
                    counterexample=_counterexample(point),
                )
        return Check(
            status="failed",
            method="numeric",
            detail=f"the two sides differ: {_fmt(va)} vs {_fmt(vb)} at the counterexample",
            counterexample=_counterexample(point),
        )
    if _with_timeout(lambda: _is_zero(d), timeout):
        return Check(status="verified", method="symbolic")
    if ok:
        return Check(
            status="numerically_verified",
            method="numeric",
            detail=f"equal at {info} random points (relative tolerance {TOLERANCE:g}); "
            f"no symbolic proof within {timeout:g} s",
        )
    return Check(status="unverified", detail=str(info))


def _kept_order(b: sp.Basic, v: sp.Symbol, point: sp.Basic) -> int:
    expr = sp.expand(b.subs(v, v + point) if point != 0 else b)
    try:
        return max(sp.Poly(expr, v).degree(), 1)
    except (sp.PolynomialError, sp.GeneratorsNeeded):
        return 1


def check_approx(
    a: sp.Basic, b: sp.Basic, expansion: Expansion | None, memory: Memory, timeout: float = 10.0
) -> Check:
    """Does ``b`` agree with ``a`` up to the order it keeps in the expansion variable?"""
    d = a - b
    point: sp.Basic = sp.Integer(0)
    if expansion and expansion.point not in ("", "0"):
        try:
            point = memory.parse(expansion.point)
        except FormalizationError:
            point = sp.Integer(0)
    if expansion and expansion.variable:
        candidates = [memory.symbol(expansion.variable)]
    else:
        candidates = sorted(d.free_symbols & b.free_symbols, key=lambda s: s.name)
    tried: list[tuple[sp.Symbol, int, sp.Basic | None]] = []
    for v in candidates:
        order = expansion.order if expansion and expansion.order else _kept_order(b, v, point)
        series = _with_timeout(lambda v=v, order=order: sp.series(d, v, point, order + 1).removeO(), timeout)
        if series is not None and _with_timeout(lambda s=series: _is_zero(s), timeout):
            detail = f"the two sides agree up to order {v}**{order}"
            if point == 0:
                lead = _with_timeout(
                    lambda v=v, order=order: sp.series(d, v, 0, order + 5).removeO(), timeout
                )
                if lead is not None and lead != 0:
                    detail += f"; leading difference {to_sympy_str(sp.simplify(lead.as_leading_term(v)))}"
            return Check(status="verified_to_order", method=f"series in {v}", detail=detail)
        tried.append((v, order, series))
    explicit = bool(expansion and expansion.variable)
    if explicit and tried and tried[0][2] is not None:
        v, order, series = tried[0]
        return Check(
            status="failed",
            method=f"series in {v}",
            detail=f"terms up to order {order} in {v} do not cancel; difference "
            f"{to_sympy_str(sp.simplify(series))}",
        )
    return Check(
        status="unverified",
        method="series",
        detail="no expansion variable makes this approximation consistent; "
        "say which quantity is small (for example 'for small $p$')",
    )


def check_solve(
    prev_lhs: sp.Basic, prev_rhs: sp.Basic, lhs: sp.Basic, rhs: sp.Basic, timeout: float = 5.0
) -> Check:
    """Does ``lhs = rhs`` (``lhs`` a symbol) solve the previous equation?"""
    if not isinstance(lhs, sp.Symbol):
        return Check(status="unverified", detail="the left-hand side of a 'solve' step must be a symbol")
    residual = (prev_lhs - prev_rhs).subs(lhs, rhs)
    check = check_equal(residual, sp.Integer(0), timeout)
    check.method = "substitution"
    if check.status == "failed":
        check.detail = f"{lhs} = {to_sympy_str(rhs)} does not satisfy the previous equation"
    return check


def check_equivalent(
    prev_lhs: sp.Basic, prev_rhs: sp.Basic, lhs: sp.Basic, rhs: sp.Basic, timeout: float = 5.0
) -> Check:
    """Is the new equation the previous one multiplied by a nonzero constant?"""
    r1, r2 = prev_lhs - prev_rhs, lhs - rhs
    if r1 == 0 or r2 == 0:
        return Check(status="unverified", detail="one of the equations is trivially true")
    ratio = _with_timeout(lambda: sp.simplify(r2 / r1), timeout)
    if ratio is not None and ratio.is_number and ratio != 0 and ratio.is_finite:
        return Check(
            status="verified",
            method="ratio",
            detail=f"new equation = ({to_sympy_str(ratio)}) x previous equation",
        )
    return Check(
        status="unverified",
        method="ratio",
        detail="the equations are not proportional; the step may still follow "
        "(squaring, dividing by a variable), check it by hand",
    )


# ---------------------------------------------------------------------------- driver


def verify_steps(
    steps: list[Step],
    memory: Memory,
    errors: dict[int, str] | None = None,
    timeout: float = 5.0,
) -> dict[int, Check]:
    """Check every step against the one it follows from. ``memory`` receives the definitions."""
    errors = errors or {}
    checks: dict[int, Check] = {}
    parsed: dict[int, tuple[sp.Basic, sp.Basic]] = {}

    # pass 1: parse everything and collect definitions, which papers often state after use
    for step in steps:
        if step.index in errors or step.lhs is None or step.rhs is None:
            checks[step.index] = Check(
                status="unparsed", detail=errors.get(step.index, "no SymPy translation")
            )
            continue
        try:
            lhs, rhs = memory.parse(step.lhs), memory.parse(step.rhs)
        except FormalizationError as exc:
            checks[step.index] = Check(status="unparsed", detail=str(exc))
            continue
        parsed[step.index] = (lhs, rhs)
        if step.relation in PREMISES and isinstance(lhs, sp.Symbol):
            memory.define(lhs.name, step.rhs)

    # pass 2: compare each step with the previous expression of its chain
    chains: dict[str, tuple[int, sp.Basic]] = {}
    last_equation: tuple[int, sp.Basic, sp.Basic] | None = None
    for step in steps:
        if step.index not in parsed:
            continue
        lhs, rhs = parsed[step.index]
        key = to_sympy_str(lhs)
        exclude = {lhs.name} if isinstance(lhs, sp.Symbol) else set()

        if step.relation in PREMISES:
            checks[step.index] = Check(status=step.relation)
            chains[key] = (step.index, rhs)

        elif step.relation in ("equals", "approx"):
            if key in chains:
                against, a = chains[key]
            elif isinstance(lhs, sp.Symbol):
                checks[step.index] = Check(status="given", detail="first appearance of this symbol")
                memory.define(lhs.name, step.rhs)
                chains[key] = (step.index, rhs)
                last_equation = (step.index, lhs, rhs)
                continue
            elif isinstance(rhs, sp.Symbol) and rhs.name not in memory.definitions:
                # "1/2 m v^2 + 1/2 k x^2 = E" names the left-hand side
                checks[step.index] = Check(status="definition", detail="names the left-hand side")
                memory.define(rhs.name, step.lhs)
                chains[to_sympy_str(rhs)] = (step.index, lhs)
                last_equation = (step.index, lhs, rhs)
                continue
            else:
                against, a = None, lhs  # self-contained claim such as (a + b)**2 = a**2 + b**2
            a_x, b_x = memory.expand(a, exclude), memory.expand(rhs, exclude)
            if step.relation == "equals":
                check = check_equal(a_x, b_x, timeout)
            else:
                check = check_approx(a_x, b_x, step.expansion, memory, 2 * timeout)
            check.against = against
            checks[step.index] = check
            chains[key] = (step.index, rhs)

        elif last_equation is None:
            checks[step.index] = Check(status="unverified", detail="no previous equation to start from")

        else:
            prev_index, prev_lhs, prev_rhs = last_equation
            prev = (memory.expand(prev_lhs, exclude), memory.expand(prev_rhs, exclude))
            new = (lhs, memory.expand(rhs, exclude))
            if step.relation == "solve":
                check = check_solve(*prev, *new, timeout=timeout)
            else:
                check = check_equivalent(*prev, memory.expand(lhs, exclude), new[1], timeout=timeout)
            check.against = prev_index
            checks[step.index] = check
            chains[key] = (step.index, rhs)

        last_equation = (step.index, lhs, rhs)
    return checks
