"""Notation memory: what the agent knows about the symbols of a derivation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import sympy as sp

from .formalize import FormalizationError, identifiers, parse_sympy

ASSUMPTIONS: dict[str, dict[str, bool]] = {
    "positive": {"positive": True},
    "negative": {"negative": True},
    "nonnegative": {"nonnegative": True},
    "real": {"real": True},
    "nonzero": {"real": True, "nonzero": True},
    "integer": {"integer": True},
    "complex": {},
}


class Memory:
    """Assumptions on symbols and definitions collected along a derivation.

    * Assumptions (``m > 0``) decide which simplifications are valid: ``sqrt(m**2*c**4)``
      is ``m*c**2`` only for positive ``m`` and ``c``. Symbols with no stated assumption are
      taken as real, which is what they are in almost every physics derivation.
    * Definitions (``omega = sqrt(k/m)``) are substituted into the expressions being
      compared, so that a step written in terms of ``omega`` can be checked against one
      written in terms of ``k`` and ``m``.

    A memory can be saved to JSON and loaded again for other derivations of the same paper.
    """

    def __init__(
        self,
        assumptions: dict[str, str] | None = None,
        definitions: dict[str, str] | None = None,
    ) -> None:
        self.assumptions: dict[str, str] = {}
        self.definitions: dict[str, str] = dict(definitions or {})
        for name, kind in (assumptions or {}).items():
            self.assume(name, kind)

    # ------------------------------------------------------------------ assumptions
    def assume(self, name: str, kind: str) -> None:
        kind = kind.strip().lower()
        if kind not in ASSUMPTIONS:
            raise ValueError(f"unknown assumption {kind!r}; use one of {sorted(ASSUMPTIONS)}")
        self.assumptions[name] = kind

    def symbol(self, name: str) -> sp.Symbol:
        return sp.Symbol(name, **ASSUMPTIONS[self.assumptions.get(name, "real")])

    def localize(self, expr: sp.Basic) -> sp.Basic:
        """Give every symbol of ``expr`` the assumptions stored for its name."""
        mapping = {s: self.symbol(s.name) for s in expr.free_symbols if isinstance(s, sp.Symbol)}
        return expr.xreplace(mapping)

    def parse(self, text: str) -> sp.Basic:
        """Parse a SymPy string with the stored assumptions and evaluate derivatives/integrals."""
        names, _ = identifiers(text)
        expr = parse_sympy(text, {name: self.symbol(name) for name in names})
        expr = self.localize(expr)
        if expr.has(sp.Derivative, sp.Integral):
            expr = expr.doit()
        return expr

    # ------------------------------------------------------------------ definitions
    def define(self, name: str, expression: str) -> None:
        self.definitions.setdefault(name, expression)

    def substitutions(self, exclude: set[str] | frozenset[str] = frozenset()) -> dict:
        subs = {}
        for name, text in self.definitions.items():
            if name in exclude:
                continue
            try:
                subs[self.symbol(name)] = self.parse(text)
            except FormalizationError:
                continue
        return subs

    def expand(self, expr: sp.Basic, exclude: set[str] | frozenset[str] = frozenset()) -> sp.Basic:
        """Substitute the definitions into ``expr`` until nothing changes (at most 6 rounds)."""
        subs = self.substitutions(exclude)
        if not subs:
            return expr
        for _ in range(6):
            new = expr.xreplace(subs)
            if new == expr:
                break
            expr = new
        return expr

    # ------------------------------------------------------------------ persistence
    def copy(self) -> Memory:
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return {"assumptions": dict(self.assumptions), "definitions": dict(self.definitions)}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> Memory:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data.get("assumptions"), data.get("definitions"))
