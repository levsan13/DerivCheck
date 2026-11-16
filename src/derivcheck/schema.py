"""Data model of a derivation and of the JSON report."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Relation = Literal["given", "definition", "equals", "approx", "solve", "equivalent"]
"""How a step relates to what came before it.

* ``given``: first expression of a chain (``E = sqrt(p**2*c**2 + m**2*c**4)``); nothing to check.
* ``definition``: a symbol introduced in the text (``omega = sqrt(k/m)``); it is substituted
  into the steps that are compared afterwards.
* ``equals``: exact rewriting of the previous expression with the same left-hand side.
* ``approx``: approximation of the previous expression (series in a small variable).
* ``solve``: the left-hand side is a symbol solved for from the previous equation.
* ``equivalent``: the previous equation with both sides transformed.
"""

Status = Literal[
    "given",
    "definition",
    "verified",
    "numerically_verified",
    "verified_to_order",
    "failed",
    "unparsed",
    "unverified",
]

VERIFIED = frozenset({"verified", "numerically_verified", "verified_to_order"})
PREMISES = frozenset({"given", "definition"})


class Expansion(BaseModel):
    """Where an approximation is taken: ``variable -> point``, keeping terms up to ``order``."""

    variable: str | None = None
    point: str = "0"
    order: int | None = None


class Step(BaseModel):
    """One step of a derivation, ``lhs <relation> rhs``."""

    index: int
    latex: str
    lhs_latex: str | None = None
    rhs_latex: str | None = None
    lhs: str | None = Field(default=None, description="left-hand side in SymPy syntax")
    rhs: str | None = Field(default=None, description="right-hand side in SymPy syntax")
    relation: Relation = "equals"
    justification: str | None = None
    expansion: Expansion | None = None


class Check(BaseModel):
    """What the verifier concluded about one step."""

    status: Status
    method: str | None = None
    against: int | None = Field(default=None, description="index of the step it was compared with")
    detail: str | None = None
    counterexample: dict[str, float] | None = None


class Attempt(BaseModel):
    """A translation of a step that the agent later replaced."""

    lhs: str | None
    rhs: str | None
    relation: Relation
    status: Status
    detail: str | None = None


class StepReport(Step):
    check: Check
    attempts: int = 0
    repaired: bool = False
    history: list[Attempt] = Field(default_factory=list)


class Summary(BaseModel):
    steps: int
    checked: int
    verified: int
    failed: int
    unparsed: int
    unverified: int
    first_failure: int | None
    verdict: Literal["consistent", "error_found", "incomplete"]
    llm_calls: int = 0
    repairs: int = 0


class TraceEvent(BaseModel):
    node: str
    message: str
    step: int | None = None
    seconds: float | None = None


class DerivationReport(BaseModel):
    derivcheck_version: str
    source: str | None = None
    backend: str
    assumptions: dict[str, str] = Field(default_factory=dict)
    definitions: dict[str, str] = Field(default_factory=dict)
    steps: list[StepReport]
    summary: Summary
    trace: list[TraceEvent] = Field(default_factory=list)
