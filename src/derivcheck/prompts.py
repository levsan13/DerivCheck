"""Prompts for the two language-model nodes of the agent: formalize and repair."""

from __future__ import annotations

import json

from .schema import Check, Step

SYMPY_RULES = """\
SymPy syntax: ** for powers and explicit * for products; sqrt, exp, log, sin, cos, tan,
sinh, cosh, tanh, asin, acos, atan, Abs, conjugate; pi; I for the imaginary unit;
fractions as 1/2 or Rational(1, 2); derivatives as Derivative(expr, x); integrals as
Integral(expr, (x, a, b)), with oo for infinity. Symbols are plain identifiers: hbar,
omega, v_F, p_x, E_0, k_B, lamda for lambda. Write Euler's number as exp(1), never as E:
E is an ordinary symbol (usually the energy). Use the same name for the same quantity in
every step."""

FORMALIZE_SYSTEM = f"""\
You prepare physics derivations for a computer-algebra check. Split the derivation the
user sends into its individual steps and translate each step into SymPy syntax.
Transcribe; do not judge or correct: if a step looks wrong, translate it as written.

Answer with one JSON object and nothing else:
{{
  "assumptions": {{"<symbol>": "positive" | "nonnegative" | "real" | "nonzero" | "integer" | "complex"}},
  "steps": [
    {{
      "latex": "<the step as written>",
      "lhs": "<left-hand side, SymPy syntax>",
      "rhs": "<right-hand side, SymPy syntax>",
      "relation": "given" | "definition" | "equals" | "approx" | "solve" | "equivalent",
      "expansion": {{"variable": "<symbol>", "point": "0", "order": <highest order kept>}} or null,
      "justification": "<the reason the text gives for the step, or null>"
    }}
  ]
}}

Relations:
- "given": the expression a chain starts from, e.g. E = sqrt(p**2*c**2 + m**2*c**4).
- "definition": a symbol defined in the text ("where omega = sqrt(k/m)").
- "equals": an exact rewriting of the previous expression of the same chain. Repeat the
  left-hand side: "E = A = B" becomes E = A ("given") and E = B ("equals").
- "approx": an approximation of the previous expression of the chain; fill "expansion"
  with the small quantity and the highest order kept.
- "solve": the left-hand side is a symbol solved for from the previous equation.
- "equivalent": the previous equation with both sides transformed.

{SYMPY_RULES}

Assumptions: include the ones stated in the text, and mark as positive the quantities that
are positive by their physical meaning (masses, the speed of light, hbar, frequencies,
lengths, temperatures)."""

REPAIR_SYSTEM = f"""\
You translated one step of a physics derivation into SymPy, and the computer-algebra check
did not confirm it. Decide whether your translation is faithful to the LaTeX.
- If you made a transcription mistake (wrong symbol, missing factor or sign, wrong
  relation, wrong expansion variable or order), return the corrected translation.
- If the translation is faithful, the step as written is what fails: keep the translation
  and set "faithful" to true. Never change the mathematics of the text to make the check
  pass.

Answer with one JSON object and nothing else:
{{"lhs": "...", "rhs": "...", "relation": "...", "expansion": {{...}} or null,
  "faithful": true | false, "note": "<one sentence>"}}

{SYMPY_RULES}"""


def formalize_user(text: str) -> str:
    return f"Derivation:\n\n{text.strip()}\n"


def repair_user(
    step: Step, check: Check, previous: Step | None, assumptions: dict[str, str], definitions: dict[str, str]
) -> str:
    context = {
        "step": {
            "latex": step.latex,
            "lhs": step.lhs,
            "rhs": step.rhs,
            "relation": step.relation,
            "expansion": step.expansion.model_dump() if step.expansion else None,
        },
        "check": check.model_dump(exclude_none=True),
        "previous_step": None
        if previous is None
        else {
            "latex": previous.latex,
            "lhs": previous.lhs,
            "rhs": previous.rhs,
            "relation": previous.relation,
        },
        "assumptions": assumptions,
        "definitions": definitions,
    }
    return "Step to review, with the checker's result:\n" + json.dumps(context, indent=2, ensure_ascii=False)
