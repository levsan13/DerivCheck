"""Turn LaTeX (offline mode) or SymPy-syntax strings (LLM mode) into SymPy expressions.

Two entry points:

* :func:`latex_to_sympy` parses what is typeset in a paper. It cleans the LaTeX that the
  SymPy parser does not know (``\\left``, ``\\mathrm``, spacing, multi-letter subscripts),
  and afterwards fixes the usual physics readings: ``e^{i x}`` is ``exp(I*x)``, ``i`` is
  the imaginary unit and ``\\omega(n + 1/2)`` is a product, not a function call.
* :func:`parse_sympy` reads the SymPy strings written by a language model. The model's
  output is untrusted text (it may echo a paper that contains anything), so the string
  is checked against a whitelist before it reaches ``parse_expr`` and Python builtins
  are switched off.
"""

from __future__ import annotations

import keyword
import re
from collections.abc import Mapping

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations
from sympy.printing.str import StrPrinter


class FormalizationError(ValueError):
    """A step could not be translated into SymPy."""


# --------------------------------------------------------------------------- printing


class _Printer(StrPrinter):
    def _print_Exp1(self, expr):  # a bare "E" would be read back as a symbol (energy)
        return "exp(1)"


def to_sympy_str(expr: sp.Basic) -> str:
    """SymPy syntax that :func:`parse_sympy` reads back to the same expression."""
    return _Printer().doprint(expr)


# --------------------------------------------------------------------- SymPy strings

_FUNCTION_NAMES = (
    "sqrt",
    "exp",
    "log",
    "sin",
    "cos",
    "tan",
    "cot",
    "sec",
    "csc",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "Abs",
    "sign",
    "conjugate",
    "re",
    "im",
    "Derivative",
    "diff",
    "Integral",
    "integrate",
    "factorial",
    "Heaviside",
    "DiracDelta",
    "erf",
    "Rational",
    "Integer",
    "Float",
    "Symbol",
    "Function",
)
NAMESPACE: dict[str, object] = {name: getattr(sp, name) for name in _FUNCTION_NAMES}
NAMESPACE.update({"pi": sp.pi, "I": sp.I, "oo": sp.oo})

_ALLOWED = re.compile(r"^[A-Za-z0-9_+\-*/^().,\s]*$")
_ATTRIBUTE = re.compile(r"\.\s*[A-Za-z_]")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CALL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
# no implicit multiplication: it would read the call psi(x) as the product psi*x
_TRANSFORMS = standard_transformations + (convert_xor,)
_MAX_LENGTH = 4000


def identifiers(text: str) -> tuple[set[str], set[str]]:
    """Names used as symbols and names used as functions in a SymPy string."""
    calls = set(_CALL.findall(text))
    names = set(_IDENT.findall(text)) - calls - set(NAMESPACE)
    return names, calls


def parse_sympy(text: str, symbols: Mapping[str, sp.Symbol] | None = None) -> sp.Expr:
    """Parse a SymPy-syntax string safely.

    ``symbols`` maps names to the symbols to use for them (with their assumptions); any
    other name becomes a plain symbol, or an undefined function when it is called.
    """
    s = (text or "").strip()
    if not s:
        raise FormalizationError("empty expression")
    if len(s) > _MAX_LENGTH:
        raise FormalizationError("expression too long")
    if not _ALLOWED.match(s):
        bad = sorted(set(re.sub(r"[A-Za-z0-9_+\-*/^().,\s]", "", s)))
        raise FormalizationError(f"characters not allowed in an expression: {''.join(bad)!r}")
    if "__" in s or _ATTRIBUTE.search(s):
        raise FormalizationError("attribute access is not allowed in an expression")
    for name in _IDENT.findall(s):
        if keyword.iskeyword(name):
            raise FormalizationError(f"{name!r} is a Python keyword, not a symbol name")
    global_dict: dict[str, object] = {"__builtins__": {}, **NAMESPACE}
    local_dict: dict[str, object] = dict(symbols or {})
    try:
        expr = parse_expr(s, local_dict=local_dict, global_dict=global_dict, transformations=_TRANSFORMS)
    except Exception as exc:  # noqa: BLE001 - any parser failure means "not parsed"
        raise FormalizationError(f"cannot parse {s!r}: {exc}") from None
    if not isinstance(expr, sp.Basic) or _is_logical(expr):
        raise FormalizationError(f"{s!r} is not an algebraic expression")
    return expr


def _is_logical(expr: sp.Basic) -> bool:
    # Symbol also derives from Boolean, so test for relations and logic operators only
    return isinstance(
        expr,
        (
            sp.core.relational.Relational,
            sp.logic.boolalg.BooleanFunction,
            sp.logic.boolalg.BooleanAtom,
        ),
    )


# ----------------------------------------------------------------------------- LaTeX

_GREEK = {
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "varepsilon",
    "zeta",
    "eta",
    "theta",
    "vartheta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "omega",
    "Gamma",
    "Delta",
    "Theta",
    "Lambda",
    "Xi",
    "Sigma",
    "Upsilon",
    "Omega",
    "hbar",
}
# Greek letters that usually name functions (wavefunctions, potentials), not constants
_FUNCTION_LIKE = {"psi", "Psi", "phi", "Phi", "varphi", "chi"}

_REMOVE = [
    r"\\label\{[^{}]*\}",
    r"\\tag\*?\{[^{}]*\}",
    r"\\(?:text|textrm|mbox|intertext)\{[^{}]*\}",
    r"\\(?:nonumber|notag|displaystyle|textstyle|limits)\b",
    r"\\(?:left|right)(?![A-Za-z])",
    r"\\(?:big|Big|bigg|Bigg)[lr]?(?![A-Za-z])",
    r"\\(?:quad|qquad)(?![A-Za-z])",
    r"\\[,;:! ]",
]


def _preprocess(latex: str) -> str:
    s = latex
    for pattern in _REMOVE:
        s = re.sub(pattern, " ", s)
    s = re.sub(r"\\[dt]frac(?![A-Za-z])", r"\\frac", s)
    s = re.sub(r"\\times(?![A-Za-z])", r"\\cdot ", s)
    s = re.sub(r"\\(?:mathrm|operatorname)\{d\}", "d", s)
    # \Delta E and \delta x name one quantity (a difference), not a product
    greek = "|".join(sorted(_GREEK | _FUNCTION_LIKE, key=len, reverse=True))
    s = re.sub(rf"\\(Delta|delta)\s*(\\(?:{greek})|[A-Za-z])(?![A-Za-z_(])", r"\\\1_{\2}", s)
    # multi-letter subscripts are names (omega_{ZB}, E_{kin}), not products of letters
    s = re.sub(
        r"_\{\s*\\(?:mathrm|text|textrm|rm|mathit|mathsf)\{([A-Za-z0-9]+)\}\s*\}",
        r"_{\\mathit{\1}}",
        s,
    )
    s = re.sub(r"_\{([A-Za-z][A-Za-z0-9]+)\}", r"_{\\mathit{\1}}", s)
    s = re.sub(r"\\(?:mathrm|mathit|mathsf|rm|mathbf|boldsymbol|vec)\{([A-Za-z])\}", r"\1", s)
    s = s.replace("&", " ")
    s = s.strip().rstrip(".,;").strip()
    return s


def _base_name(name: str) -> str:
    return name.replace("\\", "").split("_", 1)[0]


def _clean_name(name: str) -> str:
    n = re.sub(r"[^A-Za-z0-9_]", "", name.replace("\\", ""))
    n = re.sub(r"_+", "_", n).strip("_") or "x_"
    if n == "lambda":
        n = "lamda"  # SymPy's spelling; "lambda" is a Python keyword
    elif keyword.iskeyword(n) or n in NAMESPACE:
        n += "_"
    return n


def _postprocess(expr: sp.Basic) -> sp.Basic:
    e_sym, i_sym, pi_sym = sp.Symbol("e"), sp.Symbol("i"), sp.Symbol("pi")
    if pi_sym in expr.free_symbols:
        expr = expr.xreplace({pi_sym: sp.pi})
    # i is the imaginary unit (indices live in subscripts: x_i is another symbol)
    if i_sym in expr.free_symbols:
        expr = expr.xreplace({i_sym: sp.I})
    # e^{...} is the exponential; a bare e stays a symbol (elementary charge)
    expr = expr.replace(lambda x: x.is_Pow and x.base == e_sym, lambda x: sp.exp(x.exp))
    # \omega_c(n + 1/2) is read by the parser as a call of a function "omega_c"
    expr = expr.replace(
        lambda x: (
            isinstance(x, AppliedUndef)
            and _base_name(x.func.__name__) in _GREEK - _FUNCTION_LIKE
            and len(x.args) == 1
        ),
        lambda x: sp.Symbol(x.func.__name__) * x.args[0],
    )
    # functions that stay functions (psi(x), E_n(k)) need names SymPy can read back
    expr = expr.replace(
        lambda x: isinstance(x, AppliedUndef) and _clean_name(x.func.__name__) != x.func.__name__,
        lambda x: sp.Function(_clean_name(x.func.__name__))(*x.args),
    )
    rename = {
        s: sp.Symbol(_clean_name(s.name))
        for s in expr.free_symbols
        if isinstance(s, sp.Symbol) and _clean_name(s.name) != s.name
    }
    return expr.xreplace(rename) if rename else expr


def latex_to_sympy(latex: str) -> sp.Basic:
    """Parse one side of an equation written in LaTeX."""
    src = _preprocess(latex)
    if not src:
        raise FormalizationError("empty expression")
    from sympy.parsing.latex import parse_latex

    try:
        try:
            expr = parse_latex(src)  # ANTLR backend: reads implicit products such as p^2 c^2
        except ImportError:  # antlr4 runtime missing: fall back on the Lark grammar
            expr = parse_latex(src, backend="lark")
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        raise FormalizationError(f"cannot parse LaTeX {latex.strip()!r}: {msg}") from None
    if _is_logical(expr):
        raise FormalizationError(f"{latex.strip()!r} is a relation, expected an expression")
    expr = _postprocess(expr)
    # the LaTeX parser keeps the typeset grouping unevaluated, (k*x**2)/2; read the
    # expression back through the SymPy parser to get the usual canonical form
    try:
        return parse_sympy(to_sympy_str(expr))
    except FormalizationError:
        return expr


def latex_symbol_name(latex: str) -> str | None:
    """SymPy name of a single symbol written in LaTeX (``v_{F}`` -> ``v_F``), or None."""
    try:
        expr = latex_to_sympy(latex)
    except FormalizationError:
        return None
    return expr.name if isinstance(expr, sp.Symbol) else None
