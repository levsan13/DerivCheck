r"""Split a derivation written in LaTeX into steps, without a language model.

The rules follow how derivations are typeset in papers:

* display math (``align``, ``equation``, ``\[ \]``, ``$$ $$``) holds the steps, one chain
  of relations per row: ``E &= A \\ &= B \\ &\approx C``;
* an inline equality in the prose (``where $\omega = \sqrt{k/m}$``) is a definition;
* an inline inequality such as ``$m, c > 0$`` is an assumption;
* the prose before a display block tells how an approximation is taken
  ("expanding for small $p$, to second order in $p$").

The language-model mode (:mod:`derivcheck.agent`) does the same job for text that does
not follow these conventions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .formalize import latex_symbol_name
from .schema import Expansion, Relation, Step

_DISPLAY_ENVS = r"align\*?|equation\*?|gather\*?|multline\*?|eqnarray\*?|flalign\*?"
_MATH = re.compile(
    r"\\begin\{(?P<env>" + _DISPLAY_ENVS + r")\}(?P<env_body>.*?)\\end\{(?P=env)\}"
    r"|\\\[(?P<bracket>.*?)\\\]"
    r"|\$\$(?P<dollars>.*?)\$\$"
    r"|(?<![\\$])\$(?P<inline>[^$]+?)(?<!\\)\$",
    re.S,
)
_INNER_ENV = re.compile(r"\\(?:begin|end)\{(?:split|aligned|alignedat|gathered)\}(?:\{\d+\})?")
_ROW_BREAK = re.compile(r"\\\\(?:\s*\[[^\]]*\])?")
_COMMENT = re.compile(r"(?<!\\)%.*")

# relations that end a chain or make a row something other than an equation
_BLOCKER = re.compile(
    r"\\(?:neq|ne|leq|geq|le|ge|ll|gg|propto|sim|lesssim|gtrsim|to|rightarrow|Rightarrow"
    r"|implies|iff|longrightarrow|mapsto|in|subset)(?![A-Za-z])|[<>]"
)
_SYMBOL_LIKE = re.compile(
    r"^(?:\\(?:Delta|delta)\s*(?:\\[A-Za-z]+|[A-Za-z])"  # \Delta E: one quantity
    r"|(?:\\[A-Za-z]+|[A-Za-z])(?:_(?:\{[^{}]*\}|[A-Za-z0-9]))?(?:\^\{?\\?\*\}?|')?)$"
)

_DEFINITION_CUE = re.compile(r"\b(where|let|define[sd]?|defining|denote[sd]?|denoting)\b", re.I)
_SMALL_VAR = re.compile(r"small\s+\$(?P<v>[^$]+)\$", re.I)
_LL_VAR = re.compile(r"\$(?P<v>[^$<>]+?)\s*\\ll")
_TO_ZERO_VAR = re.compile(r"\$(?P<v>[^$]+?)\s*\\to\s*0\s*\$")
_ORDER_IN = re.compile(
    r"(?P<n>first|second|third|fourth|\d+)(?:st|nd|rd|th)?[\s-]+order\s+in\s+\$(?P<v>[^$]+)\$", re.I
)
_ORDER = re.compile(r"(?P<n>first|second|third|fourth|\d+)(?:st|nd|rd|th)?[\s-]+order", re.I)
_ORDER_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4}

_ASSUMPTION_PATTERNS = [
    (re.compile(r"^(?P<names>.+?)\s*>\s*0$"), "positive"),
    (re.compile(r"^(?P<names>.+?)\s*\\geq?\s*0$"), "nonnegative"),
    (re.compile(r"^(?P<names>.+?)\s*<\s*0$"), "negative"),
    (re.compile(r"^(?P<names>.+?)\s*\\neq?\s*0$"), "nonzero"),
    (re.compile(r"^(?P<names>.+?)\s*\\in\s*\\mathbb\{R\}$"), "real"),
    (re.compile(r"^(?P<names>.+?)\s*\\in\s*\\mathbb\{Z\}$"), "integer"),
]


@dataclass
class Segmentation:
    steps: list[Step] = field(default_factory=list)
    assumptions: dict[str, str] = field(default_factory=dict)


def _split_chain(row: str) -> tuple[list[str], list[Relation]] | None:
    """Split ``a = b \\approx c`` at top-level relations; None if the row is not an equation."""
    parts: list[str] = []
    rels: list[Relation] = []
    depth = 0
    current: list[str] = []
    i = 0
    while i < len(row):
        ch = row[i]
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        if depth == 0:
            for token, rel in ((r"\approx", "approx"), (r"\simeq", "approx"), (r"\equiv", "definition")):
                if row.startswith(token, i) and not row[i + len(token) : i + len(token) + 1].isalpha():
                    parts.append("".join(current))
                    rels.append(rel)  # type: ignore[arg-type]
                    current = []
                    i += len(token)
                    break
            else:
                if row.startswith(":=", i):
                    parts.append("".join(current))
                    rels.append("definition")
                    current = []
                    i += 2
                    continue
                if ch == "=" and not row.startswith("==", i):
                    parts.append("".join(current))
                    rels.append("equals")
                    current = []
                    i += 1
                    continue
                current.append(ch)
                i += 1
            continue
        current.append(ch)
        i += 1
    parts.append("".join(current))
    if not rels:
        return None
    if _BLOCKER.search(" ".join(parts)):
        return None
    return [p.strip() for p in parts], rels


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _key(latex: str) -> str:
    return re.sub(r"\s+", "", latex)


def _last_sentence(prose: str) -> str | None:
    text = _plain(re.sub(r"\\[A-Za-z]+\{([^{}]*)\}", r"\1", prose))
    pieces = [p.strip() for p in re.split(r"(?<=[.;:])\s+", text) if p.strip()]
    if not pieces:
        return None
    last = pieces[-1].rstrip(",:; ")
    return last[:200] or None


def _expansion_from(prose: str) -> Expansion | None:
    variable, order = None, None
    m = _ORDER_IN.search(prose)
    if m:
        variable = m.group("v")
        n = m.group("n").lower()
        order = _ORDER_WORDS.get(n) or int(n)
    for pattern in (_SMALL_VAR, _LL_VAR, _TO_ZERO_VAR):
        if variable is None and (m := pattern.search(prose)):
            variable = m.group("v")
    if order is None and (m := _ORDER.search(prose)):
        n = m.group("n").lower()
        order = _ORDER_WORDS.get(n) or int(n)
    if variable is None and order is None:
        return None
    name = latex_symbol_name(variable) if variable else None
    return Expansion(variable=name, order=order)


def _assumptions_from(inline: str) -> dict[str, str]:
    found: dict[str, str] = {}
    content = _plain(inline)
    for pattern, kind in _ASSUMPTION_PATTERNS:
        m = pattern.match(content)
        if not m:
            continue
        for item in m.group("names").split(","):
            item = item.strip()
            if _SYMBOL_LIKE.match(item) and (name := latex_symbol_name(item)):
                found[name] = kind
        break
    return found


def segment_latex(tex: str) -> Segmentation:
    """Find the steps of a LaTeX derivation in reading order."""
    tex = "\n".join(_COMMENT.sub("", line) for line in tex.splitlines())
    seg = Segmentation()
    chains: set[str] = set()  # left-hand sides with an open chain of equalities
    current_lhs: str | None = None
    last_display_end = 0

    def add(lhs: str, rhs: str, relation: Relation, symbol: str, justification=None, expansion=None):
        latex = f"{lhs} {symbol} {rhs}".strip()
        seg.steps.append(
            Step(
                index=len(seg.steps) + 1,
                latex=latex,
                lhs_latex=lhs,
                rhs_latex=rhs,
                relation=relation,
                justification=justification,
                expansion=expansion if relation == "approx" else None,
            )
        )

    symbols = {"equals": "=", "approx": r"\approx", "definition": r"\equiv", "given": "="}

    for m in _MATH.finditer(tex):
        inline = m.group("inline")
        if inline is not None:
            seg.assumptions.update(_assumptions_from(inline))
            chain = _split_chain(inline)
            if chain is None:
                continue
            parts, rels = chain
            lhs = parts[0]
            if not lhs or not _SYMBOL_LIKE.match(lhs):
                continue
            if _key(lhs) in chains:
                relation = rels[0] if rels[0] != "definition" else "equals"
                cues = _expansion_from(tex[last_display_end : m.start()])
                add(lhs, parts[1], relation, symbols[rels[0]], expansion=cues)
            else:
                add(lhs, parts[1], "definition", "=")
            continue

        body = m.group("env_body") or m.group("bracket") or m.group("dollars") or ""
        prose = tex[last_display_end : m.start()]
        last_display_end = m.end()
        justification = _last_sentence(prose)
        expansion = _expansion_from(prose)
        definition_cue = bool(_DEFINITION_CUE.search(justification or ""))
        body = _INNER_ENV.sub(" ", body)

        for row in _ROW_BREAK.split(body):
            row = re.sub(r"\\(?:label|tag)\*?\{[^{}]*\}|\\nonumber|\\notag", " ", row)
            row = _plain(row.replace("&", " ")).rstrip(".,;").strip()
            if not row:
                continue
            chain = _split_chain(row)
            if chain is None:
                continue
            parts, rels = chain
            first = parts[0]
            if not first:  # continuation row: "&= ..."
                if current_lhs is None:
                    continue
                lhs = current_lhs
                pairs = list(zip(parts[1:], rels, strict=True))
            else:
                lhs = first
                pairs = list(zip(parts[1:], rels, strict=True))
                if _SYMBOL_LIKE.match(lhs) and _key(lhs) not in chains:
                    rhs, rel = pairs.pop(0)
                    start = "definition" if rel == "definition" or definition_cue else "given"
                    add(lhs, rhs, start, "=", justification)
                    justification = None
            for rhs, rel in pairs:
                if not rhs:
                    continue
                add(lhs, rhs, rel, symbols[rel], justification, expansion)
                justification = None
            chains.add(_key(lhs))
            current_lhs = lhs
    return seg
