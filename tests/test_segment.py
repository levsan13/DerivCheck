from derivcheck.segment import segment_latex

TEX = r"""
For a particle with $m, c > 0$ and $\omega = \sqrt{k/m}$,
\begin{align}
  E &= \sqrt{p^2 c^2 + m^2 c^4} \label{eq:E} \\
    &= m c^2 \sqrt{1 + \frac{p^2}{m^2 c^2}} \nonumber
\end{align}
Expanding for small $p$, to second order in $p$,
\begin{equation}
  E \approx m c^2 + \frac{p^2}{2m} .
\end{equation}
Since $p < m c$, nothing else is needed. % a comment with $x = 1$
"""


def test_steps_relations_and_cues():
    seg = segment_latex(TEX)
    assert seg.assumptions == {"m": "positive", "c": "positive"}
    relations = [s.relation for s in seg.steps]
    assert relations == ["definition", "given", "equals", "approx"]
    omega, start, rewrite, approx = seg.steps
    assert omega.lhs_latex == r"\omega" and omega.rhs_latex == r"\sqrt{k/m}"
    assert start.lhs_latex == "E" and r"\label" not in start.rhs_latex
    assert rewrite.lhs_latex == "E"  # continuation row "&= ..." keeps the left-hand side
    assert approx.expansion.variable == "p" and approx.expansion.order == 2
    assert "small" in approx.justification
    assert [s.index for s in seg.steps] == [1, 2, 3, 4]


def test_compound_left_hand_side_is_a_claim():
    seg = segment_latex(r"\[ (a + b)^2 = a^2 + 2ab + b^2 = (b + a)^2 \]")
    assert [s.relation for s in seg.steps] == ["equals", "equals"]
    assert all(s.lhs_latex == "(a + b)^2" for s in seg.steps)


def test_inequalities_and_prose_are_skipped():
    seg = segment_latex(r"We need $x \ll 1$ and \begin{equation} y \leq 2 \end{equation} only.")
    assert seg.steps == []


def test_delta_quantity_opens_a_chain():
    seg = segment_latex(r"\begin{align} \Delta E &= E_2 - E_1 \\ &= \hbar\omega \end{align}")
    assert [s.relation for s in seg.steps] == ["given", "equals"]
