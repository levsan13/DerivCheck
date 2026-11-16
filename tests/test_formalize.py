import pytest
import sympy as sp

from derivcheck.formalize import (
    FormalizationError,
    latex_symbol_name,
    latex_to_sympy,
    parse_sympy,
    to_sympy_str,
)


def S(name, **kw):
    return sp.Symbol(name, **kw)


@pytest.mark.parametrize(
    "latex, expected",
    [
        (r"\frac{\hbar^2 k^2}{2m}", "hbar**2*k**2/(2*m)"),
        (r"m c^2 \sqrt{1 + \frac{p^2}{m^2 c^2}}", "c**2*m*sqrt(1 + p**2/(c**2*m**2))"),
        (r"2\pi\hbar", "2*pi*hbar"),
        (r"e^{i\theta}", "exp(I*theta)"),
        (r"\cos\theta + i\sin\theta", "I*sin(theta) + cos(theta)"),
        (r"\hbar\omega_c\left(n + \frac{1}{2}\right)", "hbar*omega_c*(n + 1/2)"),
        (r"v_{F} \sqrt{p_x^2 + p_y^2}", "v_F*sqrt(p_x**2 + p_y**2)"),
        (r"\omega_{ZB}", "omega_ZB"),
        (r"E_{\mathrm{kin}} + \Delta E", "Delta_E + E_kin"),
        (r"\lambda \nu", "lamda*nu"),
        (r"\int_{-\infty}^{\infty} e^{-x^2} dx", "Integral(exp(-x**2), (x, -oo, oo))"),
    ],
)
def test_latex_to_sympy(latex, expected):
    assert to_sympy_str(latex_to_sympy(latex)) == expected


def test_latex_errors_and_names():
    with pytest.raises(FormalizationError):
        latex_to_sympy(r"\frac{1}{")
    with pytest.raises(FormalizationError):
        latex_to_sympy("")
    assert latex_symbol_name(r"v_{F}") == "v_F"
    assert latex_symbol_name(r"\hbar") == "hbar"
    assert latex_symbol_name(r"a + b") is None


def test_parse_sympy_reads_model_output():
    expr = parse_sympy("m*c**2 + p^2/(2*m)")  # ^ is accepted as power
    assert expr == S("m") * S("c") ** 2 + S("p") ** 2 / (2 * S("m"))
    assert parse_sympy("E") == S("E")  # E is a symbol (energy), not Euler's number
    assert parse_sympy("exp(1)") == sp.E
    assert parse_sympy("Integral(exp(-x**2), (x, -oo, oo))").doit() == sp.sqrt(sp.pi)
    x = S("x", positive=True)
    assert parse_sympy("sqrt(x**2)", {"x": x}) == x


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os').system('ls')",
        "().__class__.__bases__",
        "x.real",
        "lambda: 1",
        "[x for x in range(3)]",
        "x if y else z",
        "open('secrets')",
        "a == b",
        "",
    ],
)
def test_parse_sympy_rejects_code(text):
    with pytest.raises(FormalizationError):
        parse_sympy(text)


def test_unknown_calls_stay_symbolic():
    # with builtins switched off, names such as eval are just undefined functions
    expr = parse_sympy("eval(x)")
    assert isinstance(expr, sp.core.function.AppliedUndef)


def test_round_trip_keeps_euler_number():
    expr = sp.E * S("E")
    assert parse_sympy(to_sympy_str(expr)) == expr
