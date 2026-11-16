import sympy as sp

from derivcheck.memory import Memory
from derivcheck.schema import Expansion, Step
from derivcheck.verify import check_approx, check_equal, check_equivalent, check_solve, verify_steps


def mem(**assumptions):
    return Memory(assumptions)


def test_identity_is_verified_symbolically():
    m = mem()
    check = check_equal(m.parse("cos(2*t)"), m.parse("cos(t)**2 - sin(t)**2"))
    assert check.status == "verified" and check.method == "symbolic"


def test_wrong_factor_fails_with_counterexample():
    m = mem(hbar="positive", m="positive")
    check = check_equal(m.parse("hbar**2*k**2/(2*m)"), m.parse("hbar**2*k**2/m"))
    assert check.status == "failed"
    assert set(check.counterexample) == {"hbar", "k", "m"}


def test_sign_dependent_identity_asks_for_assumptions():
    check = check_equal(mem().parse("sqrt(x**2)"), mem().parse("x"))
    assert check.status == "unverified" and "positive" in check.detail
    m = mem(x="positive")
    assert check_equal(m.parse("sqrt(x**2)"), m.parse("x")).status == "verified"


def test_approximation_to_the_order_kept():
    m = mem(m="positive", c="positive")
    exact = m.parse("m*c**2*sqrt(1 + p**2/(m**2*c**2))")
    good = check_approx(exact, m.parse("m*c**2 + p**2/(2*m)"), Expansion(variable="p"), m)
    assert good.status == "verified_to_order" and "up to order p**2" in good.detail
    assert "-p**4/(8*c**2*m**3)" in good.detail
    bad = check_approx(exact, m.parse("m*c**2 + p**2/m"), Expansion(variable="p", order=2), m)
    assert bad.status == "failed" and "-p**2/(2*m)" in bad.detail


def test_approximation_finds_the_variable():
    m = mem()
    check = check_approx(m.parse("sqrt(1 + x)"), m.parse("1 + x/2"), None, m)
    assert check.status == "verified_to_order" and check.method == "series in x"


def test_solve_and_equivalent():
    m = mem(k="positive", E_0="positive", A="positive")
    solve = check_solve(m.parse("k*A**2/2"), m.parse("E_0"), m.parse("A"), m.parse("sqrt(2*E_0/k)"))
    assert solve.status == "verified" and solve.method == "substitution"
    wrong = check_solve(m.parse("k*A**2/2"), m.parse("E_0"), m.parse("A"), m.parse("sqrt(E_0/k)"))
    assert wrong.status == "failed"
    x = mem()
    eq = check_equivalent(x.parse("2*y"), x.parse("4"), x.parse("y"), x.parse("2"))
    assert eq.status == "verified" and eq.method == "ratio"


def test_chain_uses_definitions_given_after_use():
    steps = [
        Step(index=1, latex="", lhs="E", rhs="m*v**2/2 + k*x**2/2", relation="given"),
        Step(index=2, latex="", lhs="E", rhs="k*A**2/2", relation="equals"),
        Step(index=3, latex="", lhs="x", rhs="A*cos(omega*t)", relation="definition"),
        Step(index=4, latex="", lhs="v", rhs="-A*omega*sin(omega*t)", relation="definition"),
        Step(index=5, latex="", lhs="omega", rhs="sqrt(k/m)", relation="definition"),
    ]
    m = mem(k="positive", m="positive")
    checks = verify_steps(steps, m)
    assert checks[2].status == "verified" and checks[2].against == 1
    assert m.definitions["omega"] == "sqrt(k/m)"


def test_unparsed_steps_are_reported():
    steps = [
        Step(index=1, latex="", lhs="E", rhs="m*c**2", relation="given"),
        Step(index=2, latex="", lhs="E", rhs="m c^2 +", relation="equals"),
    ]
    checks = verify_steps(steps, mem())
    assert checks[2].status == "unparsed"


def test_memory_round_trip(tmp_path):
    m = Memory({"m": "positive"}, {"omega": "sqrt(k/m)"})
    m.save(tmp_path / "notation.json")
    loaded = Memory.load(tmp_path / "notation.json")
    assert loaded.to_dict() == m.to_dict()
    assert loaded.symbol("m").is_positive and loaded.symbol("x").is_real
    assert loaded.expand(sp.Symbol("omega", real=True) ** 2) == loaded.parse("k/m")
