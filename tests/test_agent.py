import json

from conftest import ScriptedBackend

from derivcheck import DerivationReport, check_derivation
from derivcheck.llm import LLMError, parse_json

TEX = r"""
With $m, c > 0$,
\begin{align}
E &= \sqrt{p^2 c^2 + m^2 c^4} \\
  &= m c^2 \sqrt{1 + \frac{p^2}{m^2 c^2}}
\end{align}
and, for small $p$, $E \approx m c^2 + \frac{p^2}{2m}$.
"""

FORMALIZED = {
    "assumptions": {"m": "positive", "c": "positive"},
    "steps": [
        {
            "latex": r"E = \sqrt{p^2 c^2 + m^2 c^4}",
            "lhs": "E",
            "rhs": "sqrt(p**2*c**2 + m**2*c**4)",
            "relation": "given",
        },
        {
            "latex": r"= m c^2 \sqrt{1 + p^2/(m^2 c^2)}",
            "lhs": "E",
            "rhs": "m*c**2*sqrt(1 + p**2/(m**2*c**2))",
            "relation": "equals",
        },
        # transcription mistake: the 1/2 of p^2/2m was dropped
        {
            "latex": r"E \approx m c^2 + \frac{p^2}{2m}",
            "lhs": "E",
            "rhs": "m*c**2 + p**2/m",
            "relation": "approx",
            "expansion": {"variable": "p", "point": "0", "order": 2},
        },
    ],
}


def test_offline_examples(examples):
    expected = {
        "relativistic_limit": "consistent",
        "harmonic_oscillator": "consistent",
        "kinetic_energy_error": "error_found",
    }
    for name, verdict in expected.items():
        text = (examples / f"{name}.tex").read_text()
        report = check_derivation(text, "offline", source=name)
        assert report.summary.verdict == verdict, name
        assert [e.node for e in report.trace] == ["segment", "formalize", "verify"]
        # the report is plain JSON and reads back into the same model
        again = DerivationReport.model_validate(json.loads(report.model_dump_json()))
        assert again == report


def test_repair_loop_fixes_a_transcription_mistake():
    fix = {
        "lhs": "E",
        "rhs": "m*c**2 + p**2/(2*m)",
        "relation": "approx",
        "expansion": {"variable": "p", "point": "0", "order": 2},
        "faithful": False,
        "note": "the factor 1/2 was dropped",
    }
    backend = ScriptedBackend([FORMALIZED, fix])
    report = check_derivation(TEX, backend)
    step = report.steps[2]
    assert step.check.status == "verified_to_order"
    assert step.repaired and step.attempts == 1
    assert step.history[0].rhs == "c**2*m + p**2/m" and step.history[0].status == "failed"
    assert report.summary.verdict == "consistent"
    assert report.summary.llm_calls == 2 and report.summary.repairs == 1
    # the model saw the checker's result when asked to review the step
    assert "do not cancel" in backend.prompts[1][1]
    assert [e.node for e in report.trace] == ["segment", "formalize", "verify", "repair", "verify"]


def test_model_confirming_a_faithful_translation_keeps_the_failure():
    wrong = json.loads(json.dumps(FORMALIZED))
    backend = ScriptedBackend([wrong, {"faithful": True, "note": "the text drops the 1/2"}])
    report = check_derivation(TEX, backend)
    assert report.summary.verdict == "error_found" and report.summary.first_failure == 3
    assert report.summary.llm_calls == 2 and not report.steps[2].repaired


def test_repairs_are_bounded():
    wrong = json.loads(json.dumps(FORMALIZED))
    attempts = [
        {
            "lhs": "E",
            "rhs": f"m*c**2 + {n}*p**2/m",
            "relation": "approx",
            "expansion": {"variable": "p", "order": 2},
            "faithful": False,
        }
        for n in (3, 5, 7)
    ]
    backend = ScriptedBackend([wrong, *attempts])
    report = check_derivation(TEX, backend, max_repairs=2)
    assert report.steps[2].attempts == 2 and len(report.steps[2].history) == 2
    assert report.summary.llm_calls == 3 and report.summary.verdict == "error_found"


def test_falls_back_to_latex_rules_when_the_model_is_down():
    backend = ScriptedBackend([LLMError("connection refused")])
    report = check_derivation(TEX, backend)
    assert report.summary.verdict == "consistent"
    assert "LaTeX rules" in report.trace[1].message


def test_parse_json_handles_reasoning_models():
    text = '<think>let me think {not json}</think>\n```json\n{"steps": []}\n```'
    assert parse_json(text) == {"steps": []}
