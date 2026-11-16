import json

from derivcheck.cli import main


def test_cli_writes_report_and_exit_codes(examples, tmp_path, capsys):
    out = tmp_path / "report.json"
    assert main([str(examples / "relativistic_limit.tex"), "-o", str(out), "-q"]) == 0
    report = json.loads(out.read_text())
    assert report["summary"]["verdict"] == "consistent"
    assert [s["check"]["status"] for s in report["steps"]] == ["given", "verified", "verified_to_order"]

    assert main([str(examples / "kinetic_energy_error.tex"), "-o", str(tmp_path / "e.json")]) == 1
    assert "first failure at step 3" in capsys.readouterr().err


def test_cli_memory_and_assumptions(examples, tmp_path):
    notes = tmp_path / "notation.json"
    code = main(
        [
            str(examples / "harmonic_oscillator.tex"),
            "-o",
            str(tmp_path / "r.json"),
            "--positive",
            "A",
            "--memory",
            str(notes),
            "-q",
        ]
    )
    assert code == 0
    saved = json.loads(notes.read_text())
    assert saved["assumptions"]["A"] == "positive"
    assert saved["definitions"]["omega"] == "sqrt(k/m)"


def test_cli_bench(capsys):
    assert main(["bench"]) == 0
    assert "passed 10/10" in capsys.readouterr().out
