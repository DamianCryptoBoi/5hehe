import json
import re
from pathlib import Path


ROOT = Path(__file__).parents[1] / "examples" / "sample_challenges"
EXPECTED = {
    "asset-rebuild-planner": ("python", "plan_rebuild"),
    "extent-journal": ("python", "extent_journal"),
    "reactive-stat-board": ("rust", "main"),
    "revocable-verification-gate": ("rust", "main"),
    "sparse-circular-array": ("rust", "main"),
}


def test_examples_have_a_small_fixed_surface():
    directories = {
        path.name for path in ROOT.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    }
    assert directories == set(EXPECTED)
    assert {path.name for path in ROOT.iterdir() if path.is_file()} == {
        "README.md",
        "run.py",
    }
    for name in EXPECTED:
        assert {path.name for path in (ROOT / name).iterdir()} == {
            "PROBLEM.md",
            "cases.json",
        }


def test_example_cases_are_basic_public_manifests():
    for name, (language, entrypoint) in EXPECTED.items():
        payload = json.loads((ROOT / name / "cases.json").read_text())
        assert set(payload) == {"language", "entrypoint", "cases"}
        assert payload["language"] == language
        assert payload["entrypoint"] == entrypoint
        assert len(payload["cases"]) == 3
        for case in payload["cases"]:
            assert set(case) == {"name", "args", "kwargs", "expected"}
            assert isinstance(case["name"], str) and case["name"]
            assert isinstance(case["args"], list)
            assert isinstance(case["kwargs"], dict)


def test_examples_contain_no_internal_records():
    forbidden = (
        "accepted_solution",
        "created_at",
        "hidden_tests",
        "measured_pass_rate",
        "problem_id",
        "prompt_variant",
        "served_at",
    )
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".md", ".py"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert not any(term in text for term in forbidden), path
        assert not re.search(r"\b[0-9a-f]{32}\b", text), path
        assert not re.search(r"20\d\d-\d\d-\d\dT", text), path
