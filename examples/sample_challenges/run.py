#!/usr/bin/env python3
"""Run a submitted solution against a challenge's public cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlvr.execution.rust_docker_executor import RustDockerExecutor
from rlvr.execution.subprocess_executor import SubprocessExecutor
from rlvr.types import TestCase


ROOT = Path(__file__).resolve().parent


def load_problem(name: str) -> tuple[dict, list[TestCase]]:
    path = ROOT / name / "cases.json"
    if not path.is_file() or path.parent.parent != ROOT:
        raise ValueError(f"unknown example: {name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    tests = [TestCase.model_validate(case) for case in payload["cases"]]
    return payload, tests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", help="example name from README.md")
    parser.add_argument("submission", type=Path, help="source file to test")
    parser.add_argument("--timeout", type=float, default=5.0, help="seconds per case")
    args = parser.parse_args()

    try:
        payload, tests = load_problem(args.example)
        code = args.submission.read_text(encoding="utf-8")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))

    executor = (
        RustDockerExecutor()
        if payload["language"] == "rust"
        else SubprocessExecutor()
    )
    results = executor.run_tests(code, payload["entrypoint"], tests, args.timeout)

    passed = 0
    for index, (case, result) in enumerate(zip(payload["cases"], results, strict=True), 1):
        label = case.get("name", f"case {index}")
        if result.passed:
            passed += 1
            print(f"PASS {index}: {label}")
        else:
            detail = result.error or f"returned {result.actual_repr}"
            print(f"FAIL {index}: {label} ({detail})")
    print(f"{passed}/{len(tests)} cases passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
