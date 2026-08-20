"""Verifier: turn a solution + hidden tests into a Verification verdict.

Pipeline per solution:
  1. Run the candidate against all hidden tests for the release-policy number
     of verification passes (one in the current release).
  2. Detect a COMPILE failure from the executor's typed error signal. Syntax,
     import, and missing-entrypoint failures are surfaced as `compile_error`
     (reward 0); a callable that raises on some or all inputs remains a normal
     runtime failure.
  3. Determinism: a test whose pass/fail verdict FLIPS across runs is
     nondeterministic. A flaky test counts as a FAILURE (and the verdict is
     flagged flaky); it is NEVER silently excluded, so a miner cannot make a
     hidden test nondeterministic to get it dropped from the suite.
  4. A test PASSES only if it passed deterministically (in every run). Reward:
     near-binary -> 1.0 iff every test passed deterministically, else 0.0. With
     `reward_partial_credit`, reward = fraction of tests that passed deterministically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..policy import RELEASE_POLICY, ValidatorPolicy
from ..types import (
    ExecutionResult,
    Problem,
    SolutionResponse,
    Verification,
)

if TYPE_CHECKING:  # avoid a hard runtime dep on the execution module
    from ..execution.executor import Executor


class Verifier:
    def __init__(
        self,
        executor: "Executor",
        settings,
        policy: ValidatorPolicy = RELEASE_POLICY,
    ):
        self.executor = executor
        self.settings = settings
        self.policy = policy
        self._language_executors = {"python": executor}

    # ------------------------------------------------------------------ #
    def verify(self, problem: Problem, solution: SolutionResponse) -> Verification:
        tests = problem.tests
        num_tests = len(tests)
        timeout_s = float(self.settings.per_test_timeout_s)
        runs = self.policy.verification_runs

        executor = self._language_executors.get(problem.language)
        if executor is None:
            from ..execution.executor import get_executor

            executor = get_executor(self.settings, language=problem.language)
            self._language_executors[problem.language] = executor

        # 1. Execute the candidate `runs` times over the full hidden suite.
        all_runs: list[list[ExecutionResult]] = []
        for _ in range(runs):
            results = executor.run_tests(
                solution.code, problem.entrypoint, tests, timeout_s
            )
            all_runs.append(list(results))

        first_run = all_runs[0] if all_runs else []

        # 2. Compile failure: rely on the executor's explicit error kind. A
        #    callable that raises for every authored input is a runtime failure,
        #    not a compile failure; the old "no value anywhere" heuristic could
        #    not distinguish those cases.
        compile_error = self._detect_compile_error(all_runs, num_tests)
        if compile_error is not None:
            return Verification(
                problem_id=problem.problem_id,
                num_tests=num_tests,
                num_passed=0,
                all_passed=False,
                reward=0.0,
                compile_error=compile_error,
                results=first_run,
                flagged_flaky=False,
            )

        # 3. Per-test determinism: a test that flips verdict across runs is flaky.
        #    A flaky test is NOT excluded — it is treated as a FAILURE so a miner
        #    cannot make a hidden test nondeterministic to get it dropped.
        flaky_idx: set[int] = set()
        if runs > 1:
            for i in range(num_tests):
                # `None` means the run had no result at that index; treat presence
                # of >1 distinct (non-None) verdict as a flip.
                concrete = {
                    v for v in (self._verdict_for(run, i) for run in all_runs)
                    if v is not None
                }
                if len(concrete) > 1:
                    flaky_idx.add(i)

        flagged_flaky = len(flaky_idx) > 0

        # 4. A test passes only if it passed DETERMINISTICALLY (in every run) and is
        #    not flaky. Flaky tests count as failures, so all_passed requires every
        #    test to pass deterministically.
        num_passed = sum(
            1
            for i in range(num_tests)
            if i not in flaky_idx and self._passed_every_run(all_runs, i)
        )
        all_passed = num_tests > 0 and num_passed == num_tests

        reward = self._reward(num_passed, num_tests, all_passed)

        return Verification(
            problem_id=problem.problem_id,
            num_tests=num_tests,
            num_passed=num_passed,
            all_passed=all_passed,
            reward=reward,
            compile_error=None,
            results=first_run,
            flagged_flaky=flagged_flaky,
        )

    def _reward(self, num_passed: int, num_scored: int, all_passed: bool) -> float:
        if getattr(self.settings, "reward_partial_credit", False):
            if num_scored == 0:
                return 0.0
            return num_passed / num_scored
        return 1.0 if all_passed else 0.0

    @staticmethod
    def _verdict_for(run: list[ExecutionResult], test_index: int):
        """The pass/fail verdict for a given test_index within one run, or None."""
        for r in run:
            if r.test_index == test_index:
                return bool(r.passed)
        return None

    @staticmethod
    def _passed_every_run(runs: list[list[ExecutionResult]], test_index: int) -> bool:
        """True iff `test_index` has a result that PASSED in every run (no run is
        missing it and none failed it)."""
        for run in runs:
            verdict = Verifier._verdict_for(run, test_index)
            if verdict is not True:
                return False
        return True

    @staticmethod
    def _detect_compile_error(runs: list[list[ExecutionResult]], num_tests: int):
        """Return a compile-error string if the executor signals a genuine
        compile/define failure, else None.

        The executor labels syntax/import/entrypoint-definition failures as
        ``compile_error`` and ordinary call exceptions as ``runtime_error``.
        """
        if num_tests == 0 or not runs:
            return None
        run = runs[0]
        if not run or len(run) != num_tests:
            return None
        if not all(r.error and r.error_kind == "compile_error" for r in run):
            return None
        return next((r.error for r in run if r.error), "compile error")
