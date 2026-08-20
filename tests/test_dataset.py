"""Tests for rlvr.dataset.writer.RolloutWriter.

These tests construct synthetic ChallengeResults (no network), write
Rollouts to a tmp dir, and assert JSONL round-trips and restart behavior.
"""

from __future__ import annotations

import json
import os

import pytest

from rlvr.dataset.writer import RolloutWriter
from rlvr.types import (
    ChallengeResult,
    DifficultyBand,
    ExecutionResult,
    MinerOutcome,
    Problem,
    SolutionResponse,
    TestCase as Case,
    Verification,
)


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #
def make_problem(problem_id: str = "p1") -> Problem:
    return Problem(
        problem_id=problem_id,
        language="python",
        statement=(
            "Return the sum of two integers a and b.\n"
            "Implement a function `add`. For example, add(1, 2) == 3."
        ),
        entrypoint="add",
        tests=[
            Case(args=[1, 2], expected=3),
            Case(args=[10, 5], expected=15),
        ],
        public_examples=[Case(args=[1, 2], expected=3)],
        created_at=123.0,
    )


def make_outcome(uid: int, passed: bool, num_tests: int = 2) -> MinerOutcome:
    num_passed = num_tests if passed else 0
    reward = 1.0 if passed else 0.0
    results = [
        ExecutionResult(test_index=i, passed=passed) for i in range(num_tests)
    ]
    verification = Verification(
        problem_id="p1",
        num_tests=num_tests,
        num_passed=num_passed,
        all_passed=passed,
        reward=reward,
        results=results,
    )
    solution = SolutionResponse(
        problem_id="p1",
        code="def add(a, b):\n    return a + b\n",
        raw_response=f"```python\ndef add(a, b):\n    return a + b\n```  # uid={uid}",
    )
    return MinerOutcome(
        uid=uid,
        hotkey=f"hk{uid}",
        solution=solution,
        verification=verification,
        reward=reward,
    )


def make_challenge_result(pass_rate: float, band: DifficultyBand) -> ChallengeResult:
    # 2 pass, 1 fail -> pass_rate driven by caller for clarity.
    outcomes = [
        make_outcome(uid=0, passed=True),
        make_outcome(uid=1, passed=True),
        make_outcome(uid=2, passed=False),
    ]
    return ChallengeResult(
        problem_id="p1",
        outcomes=outcomes,
        pass_rate=pass_rate,
        band=band,
        num_responses=len(outcomes),
    )


# --------------------------------------------------------------------------- #
# build_prompt
# --------------------------------------------------------------------------- #
def test_build_prompt_contains_all_public_content_and_no_hidden_tests(tmp_path):
    writer = RolloutWriter(str(tmp_path))
    problem = make_problem()
    prompt = writer.build_prompt(problem)
    assert prompt.startswith(problem.statement.strip())
    assert "Return the sum of two integers" in prompt
    assert "add(1, 2) == 3" in prompt  # example embedded by the author
    assert "Public examples (args, kwargs, expected)" in prompt
    assert '"expected": 3' in prompt
    # hidden test value (15) must NOT leak into the prompt
    assert "15" not in prompt


def test_build_prompt_is_deterministic(tmp_path):
    writer = RolloutWriter(str(tmp_path))
    problem = make_problem()
    assert writer.build_prompt(problem) == writer.build_prompt(problem)


# --------------------------------------------------------------------------- #
# to_rollouts
# --------------------------------------------------------------------------- #
def test_to_rollouts_one_per_outcome(tmp_path):
    writer = RolloutWriter(str(tmp_path))
    problem = make_problem()
    result = make_challenge_result(pass_rate=2 / 3, band=DifficultyBand.IN_BAND)
    rollouts = writer.to_rollouts(problem, result)

    assert len(rollouts) == 3
    for r in rollouts:
        assert r.problem_id == "p1"
        assert r.prompt == writer.build_prompt(problem)
        assert r.pool_pass_rate == pytest.approx(2 / 3)
        assert r.band == DifficultyBand.IN_BAND
        assert r.num_tests == 2

    passed = [r for r in rollouts if r.all_passed]
    failed = [r for r in rollouts if not r.all_passed]
    assert len(passed) == 2 and len(failed) == 1
    assert passed[0].reward == 1.0 and passed[0].num_passed == 2
    assert failed[0].reward == 0.0 and failed[0].num_passed == 0
    # completion carries the miner raw response
    assert "uid=0" in rollouts[0].completion


# --------------------------------------------------------------------------- #
# write + JSONL round-trip
# --------------------------------------------------------------------------- #
def test_write_jsonl_roundtrips(tmp_path):
    writer = RolloutWriter(str(tmp_path))
    problem = make_problem()
    result = make_challenge_result(pass_rate=2 / 3, band=DifficultyBand.IN_BAND)
    rollouts = writer.to_rollouts(problem, result)
    writer.write(rollouts)

    shard = os.path.join(str(tmp_path), "rollouts-00000.jsonl")
    assert os.path.exists(shard)

    with open(shard, "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]
    assert len(lines) == 3

    parsed = [json.loads(ln) for ln in lines]
    # every line is a full Rollout-shaped object
    for obj, src in zip(parsed, rollouts):
        assert obj["problem_id"] == src.problem_id
        assert obj["reward"] == src.reward
        assert obj["band"] == src.band.value  # enum serialized to string
        assert obj["prompt"] == src.prompt
        assert obj["completion"] == src.completion
        assert obj["num_tests"] == src.num_tests
        assert obj["miner_uid"] == src.miner_uid


def test_write_appends_across_calls(tmp_path):
    writer = RolloutWriter(str(tmp_path))
    problem = make_problem()
    result = make_challenge_result(pass_rate=2 / 3, band=DifficultyBand.IN_BAND)
    rollouts = writer.to_rollouts(problem, result)
    writer.write(rollouts)
    writer.write(rollouts)  # append again

    shard = os.path.join(str(tmp_path), "rollouts-00000.jsonl")
    with open(shard, "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]
    assert len(lines) == 6


# --------------------------------------------------------------------------- #
# Sharding
# --------------------------------------------------------------------------- #
def test_sharding_rolls_over(tmp_path):
    writer = RolloutWriter(str(tmp_path), shard_size=2)
    problem = make_problem()
    result = make_challenge_result(pass_rate=2 / 3, band=DifficultyBand.IN_BAND)
    rollouts = writer.to_rollouts(problem, result)
    writer.write(rollouts)  # 3 records, shard_size=2 -> 2 shards

    shard0 = os.path.join(str(tmp_path), "rollouts-00000.jsonl")
    shard1 = os.path.join(str(tmp_path), "rollouts-00001.jsonl")
    assert os.path.exists(shard0)
    assert os.path.exists(shard1)
    with open(shard0) as fh:
        assert len([ln for ln in fh if ln.strip()]) == 2
    with open(shard1) as fh:
        assert len([ln for ln in fh if ln.strip()]) == 1


def test_writer_resumes_appending_existing_shard(tmp_path):
    writer = RolloutWriter(str(tmp_path))
    problem = make_problem()
    result = make_challenge_result(pass_rate=2 / 3, band=DifficultyBand.IN_BAND)
    writer.write(writer.to_rollouts(problem, result))

    writer2 = RolloutWriter(str(tmp_path))
    writer2.write(writer2.to_rollouts(problem, result))
    shard = os.path.join(str(tmp_path), "rollouts-00000.jsonl")
    with open(shard) as fh:
        assert len([ln for ln in fh if ln.strip()]) == 6


def make_result(problem: Problem, passed_flags: list[bool]) -> ChallengeResult:
    """A ChallengeResult with one outcome per flag and a consistent pass_rate."""
    outcomes = [make_outcome(uid=i, passed=p) for i, p in enumerate(passed_flags)]
    rate = sum(passed_flags) / len(passed_flags)
    return ChallengeResult(
        problem_id=problem.problem_id,
        outcomes=outcomes,
        pass_rate=rate,
        band=DifficultyBand.IN_BAND,
        num_responses=len(outcomes),
    )


# --------------------------------------------------------------------------- #
# write_problem: the validator-private problem store (verifiable environments)
# --------------------------------------------------------------------------- #
def test_write_problem_persists_hidden_tests_and_stats(tmp_path):
    writer = RolloutWriter(str(tmp_path))
    problem = make_problem("p-store")
    problem.measured_pass_rate = 0.5
    problem.band = DifficultyBand.IN_BAND
    writer.write_problem(problem)
    writer.write_problem(make_problem("p-store-2"))

    path = os.path.join(str(tmp_path), "problems-00000.jsonl")
    assert os.path.exists(path)
    with open(path) as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    assert [r["problem_id"] for r in records] == ["p-store", "p-store-2"]
    # The audit store includes the revealed cases and measured difficulty.
    r = records[0]
    assert r["statement"] and r["entrypoint"] == "add"
    assert r["tests"][0]["expected"] == 3
    assert r["measured_pass_rate"] == 0.5
    assert r["band"] == "in_band"


def test_problem_store_shards_are_separate_from_rollouts(tmp_path):
    writer = RolloutWriter(str(tmp_path), shard_size=1)
    problem = make_problem("p1")
    result = make_result(problem, passed_flags=[True])
    writer.write(writer.to_rollouts(problem, result))
    writer.write_problem(problem)
    writer.write_problem(make_problem("p2"))
    # rollovers are tracked per prefix: rollouts and problems shard independently
    assert os.path.exists(os.path.join(str(tmp_path), "rollouts-00000.jsonl"))
    assert os.path.exists(os.path.join(str(tmp_path), "problems-00000.jsonl"))
    assert os.path.exists(os.path.join(str(tmp_path), "problems-00001.jsonl"))


def test_append_recovers_after_torn_jsonl_line(tmp_path):
    shard = tmp_path / "rollouts-00000.jsonl"
    shard.write_bytes(b'{"truncated":')
    writer = RolloutWriter(str(tmp_path))
    problem = make_problem("after-crash")
    writer.write(
        writer.to_rollouts(problem, make_result(problem, passed_flags=[True]))
    )

    lines = shard.read_text().splitlines()
    assert lines[0] == '{"truncated":'
    assert json.loads(lines[1])["problem_id"] == "after-crash"
