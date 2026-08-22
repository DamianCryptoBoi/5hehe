#!/usr/bin/env python3
"""Send sample Python and Rust tasks through the signed real miner flow.

The first sample case is included as public context. All cases are executed only
after the signed response, so this exercises provider inference, optional
self-review, the signed endpoint, and the configured language executor without
feeding evaluation results back into the solver. Pass ``--problem-only`` to
hide every authored sample case from the miner. The default selection covers
all five samples.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import shutil
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

import httpx

from rlvr.config import Settings
from rlvr.neurons.demo_miner import (
    DemoMiner,
    DemoMinerSettings,
    GLM52Client,
    build_demo_miner_app,
    extract_python,
    extract_rust,
)
from rlvr.neurons.openai_miner import (
    OpenAICompatibleClient,
    OpenAICompatibleSettings,
)
from rlvr.neurons.live import LiveSolverClient
from rlvr.protocol import TaskRequest
from rlvr.types import Problem, SolutionResponse, TestCase


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = REPO_ROOT / "examples" / "sample_challenges"
DEFAULT_EXAMPLES = (
    "asset-rebuild-planner",
    "extent-journal",
    "reactive-stat-board",
    "revocable-verification-gate",
    "sparse-circular-array",
)


def load_sample(
    name: str,
) -> tuple[dict[str, Any], str, list[TestCase], list[str]]:
    sample_dir = SAMPLE_ROOT / name
    problem_path = sample_dir / "PROBLEM.md"
    cases_path = sample_dir / "cases.json"
    if (
        not sample_dir.is_dir()
        or not problem_path.is_file()
        or not cases_path.is_file()
        or sample_dir.parent != SAMPLE_ROOT
    ):
        raise ValueError(f"unknown sample challenge: {name}")
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    raw_cases = payload["cases"]
    cases = [TestCase.model_validate(case) for case in raw_cases]
    if len(cases) < 2:
        raise ValueError(f"sample challenge {name!r} needs at least two cases")
    labels = [
        str(case.get("name", f"case {index}"))
        for index, case in enumerate(raw_cases, 1)
    ]
    return payload, problem_path.read_text(encoding="utf-8"), cases, labels


def build_provider(
    provider: str,
    env_file: str,
    archive_file: str,
    *,
    review: bool,
):
    common = {
        "_env_file": env_file,
        "miner_task_archive_file": archive_file,
        "miner_self_verify": review,
    }
    if provider == "openai":
        settings = OpenAICompatibleSettings(**common)
        return settings, OpenAICompatibleClient(settings)
    settings = DemoMinerSettings(**common)
    return settings, GLM52Client(settings)


class _WalletLike:
    def __init__(self, hotkey: Any):
        self.hotkey = hotkey


class _FakeMetagraph:
    def __init__(self, validator_address: str):
        self.hotkeys = [validator_address]
        self.S = [100.0]
        self.validator_permit = [True]


class _Timeline:
    def __init__(self):
        self.events: list[tuple[str, float, float]] = []
        self.ai_outputs: list[tuple[str, str]] = []

    def add(self, label: str, started: float, finished: float) -> None:
        self.events.append((label, started, finished))

    def clear(self) -> None:
        self.events.clear()
        self.ai_outputs.clear()


class _TimedProvider:
    def __init__(self, inner: Any, timeline: _Timeline):
        self.inner = inner
        self.timeline = timeline

    @property
    def model_name(self):
        return self.inner.model_name

    @property
    def request_timeout_s(self):
        return self.inner.request_timeout_s

    @property
    def log_prefix(self):
        return self.inner.log_prefix

    async def complete(self, messages, *, timeout_s):
        if any(message.get("role") == "assistant" for message in messages):
            stage = "review-ai"
        else:
            stage = "draft-ai"
        started = time.perf_counter()
        try:
            output = await self.inner.complete(messages, timeout_s=timeout_s)
            self.timeline.ai_outputs.append((stage, output))
            return output
        finally:
            self.timeline.add(stage, started, time.perf_counter())

    async def aclose(self):
        await self.inner.aclose()


def _extract_sample_code(language: str, raw: str) -> str:
    return extract_rust(raw) if language == "rust" else extract_python(raw)


def _write_code(path: Path, code: str) -> str:
    if not code:
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code.rstrip() + "\n", encoding="utf-8")
    return str(path)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def preflight(selected: list[str], *, skip_docker: bool) -> None:
    if not skip_docker:
        docker = shutil.which("docker")
        if not docker:
            raise RuntimeError("Docker CLI not found on PATH")
        try:
            result = subprocess.run(
                [docker, "info"], capture_output=True, text=True, timeout=20
            )
        except OSError as error:
            raise RuntimeError(f"could not run docker info: {error}") from error
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-500:]
            raise RuntimeError(
                "Docker daemon is unavailable; start Docker before running the "
                f"real smoke test ({detail})"
            )
    if not selected:
        raise ValueError("select at least one sample")


def _request_for_sample(
    name: str,
    payload: dict[str, Any],
    statement: str,
    cases: list[TestCase],
    deadline_s: float,
    *,
    problem_only: bool = False,
) -> TaskRequest:
    return TaskRequest(
        problem_id=f"sample-smoke-{name}",
        language=payload["language"],
        statement=statement,
        entrypoint=payload["entrypoint"],
        public_examples=[] if problem_only else [cases[0]],
        deadline_s=deadline_s,
    )


async def run_sample(
    miner: DemoMiner,
    validator_client: LiveSolverClient,
    name: str,
    payload: dict[str, Any],
    statement: str,
    cases: list[TestCase],
    labels: list[str],
    deadline_s: float,
    timeline: _Timeline,
    run_id: str,
    log_file: Path,
    artifacts_root: Path,
    *,
    problem_only: bool,
) -> bool:
    public_examples = [] if problem_only else [cases[0]]
    problem = Problem(
        problem_id=f"sample-smoke-{name}",
        language=payload["language"],
        statement=statement,
        entrypoint=payload["entrypoint"],
        tests=cases,
        public_examples=public_examples,
    )
    timeline.clear()
    sample_started_at = datetime.now(timezone.utc).isoformat()
    wire_started = time.perf_counter()
    artifact = await validator_client.solve_signed(
        problem,
        request_id=f"sample-smoke-{name}-{uuid4().hex}",
    )
    timeline.add("signed-submit", wire_started, time.perf_counter())
    solution: SolutionResponse = artifact.to_solution(problem.problem_id)
    request = _request_for_sample(
        name,
        payload,
        statement,
        cases,
        deadline_s,
        problem_only=problem_only,
    )
    from rlvr.execution.executor import get_executor
    from types import SimpleNamespace

    executor = get_executor(
        SimpleNamespace(executor="docker"), language=payload["language"]
    )
    report_started = time.perf_counter()
    results = await asyncio.to_thread(
        executor.run_tests,
        solution.code,
        request.entrypoint,
        cases,
        5.0,
    )
    report_finished = time.perf_counter()
    sample_finished = time.perf_counter()
    passed = sum(result.passed for result in results)
    submit_status = "PASS" if not artifact.error else "FAIL"
    ordered_events = sorted(timeline.events, key=lambda event: event[1])
    stage_records: list[dict[str, Any]] = []
    for label, started, finished in ordered_events:
        if label in {"draft-ai", "review-ai"}:
            display_label = label
        else:
            continue
        duration_ms = (finished - started) * 1000.0
        stage = {"stage": display_label, "duration_ms": round(duration_ms, 3)}
        stage_records.append(stage)
        print(f"  {display_label}: {duration_ms:.0f} ms")
    submit_detail = artifact.error or "validator accepted signed response"
    submit_duration_ms = solution.latency_ms
    stage_records.append(
        {
            "stage": "signed-submit",
            "duration_ms": round(submit_duration_ms, 3),
            "accepted": not bool(artifact.error),
        }
    )
    print(
        f"  signed-submit: {submit_status} ({submit_duration_ms:.0f} ms; "
        f"{submit_detail})"
    )
    report_duration_ms = (report_finished - report_started) * 1000.0
    stage_records.append(
        {
            "stage": "tests-report",
            "duration_ms": round(report_duration_ms, 3),
            "passed": passed,
            "total": len(results),
        }
    )
    print(f"  tests-report: {report_duration_ms:.0f} ms")
    print(
        f"[{name}] language={payload['language']} model={miner.model_name} "
        f"response_bytes={len(solution.code.encode('utf-8'))} "
        f"wire_latency_ms={solution.latency_ms:.0f} "
        f"cases={passed}/{len(results)}"
    )
    for index, (result, label) in enumerate(zip(results, labels, strict=True), 1):
        if result.passed:
            print(f"  PASS {index}: {label}")
        else:
            detail = result.error or f"returned {result.actual_repr}"
            print(f"  FAIL {index}: {label} ({detail[:300]})")

    sample_artifacts = artifacts_root / run_id / name
    suffix = ".rs" if payload["language"] == "rust" else ".py"
    draft_output = next(
        (
            model_output
            for stage, model_output in timeline.ai_outputs
            if stage == "draft-ai"
        ),
        "",
    )
    draft_code = _extract_sample_code(payload["language"], draft_output)
    review_codes = [
        _extract_sample_code(payload["language"], model_output)
        for stage, model_output in timeline.ai_outputs
        if stage == "review-ai"
    ]
    code_files = {
        "request": _write_code(
            sample_artifacts / "request.json",
            request.model_dump_json(indent=2),
        ),
        "draft": _write_code(sample_artifacts / f"draft{suffix}", draft_code),
        "review": _write_code(
            sample_artifacts / f"review{suffix}",
            review_codes[0] if review_codes else "",
        ),
        "submitted": _write_code(
            sample_artifacts / f"submitted{suffix}", solution.code
        ),
    }
    accepted = (
        not artifact.error
        and bool(solution.code)
        and bool(results)
        and passed == len(results)
    )
    _append_jsonl(
        log_file,
        {
            "schema_version": 1,
            "run_id": run_id,
            "started_at": sample_started_at,
            "sample": name,
            "language": payload["language"],
            "model": miner.model_name,
            "problem_only": problem_only,
            "accepted": accepted,
            "response_bytes": len(solution.code.encode("utf-8")),
            "wire_latency_ms": round(submit_duration_ms, 3),
            "total_elapsed_ms": round((sample_finished - wire_started) * 1000.0, 3),
            "signed_response_error": artifact.error,
            "stages": stage_records,
            "cases": [
                {
                    "index": index,
                    "name": label,
                    "passed": result.passed,
                    "runtime_ms": round(result.runtime_ms, 3),
                    "error": result.error,
                    "actual_repr": result.actual_repr,
                }
                for index, (result, label) in enumerate(
                    zip(results, labels, strict=True), 1
                )
            ],
            "code_files": code_files,
        },
    )
    return accepted


async def async_main(args: argparse.Namespace) -> int:
    selected = list(args.examples or DEFAULT_EXAMPLES)
    if args.all:
        selected = sorted(
            path.name
            for path in SAMPLE_ROOT.iterdir()
            if path.is_dir() and (path / "PROBLEM.md").is_file()
        )
    preflight(selected, skip_docker=args.skip_docker_preflight)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid4().hex[:8]}"
    log_file = Path(args.log_file).resolve()
    artifacts_root = Path(args.artifacts_dir).resolve()
    run_started = time.perf_counter()
    print(
        f"run_id={run_id} provider={args.provider} samples={len(selected)} "
        f"review={'on' if not args.no_review else 'off'} "
        f"input={'problem-only' if args.problem_only else 'one-public-example'}"
    )
    print(f"log_file={log_file}")
    print(f"artifacts_dir={artifacts_root / run_id}")

    with TemporaryDirectory(prefix="hone-miner-samples-") as directory:
        loaded: list[tuple[str, dict[str, Any], str, list[TestCase], list[str]]] = []
        for name in selected:
            payload, statement, cases, labels = load_sample(name)
            loaded.append((name, payload, statement, cases, labels))

        archive_file = str(Path(directory) / "miner_tasks.jsonl")
        settings, client = build_provider(
            args.provider,
            args.env_file,
            archive_file,
            review=not args.no_review,
        )
        try:
            from bittensor_wallet import Keypair
        except ImportError as error:
            raise RuntimeError(
                "the signed smoke test requires bittensor_wallet; install the chain extras"
            ) from error

        validator_key = Keypair.create_from_uri("//Alice")
        miner_key = Keypair.create_from_uri("//Bob")
        timeline = _Timeline()
        timed_client = _TimedProvider(client, timeline)
        miner = DemoMiner(
            settings,
            timed_client,
            wallet=_WalletLike(miner_key),
            metagraph=_FakeMetagraph(validator_key.ss58_address),
        )
        app = build_demo_miner_app(miner)
        http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://sample-miner",
        )
        validator_settings = Settings(
            _env_file=None,
            solve_deadline_s=args.deadline,
        )
        validator_client = LiveSolverClient(
            uid=0,
            hotkey=miner_key.ss58_address,
            url="http://sample-miner",
            wallet=_WalletLike(validator_key),
            settings=validator_settings,
            http=http,
        )
        try:
            outcomes: list[bool] = []
            for name, payload, statement, cases, labels in loaded:
                print(f"\n[{name}] starting")
                try:
                    outcome = await run_sample(
                        miner,
                        validator_client,
                        name,
                        payload,
                        statement,
                        cases,
                        labels,
                        args.deadline,
                        timeline,
                        run_id,
                        log_file,
                        artifacts_root,
                        problem_only=args.problem_only,
                    )
                except Exception as error:  # noqa: BLE001 - continue remaining samples
                    outcome = False
                    print(f"[{name}] ERROR: {type(error).__name__}: {error}")
                    _append_jsonl(
                        log_file,
                        {
                            "schema_version": 1,
                            "run_id": run_id,
                            "started_at": datetime.now(timezone.utc).isoformat(),
                            "sample": name,
                            "language": payload["language"],
                            "model": miner.model_name,
                            "accepted": False,
                            "error": f"{type(error).__name__}: {error}",
                        },
                    )
                outcomes.append(outcome)

            archive_path = Path(archive_file)
            archived = (
                archive_path.read_text(encoding="utf-8").splitlines()
                if archive_path.is_file()
                else []
            )
            archive_ok = len(archived) == len(loaded)
            archive_status = "PASS" if archive_ok else "FAIL"
            print(
                f"\nsigned request/archive path: {archive_status} "
                f"({len(archived)}/{len(loaded)} task(s))"
            )
        finally:
            await http.aclose()
            await miner.aclose()
    total_elapsed_ms = (time.perf_counter() - run_started) * 1000.0
    passed_samples = sum(outcomes)
    _append_jsonl(
        log_file,
        {
            "schema_version": 1,
            "record_type": "summary",
            "run_id": run_id,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "provider": args.provider,
            "problem_only": args.problem_only,
            "samples_passed": passed_samples,
            "samples_total": len(outcomes),
            "archive_ok": archive_ok,
            "total_elapsed_ms": round(total_elapsed_ms, 3),
        },
    )
    status = "PASS" if all(outcomes) and archive_ok else "FAIL"
    print(
        f"run {status}: samples={passed_samples}/{len(outcomes)} "
        f"total_ms={total_elapsed_ms:.0f}"
    )
    return 0 if all(outcomes) and archive_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=("openai", "glm"),
        default="openai",
        help="real provider configured by .env (default: openai)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="settings file passed to the selected provider",
    )
    parser.add_argument(
        "--example",
        dest="examples",
        action="append",
        help="sample name; repeat to select multiple (default: all five samples)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="run every sample challenge (more provider requests)",
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="disable the second model verification request",
    )
    parser.add_argument(
        "--problem-only",
        action="store_true",
        help=(
            "send no authored cases to the miner; sample cases run afterward "
            "for reporting only"
        ),
    )
    parser.add_argument("--deadline", type=float, default=300.0)
    parser.add_argument(
        "--log-file",
        default="data/miner_sample_smoke.jsonl",
        help="append structured results and timings to this JSONL file",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="data/miner_sample_smoke_artifacts",
        help="directory for generated draft, review, and submitted source",
    )
    parser.add_argument(
        "--skip-docker-preflight",
        action="store_true",
        help="skip the fail-fast Docker daemon check",
    )
    args = parser.parse_args()
    try:
        return asyncio.run(async_main(args))
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
