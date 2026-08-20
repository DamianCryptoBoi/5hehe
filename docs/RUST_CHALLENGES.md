# Rust challenge design

## Scope

Protocol V2 adds complete Rust programs to the existing Python challenge system.
A Rust submission defines `fn main()`, reads one test case from standard input,
and writes its answer to standard output. The Rust profile permits the standard library
only. Miners submit source code, never Cargo projects or compiled artifacts.

The protocol uses one coordinated cutover. Validators, miners, and the problem
service require an explicit language after that cutover; older messages are not
accepted.

## Challenge profiles

The problem service stores an explicit language for every challenge. Existing
records are migrated internally to `python`.

The language value `rust` identifies this complete profile:

- complete `fn main()` source;
- stdin/stdout execution;
- `rust-exact-token-v1` comparison;
- a release-pinned compiler environment and resource envelope.

Judge semantics and the wire contract must not change under the same token. An
incompatible judge or execution contract requires a new language/profile value
or a new protocol version. Compiler and limit values may advance through coordinated
validator releases when judge and wire semantics remain unchanged. Reference
solutions must be revalidated against the fleet's current compiler policy.

## Wire contract

### Lease request

The signed lease request remains:

```json
{"request_id": "..."}
```

There is no capability field or capability negotiation.

### Lease and reveal responses

Every lease and reveal response requires `language`, exactly `python` or
`rust`. A Rust lease also uses the `main` entrypoint:

```json
{"language": "rust", "entrypoint": "main"}
```

The challenge identity and language are immutable between lease and reveal.
Commit and feedback requests do not gain language fields; the server derives
language from the immutable challenge record keyed by `challenge_id`.

### Miner request

Every miner `TaskRequest` requires the same explicit `language`. Missing or
unknown values are invalid.

## Lease idempotency

Lease request state follows:

```text
UNBOUND -> ISSUED(challenge)
```

- A retry may attempt to lease again after pacing, an empty pool, or a
  temporary failure when no challenge was issued.
- Once `ISSUED`, every retry returns the exact original challenge.
- The challenge/request-ID association is committed durably before the issued
  response leaves the server.
- One request ID can never issue two challenges.
- Idempotency records expire after challenge expiry plus a documented margin.

A transient failure response is not cached permanently. Concurrent lease
requests cannot receive the same challenge.

## Test representation

Rust hidden tests and public examples retain the existing `TestCase` shape:

```json
{
  "args": ["complete stdin text"],
  "kwargs": {},
  "expected": "complete expected stdout text"
}
```

For Rust, `args` contains exactly one UTF-8 string, `kwargs` is empty, and
`expected` is a UTF-8 string. The statement completely specifies the stdin and
stdout formats.

Initial authoring ceilings are:

- at most 100 hidden tests;
- stdin at most 64 KiB per test;
- expected stdout at most 32 KiB per test;
- serialized body caps described below.

Total stdin and expected output are bounded in practice by the 2 MiB serialized
reveal cap, its 4 KiB headroom, and JSON escaping.

The server validates hidden tests and public examples before a challenge enters
the pool. The validator repeats public-example validation at lease time and
hidden-test validation after reveal.

## Exact-token judge

`rust-exact-token-v1` performs these steps:

1. Strictly decode candidate stdout as UTF-8. Invalid UTF-8 fails the test.
2. Split actual and expected output on runs of ASCII whitespace bytes `0x09`
   through `0x0D` and `0x20`.
3. Compare tokens as exact strings in order.
4. Require equal token counts.

Leading, trailing, and repeated ASCII whitespace are ignored. Empty output
matches empty expected output. Values such as `007` and `7` differ.

The initial Rust profile must not require floating-point-valued answer tokens. Problems use
integers, strings, or exact rational representations and require deterministic,
canonical output. Statements must not depend on hash iteration order, timing,
randomness, platform details, or undefined behavior.

The problem service runs the reference solution through the same judge and
pinned compiler policy. Every public example and hidden test must pass before
the challenge becomes leasable. Problems must be comfortably solvable within
the fixed per-test deadline; the Rust profile does not carry a challenge-specific time
limit.

## Serialized response limits

Limits apply to the final UTF-8 JSON body after field names, brackets, and JSON
escaping, excluding HTTP headers and transport chunking.

- Python lease body: at most 512,000 bytes.
- Python reveal body: at most 512,000 bytes.
- Rust lease body: at most 2 MiB.
- Rust reveal body: at most 2 MiB.

Authoring validation requires the serialized draft body to fit within the
applicable cap minus 4 KiB. The fixed headroom covers all late-bound fields,
including maximum-length identifiers and timestamps.

The validator response reader is release-controlled and accepts at least the
Rust cap. Candidate stdout and aggregate runner framing have separate limits.
Runner tests cover worst-case JSON escaping so legal raw output cannot overflow
the framed batch and fail unrelated cases.

## Rust sandbox

The Rust grader uses one digest-pinned image containing a pinned Rust toolchain
and the trusted Python supervisor. It contains no third-party Rust crates and
does not use Cargo for candidate compilation.

Each submission uses one network-disabled container:

1. Write the bounded source into ephemeral storage.
2. Invoke `rustc` once under a separate compilation deadline and resource
   envelope.
3. Reject an oversized artifact.
4. For each test, start the artifact in a fresh process group, send only that
   test's stdin, collect bounded stdout and stderr, enforce the per-test
   deadline, and kill and reap the process group.
5. Strictly decode stdout and return its text value to the host.
6. Compare on the host; expected output never enters the container.
7. Force-remove the container and artifact after grading.

Compilation failure produces a uniform compile failure for the submission.
One runtime failure, nonzero exit, timeout, invalid UTF-8 result, or oversized
output fails only that case. A nonzero exit fails regardless of stdout; a
program cannot print an answer and then crash successfully. Candidate output
never controls a trusted status frame.

The release policy pins:

- image digest;
- `rustc` version;
- Rust edition;
- exact compiler flags;
- compiler and execution deadlines;
- CPU, memory, process, temporary-storage, output, and artifact limits;
- response-read limit;
- judge version.

The release uses a 60-second compilation deadline, 2 GiB memory per container,
and 256 MiB temporary storage. Rust verification concurrency is reduced when
needed so the container caps fit in host memory with a 2 GiB reserve. Rust
grading requires a nominal 16 GiB host. Larger hosts permit more concurrent
Rust verification.

## Failure disposition

The validator abandons a challenge without grading or changing miner scores
when it sees:

- an unsupported challenge profile;
- malformed Rust public examples or hidden tests;
- a lease/reveal language or identity mismatch;
- a response beyond the release-controlled body limit;
- an unavailable or incorrect release-pinned Rust environment.

These are protocol, server, or validator faults and must not mass-zero miners.
Validly formed challenges grade individual miner compile and execution failures
normally.

## Coordinated cutover and Rust activation

1. Pause challenge issuance.
2. Wait at least the 900-second challenge TTL so no old lease remains live.
3. Deploy the V2 problem service, validators, and miners as one coordinated
   release. Archive the old problem-service lease-state file.
4. Verify an explicit-language Python challenge end to end before resuming
   issuance.
5. Resume Python issuance. Only append Rust challenges after the custom image
   is digest-pinned, preflight passes, the benchmark gate passes, miners are
   ready, and activation is explicitly approved.

Any miner that has not upgraded rejects the mandatory language field and can
receive zero on Python as well as Rust rounds. This consequence must be
accepted and communicated before scheduling the cutover.

A validator whose Rust canary fails abandons a prematurely issued Rust round
without dispatching or scoring it. This is a safety fallback, not capability
negotiation.

## Required validation

Wire and server tests cover:

- mandatory explicit language on lease, reveal, miner, and local problem
  models;
- a lease request containing only `request_id`;
- request-ID idempotency across pacing, empty pool, temporary failure, issue,
  retry, expiry, and concurrent requests;
- immutable language and judge version;
- the shared 2 MiB lease/reveal read limit, escaping, and headroom;
- Rust public-example and hidden-test shape and size validation;
- reference-solution gating under the pinned judge and compiler.

Validator tests cover:

- unsupported-profile and malformed-challenge abandonment without scoring;
- Rust and Python task signing with explicit language;
- compile success, compile failure, compile timeout, and artifact cap;
- fresh process state, runtime failure, timeout, process-group cleanup, stdout
  and stderr flooding, invalid UTF-8, and sibling-case continuation;
- expected values absent from all container payloads;
- exact-token judge vectors, including empty output and ASCII/non-ASCII
  whitespace boundaries;
- aggregate framing under worst-case JSON escaping;
- pinned release-policy environment and preflight behavior.

## Benchmark gate

The benchmark covers source sizes from small programs through approximately
50 KiB, 10/100 test cases, half-pool submission count, and concurrency 8/16/32.
It records end-to-end challenge wall time, throughput, and failure counts.
Host CPU, memory, temporary storage, and artifact-size margins require separate
monitoring. Adversarial cases include compile timeout, runaway execution,
output flooding, invalid UTF-8, and maximum escaped output.

No Rust challenge enters the active pool until the pinned environment and
limits pass this benchmark on validator-class hardware.
