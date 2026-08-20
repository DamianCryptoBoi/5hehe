# Validator protocol and trust model

## Scope

This repository implements the validator portion of the subnet and includes a
small demo miner as a protocol reference. The validator treats the challenge
source and every miner as untrusted remote systems. It accepts no remote grading
claim: it executes committed candidate code against the revealed evaluation
cases in its own sandbox and derives rewards locally.

## Challenge lifecycle

1. The validator signs a retry-safe lease request to the configured HTTPS
   problem service.
2. The service returns a public task, a challenge ID, a deadline, and
   response-commit thresholds. Hidden evaluation cases are absent.
3. The validator rotation-deals each signed `TaskRequest` to half of the serving
   miners. A challenge's commit quorum can raise that sample, and operators may
   configure a fixed count.
4. It retains each exact signed response body and authentication headers.
5. It commits that immutable response set to the problem service.
6. A valid commit token unlocks the hidden cases for that challenge.
7. The validator executes every committed candidate locally and computes its
   pass/fail result.
8. Verified outcomes update the validator's recent score window and eventually its chain
   weights. Aggregate feedback to the service is advisory only.

Lease and commit operations are idempotent. Challenge ownership, expiry,
registered UID/hotkey pairs, response signatures, request IDs, uniqueness, and
minimum response counts are checked before reveal.
A rejected lease consumes no problem. A server `Retry-After` defers only future
leases on a monotonic clock; the validator continues its event loop and weight
schedule during that delay. Ordinary failed miner responses remain part of the
committed response set so contact count, returned responses, and signed
responses stay auditable.

A rejected commit reports the service's stated reason and is not retried. An
accepted commit is never repeated with another response set. A successful
receipt whose count differs from the sent set is treated as a protocol error.

## Security boundaries

- Problem-service requests are signed and bound to the validator hotkey. HTTPS
  authenticates responses in production; plain HTTP is an explicit local-test
  escape hatch.
- Miner requests and responses are signed for their intended recipient.
- Hidden cases are unavailable until miner responses have been committed.
- Candidate code runs without network access and with time, memory, process,
  and output limits when Docker execution is selected.
- Pass/fail comparison occurs in the trusted parent process rather than inside
  candidate code.
- The Docker grader starts one container per submitted program. Source is
  compiled once, and each hidden case runs in a fresh child process with its
  own deadline. The current release uses one verification pass.
- Invalid, missing, oversized, unauthenticated, or timed-out responses receive
  zero verified reward and cannot crash an entire round.
- Score state is keyed by hotkey so UID reassignment does not inherit another
  participant's history.

## Reward and weights

Correctness is determined by the complete hidden suite. A fully correct
submission receives a score between 0.95 and 1.0 using observed response speed
as a small tiebreaker. Wrong, partial, invalid, missing, or timed-out responses
receive zero. Peer pass-rates do not change an individual miner's score.
Optional partial-credit rewards affect exported dataset labels only. Validator
weights require the complete hidden suite to pass.

TODO(V2): Revisit duplication-resistant scoring only with a robust signal.
Simple AST fingerprints encourage irrelevant structural changes in submitted
solutions, so V1 deliberately excludes them from scoring and difficulty
classification.

The resulting per-UID payments update a rolling score containing the latest
200 completed-problem observations for that registration. Until 200 problems
have been attempted, every available observation is retained. A minimum
denominator of four prevents a few early responses from receiving a full score,
and weight submission begins after four completed observations. A UID's history
is cleared when its registered hotkey changes, so a new registration never
inherits the previous miner's results.
Validators periodically normalize those scores and submit weights through
Bittensor, but only after the release-policy minimum number of completed
observations is present in the authoritative score histories.
The SN5 release cadence is 38 blocks (about 7.5 minutes) for challenge
attempts and 180 blocks for weights. The problem service uses `Retry-After` to
enforce per-validator pacing and shared global-slot contention. The effective weight
cadence is `max(release interval, chain rate limit + 20)`; 180 is safely
above the known 100-block limit and allows roughly two attempts per 360-block
tempo.

The release owner burn share is 0%. Validators submit the normalized miner
score vector without an owner reservation and skip an all-zero vector. Weight
submission waits for inclusion in a worker thread, but not finalization, so the
synchronous SDK call does not block the event loop.

Rounds with an elevated sandbox-error rate run an independent known-good
sandbox probe before scores are changed. A failed probe discards the grading
event; a successful probe treats those errors as candidate failures. This keeps
a broken local executor from manufacturing an all-zero miner history without
giving candidate code the authority to declare the sandbox unhealthy.

## Availability behavior

The validator fails closed when no problem-service URL is configured. An
exhausted or unavailable service produces no challenge and therefore no grading
event. On a completed challenge, previously observed non-serving miners may
receive a zero according to validator configuration; paced or empty rounds do
not change scores. Remote systems cannot directly set or report scores.
