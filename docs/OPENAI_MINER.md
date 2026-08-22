# OpenAI-compatible Hone miner

This miner keeps Hone's validator authentication, replay protection, axon
advertisement, response signing, and Python/Rust wire contract. Only the model
provider boundary changes: public tasks are sent to an OpenAI-compatible
`POST /chat/completions` endpoint.

The implementation uses `httpx` directly rather than a provider SDK, so the
same code can target OpenAI or another service implementing the common chat
completions request and response shape. The standard endpoint accepts a model
and a list of messages and returns assistant content in
`choices[0].message.content`; see the
[official API reference](https://developers.openai.com/api/reference/cli/resources/chat/subresources/completions).

## Install

Python 3.10 through 3.12 is supported:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e '.[chain,miner,dev]'
cp .env.example .env
```

## Configure

Set the provider, chain identity, and public endpoint in `.env`:

```dotenv
OPENAI_API_KEY=<provider-key>
OPENAI_REQUIRE_API_KEY=true
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=<chat-completions-model-id>
OPENAI_MAX_TOKENS=16384
OPENAI_MAX_TOKENS_PARAM=max_tokens
OPENAI_TEMPERATURE=
OPENAI_REASONING_EFFORT=
OPENAI_REQUEST_TIMEOUT_S=280
OPENAI_MAX_RETRIES=2
MINER_SELF_VERIFY=true
MINER_SELF_VERIFY_RESERVE_S=90
MINER_SELF_VERIFY_MAX_ATTEMPTS=3

# Optional secondary OpenAI-compatible endpoint
OPENAI_FALLBACK_BASE_URL=https://second-provider.example/v1
OPENAI_FALLBACK_MODEL=<fallback-chat-completions-model-id>
OPENAI_FALLBACK_API_KEY=<fallback-provider-key>
OPENAI_FALLBACK_RESERVE_S=60

NETUID=5
SUBTENSOR_NETWORK=finney
WALLET_NAME=<wallet-name>
WALLET_HOTKEY=<registered-miner-hotkey>

AXON_HOST=0.0.0.0
AXON_PORT=8091
AXON_EXTERNAL_IP=<public-ip-if-auto-detection-is-unsuitable>
```

`OPENAI_BASE_URL` may either be the API root, such as
`https://provider.example/v1`, or the complete
`https://provider.example/v1/chat/completions` URL. The miner appends
`/chat/completions` only when it is absent.

Fallback is disabled when `OPENAI_FALLBACK_BASE_URL` and
`OPENAI_FALLBACK_MODEL` are both empty. Set both to enable it. The fallback key
is optional: an empty `OPENAI_FALLBACK_API_KEY` reuses `OPENAI_API_KEY`, which
is convenient when both models are served by the same provider. Use a separate
key when the URLs belong to different providers.

The primary endpoint receives the first attempt and all retries. Only after it
has exhausted that policy does the client call the fallback endpoint. The
fallback uses the same `OPENAI_MAX_RETRIES` policy. A fast primary failure
leaves almost the entire request budget for fallback; a slow primary cannot
consume the last `OPENAI_FALLBACK_RESERVE_S` seconds. The reserve is capped at
half of the actual validator/provider budget, and setting it to `0` disables
the reservation without disabling fallback.

The default omits `temperature` and `reasoning_effort` because those optional
fields are not uniformly supported by compatible servers. Configure them only
when the selected provider/model accepts them. Change
`OPENAI_MAX_TOKENS_PARAM` to `max_completion_tokens` when required by the
model. Optional provider extensions can be supplied as JSON objects:

```dotenv
OPENAI_EXTRA_HEADERS_JSON='{"X-Provider-Feature":"enabled"}'
OPENAI_EXTRA_BODY_JSON='{"top_p":0.95}'
```

Core fields such as `model`, `messages`, `stream`, token limits,
`Authorization`, and `Content-Type` cannot be overridden through these escape
hatches.

For a local server without authentication, set a placeholder key or explicitly
use:

```dotenv
OPENAI_REQUIRE_API_KEY=false
OPENAI_API_KEY=
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_ALLOW_INSECURE_HTTP=true
```

Keep `OPENAI_ALLOW_INSECURE_HTTP=false` for remote providers so bearer tokens
are never sent over plaintext HTTP.

## Run

The hotkey must already be registered on the configured subnet. Start the
miner from the repository root:

```bash
./start_miner.sh
```

The process verifies registration, advertises the configured axon, and serves
the signed Hone endpoint. Check the local health route with:

```bash
curl http://127.0.0.1:8091/health
```

The response names the configured model but never exposes the API key.

The miner also archives each authenticated, authorized, schema-valid public task
to `MINER_TASK_ARCHIVE_FILE` (default `data/miner_tasks.jsonl`) before model
inference. Records are JSONL and include a semantic task fingerprint plus
non-secret receipt metadata. Hidden tests are not sent to miners, so this archive
contains only the public task and cannot recover the private evaluation suite.

With `MINER_SELF_TEST=true`, the initial response contains source and a fixed
model-generated test suite. The miner executes it with public examples and
operator-owned tests from `MINER_SELF_TEST_FILE`. A pass is returned immediately;
a failure enters a repair/retest loop bounded by
`MINER_SELF_VERIFY_MAX_ATTEMPTS` and the signed request deadline. Tests remain
fixed across repairs. This is local validation only and does not expose
validator hidden tests or results. A missing combined-response suite triggers a
bounded test-only request, and schema-valid generic `json` fences are accepted.
If the loop exhausts its attempts or deadline, it submits the candidate with the
fewest local failures rather than blank source.

To exercise the real OpenAI-compatible provider with all five samples, including
the signed miner endpoint, self-review, and Docker self-tests, run from the
repository root:

```bash
PYTHONPATH=. .venv/bin/python scripts/smoke_miner_samples.py --provider openai
```

The command places only the first case in each model prompt and loads the
remaining sample cases as operator-owned local tests. It logs provider-call
times, local-test times, signed-response latency, and per-case results. Use
`--example NAME` to run a subset. It exits nonzero when a returned solution
fails any case. Structured results are appended to
`data/miner_sample_smoke.jsonl`; generated draft, review, and submitted source
files are retained under `data/miner_sample_smoke_artifacts/<run-id>/`.

To expose only the statement and let the miner generate every solve-time case:

```bash
PYTHONPATH=. .venv/bin/python scripts/smoke_miner_samples.py \
  --provider openai --example extent-journal --problem-only
```

The sample's authored cases are withheld until post-response reporting. Inspect
`request.json`, `generated_tests.json`, the draft/review source, and submitted
source under the printed artifact directory. In this mode, `request.json` has an
empty `public_examples` list; repeated repairs are retained as `repair_N.*`.

## Solver prompt

Python and Rust use separate, compact system prompts that mirror their actual
validator sandboxes. The initial call asks for a fenced `self-tests` JSON block
and a fenced source block; repair calls return one complete source block. The
public statement,
entrypoint metadata, and examples are placed in separate tagged sections so
their roles remain clear on long challenges.

Public examples are explicitly described as non-exhaustive. The prompt does
not claim access to hidden cases and no hidden challenge data crosses the miner
wire contract. Evaluate future prompt revisions on a fixed held-out task set;
change one instruction group at a time and compare full-suite accuracy,
latency, output validity, and cost.

By default, the miner reserves up to `MINER_SELF_VERIFY_RESERVE_S` seconds of
the validator deadline for repair. Each failed candidate and its test evidence
are reviewed and retested until it passes, the attempt cap is reached, or the
deadline margin is exhausted. A candidate that already passed skips review.
When no tests execute, the existing single independent review remains as a
fail-open fallback. Set `MINER_SELF_VERIFY=false` for one-pass operation.

Reasoning remains available to models through `OPENAI_REASONING_EFFORT`, but
provider-visible reasoning is suppressed before the miner signs its response.
When a Python or Rust fenced block is present, the first matching solution and
optional `self-tests` block are retained; prose and other fenced planning
artifacts are discarded.
For unfenced responses, common `<think>`, `<thinking>`, `<thought>`,
`<reasoning>`, and `<analysis>` blocks are removed. Provider-specific reasoning
fields outside `message.content` are never forwarded. A response containing
only reasoning is treated as a provider failure so configured fallback can run.

## Request safety and capacity

The default authorization policy accepts only registered hotkeys that
currently hold a validator permit. Requests are recipient-bound,
signature-checked, replay-protected, body-limited, and concurrency-limited.

```dotenv
MINER_REQUIRE_VALIDATOR_PERMIT=true
MINER_MIN_STAKE=0
MINER_MAX_CONCURRENT_REQUESTS=4
MINER_MAX_REQUEST_BYTES=1000000
MINER_TASK_ARCHIVE_FILE=data/miner_tasks.jsonl
MINER_SELF_TEST=true
MINER_SELF_TEST_FILE=data/miner_tests.jsonl
MINER_SELF_TEST_MAX_GENERATED_CASES=8
MINER_SELF_TEST_EXECUTOR=docker
MINER_SELF_TEST_TIMEOUT_S=5
MINER_METAGRAPH_SYNC_S=300
```

Provider calls use one absolute budget bounded by both the validator's task
deadline and `OPENAI_REQUEST_TIMEOUT_S`. HTTP 408, 409, 429, and 5xx responses,
timeouts, and network transport failures are retried with short exponential
backoff within that same budget. Other terminal primary errors also trigger
fallback immediately. If fallback also fails, the miner returns a valid signed
empty solution, which safely scores zero without breaking the validator round.
