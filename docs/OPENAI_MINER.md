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

# Optional secondary OpenAI-compatible endpoint
OPENAI_FALLBACK_BASE_URL=https://second-provider.example/v1
OPENAI_FALLBACK_MODEL=<fallback-chat-completions-model-id>
OPENAI_FALLBACK_API_KEY=<fallback-provider-key>

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

When configured, the primary and fallback endpoints are called concurrently
with the same absolute request deadline and retry policy. The first valid
completion is used; if both complete in the same event-loop turn, the primary
result is preferred. This means a healthy fallback can answer immediately when
the primary is slow or unavailable, without waiting for the primary retries to
exhaust first.

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

The miner makes one draft request and, when `MINER_SELF_VERIFY=true`, one
independent review request. A malformed or timed-out review preserves the draft;
there is no model-generated self-test or repair loop.

To exercise the real OpenAI-compatible provider with all five samples, including
the signed miner endpoint and one review, run from the repository root:

```bash
PYTHONPATH=. .venv/bin/python scripts/smoke_miner_samples.py --provider openai
```

The command places only the first case in each model prompt and runs all sample
cases after the signed response for reporting. It logs provider-call times,
signed-response latency, and per-case results. Use
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
`request.json`, the draft/review source, and submitted source under the printed
artifact directory. In this mode, `request.json` has an empty `public_examples`
list.

## Solver prompt

Python and Rust use separate, compact system prompts that mirror their actual
validator sandboxes. The draft and review each return one complete source block.
The public statement,
entrypoint metadata, and examples are placed in separate tagged sections so
their roles remain clear on long challenges.

Public examples are explicitly described as non-exhaustive. The prompt does
not claim access to hidden cases and no hidden challenge data crosses the miner
wire contract. Evaluate future prompt revisions on a fixed held-out task set;
change one instruction group at a time and compare full-suite accuracy,
latency, output validity, and cost.

By default, the miner reserves up to `MINER_SELF_VERIFY_RESERVE_S` seconds of
the validator deadline for one independent review. Set
`MINER_SELF_VERIFY=false` for one-pass operation.

Reasoning remains available to models through `OPENAI_REASONING_EFFORT`, but
provider-visible reasoning is suppressed before the miner signs its response.
When a Python or Rust fenced block is present, the first matching solution is
retained; prose and other fenced planning artifacts are discarded.
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
MINER_METAGRAPH_SYNC_S=300
```

Provider calls use one absolute budget bounded by both the validator's task
deadline and `OPENAI_REQUEST_TIMEOUT_S`. HTTP 408, 409, 429, and 5xx responses,
timeouts, and network transport failures are retried with short exponential
backoff within that same budget. If both concurrent providers fail, the miner
returns a valid signed empty solution, which safely scores zero without breaking
the validator round.
