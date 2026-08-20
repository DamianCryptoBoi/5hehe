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

## Solver prompt

Python and Rust use separate, compact system prompts that mirror their actual
validator sandboxes. The model is asked to reason privately about the full
specification, algorithm, complexity, public examples, and applicable boundary
cases before returning exactly one fenced source block. The public statement,
entrypoint metadata, and examples are placed in separate tagged sections so
their roles remain clear on long challenges.

Public examples are explicitly described as non-exhaustive. The prompt does
not claim access to hidden cases and no hidden challenge data crosses the miner
wire contract. Evaluate future prompt revisions on a fixed held-out task set;
change one instruction group at a time and compare full-suite accuracy,
latency, output validity, and cost.

## Request safety and capacity

The default authorization policy accepts only registered hotkeys that
currently hold a validator permit. Requests are recipient-bound,
signature-checked, replay-protected, body-limited, and concurrency-limited.

```dotenv
MINER_REQUIRE_VALIDATOR_PERMIT=true
MINER_MIN_STAKE=0
MINER_MAX_CONCURRENT_REQUESTS=4
MINER_MAX_REQUEST_BYTES=1000000
MINER_METAGRAPH_SYNC_S=300
```

Provider calls use one absolute budget bounded by both the validator's task
deadline and `OPENAI_REQUEST_TIMEOUT_S`. HTTP 429 and 5xx responses and network
transport failures are retried within that same budget. A provider failure
returns a valid signed empty solution, which safely scores zero without
breaking the validator round.
