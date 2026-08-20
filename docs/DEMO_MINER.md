# GLM-5.2 demo miner

The demo miner is a minimal reference implementation of the subnet's
validator/miner wire protocol. It:

1. advertises the configured IP and port on Bittensor;
2. accepts signed `POST /solve` requests from authorized validators;
3. sends the public task to GLM-5.2 through Z.ai's chat-completions API;
4. extracts the Python solution and signs the exact response bytes with the
   miner hotkey.

It does not receive hidden evaluation cases and does not execute generated
code. Validators reveal tests only after committing miner responses, then run
the code in their own sandboxes. Public task statements and examples are sent
to the configured model provider, so operators should review that provider's
data-handling terms.

## Prerequisites

- Python 3.10–3.12
- a funded hotkey registered on the target subnet
- a Z.ai API key with access to `glm-5.2`
- a public TCP port reachable by validators

Install the miner extras:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e '.[chain,miner]'
cp .env.example .env
```

Set at least:

```dotenv
GLM_API_KEY=<your-key>
GLM_BASE_URL=https://api.z.ai/api/paas/v4
GLM_MODEL=glm-5.2

NETUID=<subnet-id>
SUBTENSOR_NETWORK=test
WALLET_NAME=miner
WALLET_HOTKEY=default

AXON_PORT=8091
# Set this when automatic public-IP detection is unsuitable.
AXON_EXTERNAL_IP=
```

`GLM_BASE_URL` may be changed to another endpoint supported by the account.
The miner appends `/chat/completions` unless the configured URL already ends
with that path. See the
[Z.ai chat-completions API reference](https://docs.z.ai/api-reference/llm/chat-completion)
for the provider request contract. Accounts using a Coding Plan endpoint can
set `GLM_BASE_URL=https://api.z.ai/api/coding/paas/v4` when permitted by their
plan.

## Run

Register the hotkey first if necessary, then start the service:

```bash
./start_demo_miner.sh
```

On startup the miner verifies registration, advertises its endpoint on the
configured subnet, and starts the HTTP server. A local health check is
available at:

```bash
curl http://127.0.0.1:8091/health
```

## Authorization and cost controls

The default policy accepts only registered hotkeys that currently have a
validator permit. Requests are signature-checked, recipient-bound, protected
against replay, and limited in body size and concurrency.

Useful controls:

```dotenv
MINER_REQUIRE_VALIDATOR_PERMIT=true
MINER_MIN_STAKE=0
MINER_MAX_CONCURRENT_REQUESTS=4
MINER_MAX_REQUEST_BYTES=1000000
MINER_METAGRAPH_SYNC_S=300
```

Relaxing the permit or stake policy lets more registered hotkeys spend the
configured model account's quota. Keep the API key only in `.env` or the
process environment; `.env` is ignored by Git.

The model defaults are intentionally configurable:

```dotenv
GLM_MAX_TOKENS=16384
GLM_TEMPERATURE=1.0
GLM_THINKING=true
GLM_REASONING_EFFORT=high
GLM_REQUEST_TIMEOUT_S=280
GLM_MAX_RETRIES=2
```

Lower output or reasoning budgets reduce cost and latency, but may reduce the
pass rate on harder tasks.
