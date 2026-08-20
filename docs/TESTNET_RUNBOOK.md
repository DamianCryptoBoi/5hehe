# Validator testnet runbook

## Prerequisites

- Python 3.10–3.12
- Docker for production-style sandboxing
- A funded and registered validator hotkey
- The subnet ID and Bittensor network endpoint
- The HTTPS URL of the private problem service

## Install

```bash
git clone <validator-repository-url> rlvr-subnet
cd rlvr-subnet
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e '.[chain,dev]'
cp .env.example .env
```

## Configure

Set at least:

```dotenv
PROBLEM_SERVER_URL=https://problems.example.com
PROBLEM_SERVER_ALLOW_INSECURE_HTTP=false
EXECUTOR=docker
DOCKER_IMAGE=python:3.12-slim@sha256:<approved-digest>
NETUID=0
SUBTENSOR_NETWORK=test
WALLET_NAME=validator
WALLET_HOTKEY=default
```

The problem service verifies requests against the current on-chain metagraph.
The signing hotkey must hold a validator permit on the configured NETUID; no
separate service-side enrollment is required. Keep
`PROBLEM_SERVER_ALLOW_INSECURE_HTTP=false` outside local integration tests.
Pin `DOCKER_IMAGE` to a tested immutable digest in production; the sandbox
image must provide Python 3.12 and GNU `timeout`.

## Verify locally

```bash
pytest -q
docker run --rm --network=none python:3.12-slim python -c 'print("sandbox ready")'
```

## Run

```bash
./start_validator.sh
```

Expected round behavior is: synchronize metagraph, lease a challenge, dispatch
signed requests, commit signed responses, reveal evaluation cases, execute
locally, update scores, and periodically submit weights.

Never run the validator as root, mount sensitive host directories into the
sandbox, or enable insecure problem-service transport in production.
