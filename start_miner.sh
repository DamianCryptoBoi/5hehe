#!/usr/bin/env bash
# Launch the OpenAI-compatible Hone miner.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "[openai-miner] ERROR: .venv not found; create it and install '.[chain,miner]' first." >&2
  exit 1
fi
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
else
  echo "[openai-miner] ERROR: .env not found; copy .env.example and configure it first." >&2
  exit 1
fi

: "${OPENAI_MODEL:?set OPENAI_MODEL in .env}"
: "${NETUID:?set NETUID in .env}"
: "${WALLET_NAME:?set WALLET_NAME in .env}"
: "${WALLET_HOTKEY:?set WALLET_HOTKEY in .env}"
: "${SUBTENSOR_NETWORK:=finney}"
: "${OPENAI_REQUIRE_API_KEY:=true}"
: "${OPENAI_API_KEY:=}"

case "${OPENAI_REQUIRE_API_KEY}" in
  true|TRUE|True|1|yes|YES|Yes)
    : "${OPENAI_API_KEY:?set OPENAI_API_KEY in .env}"
    ;;
esac

export OPENAI_MODEL NETUID WALLET_NAME WALLET_HOTKEY SUBTENSOR_NETWORK
export OPENAI_REQUIRE_API_KEY OPENAI_API_KEY

echo "[openai-miner] netuid=${NETUID} network=${SUBTENSOR_NETWORK} wallet=${WALLET_NAME}/${WALLET_HOTKEY} model=${OPENAI_MODEL}"
exec python scripts/run_openai_miner.py
