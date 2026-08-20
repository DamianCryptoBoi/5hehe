#!/usr/bin/env bash
#
# register_testnet.sh — Bittensor TESTNET onboarding for the RLVR code-data subnet.
#
# This is a COMMENTED RUNBOOK, not an automated installer. Read it, fill in the
# placeholders (<...>), and run the btcli commands one at a time so you can react
# to prompts and confirm balances/UIDs between steps. Testnet TAO has no value;
# use it freely. Nothing here touches mainnet (finney).
#
# Prereqs:
#   - Python env active:  . .venv/bin/activate
#   - SDK + btcli:        pip install -e '.[chain]'
#                         btcli --version
#   - Endpoint:           testnet is `--subtensor.network test`
#                         (chain endpoint wss://test.finney.opentensor.ai:443)
#
# Settings in your .env that should match what you register here:
#   NETUID, SUBTENSOR_NETWORK=test, WALLET_NAME, WALLET_HOTKEY
# -----------------------------------------------------------------------------

set -euo pipefail

# === EDIT THESE PLACEHOLDERS ================================================
WALLET_NAME="${WALLET_NAME:-rlvr_cold}"        # coldkey name (holds TAO)
WALLET_HOTKEY="${WALLET_HOTKEY:-rlvr_hot}"      # hotkey (signs / mines / validates)
NETWORK="test"                                   # testnet
NETUID="${NETUID:-<YOUR_TESTNET_NETUID>}"       # the subnet's netuid on testnet
# ============================================================================

echo "This script is a runbook. Review it, then run the steps below manually."
echo "WALLET_NAME=${WALLET_NAME}  WALLET_HOTKEY=${WALLET_HOTKEY}  NETUID=${NETUID}"
exit 0
# -----------------------------------------------------------------------------
# Everything below this `exit 0` is reference. Uncomment / copy-paste per step.
# -----------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1) CREATE A COLDKEY (your wallet's "savings" key — keep the mnemonic safe).
#    Encrypted on disk under ~/.bittensor/wallets/<WALLET_NAME>/.
# ---------------------------------------------------------------------------
# btcli wallet new_coldkey --wallet.name "${WALLET_NAME}"

# ---------------------------------------------------------------------------
# 2) CREATE A HOTKEY (the "hot" signing key your neuron uses on chain).
# ---------------------------------------------------------------------------
# btcli wallet new_hotkey --wallet.name "${WALLET_NAME}" --wallet.hotkey "${WALLET_HOTKEY}"

# ---------------------------------------------------------------------------
# 3) GET TESTNET TAO.
#    Faucet availability varies; two common routes:
#      a) On-chain faucet (if enabled on testnet):
#         btcli wallet faucet --wallet.name "${WALLET_NAME}" --subtensor.network "${NETWORK}"
#      b) Ask in the Bittensor Discord #testnet-faucet channel and paste your
#         coldkey SS58 address (printed by step 1, or via:)
#         btcli wallet overview --wallet.name "${WALLET_NAME}" --subtensor.network "${NETWORK}"
#    Confirm the balance landed before continuing:
#         btcli wallet balance --wallet.name "${WALLET_NAME}" --subtensor.network "${NETWORK}"

# ---------------------------------------------------------------------------
# 4) (SUBNET OWNER ONLY) CREATE A SUBNET on testnet to get a fresh NETUID.
#    Most contributors SKIP this and instead register onto an existing NETUID.
#    Creating a subnet locks a registration cost in TAO.
# ---------------------------------------------------------------------------
# btcli subnet create --wallet.name "${WALLET_NAME}" --subtensor.network "${NETWORK}"
#    -> note the NETUID it prints; put it in .env (NETUID=...) and at the top here.

# ---------------------------------------------------------------------------
# 5) REGISTER YOUR NEURON (hotkey) onto the subnet's NETUID.
#    This claims a UID slot. Costs a small registration fee (recycled TAO) or,
#    on some subnets, requires proof-of-work (`--pow-register`).
# ---------------------------------------------------------------------------
# btcli subnet register \
#   --wallet.name "${WALLET_NAME}" \
#   --wallet.hotkey "${WALLET_HOTKEY}" \
#   --netuid "${NETUID}" \
#   --subtensor.network "${NETWORK}"

# ---------------------------------------------------------------------------
# 6) CONFIRM REGISTRATION — find your UID in the metagraph.
# ---------------------------------------------------------------------------
# btcli subnet metagraph --netuid "${NETUID}" --subtensor.network "${NETWORK}"
# btcli wallet overview  --wallet.name "${WALLET_NAME}" --subtensor.network "${NETWORK}"

# ---------------------------------------------------------------------------
# 7) STAKE so the validator's weights count, then run start_validator.sh.
# ---------------------------------------------------------------------------
# btcli stake add \
#   --wallet.name "${WALLET_NAME}" \
#   --wallet.hotkey "${WALLET_HOTKEY}" \
#   --netuid "${NETUID}" \
#   --subtensor.network "${NETWORK}" \
#   --amount <TAO_AMOUNT>

# ---------------------------------------------------------------------------
# 8) RUN THE VALIDATOR. Make sure .env matches (NETUID,
#    SUBTENSOR_NETWORK=test, WALLET_NAME, WALLET_HOTKEY, and
#    PROBLEM_SERVER_URL). Then:
#       ./start_validator.sh    # to validate
# ---------------------------------------------------------------------------
