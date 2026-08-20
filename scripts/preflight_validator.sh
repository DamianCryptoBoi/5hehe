#!/usr/bin/env bash
#
# Verify the sandbox image and problem service before any challenge is leased.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

pull_image=false
require_rust=false
while (( $# )); do
  case "$1" in
    --pull) pull_image=true ;;
    --require-rust) require_rust=true ;;
    *) echo "[preflight] ERROR: usage: $0 [--pull] [--require-rust]" >&2; exit 2 ;;
  esac
  shift
done
if (( $# != 0 )); then
  echo "[preflight] ERROR: usage: $0 [--pull] [--require-rust]" >&2
  exit 2
fi

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

python_bin="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="$(command -v python3 || true)"
fi
if [[ -z "${python_bin}" ]]; then
  echo "[preflight] ERROR: Python 3 is required for bounded preflight checks." >&2
  exit 1
fi

run_timed() {
  local timeout_s="$1"
  shift
  "${python_bin}" - "${timeout_s}" "$@" <<'PY'
import subprocess
import sys

timeout_s = float(sys.argv[1])
command = sys.argv[2:]
try:
    completed = subprocess.run(command, timeout=timeout_s, check=False)
except subprocess.TimeoutExpired:
    print(
        f"[preflight] ERROR: command timed out after {timeout_s:g}s: {command[0]}",
        file=sys.stderr,
    )
    raise SystemExit(124)
raise SystemExit(completed.returncode)
PY
}

: "${EXECUTOR:=docker}"
: "${PROBLEM_SERVER_URL:?set PROBLEM_SERVER_URL in .env}"

if [[ "${SUBTENSOR_NETWORK:-}" == "finney" && "${NETUID:-}" != "5" ]]; then
  echo "[preflight] ERROR: this Finney release is configured for NETUID=5." >&2
  exit 1
fi
if [[ "${SUBTENSOR_NETWORK:-}" == "finney" && "${EXECUTOR}" != "docker" ]]; then
  echo "[preflight] ERROR: Finney validation requires EXECUTOR=docker." >&2
  exit 1
fi
if [[ "${EXECUTOR}" != "docker" ]]; then
  echo "[preflight] development executor=${EXECUTOR}; Docker check skipped"
else
  : "${DOCKER_IMAGE:?set DOCKER_IMAGE in .env}"
  if [[ ! "${DOCKER_IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]]; then
    echo "[preflight] ERROR: DOCKER_IMAGE must use an immutable sha256 digest." >&2
    exit 1
  fi
  command -v docker >/dev/null 2>&1 || {
    echo "[preflight] ERROR: Docker CLI not found. Install and start Docker." >&2
    exit 1
  }
  run_timed 20 docker info >/dev/null
  if [[ "${pull_image}" == true ]]; then
    echo "[preflight] pulling sandbox image ${DOCKER_IMAGE}"
    run_timed 600 docker pull "${DOCKER_IMAGE}"
  elif ! run_timed 20 docker image inspect "${DOCKER_IMAGE}" >/dev/null 2>&1; then
    echo "[preflight] ERROR: sandbox image is missing; rerun ./setup_validator.sh." >&2
    exit 1
  fi

  smoke="$(
    run_timed 45 docker run --rm --pull=never --network=none --read-only \
      --cap-drop=ALL --security-opt=no-new-privileges \
      --user=65534:65534 "${DOCKER_IMAGE}" \
      sh -ec 'command -v timeout >/dev/null; timeout 5 python -I -S -c "print(\"rlvr-preflight-ok\")"'
  )"
  if [[ "${smoke}" != "rlvr-preflight-ok" ]]; then
    echo "[preflight] ERROR: sandbox Python/GNU timeout smoke test failed." >&2
    exit 1
  fi
  echo "[preflight] sandbox image ready"
fi

"${python_bin}" - <<'PY'
import email.utils
import os
import time
import urllib.parse
import urllib.error
import urllib.request

base = os.environ["PROBLEM_SERVER_URL"].rstrip("/")
parsed = urllib.parse.urlparse(base)
if os.environ.get("SUBTENSOR_NETWORK") == "finney" and parsed.scheme != "https":
    raise SystemExit("[preflight] ERROR: Finney problem server must use HTTPS")
url = f"{base}/v1/challenges/lease"
request = urllib.request.Request(
    url,
    data=b"{}",
    headers={"Content-Type": "application/json"},
    method="POST",
)


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


opener = urllib.request.build_opener(NoRedirects)
try:
    with opener.open(request, timeout=20) as response:
        status = response.status
        date_header = response.headers.get("Date", "")
except urllib.error.HTTPError as error:
    status = error.code
    date_header = error.headers.get("Date", "")
except Exception as error:
    raise SystemExit(
        f"[preflight] ERROR: problem server reachability check failed: {error}; "
        "check network access and NTP"
    )
if status != 401:
    raise SystemExit(
        f"[preflight] ERROR: unsigned lease probe returned HTTP {status}; expected 401"
    )
if not date_header:
    raise SystemExit("[preflight] ERROR: server omitted Date; cannot verify clock")
try:
    server_time = email.utils.parsedate_to_datetime(date_header).timestamp()
except (TypeError, ValueError, OverflowError):
    raise SystemExit("[preflight] ERROR: server returned an invalid Date header")
skew = abs(time.time() - server_time)
if skew > 5:
    raise SystemExit(
        f"[preflight] ERROR: system clock differs from server by {skew:.1f}s; "
        "enable NTP before starting"
    )
print("[preflight] problem server reachable; unsigned lease rejected; clock synchronized")
PY

rust_image="$("${python_bin}" - <<'PY'
from rlvr.policy import RELEASE_POLICY
print(RELEASE_POLICY.rust_image)
PY
)"
if [[ "${EXECUTOR}" == "docker" ]]; then
  rust_capacity="$("${python_bin}" - <<'PY'
from rlvr.config import get_settings
from rlvr.neurons.decentralized import _host_memory_bytes, _rust_verify_concurrency

settings = get_settings()
print(
    _host_memory_bytes(),
    settings.validator_verify_concurrency,
    _rust_verify_concurrency(settings),
)
PY
)"
  read -r host_memory_bytes VALIDATOR_VERIFY_CONCURRENCY effective_rust_slots \
    <<<"${rust_capacity}"
  echo "[preflight] rust memory MemTotal=${host_memory_bytes} configured_slots=${VALIDATOR_VERIFY_CONCURRENCY} effective_slots=${effective_rust_slots}"
  rust_required_memory=$((14 * 1024 * 1024 * 1024))
  if (( host_memory_bytes < rust_required_memory )); then
    if [[ "${require_rust}" == true ]]; then
      echo "[preflight] ERROR: Rust grading requires a nominal 16 GiB host." >&2
      exit 1
    fi
    echo "[preflight] WARN: Rust grading requires a nominal 16 GiB host." >&2
  fi
fi
rust_ready=false
if [[ "${EXECUTOR}" == "docker" ]]; then
  if [[ "${pull_image}" == true ]]; then
    run_timed 600 docker pull "${rust_image}" >/dev/null 2>&1 || true
  fi
  if run_timed 20 docker image inspect "${rust_image}" >/dev/null 2>&1 \
    && run_timed 20 docker run --rm --pull=never --network=none --read-only \
      --cap-drop=ALL --security-opt=no-new-privileges \
      "${rust_image}" sh -ec 'rustc --version; command -v python3; command -v timeout' \
      >/dev/null 2>&1 \
    && run_timed 120 "${python_bin}" - <<'PY'
from rlvr.config import get_settings
from rlvr.neurons.decentralized import _rust_ready
raise SystemExit(0 if _rust_ready(get_settings()) else 1)
PY
  then
    rust_ready=true
  fi
fi

if [[ "${rust_ready}" == true ]]; then
  echo "[preflight] rust ready; validator will advertise rust support"
elif [[ "${require_rust}" == true ]]; then
  echo "[preflight] ERROR: rust sandbox is not ready: ${rust_image}" >&2
  exit 1
else
  echo "[preflight] rust not ready; continuing in python-only mode"
fi
