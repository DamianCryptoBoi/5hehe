#!/usr/bin/env bash
# Build and verify the Rust grading image. Publishing requires --push REF.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

push_ref=""
if [[ "${1:-}" == "--push" ]]; then
  [[ $# -eq 2 ]] || { echo "usage: $0 [--push REGISTRY/IMAGE:TAG]" >&2; exit 2; }
  push_ref="$2"
elif (( $# != 0 )); then
  echo "usage: $0 [--push REGISTRY/IMAGE:TAG]" >&2
  exit 2
fi

readarray -t policy_values < <(
  .venv/bin/python - <<'PY'
from rlvr.policy import RELEASE_POLICY
print(RELEASE_POLICY.rustc_version)
PY
)
rustc_version="${policy_values[0]}"
local_tag="hone-rust-sandbox:rustc-${rustc_version}"

docker build --pull -f docker/rust-sandbox/Dockerfile -t "${local_tag}" .
probe="$(docker run --rm --network=none --read-only "${local_tag}" \
  sh -ec 'rustc --version; command -v python3; command -v timeout')"
grep -F "rustc ${rustc_version} " <<<"${probe}" >/dev/null

if [[ -z "${push_ref}" ]]; then
  image_id="$(docker image inspect --format '{{.Id}}' "${local_tag}")"
  echo "Local image verified: ${image_id}"
  echo "A local image ID is not a fleet identity. Use --push to obtain a RepoDigest."
  exit 0
fi

docker tag "${local_tag}" "${push_ref}"
docker push "${push_ref}"
repo_name="${push_ref%:*}"
repo_digest="$(
  docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' \
    "${push_ref}" | grep -F "${repo_name}@" | head -n 1
)"
digest_marker='@sha256'
[[ -n "${repo_digest}" && "${repo_digest}" == *"${digest_marker}":* ]] || {
  echo "published image has no RepoDigests entry" >&2
  exit 1
}
echo "Published and verified: ${repo_digest}"
echo "Update RELEASE_POLICY.rust_image in rlvr/policy.py to: ${repo_digest}"
