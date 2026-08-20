# Grading benchmark

The benchmark measures the Docker grader with synthetic submissions. It does
not use leased challenges, hidden tests, miner responses, wallets, or network
services.

Run it on validator-class hardware from the repository root:

```bash
. .venv/bin/activate
python scripts/benchmark_grading.py \
  --submissions 123 \
  --tests 10 \
  --concurrency 16 \
  --repeats 5
```

Repeat with 62 and 123 submissions and with each concurrency supported by the
host. Record the instance type, CPU count, memory, storage type, architecture,
Docker version, image digest, and the complete JSON output.

The report includes per-challenge wall time, submission latency, submissions
per second, test processes per second, and failure counts. Monitor host CPU,
memory, throttling, and disk use separately during each run.

The synthetic workload measures sandbox overhead rather than model-generated
program complexity. A deployment decision should also use a run over a
representative local archive. Test cases and submitted code should remain on
the benchmark host.

## Rust activation matrix

Build and verify the Rust sandbox locally before running Rust measurements:

```bash
./scripts/build_rust_sandbox.sh
```

A local image ID may be supplied to the benchmark-only `--image` flag. The
result records the override explicitly and does not alter validator policy.
Benchmark the exact local image object that passed the canary. If it is later
approved for publication, push that same object without rebuilding it; a
rebuild invalidates the measurements. Fleet activation then requires reading
the registry `RepoDigest`, updating the release-policy image pin, and rerunning
preflight with `--require-rust`.

Run the clean Rust matrix with 10 and 100 tests, concurrency 8, 16, and 32,
and the expected per-challenge submission count. Repeat each configuration
with small source and approximately 50 KiB of comment padding:

```bash
python scripts/benchmark_grading.py \
  --language rust \
  --submissions 123 \
  --tests 100 \
  --concurrency 16 \
  --repeats 5 \
  --source-pad-bytes 50000 \
  --image sha256:LOCAL_IMAGE_ID
```

Run `--adversarial` separately. Its compile failure, runaway execution,
oversized output, invalid UTF-8, and heavily escaped output cases are reported
outside the clean throughput measurements.

## Reference Rust run

The release image was tested on an EC2 c6a.2xlarge host with 8 vCPUs and
16 GiB memory using Ubuntu 24.04 and Docker 29.1.3. The matrix used 123
submissions, 10/100 tests, 0/50 KiB source padding, concurrency 8/16/32, and
five repetitions. All clean runs completed without grading failures. At
concurrency 16, the 100-test challenge p95 was 13.48 seconds without padding
and 13.40 seconds with padding.

The JSON report does not measure Docker child peak memory, temporary-storage
high-water marks, or artifact-size margins. Monitor those separately on each
deployment host. Rust verification concurrency is bounded from host memory,
the 2 GiB per-container cap, and a 2 GiB host reserve.

Rust grading requires a nominal 16 GiB host. Preflight enforces this minimum
when Rust is required and reports the effective memory-bounded concurrency on
every run. Larger hosts permit more concurrent Rust verification.
