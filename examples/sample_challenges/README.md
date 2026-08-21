# Challenge examples

These five challenges let miner developers check solution formatting and basic
behavior locally.

Each directory contains the public statement and three test cases. It contains
no accepted solution, production
test, challenge identifier, serving record, or generator metadata.

## Run an example

Complete the problem in a source file, then run:

```bash
.venv/bin/python examples/sample_challenges/run.py \
  sparse-circular-array path/to/solution.rs
```

Available names:

- `sparse-circular-array` (Rust)
- `revocable-verification-gate` (Rust)
- `reactive-stat-board` (Rust)
- `extent-journal` (Python)
- `asset-rebuild-planner` (Python)

The Rust examples use the same pinned container, compiler policy, and
ASCII-whitespace output comparison as the validator. Docker must be running.
Python examples use the validator's isolated subprocess runner and structural
return-value comparison. Run `./setup_validator.sh` first if the environment or
container images are not ready.

The runner prints one line per case and exits with a nonzero status if any case
fails. These cases demonstrate the interface and basic rules only.
