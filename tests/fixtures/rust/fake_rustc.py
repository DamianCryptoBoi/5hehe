"""Fake rustc for the rust runner tests.

The runner invokes ``compiler_argv + [source_path, "-o", artifact_path]``.
This stub stands in for rustc so the trusted supervisor's compile/execute
logic is testable on hosts with no Rust toolchain; the real compiler axis
belongs to the deploy-host benchmark gate.

Modes (passed inside compiler_argv, before the positionals):
  --mode=ok      write an executable artifact running fake_artifact.py logic
  --mode=fail    exit 1 with a compiler-style error on stderr
  --mode=sleep   hang far past any sane compile deadline
  --mode=bigout  write an artifact of --size bytes (artifact-cap tests)
  --mode=noexec  write a size-legal artifact that cannot be executed
"""

import argparse
import pathlib
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="ok")
    parser.add_argument("--size", type=int, default=0)
    parser.add_argument("source")
    parser.add_argument("-o", dest="out", required=True)
    args = parser.parse_args()

    if not pathlib.Path(args.source).exists():
        print("fake-rustc: source file missing", file=sys.stderr)
        sys.exit(2)

    if args.mode == "fail":
        print("error[E0999]: fake compilation failure", file=sys.stderr)
        sys.exit(1)
    if args.mode == "sleep":
        time.sleep(30)
        sys.exit(1)

    out = pathlib.Path(args.out)
    if args.mode == "noexec":
        # No shebang, no exec bit: exec() of this artifact raises OSError.
        out.write_text("not an executable\n", encoding="utf-8")
        sys.exit(0)
    if args.mode == "bigout":
        stub = b"#!/bin/false\n"
        out.write_bytes(stub + b"\0" * max(0, args.size - len(stub)))
        out.chmod(0o755)
        sys.exit(0)

    # ok: install fake_artifact.py as the "compiled" executable, with a
    # shebang pointing at THIS interpreter so no python3 needs to be on PATH.
    logic = (pathlib.Path(__file__).parent / "fake_artifact.py").read_text(
        encoding="utf-8"
    )
    out.write_text(f"#!{sys.executable}\n" + logic, encoding="utf-8")
    out.chmod(0o755)
    sys.exit(0)


if __name__ == "__main__":
    main()
