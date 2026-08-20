"""Fake compiled artifact: obeys a one-line command protocol on stdin.

The runner treats the artifact as an opaque program: feed stdin, read stdout.
Each test case selects a behavior through its stdin text:

  echo\n<payload>       write <payload> exactly, exit 0
  sleep <seconds>       sleep, then write "late"
  flood <n>             write n bytes of "A"
  quotes <n>            write n double-quote characters (JSON-escaping tests)
  badutf8               write bytes that are not valid UTF-8
  exitcode <c>\n<pay>   write <pay>, then exit with code c
  spawn                 start a detached 30s child, write "spawned", exit 0
"""

import subprocess
import sys
import time


def main() -> None:
    data = sys.stdin.read()
    head, _, rest = data.partition("\n")
    parts = head.split()
    command = parts[0] if parts else "echo"

    if command == "echo":
        sys.stdout.write(rest)
    elif command == "sleep":
        time.sleep(float(parts[1]))
        sys.stdout.write("late")
    elif command == "flood":
        sys.stdout.write("A" * int(parts[1]))
    elif command == "quotes":
        sys.stdout.write('"' * int(parts[1]))
    elif command == "badutf8":
        sys.stdout.flush()
        sys.stdout.buffer.write(b"\xff\xfe\xfd")
    elif command == "exitcode":
        sys.stdout.write(rest)
        sys.stdout.flush()
        sys.exit(int(parts[1]))
    elif command == "spawn":
        subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            start_new_session=True,
        )
        sys.stdout.write("spawned")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
