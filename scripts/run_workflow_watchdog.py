#!/usr/bin/env python3
"""Run cli.py with log growth monitoring; kill if stalled (no new bytes) too long."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = Path("/tmp/agentic_watch.log")
MAX_ITER = os.environ.get("MAX_WORKFLOW_ITERATIONS", "8")
STALL_SEC = int(os.environ.get("WATCHDOG_STALL_SEC", "300"))
ABS_MAX_SEC = int(os.environ.get("WATCHDOG_MAX_SEC", "900"))
POLL = 15


def main() -> int:
    for py in (ROOT / "venv/bin/python", ROOT / ".venv/bin/python"):
        if py.is_file():
            break
    else:
        py = Path(sys.executable)
    env = {**os.environ, "PYTHONPATH": str(ROOT), "MAX_WORKFLOW_ITERATIONS": MAX_ITER}
    LOG.write_bytes(b"")
    print(f"watchdog: log={LOG} stall_limit={STALL_SEC}s abs_max={ABS_MAX_SEC}s", flush=True)
    with LOG.open("wb", buffering=0) as logf:
        p = subprocess.Popen(
            [str(py), "-u", str(ROOT / "cli.py")],
            cwd=str(ROOT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    t0 = time.time()
    last = 0
    stall = 0
    while p.poll() is None:
        time.sleep(POLL)
        sz = LOG.stat().st_size
        if sz == last:
            stall += POLL
            print(f"watchdog: no log growth {stall}s (last {sz} bytes)", flush=True)
            if stall >= STALL_SEC:
                print("watchdog: STALL — sending SIGTERM", flush=True)
                p.terminate()
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
                print_tail()
                return 124
        else:
            stall = 0
            last = sz
        if time.time() - t0 >= ABS_MAX_SEC:
            print("watchdog: ABS TIME — sending SIGTERM", flush=True)
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
            print_tail()
            return 124
    rc = p.wait()
    print_tail()
    return int(rc or 0)


def print_tail() -> None:
    if not LOG.is_file():
        return
    text = LOG.read_text(errors="replace")
    lines = text.splitlines()
    tail = "\n".join(lines[-60:])
    print("\n--- last 60 log lines ---\n", tail, sep="", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
