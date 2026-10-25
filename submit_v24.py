#!/usr/bin/env python3
"""Poll kernel v24 until complete, then auto-submit to competition."""
import subprocess, time, sys

KERNEL      = "joyjeni/arc-agi-2-solver-three-stage-pipeline"
COMPETITION = "arc-prize-2026-arc-agi-2"
POLL_INTERVAL = 90   # seconds

def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()

print(f"[{time.strftime('%H:%M:%S UTC', time.gmtime())}] Polling kernel v24...", flush=True)
attempt = 0
while True:
    attempt += 1
    status = run(f"kaggle kernels status {KERNEL}")
    ts = time.strftime('%H:%M:%S UTC', time.gmtime())
    print(f"[{ts}] #{attempt}: {status}", flush=True)

    if "complete" in status.lower():
        print("KERNEL COMPLETE — submitting...", flush=True)
        result = run(
            f'kaggle competitions submit -c {COMPETITION} '
            f'-f /kaggle/working/submission.json '
            f'-m "v24: all 301 solvers (adv5-adv8 fixed, zlib-embedded)"'
        )
        print("Submit result:", result, flush=True)
        break
    elif "error" in status.lower() or "failed" in status.lower() or "cancel" in status.lower():
        print("KERNEL FAILED — check log:", flush=True)
        log = run(f"kaggle kernels output {KERNEL}")
        print(log, flush=True)
        sys.exit(1)

    time.sleep(POLL_INTERVAL)
