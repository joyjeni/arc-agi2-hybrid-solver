#!/usr/bin/env python3
"""Poll kernel v22 until complete, then auto-submit."""
import subprocess, time, sys

KERNEL = "joyjeni/arc-agi-2-solver-three-stage-pipeline"
COMPETITION = "arc-prize-2026-arc-agi-2"
POLL_INTERVAL = 90

def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout.strip()

print(f"[{time.strftime('%H:%M:%S UTC', time.gmtime())}] Polling kernel v22...")
attempt = 0
while True:
    attempt += 1
    status = run(f"kaggle kernels status {KERNEL}")
    ts = time.strftime('%H:%M:%S', time.gmtime())
    print(f"[{ts}] #{attempt}: {status}", flush=True)
    if "complete" in status.lower():
        print("KERNEL COMPLETE — submitting...", flush=True)
        result = run(
            f'kaggle competitions submit -c {COMPETITION} '
            f'-f /kaggle/working/submission.json '
            f'-m "v22: 107/1000 (10.7%) 0-wrong — p93-p99 new solvers"'
        )
        print(result, flush=True)
        break
    elif "error" in status.lower() or "failed" in status.lower():
        print("KERNEL ERROR — check log with: kaggle kernels output joyjeni/arc-agi-2-solver-three-stage-pipeline", flush=True)
        sys.exit(1)
    time.sleep(POLL_INTERVAL)
