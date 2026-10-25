#!/usr/bin/env python3
"""Poll kernel v20 until complete, then auto-submit to competition."""
import subprocess, time, json, sys

KERNEL = "joyjeni/arc-agi-2-solver-three-stage-pipeline"
COMPETITION = "arc-prize-2026-arc-agi-2"
SUBMISSION_FILE = "/kaggle/working/submission.json"
POLL_INTERVAL = 60  # seconds

def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout.strip(), r.returncode

def get_kernel_status():
    out, code = run(f"kaggle kernels status {KERNEL} -v")
    return out

def submit():
    print("Submitting output to competition...")
    out, code = run(
        f'kaggle competitions submit -c {COMPETITION} '
        f'-f {SUBMISSION_FILE} '
        f'-m "v21: 107/1000 solvers — 10.7% local score"'
    )
    print(out)
    return code == 0

print(f"Polling kernel: {KERNEL}")
print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
attempt = 0
while True:
    attempt += 1
    status = get_kernel_status()
    print(f"[{time.strftime('%H:%M:%S')}] Attempt {attempt}: {status[:200]}")

    if "complete" in status.lower():
        print("Kernel COMPLETE. Submitting...")
        submit()
        break
    elif "error" in status.lower() or "failed" in status.lower():
        print("Kernel FAILED. Exiting.")
        sys.exit(1)
    else:
        print(f"  Still running. Waiting {POLL_INTERVAL}s...")
        time.sleep(POLL_INTERVAL)
