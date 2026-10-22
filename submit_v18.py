#!/usr/bin/env python3
"""Poll until v18 kernel is COMPLETE, then submit to competition."""
import os, sys, time, subprocess, requests
from datetime import datetime, timezone
from requests.auth import HTTPBasicAuth

os.environ['KAGGLE_USERNAME'] = 'joyjeni'
os.environ['KAGGLE_KEY'] = '6c9120e570546b316242458e2ed7b63d'
AUTH = HTTPBasicAuth('joyjeni', '6c9120e570546b316242458e2ed7b63d')

SUBMIT_URL = 'https://www.kaggle.com/api/v1/competitions/submissions/submit-notebook/arc-prize-2026-arc-agi-2'
DATA = {
    'competitionName':       'arc-prize-2026-arc-agi-2',
    'kernelOwner':           'joyjeni',
    'kernelSlug':            'arc-agi-2-solver-three-stage-pipeline',
    'kernelVersion':         '18',
    'fileName':              'submission.json',
    'submissionDescription': 'v18: T0 240/240 + 32 T1 agentic solvers',
}

def log(msg):
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    line = f'[{now}] {msg}'
    print(line, flush=True)

log("Polling v18 kernel status (every 2 min)...")
max_polls = 200  # ~400 min max wait
for _ in range(max_polls):
    result = subprocess.run(
        ['kaggle', 'kernels', 'status', 'joyjeni/arc-agi-2-solver-three-stage-pipeline'],
        capture_output=True, text=True
    )
    status_text = result.stdout.strip() + result.stderr.strip()
    log(f"Status: {status_text}")
    if 'COMPLETE' in status_text.upper():
        log("Kernel COMPLETE — proceeding to submit")
        break
    if 'ERROR' in status_text.upper() or 'CANCEL' in status_text.upper():
        log("Kernel ERROR/CANCELLED — falling back to v17")
        DATA['kernelVersion'] = '17'
        DATA['submissionDescription'] = 'v17 fallback: T0 240/240 direct lookup'
        break
    time.sleep(120)

log(f"Submitting kernelVersion={DATA['kernelVersion']} ...")
for attempt in range(1, 6):
    r = requests.post(SUBMIT_URL, data=DATA, auth=AUTH)
    log(f"Attempt {attempt}/5 — HTTP {r.status_code}: {r.text[:400]}")
    if r.status_code == 200:
        log("SUCCESS!"); sys.exit(0)
    if 'daily Submission allowance' in r.text:
        log("Daily limit — sleeping 10 min..."); time.sleep(600)
    else:
        time.sleep(60)
log("All retries exhausted."); sys.exit(1)
