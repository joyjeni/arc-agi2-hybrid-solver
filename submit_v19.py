#!/usr/bin/env python3
"""Submit v19 kernel output to competition — retries until daily limit resets at midnight UTC."""
import os, sys, time, subprocess, requests
from datetime import datetime, timezone, timedelta
from requests.auth import HTTPBasicAuth

os.environ['KAGGLE_USERNAME'] = 'joyjeni'
os.environ['KAGGLE_KEY'] = '6c9120e570546b316242458e2ed7b63d'
AUTH = HTTPBasicAuth('joyjeni', '6c9120e570546b316242458e2ed7b63d')
KAGGLE_API = 'https://www.kaggle.com/api/v1'
KERNEL_SLUG = 'joyjeni/arc-agi-2-solver-three-stage-pipeline'
SUBMIT_URL  = f'{KAGGLE_API}/competitions/submissions/submit-notebook/arc-prize-2026-arc-agi-2'

DATA = {
    'competitionName':       'arc-prize-2026-arc-agi-2',
    'kernelOwner':           'joyjeni',
    'kernelSlug':            'arc-agi-2-solver-three-stage-pipeline',
    'kernelVersion':         '19',
    'fileName':              'submission.json',
    'submissionDescription': 'v19: PaletteSwapSolver fix + 38 T1 solvers (34/1000 train, 0 wrong)',
}

def log(msg):
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    line = f'[{now}] {msg}'
    print(line, flush=True)

def secs_until_utc_midnight():
    now = datetime.now(timezone.utc)
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)
    return max(0, (midnight - now).total_seconds())

def wait_for_kernel():
    """Poll until kernel v19 is COMPLETE."""
    url = f"{KAGGLE_API}/kernels/output/{KERNEL_SLUG}?kernelVersionNumber=19"
    status_url = f"https://www.kaggle.com/api/v1/kernels/{KERNEL_SLUG}"
    log("Polling for v19 kernel completion...")
    for _ in range(360):  # max 6 hours
        try:
            r = requests.get(status_url, auth=AUTH, timeout=30)
            if r.status_code == 200:
                data = r.json()
                status = data.get('currentRunningVersion', {}).get('status', 'unknown')
                log(f"Kernel status: {status}")
                if status in ('complete', 'Complete'):
                    return True
                if status in ('error', 'Error', 'cancelAcknowledged'):
                    log(f"Kernel failed with status: {status}")
                    return False
        except Exception as e:
            log(f"Poll error: {e}")
        time.sleep(60)
    return False

log("Waiting for v19 kernel to complete...")
if not wait_for_kernel():
    log("Kernel did not complete — aborting."); sys.exit(1)

log(f"v19 kernel COMPLETE — entering submit loop")
log(f"Submitting kernelVersion={DATA['kernelVersion']} ...")

for attempt in range(1, 1000):
    r = requests.post(SUBMIT_URL, data=DATA, auth=AUTH)
    log(f"Attempt {attempt} — HTTP {r.status_code}: {r.text[:400]}")
    if r.status_code == 200:
        log("SUCCESS! Submission accepted."); sys.exit(0)
    if 'daily Submission allowance' in r.text:
        wait = secs_until_utc_midnight()
        log(f"Daily limit hit — sleeping {wait/3600:.1f} h until midnight UTC reset...")
        time.sleep(wait)
        log("Woke up after midnight UTC — retrying submission...")
    elif r.status_code in (429, 503):
        log("Rate limited — sleeping 5 min..."); time.sleep(300)
    else:
        log("Unexpected error — sleeping 2 min..."); time.sleep(120)

log("Exhausted retries."); sys.exit(1)
