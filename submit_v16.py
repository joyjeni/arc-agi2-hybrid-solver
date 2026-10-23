#!/usr/bin/env python3
import os, sys, time, requests, json
from datetime import datetime, timezone
from requests.auth import HTTPBasicAuth

AUTH = HTTPBasicAuth('joyjeni', '6c9120e570546b316242458e2ed7b63d')
URL  = 'https://www.kaggle.com/api/v1/competitions/submissions/submit-notebook/arc-prize-2026-arc-agi-2'
DATA = {
    'competitionName':       'arc-prize-2026-arc-agi-2',
    'kernelOwner':           'joyjeni',
    'kernelSlug':            'arc-agi-2-solver-three-stage-pipeline',
    'kernelVersion':         '16',
    'fileName':              'submission.json',
    'submissionDescription': 'v16: NN fallback + T1 DSA, T0 disabled',
}
for attempt in range(1, 6):
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f'[{now}] Attempt {attempt}/5...', flush=True)
    r = requests.post(URL, data=DATA, auth=AUTH)
    print(f'  HTTP {r.status_code}: {r.text[:200]}', flush=True)
    if r.status_code == 200:
        print('SUCCESS!', flush=True); sys.exit(0)
    if 'daily Submission allowance' in r.text:
        print('Daily limit — sleeping 10 min...', flush=True); time.sleep(600)
    else:
        time.sleep(60)
print('All retries exhausted.'); sys.exit(1)
