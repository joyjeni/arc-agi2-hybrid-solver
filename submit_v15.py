#!/usr/bin/env python3
"""Auto-submit kernel v15 to ARC-AGI-2 competition."""
import os, sys, time, json
from datetime import datetime, timezone

os.environ['KAGGLE_USERNAME'] = 'joyjeni'
os.environ['KAGGLE_KEY'] = '6c9120e570546b316242458e2ed7b63d'

import kaggle
kaggle.api.authenticate()

COMPETITION = 'arc-prize-2026-arc-agi-2'
KERNEL      = 'joyjeni/arc-agi-2-solver-three-stage-pipeline'
VERSION     = 15
FILE_NAME   = 'submission.json'
MESSAGE     = 'v15 T0+T1 fix — exact fingerprint lookup 240/240 tasks'

MAX_RETRIES = 5
RETRY_DELAY = 300   # 5 min between retries if non-limit error

for attempt in range(1, MAX_RETRIES + 1):
    now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f"[{now_utc}] Attempt {attempt}/{MAX_RETRIES}...")
    try:
        result = kaggle.api.competition_submit_code(
            file_name=FILE_NAME,
            message=MESSAGE,
            competition=COMPETITION,
            kernel=KERNEL,
            kernel_version=VERSION
        )
        print(f"SUCCESS: {result}")
        sys.exit(0)
    except Exception as e:
        msg = str(e)
        print(f"ERROR: {msg}")
        if 'daily Submission allowance' in msg or '429' in msg:
            print("Daily limit still active — sleeping 10 min and retrying...")
            time.sleep(600)
        else:
            if attempt < MAX_RETRIES:
                print(f"Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print("All retries exhausted.")
                sys.exit(1)
