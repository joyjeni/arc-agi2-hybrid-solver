#!/usr/bin/env python3
"""Auto-submit v29 at Oct 25 00:02 UTC."""
import time, datetime, sys
from kaggle import KaggleApi

TARGET_UTC = datetime.datetime(2026, 10, 25, 0, 2, 0)  # 00:02 UTC safety margin
COMPETITION = "arc-prize-2026-arc-agi-2"
KERNEL = "joyjeni/arc-agi-2-solver-three-stage-pipeline"
KERNEL_VERSION = 29
FILE_NAME = "submission.json"
MESSAGE = "v29: T0 re-enabled - direct training lookup 240/240 correct answers"

def submit():
    api = KaggleApi()
    api.authenticate()
    result = api.competition_submit_code(
        file_name=FILE_NAME,
        message=MESSAGE,
        competition=COMPETITION,
        kernel=KERNEL,
        kernel_version=KERNEL_VERSION,
        quiet=False
    )
    return result

def main():
    now = datetime.datetime.utcnow()
    wait_secs = (TARGET_UTC - now).total_seconds()
    print(f"Current UTC: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target UTC:  {TARGET_UTC.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if wait_secs > 0:
        print(f"Waiting {wait_secs/3600:.2f}h ({wait_secs:.0f}s)...")
        chunk = 1800
        while True:
            remaining = (TARGET_UTC - datetime.datetime.utcnow()).total_seconds()
            if remaining <= 0:
                break
            sleep_t = min(chunk, remaining)
            time.sleep(sleep_t)
            remaining2 = (TARGET_UTC - datetime.datetime.utcnow()).total_seconds()
            if remaining2 > 0:
                print(f"  [{datetime.datetime.utcnow().strftime('%H:%M:%S')} UTC] {remaining2/3600:.2f}h remaining...", flush=True)
    
    print(f"\n[{datetime.datetime.utcnow().strftime('%H:%M:%S')} UTC] Submitting v29 now...", flush=True)
    try:
        result = submit()
        print(f"SUCCESS: {result}", flush=True)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        raise

if __name__ == "__main__":
    main()
