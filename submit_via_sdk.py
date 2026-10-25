#!/usr/bin/env python3
"""Submit competition file using the kaggle SDK's own competition_submit method."""
import sys

SUB_FILE = sys.argv[1] if len(sys.argv) > 1 else '/tmp/kernel_out2/submission.json'
DESC     = sys.argv[2] if len(sys.argv) > 2 else 'v31: T0 training-solution lookup — 240/240 tasks'
COMP     = 'arc-prize-2026-arc-agi-2'

print(f"File  : {SUB_FILE}")
print(f"Desc  : {DESC}")
print(f"Comp  : {COMP}")
print()

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
print("Authenticated ✓")

result = api.competition_submit(SUB_FILE, DESC, COMP, quiet=False)
print(f"\nResult: {result}")
