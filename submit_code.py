#!/usr/bin/env python3
"""Submit latest kernel version output to competition using code submission."""
import sys

COMP    = 'arc-prize-2026-arc-agi-2'
KERNEL  = 'joyjeni/arc-agi-2-solver-three-stage-pipeline'
FILE    = 'submission.json'

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
print("Authenticated ✓")

# Auto-detect latest kernel version
try:
    status = api.kernels_status(KERNEL)
    # status is a dict with currentRunningVersion or similar
    import subprocess, json
    out = subprocess.check_output(['kaggle', 'kernels', 'list', '--mine', '-s', 'arc-agi-2-solver', '--csv'], text=True)
    # parse version from kernels output
    VERSION = None
    for line in out.strip().split('\n')[1:]:
        parts = line.split(',')
        if 'arc-agi-2-solver-three-stage-pipeline' in parts[0]:
            VERSION = int(parts[3]) if len(parts) > 3 else None
            break
except Exception:
    VERSION = None

# Fallback: hard-code latest known version
if not VERSION:
    VERSION = 32
    print(f"(auto-detect failed — using hard-coded VERSION={VERSION})")

DESC = f'v{VERSION}: T0 disabled, NN base + T1 DSA — fix 0.00 root cause'

print(f"Kernel  : {KERNEL}")
print(f"Version : {VERSION}")
print(f"File    : {FILE}")
print()

try:
    result = api.competition_submit_code(
        file_name=FILE,
        message=DESC,
        competition=COMP,
        kernel=KERNEL,
        kernel_version=VERSION,
        quiet=False,
    )
    print(f"\n✅ Submit result: {result}")
except Exception as e:
    print(f"\n❌ Exception: {e}")
