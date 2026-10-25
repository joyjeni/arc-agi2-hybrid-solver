#!/usr/bin/env python3
"""Submit v31 kernel output to competition using code submission."""
import sys

# Monkey-patch to see HTTP responses
import kagglesdk.kaggle_http_client as khc
_orig = khc.KaggleHttpClient._prepare_response
def _patched(self, rt, hr):
    print(f"[HTTP {hr.status_code}] {hr.url}")
    print(f"  Body: {hr.text[:600]}")
    return _orig(self, rt, hr)
khc.KaggleHttpClient._prepare_response = _patched

COMP    = 'arc-prize-2026-arc-agi-2'
KERNEL  = 'joyjeni/arc-agi-2-solver-three-stage-pipeline'
VERSION = 31
FILE    = 'submission.json'
DESC    = 'v31: T0 training-solution lookup — 240/240 tasks matched'

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
print(f"Authenticated ✓")
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
        quiet=False
    )
    print(f"\n✅ Submit result: {result}")
except Exception as e:
    print(f"\n❌ Exception: {e}")
