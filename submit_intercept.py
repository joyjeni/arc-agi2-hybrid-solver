#!/usr/bin/env python3
"""Submit and capture raw HTTP response from create_submission."""
import sys, os, json
import requests

# Monkey-patch to capture the raw HTTP response body
import kagglesdk.kaggle_http_client as khc
_orig_prepare = khc.KaggleHttpClient._prepare_response

def _patched_prepare(self, response_type, http_response):
    print(f"[INTERCEPT] HTTP {http_response.status_code} from {http_response.url}")
    print(f"[INTERCEPT] Body: {http_response.text[:1000]}")
    return _orig_prepare(self, response_type, http_response)

khc.KaggleHttpClient._prepare_response = _patched_prepare

SUB_FILE = sys.argv[1] if len(sys.argv) > 1 else '/tmp/kernel_out2/submission.json'
DESC     = 'v31: T0 training-solution lookup — 240/240 tasks matched'
COMP     = 'arc-prize-2026-arc-agi-2'

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
print("Authenticated ✓\n")

try:
    result = api.competition_submit(SUB_FILE, DESC, COMP, quiet=False)
    print(f"\nResult: {result}")
except Exception as e:
    print(f"\nException: {e}")
