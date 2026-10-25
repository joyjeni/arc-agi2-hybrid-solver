#!/usr/bin/env python3
"""Debug submission — intercept raw response from create_submission."""
import sys, os, json
import requests

SUB_FILE = sys.argv[1] if len(sys.argv) > 1 else '/tmp/kernel_out2/submission.json'
DESC     = sys.argv[2] if len(sys.argv) > 2 else 'v31: T0 training-solution lookup 240/240'
COMP     = 'arc-prize-2026-arc-agi-2'
BASE     = 'https://www.kaggle.com/api/v1'

creds = json.loads(open('/root/.kaggle/kaggle.json').read())
AUTH  = (creds['username'], creds['key'])

# ── Step 1: Get upload URL ────────────────────────────────────────────────────
print("Step 1: Getting upload URL …")
file_size  = os.path.getsize(SUB_FILE)
file_mtime = int(os.path.getmtime(SUB_FILE))
payload1 = {
    'competitionName': COMP,
    'fileName': os.path.basename(SUB_FILE),
    'contentLength': file_size,
    'lastModifiedEpochSeconds': file_mtime,
}
r1 = requests.post(f'{BASE}/competitions/submission-url',
                   json=payload1, auth=AUTH)
print(f"  HTTP {r1.status_code}")
print(f"  Body: {r1.text[:500]}")
r1.raise_for_status()
data1 = r1.json()
token      = data1.get('token') or data1.get('createToken')
create_url = data1.get('createUrl') or data1.get('create_url')
print(f"  token      = {token}")
print(f"  create_url = {create_url}")

# ── Step 2: Upload file ───────────────────────────────────────────────────────
print("\nStep 2: Uploading file to blob URL …")
with open(SUB_FILE, 'rb') as f:
    file_bytes = f.read()
headers2 = {'Content-Type': 'application/octet-stream',
             'Content-Length': str(file_size)}
r2 = requests.put(create_url, data=file_bytes, headers=headers2)
print(f"  HTTP {r2.status_code}: {r2.text[:200]}")

# ── Step 3: Create submission ─────────────────────────────────────────────────
print("\nStep 3: Creating submission record …")
payload3 = {
    'competitionName': COMP,
    'blobFileTokens': token,
    'submissionDescription': DESC,
}
print(f"  Payload: {json.dumps(payload3)}")
r3 = requests.post(f'{BASE}/competitions/submissions/submit/{COMP}',
                   json=payload3, auth=AUTH)
print(f"  HTTP {r3.status_code}")
print(f"  Body: {r3.text[:800]}")
