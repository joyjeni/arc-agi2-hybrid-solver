#!/usr/bin/env python3
"""
Kaggle competition submission via 2-step staged blob upload API.
Usage: python3 submit_blob.py <submission_file> <description>
"""
import sys, os, json, time, hashlib, requests
from pathlib import Path

# ── Credentials ──────────────────────────────────────────────────────────────
CRED_PATH = Path('/root/.kaggle/kaggle.json')
creds = json.loads(CRED_PATH.read_text())
USERNAME = creds['username']
API_KEY   = creds['key']
AUTH      = (USERNAME, API_KEY)
COMP      = 'arc-prize-2026-arc-agi-2'
BASE      = 'https://www.kaggle.com/api/v1'

# ── Args ─────────────────────────────────────────────────────────────────────
SUB_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/tmp/kernel_out2/submission.json')
DESC     = sys.argv[2] if len(sys.argv) > 2 else 'v31: T0+T1 hybrid solver — 240 tasks — trained solutions'

print(f"Submission file : {SUB_FILE}")
print(f"File size       : {SUB_FILE.stat().st_size:,} bytes")
print(f"Description     : {DESC}")
print()

file_size  = SUB_FILE.stat().st_size
file_mtime = int(SUB_FILE.stat().st_mtime * 1000)   # milliseconds

# ── Step 1: Get blob upload URL ───────────────────────────────────────────────
print("Step 1: Requesting blob upload token …")
url1 = f"{BASE}/competitions/submissions/url/{COMP}/{file_size}/{file_mtime}"
r1 = requests.get(url1, auth=AUTH)
print(f"  HTTP {r1.status_code}: {r1.text[:300]}")
r1.raise_for_status()
data1 = r1.json()
blob_url        = data1.get('createUrl') or data1.get('url') or data1.get('blobUrl')
blob_file_token = data1.get('token') or data1.get('blobFileToken')
print(f"  blob_url        = {blob_url}")
print(f"  blob_file_token = {blob_file_token}")

# ── Step 2: Upload file to blob URL ──────────────────────────────────────────
if blob_url:
    print("\nStep 2: Uploading file to blob URL …")
    with SUB_FILE.open('rb') as f:
        file_data = f.read()
    r2 = requests.put(blob_url, data=file_data,
                      headers={'Content-Type': 'application/octet-stream'})
    print(f"  HTTP {r2.status_code}: {r2.text[:200]}")
    if r2.status_code not in (200, 201):
        print(f"  WARNING: blob upload may have failed (status {r2.status_code})")
else:
    print("  No blob_url returned — skipping direct PUT")

# ── Step 3: Create submission record ─────────────────────────────────────────
print("\nStep 3: Submitting to competition …")
url3  = f"{BASE}/competitions/submissions/submit/{COMP}"
# Try with blob token first
if blob_file_token:
    payload = {
        'blobFileTokens': [blob_file_token],
        'submissionDescription': DESC
    }
    r3 = requests.post(url3, json=payload, auth=AUTH)
    print(f"  (json payload) HTTP {r3.status_code}: {r3.text[:400]}")
    if r3.status_code in (200, 201):
        print("\n✅ Submission created successfully!")
        print(json.dumps(r3.json(), indent=2)[:600])
        sys.exit(0)

# Fallback: form-encoded
print("  Trying form-encoded …")
r3b = requests.post(url3,
                    data={'blobFileTokens': blob_file_token or '',
                          'submissionDescription': DESC},
                    auth=AUTH)
print(f"  (form) HTTP {r3b.status_code}: {r3b.text[:400]}")
if r3b.status_code in (200, 201):
    print("\n✅ Submission created successfully!")
    sys.exit(0)

# Fallback: multipart form upload
print("\nFallback: direct multipart upload …")
url_upload = f"{BASE}/competitions/{COMP}/submissions"
with SUB_FILE.open('rb') as f:
    files = {'file': (SUB_FILE.name, f, 'application/json')}
    data  = {'submissionDescription': DESC}
    r4 = requests.post(url_upload, files=files, data=data, auth=AUTH)
print(f"  HTTP {r4.status_code}: {r4.text[:400]}")

print("\n❌ All submission paths attempted — check responses above")
