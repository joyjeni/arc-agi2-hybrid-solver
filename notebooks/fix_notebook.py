"""Apply all 3 bug fixes to arc_jeni_submission.ipynb and write v10."""
import json, uuid, pathlib

NB_PATH = pathlib.Path(__file__).parent / "arc_jeni_submission.ipynb"
nb = json.load(open(NB_PATH))

# ── Bug 1: Add id field to every cell that lacks one ─────────────────────────
for cell in nb["cells"]:
    if "id" not in cell:
        cell["id"] = uuid.uuid4().hex[:8]

# ── Bug 2: Replace Cell 1 — minimal pip install (no pre-installed packages) ──
CELL1_NEW = """\
# ── Install dependencies (Kaggle pre-installs torch/transformers/etc) ────────
import subprocess, sys

_extra = []
try:
    import yaml          # pyyaml check
except ImportError:
    _extra.append('pyyaml')

if _extra:
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q'] + _extra,
        check=True,
    )
    print('Installed:', _extra)
else:
    print('All dependencies already satisfied — skipping pip install.')
"""
nb["cells"][1]["source"] = CELL1_NEW

# ── Bug 3: Replace Cell 5 — robust data-file discovery ───────────────────────
CELL5_NEW = """\
# ── Load challenge data ──────────────────────────────────────────────────────
import os, glob as _glob
from pathlib import Path
from src.pipeline import load_challenges

# Diagnostic: show what is actually in /kaggle/input/
KAGGLE_INPUT = Path('/kaggle/input')
print('── /kaggle/input/ tree ──────────────────────────────────────────────')
for root, dirs, files in os.walk(KAGGLE_INPUT):
    depth = root.replace(str(KAGGLE_INPUT), '').count(os.sep)
    indent = '  ' * depth
    print(f'{indent}{Path(root).name}/')
    sub = '  ' * (depth + 1)
    for f in files:
        print(f'{sub}{f}')
    if depth >= 3:          # don't recurse too deep
        dirs.clear()
print('────────────────────────────────────────────────────────────────────')

def _find_json(patterns):
    \"\"\"Search /kaggle/input recursively for the first matching filename pattern.\"\"\"
    for pat in patterns:
        hits = list(KAGGLE_INPUT.rglob(pat))
        if hits:
            return hits[0]
    return None

# Priority 1 — test challenges (competition scoring)
CHALLENGES_FILE = _find_json(['arc-agi_test_challenges.json',
                               'arc-agi-test-challenges.json',
                               '*test*challenges*.json'])
if CHALLENGES_FILE is None:
    # Priority 2 — evaluation challenges (dev/testing fallback)
    CHALLENGES_FILE = _find_json(['arc-agi_evaluation_challenges.json',
                                   'arc-agi-evaluation-challenges.json',
                                   '*evaluation*challenges*.json'])
    if CHALLENGES_FILE is None:
        all_json = list(KAGGLE_INPUT.rglob('*.json'))
        raise FileNotFoundError(
            'No test or evaluation challenges file found in /kaggle/input/\\n'
            f'All JSON files present: {[str(p) for p in all_json]}'
        )
    logger.warning('Test challenges not found — using evaluation set as dev fallback')

print(f'Using challenges file: {CHALLENGES_FILE}')
tasks = load_challenges(CHALLENGES_FILE)
print(f'Loaded {len(tasks)} tasks from {CHALLENGES_FILE.name}')
print('Sample task IDs:', list(tasks.keys())[:5])
print(f'Remaining budget at data load: {_remaining()/3600:.2f}h')
"""
nb["cells"][5]["source"] = CELL5_NEW

# ── Write updated notebook ────────────────────────────────────────────────────
json.dump(nb, open(NB_PATH, "w"), indent=1)
print("Fixed notebook written to", NB_PATH)

# Verify all cells now have ids
missing = [i for i, c in enumerate(nb["cells"]) if "id" not in c]
if missing:
    print("WARNING — cells still missing id:", missing)
else:
    print(f"All {len(nb['cells'])} cells have id fields.")
