"""Build the two cells that add the verified symbolic layer to an NVARC fork.

    python scripts/build_nvarc_addon.py      # writes notebooks/nvarc_addon/cell_A_start.py, cell_B_merge.py

Cell A — paste right after the NVARC notebook's first cell (the one defining RERUN, T0,
         global_end_time). Unpacks the code and starts the verifier in the background
         (CPU only, lowest priority), so NVARC loses no GPU time.
Cell B — paste right after the cell that writes submission.json. Merges verified answers
         into it. Fail-safe: on any problem the NVARC submission is left untouched.
"""

from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "nvarc_addon"

# Only what the verifier needs: no LLM code, no torch.
INCLUDE = ["src/__init__.py", "src/core", "src/solvers", "src/addon"]


def collect() -> dict[str, str]:
    files: dict[str, str] = {}
    for item in INCLUDE:
        p = ROOT / item
        paths = [p] if p.is_file() else sorted(p.rglob("*.py"))
        for f in paths:
            files[str(f.relative_to(ROOT))] = f.read_text(encoding="utf-8")
    return files


CELL_A = '''# ── Jeni add-on, cell A: start the verified symbolic layer in the background ──
# Paste this right AFTER the first NVARC cell (it needs RERUN from that cell).
# CPU only, lowest priority, no GPU: NVARC's time budget is unaffected.
# Source: https://github.com/joyjeni/arc-agi2-hybrid-solver (src/addon)
import base64, json, os, subprocess, sys, zlib
from pathlib import Path

JENI_DIR = Path("/kaggle/working/jeni_addon")
JENI_OUT = JENI_DIR / "verified.json"
JENI_BUNDLE = (
{chunks}
)
try:
    for _rel, _src in json.loads(zlib.decompress(base64.b64decode(JENI_BUNDLE))).items():
        _p = JENI_DIR / _rel
        _p.parent.mkdir(parents=True, exist_ok=True)
        _p.write_text(_src, encoding="utf-8")
    _name = "arc-agi_test_challenges.json" if RERUN else "arc-agi_evaluation_challenges.json"
    _hits = sorted(Path("/kaggle/input").rglob(_name))
    if not _hits:
        raise FileNotFoundError(_name)
    _env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    jeni_proc = subprocess.Popen(
        ["nice", "-n", "19", sys.executable, "-m", "src.addon.verified",
         str(_hits[0]), str(JENI_OUT), "--budget-s", "3600" if RERUN else "600"],
        cwd=JENI_DIR, env=_env,
        stdout=open(JENI_DIR / "verified.log", "w"), stderr=subprocess.STDOUT,
    )
    print(f"[jeni] verifier started in background on {{_hits[0].name}} (pid {{jeni_proc.pid}})")
except Exception as _e:
    jeni_proc = None
    print("[jeni] add-on disabled, NVARC continues unchanged:", repr(_e))
'''

CELL_B = '''# ── Jeni add-on, cell B: merge verified answers into submission.json ──────────
# Paste this right AFTER the cell that writes submission.json.
# Rule: a verified answer replaces attempt_2 only when it differs from both attempts;
# NVARC's top pick stays as attempt_1. Any error leaves submission.json untouched.
import json, sys, time
from pathlib import Path

try:
    _jdir = Path("/kaggle/working/jeni_addon")
    _p = globals().get("jeni_proc")
    if _p is not None and _p.poll() is None:
        _t = time.time()
        while _p.poll() is None and time.time() - _t < 120:  # normally finished hours ago
            time.sleep(2)
        if _p.poll() is None:
            _p.kill()
            print("[jeni] verifier still running after 120 s; using its partial results")
    if str(_jdir) not in sys.path:
        sys.path.insert(0, str(_jdir))
    from src.addon.merge import merge_files
    _sub = Path("/kaggle/working/submission.json")
    print("[jeni] merge:", merge_files(_sub, _jdir / "verified.json"))
    if not RERUN:  # commit run: show the effect on the public evaluation tasks
        _sol = sorted(Path("/kaggle/input").rglob("arc-agi_evaluation_solutions.json"))
        if _sol:
            _s, _g = json.loads(_sub.read_text()), json.loads(_sol[0].read_text())
            _ok = sum(1 for t, gs in _g.items() for k, g in enumerate(gs)
                      if t in _s and k < len(_s[t]) and g in (_s[t][k]["attempt_1"], _s[t][k]["attempt_2"]))
            print(f"[jeni] eval outputs correct after merge: {{_ok}}/{{sum(len(v) for v in _g.values())}}")
except Exception as _e:
    print("[jeni] merge skipped, submission unchanged:", repr(_e))
'''


def main() -> None:
    files = collect()
    blob = base64.b64encode(zlib.compress(json.dumps(files).encode(), 9)).decode()
    chunks = "\n".join(f'    "{blob[i:i + 100]}"' for i in range(0, len(blob), 100))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cell_A_start.py").write_text(CELL_A.format(chunks=chunks), encoding="utf-8")
    (OUT_DIR / "cell_B_merge.py").write_text(CELL_B.format(), encoding="utf-8")
    print(f"{len(files)} files bundled ({len(blob) / 1024:.0f} KB) -> {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
