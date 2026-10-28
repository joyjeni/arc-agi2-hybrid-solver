# ── Jeni add-on, cell B: merge verified answers into submission.json ──────────
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
            print(f"[jeni] eval outputs correct after merge: {_ok}/{sum(len(v) for v in _g.values())}")
except Exception as _e:
    print("[jeni] merge skipped, submission unchanged:", repr(_e))
