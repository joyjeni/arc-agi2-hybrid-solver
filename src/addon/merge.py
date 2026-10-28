"""Merge verified symbolic answers into an existing (NVARC) submission.json.

Policy, per test output:
  * verified answer already equals attempt_1 or attempt_2  -> unchanged
  * otherwise -> attempt_2 := verified answer (NVARC's top pick stays as attempt_1)

Kaggle counts an output as solved if either attempt is right, so this can only lose
a point when NVARC's *second* guess was right and the verified answer is wrong.

The merge is fail-safe: any problem with the inputs leaves the submission untouched.
"""

from __future__ import annotations

import json
from pathlib import Path


def _is_grid(g) -> bool:
    return (
        isinstance(g, list) and len(g) > 0
        and all(isinstance(r, list) and len(r) == len(g[0]) and len(r) > 0 for r in g)
        and all(type(v) is int and 0 <= v <= 9 for r in g for v in r)
    )


def merge(submission: dict, verified: dict) -> tuple[dict, dict]:
    """Return (new_submission, stats). Inputs are not modified."""
    out = json.loads(json.dumps(submission))  # deep copy, plain types only
    stats = {"verified": 0, "agreed": 0, "added": 0, "skipped": 0}
    for tid, entries in verified.items():
        sub_entries = out.get(tid)
        if not isinstance(sub_entries, list) or len(sub_entries) != len(entries):
            stats["skipped"] += sum(1 for e in entries if e.get("grid") is not None)
            continue
        for k, e in enumerate(entries):
            g = e.get("grid")
            if g is None:
                continue
            stats["verified"] += 1
            slot = sub_entries[k]
            if not _is_grid(g) or not isinstance(slot, dict):
                stats["skipped"] += 1
                continue
            if g == slot.get("attempt_1") or g == slot.get("attempt_2"):
                stats["agreed"] += 1
                continue
            if not _is_grid(slot.get("attempt_1")):
                slot["attempt_1"] = g  # NVARC produced nothing usable here
            slot["attempt_2"] = g
            stats["added"] += 1
    return out, stats


def merge_files(submission_path: str | Path, verified_path: str | Path) -> dict:
    """Merge in place. Never raises: on any error the submission file is left as it was."""
    try:
        sub_p, ver_p = Path(submission_path), Path(verified_path)
        if not sub_p.exists() or not ver_p.exists():
            return {"error": "missing file"}
        submission = json.loads(sub_p.read_text())
        verified = json.loads(ver_p.read_text())
        new, stats = merge(submission, verified)
        tmp = sub_p.with_suffix(".merged.tmp")
        tmp.write_text(json.dumps(new))
        json.loads(tmp.read_text())  # re-read check before replacing
        tmp.replace(sub_p)
        return stats
    except Exception as exc:  # fail-safe by design
        return {"error": f"{type(exc).__name__}: {exc}"}
