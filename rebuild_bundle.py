#!/usr/bin/env python3
"""
rebuild_bundle.py — Rebuild cell 11 bundle in arc_jeni_submission.ipynb
with ALL current src/ files including sessions 32-51.
"""
import json, bz2, base64, os

NOTEBOOK  = "notebooks/arc_jeni_submission.ipynb"
SRC_ROOT  = "src"

# All files to include (relative to src/, mapped to bundle key)
# Bundle keys are RELATIVE to jeni_dsa/ (i.e. no "src/" prefix)
SRC_FILES = [
    ("__init__.py",),
    ("core/__init__.py",),
    ("core/augmentation.py",),
    ("core/grid_codec.py",),
    ("core/interfaces.py",),
    ("core/submission.py",),
    ("llm/__init__.py",),
    ("llm/chakra_ensemble.py",),
    ("llm/engine.py",),
    ("llm/program_synth.py",),
    ("llm/prompts.py",),
    ("llm/qwen3_ttt_solver.py",),
    ("pipeline.py",),
    ("solvers/__init__.py",),
    ("solvers/registry.py",),
    ("solvers/advanced_solvers.py",),
    ("solvers/advanced_solvers2.py",),
    ("solvers/advanced_solvers3.py",),
    ("solvers/advanced_solvers4.py",),
    ("solvers/advanced_solvers5.py",),
    ("solvers/advanced_solvers6.py",),
    ("solvers/advanced_solvers7.py",),
    ("solvers/advanced_solvers8.py",),
    ("solvers/advanced_solvers9.py",),
    ("solvers/advanced_solvers10.py",),
    ("solvers/advanced_solvers11.py",),
    ("solvers/advanced_solvers12.py",),
    ("solvers/advanced_solvers13.py",),
    ("solvers/advanced_solvers14.py",),
    ("solvers/advanced_solvers15.py",),
    ("solvers/advanced_solvers16.py",),
    ("solvers/advanced_solvers17.py",),
    ("solvers/advanced_solvers18.py",),
    ("solvers/advanced_solvers19.py",),
    ("solvers/advanced_solvers20.py",),
    ("solvers/advanced_solvers21.py",),
    ("solvers/advanced_solvers22.py",),
    ("solvers/advanced_solvers23.py",),
    ("solvers/advanced_solvers24.py",),
    ("solvers/advanced_solvers25.py",),
    ("solvers/advanced_solvers26.py",),
    ("solvers/advanced_solvers27.py",),
    ("solvers/advanced_solvers28.py",),
    ("solvers/advanced_solvers29.py",),
    ("solvers/advanced_solvers30.py",),
    ("solvers/advanced_solvers31.py",),
    ("solvers/advanced_solvers32.py",),
    ("solvers/advanced_solvers33.py",),
    ("solvers/advanced_solvers34.py",),
    ("solvers/advanced_solvers35.py",),
    ("solvers/advanced_solvers36.py",),
    ("solvers/advanced_solvers37.py",),
    ("solvers/advanced_solvers38.py",),
    ("solvers/advanced_solvers42.py",),
    ("solvers/advanced_solvers43.py",),
    ("solvers/advanced_solvers44.py",),
    ("solvers/advanced_solvers45.py",),
    ("solvers/advanced_solvers46.py",),
    ("solvers/advanced_solvers47.py",),
    ("solvers/advanced_solvers48.py",),
    ("solvers/advanced_solvers49.py",),
    ("solvers/advanced_solvers50.py",),
    ("solvers/advanced_solvers51.py",),
    ("solvers/agentic_solvers.py",),
    ("solvers/dsa_solvers.py",),
    ("solvers/generalized_solvers.py",),
    ("solvers/new_solvers.py",),
    ("solvers/vedic_solvers.py",),
]

def build_bundle():
    bundle = {}
    missing = []
    for (rel_path,) in SRC_FILES:
        abs_path = os.path.join(SRC_ROOT, rel_path)
        if not os.path.exists(abs_path):
            missing.append(rel_path)
            continue
        with open(abs_path, encoding="utf-8") as f:
            bundle[rel_path] = f.read()
        print(f"  + {rel_path} ({len(bundle[rel_path])} chars)")
    if missing:
        print(f"\nWARNING: {len(missing)} files not found: {missing}")
    raw = json.dumps(bundle).encode("utf-8")
    compressed = bz2.compress(raw, compresslevel=9)
    encoded = base64.b64encode(compressed).decode("ascii")
    print(f"\nBundle: {len(bundle)} files, {len(raw)} raw bytes → {len(compressed)} bz2 → {len(encoded)} b64 chars")
    return encoded, len(bundle)

def update_notebook(encoded):
    with open(NOTEBOOK, encoding="utf-8") as f:
        nb = json.load(f)

    cell11 = nb["cells"][11]
    src = "".join(cell11["source"])

    # Find the _BUNDLE = (...) block
    bundle_start = src.find("_BUNDLE = (\n")
    bundle_end_marker = "\n)\n_files = json.loads"
    bundle_end = src.find(bundle_end_marker)
    if bundle_start < 0 or bundle_end < 0:
        raise RuntimeError(f"Cannot find _BUNDLE markers. start={bundle_start}, end={bundle_end}")

    # Build new bundle lines (76 chars per line)
    chunk = 76
    parts = [encoded[i:i+chunk] for i in range(0, len(encoded), chunk)]
    new_bundle_lines = "_BUNDLE = (\n"
    for part in parts:
        new_bundle_lines += f'    "{part}"\n'

    # Replace
    new_src = src[:bundle_start] + new_bundle_lines + src[bundle_end:]
    cell11["source"] = [new_src]
    cell11["execution_count"] = None
    if "outputs" not in cell11:
        cell11["outputs"] = []

    with open(NOTEBOOK, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"\nNotebook updated: {NOTEBOOK}")

def main():
    print("Building bundle from current src/ files...")
    encoded, n_files = build_bundle()
    print(f"\nUpdating notebook cell 11...")
    update_notebook(encoded)
    print("Done.")

if __name__ == "__main__":
    main()
