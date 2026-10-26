#!/usr/bin/env python3
"""
regen_cell4_v3.py — Regenerate Cell 4 of the Kaggle notebook
with current src/ files embedded as zlib+base64.

Run from project root:
    python3 regen_cell4_v3.py
"""
import json, zlib, base64, os, sys

NOTEBOOK = "notebooks/arc_jeni_submission.ipynb"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# All source files to embed (relative to project root)
SRC_FILES = [
    "src/__init__.py",
    "src/llm/__init__.py",
    "src/llm/engine.py",
    "src/llm/program_synth.py",
    "src/llm/chakra_ensemble.py",
    "src/llm/prompts.py",
    "src/llm/qwen3_ttt_solver.py",
    "src/core/__init__.py",
    "src/core/interfaces.py",
    "src/core/grid_codec.py",
    "src/core/augmentation.py",
    "src/core/submission.py",
    "src/pipeline.py",
    "src/solvers/__init__.py",
    "src/solvers/registry.py",
    "src/solvers/advanced_solvers.py",
    "src/solvers/advanced_solvers2.py",
    "src/solvers/advanced_solvers3.py",
    "src/solvers/advanced_solvers4.py",
    "src/solvers/advanced_solvers5.py",
    "src/solvers/advanced_solvers6.py",
    "src/solvers/advanced_solvers7.py",
    "src/solvers/advanced_solvers8.py",
    "src/solvers/advanced_solvers9.py",
    "src/solvers/advanced_solvers10.py",
    "src/solvers/advanced_solvers11.py",
    "src/solvers/advanced_solvers12.py",
    "src/solvers/advanced_solvers13.py",
    "src/solvers/advanced_solvers14.py",
    "src/solvers/advanced_solvers15.py",
    "src/solvers/advanced_solvers16.py",
    "src/solvers/advanced_solvers17.py",
    "src/solvers/advanced_solvers18.py",
    "src/solvers/advanced_solvers19.py",
    "src/solvers/advanced_solvers20.py",
    "src/solvers/advanced_solvers21.py",
    "src/solvers/advanced_solvers22.py",
    "src/solvers/advanced_solvers23.py",
    "src/solvers/advanced_solvers24.py",
    "src/solvers/advanced_solvers25.py",
    "src/solvers/agentic_solvers.py",
    "src/solvers/dsa_solvers.py",
    "src/solvers/generalized_solvers.py",
    "src/solvers/new_solvers.py",
    "src/solvers/vedic_solvers.py",
]

def build_cell4():
    """Build the Cell 4 source lines."""
    # Build the file bundle as a dict path->content
    bundle = {}
    for rel_path in SRC_FILES:
        abs_path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(abs_path):
            print(f"  WARNING: {rel_path} not found, skipping")
            continue
        with open(abs_path, "r", encoding="utf-8") as f:
            bundle[rel_path] = f.read()
        print(f"  Packed: {rel_path} ({len(bundle[rel_path])} chars)")

    # Serialize → zlib compress → base64 encode
    raw = json.dumps(bundle).encode("utf-8")
    compressed = zlib.compress(raw, level=9)
    encoded = base64.b64encode(compressed).decode("ascii")

    print(f"\nBundle: {len(bundle)} files, {len(raw)} raw bytes → {len(compressed)} compressed → {len(encoded)} base64 chars")

    cell_lines = [
        "# Cell 4: Embed source files from zlib+base64 bundle\n",
        "import json, zlib, base64, os, sys, textwrap\n",
        "\n",
        "BUNDLE_B64 = (\n",
    ]

    # Split into 80-char lines for readability
    chunk = 76
    parts = [encoded[i:i+chunk] for i in range(0, len(encoded), chunk)]
    for part in parts:
        cell_lines.append(f'    "{part}"\n')
    cell_lines.append(")\n")
    cell_lines.append("\n")
    cell_lines.append("_raw = zlib.decompress(base64.b64decode(BUNDLE_B64))\n")
    cell_lines.append("_bundle = json.loads(_raw.decode('utf-8'))\n")
    cell_lines.append("\n")
    cell_lines.append("# Write files to /kaggle/working/src/\n")
    cell_lines.append("for rel_path, content in _bundle.items():\n")
    cell_lines.append("    abs_path = os.path.join('/kaggle/working', rel_path)\n")
    cell_lines.append("    os.makedirs(os.path.dirname(abs_path), exist_ok=True)\n")
    cell_lines.append("    with open(abs_path, 'w', encoding='utf-8') as f:\n")
    cell_lines.append("        f.write(content)\n")
    cell_lines.append("\n")
    cell_lines.append("# Add to sys.path\n")
    cell_lines.append("if '/kaggle/working' not in sys.path:\n")
    cell_lines.append("    sys.path.insert(0, '/kaggle/working')\n")
    cell_lines.append("\n")
    cell_lines.append("print(f'Extracted {len(_bundle)} source files')\n")

    return cell_lines


def main():
    nb_path = os.path.join(PROJECT_ROOT, NOTEBOOK)
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]
    print(f"Notebook has {len(cells)} cells")

    # Find Cell 4 (index 3, 0-based) — the embed cell
    # Look for the cell that starts with "# Cell 4" or has BUNDLE_B64
    target_idx = None
    for i, cell in enumerate(cells):
        src = "".join(cell.get("source", []))
        if "BUNDLE_B64" in src or "# Cell 4" in src:
            target_idx = i
            break

    if target_idx is None:
        # Default to index 3
        target_idx = 3
        print(f"  Cell 4 marker not found; using index {target_idx}")
    else:
        print(f"  Found embed cell at index {target_idx}")

    print("\nBuilding new Cell 4...")
    new_source = build_cell4()

    cells[target_idx]["source"] = new_source
    cells[target_idx]["cell_type"] = "code"
    if "outputs" not in cells[target_idx]:
        cells[target_idx]["outputs"] = []
    cells[target_idx]["execution_count"] = None

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\nCell {target_idx} updated in {nb_path}")
    print("Done.")


if __name__ == "__main__":
    main()
