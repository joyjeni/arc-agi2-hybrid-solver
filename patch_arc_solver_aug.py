import json, re

NB_PATH = "/home/sandbox/arc-jeni-2026/notebooks/arc_jeni_submission.ipynb"
nb = json.load(open(NB_PATH))

# Cell 6 is arc_solver.py
cell = nb["cells"][6]
src = "".join(cell["source"])

# ── 1. Add `import random` after `import numpy as np` ─────────────────────────
OLD1 = "import numpy as np\nfrom datasets import Dataset"
NEW1 = "import random\nimport numpy as np\nfrom datasets import Dataset"
assert OLD1 in src, "PATCH1: anchor not found"
src = src.replace(OLD1, NEW1, 1)

# ── 2. Insert augmentation functions after the imports block ──────────────────
#    Anchor: just before the first "def " that follows the imports
AUG_FUNCS = '''

# ═══════════════════════════════════════════════════════════════════════════════
# D4 SPATIAL + COLOR AUGMENTATION  (Dihedral Group, 8 symmetries)
# Used to expand the TTT training dataset for each puzzle.
# ═══════════════════════════════════════════════════════════════════════════════

def augment_grid_spatial(input_grid, output_grid):
    """All 8 D4 symmetries (4 rotations × flip) for one input/output pair."""
    inp_np = np.array(input_grid)
    out_np = np.array(output_grid)
    pairs = []
    for k in range(4):
        rot_in  = np.rot90(inp_np, k)
        rot_out = np.rot90(out_np, k)
        pairs.append((rot_in.tolist(),          rot_out.tolist()))
        pairs.append((np.fliplr(rot_in).tolist(), np.fliplr(rot_out).tolist()))
    return pairs


def augment_grid_colors(input_grid, output_grid, num_permutations=3):
    """Randomly swap non-black colors to make the model color-invariant."""
    inp_np = np.array(input_grid)
    out_np = np.array(output_grid)
    unique_colors = list(
        set(np.unique(inp_np)).union(set(np.unique(out_np))) - {0}
    )
    if len(unique_colors) < 2:
        return [(input_grid, output_grid)]
    pairs = []
    available = list(range(1, 10))
    for _ in range(num_permutations):
        shuffled = random.sample(available, len(unique_colors))
        color_map = {orig: new for orig, new in zip(unique_colors, shuffled)}
        color_map[0] = 0
        perm_in  = np.vectorize(color_map.get)(inp_np)
        perm_out = np.vectorize(color_map.get)(out_np)
        pairs.append((perm_in.tolist(), perm_out.tolist()))
    return pairs


def build_augmented_ttt_dataset(train_pairs):
    """
    Combine spatial D4 + color permutations for all train pairs.
    Returns deduplicated, shuffled list of {\"input\": grid, \"output\": grid}.
    """
    final = []
    for pair in train_pairs:
        inp, out = pair["input"], pair["output"]
        for s_in, s_out in augment_grid_spatial(inp, out):
            final.append({"input": s_in, "output": s_out})
            for c_in, c_out in augment_grid_colors(s_in, s_out, num_permutations=2):
                final.append({"input": c_in, "output": c_out})
    # Deduplicate
    seen, deduped = set(), []
    for p in final:
        key = f"{p['input']}-{p['output']}"
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    random.shuffle(deduped)
    return deduped


def _d4_augmented_train_rows(train_pairs, formatter, max_seq_length):
    """
    Convert augmented pairs into NVARC training text rows ready for the
    HuggingFace Dataset.  Each row: original pairs as context + augmented
    pair as the challenge (last_is_challenge=True).
    Rows that exceed max_seq_length characters are silently dropped.
    """
    aug_pairs = build_augmented_ttt_dataset(train_pairs)
    rows = []
    for aug_pair in aug_pairs:
        try:
            text = formatter.fmt_train(
                train_pairs + [aug_pair], last_is_challenge=True
            )
            # Rough character-length guard (not token-exact, but fast)
            if len(text) <= max_seq_length * 6:
                rows.append({"text": text})
        except Exception:
            pass
    return rows

'''

# Insert BEFORE the first class/def that is NOT inside the %%writefile header block.
# Safe anchor: "class UnslothFixedTrainer"
ANCHOR2 = "class UnslothFixedTrainer"
assert ANCHOR2 in src, f"PATCH2 anchor not found: {ANCHOR2!r}"
src = src.replace(ANCHOR2, AUG_FUNCS.lstrip("\n") + ANCHOR2, 1)

# ── 3. Inject augmented rows into Dataset.from_list ───────────────────────────
OLD3 = (
    "            trainer = UnslothFixedTrainer(\n"
    "                model=model,\n"
    "                tokenizer=tokenizer,\n"
    "                data_collator=collator,\n"
    "                train_dataset=Dataset.from_list(train_ds.as_list(formatter)),"
)
NEW3 = (
    "            # Build extra D4 + color-augmented TTT rows and merge\n"
    "            _base_train_pairs = puzzle_ds.queries[key]['train']\n"
    "            _aug_rows = _d4_augmented_train_rows(\n"
    "                _base_train_pairs, formatter, max_seq_length\n"
    "            )\n"
    "            _base_rows = train_ds.as_list(formatter)\n"
    "            _merged_rows = _base_rows + _aug_rows\n"
    "            random.shuffle(_merged_rows)\n"
    "            trainer = UnslothFixedTrainer(\n"
    "                model=model,\n"
    "                tokenizer=tokenizer,\n"
    "                data_collator=collator,\n"
    "                train_dataset=Dataset.from_list(_merged_rows),"
)
assert OLD3 in src, f"PATCH3 anchor not found"
src = src.replace(OLD3, NEW3, 1)

# ── Write back ─────────────────────────────────────────────────────────────────
cell["source"] = src
with open(NB_PATH, "w") as fh:
    json.dump(nb, fh, indent=1)

print("arc_solver.py patched OK")
print(f"New cell length: {len(src)} chars / {src.count(chr(10))} lines")
