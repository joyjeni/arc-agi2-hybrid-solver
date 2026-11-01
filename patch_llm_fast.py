import re

with open("/home/sandbox/arc-jeni-2026/jeni_llm_fast.py", "r") as f:
    src = f.read()

# ── 1. Update docstring ────────────────────────────────────────────────────────
old_doc_start = 'jeni_llm_fast.py'
old_doc_end = 'parse_grid'
# Replace the Design section in the docstring
src = re.sub(
    r'(Design:.*?hard fallback.*?all parsing fails)',
    'Design (matches SFT training format):\n'
    '  \u2022 attempt_1 \u2190 greedy NVARC-format (T=0, compact multi-turn, no system msg)\n'
    '  \u2022 attempt_2 \u2190 compact+CoT (T=0.7, thinking mode, 512 tok reason)\n'
    '  \u2022 fallback   \u2190 compact prompt (T=0.4) when either failed\n'
    '  \u2022 hard fallback \u2190 copy of test input when all parsing fails',
    src, flags=re.DOTALL
)

# ── 2. Add convert_grid_to_string (NVARC exact format) after grid_to_colored ──
nvarc_grid_fn = '''
def convert_grid_to_string(grid) -> str:
    """NVARC/SFT format: rows of compact digits separated by newlines (no spaces).
    Matches the exact format the qwen3_4b_grids15_sft139 model was SFT-trained on.
    """
    text = ""
    for row in grid:
        for cell in row:
            text += str(int(cell))
        text += "\\n"
    return text.strip()

'''
src = src.replace(
    'def parse_grid(',
    nvarc_grid_fn + 'def parse_grid('
)

# ── 3. Add NVARC style to make_prompt ─────────────────────────────────────────
# Find the make_prompt function and add nvarc style before "else:"
nvarc_style_block = '''
    elif style == "nvarc":
        # Exact SFT training format: multi-turn, no system msg, compact grids
        # Each train pair = one user turn (input) + one assistant turn (output)
        # Returns ("", messages_list) — caller must handle list case
        messages = []
        for ex in train:
            messages.append({"role": "user",      "content": convert_grid_to_string(ex["input"])})
            messages.append({"role": "assistant",  "content": convert_grid_to_string(ex["output"])})
        messages.append({"role": "user", "content": convert_grid_to_string(test_input)})
        return ("__NVARC_MULTITURN__", messages)

'''
src = src.replace(
    '    else:\n        return make_prompt(task_view, "direct")',
    nvarc_style_block + '    else:\n        return make_prompt(task_view, "nvarc")'
)

# ── 4. Update _apply_chat to handle nvarc multi-turn messages ─────────────────
# The worker already calls _apply_chat; just make sure it handles lists.
# No change needed — it already accepts `messages: list`.

# ── 5. Update strategy order in worker_inference: nvarc first, cot second ─────
# Replace the "Strategy 1: greedy direct → attempt_1" block
src = src.replace(
    '# ── Strategy 1: greedy direct → attempt_1 ────────────────────\n                try:\n                    sys_m, usr_m = make_prompt(task_view, "direct")\n                    msgs = [\n                        {"role": "system",  "content": sys_m},\n                        {"role": "user",    "content": usr_m},\n                    ]',
    '# ── Strategy 1: greedy NVARC format → attempt_1 ─────────────────\n                # Uses the exact SFT training format: compact multi-turn, no system msg\n                try:\n                    _style_ret = make_prompt(task_view, "nvarc")\n                    if _style_ret[0] == "__NVARC_MULTITURN__":\n                        msgs = _style_ret[1]\n                    else:\n                        sys_m, usr_m = _style_ret\n                        msgs = [\n                            {"role": "system", "content": sys_m},\n                            {"role": "user",   "content": usr_m},\n                        ]'
)

# ── 6. Update strategy 2 (CoT) to use compact format ─────────────────────────
src = src.replace(
    'sys_m, usr_m = make_prompt(task_view, "cot")\n                    msgs = [\n                        {"role": "system",  "content": sys_m},\n                        {"role": "user",    "content": usr_m},\n                    ]',
    'sys_m, usr_m = make_prompt(task_view, "compact")\n                    msgs = [\n                        {"role": "system",  "content": sys_m},\n                        {"role": "user",    "content": usr_m},\n                    ]'
)

# ── 7. Update discover_model_path to use NVARC hard-coded paths first ─────────
old_discover = '''def discover_model_path() -> str:
    """
    Find the Qwen3-4B model directory.
    Priority: ARC_MODEL_PATH env → search /kaggle/input for LM config.
    """
    env_path = os.getenv("ARC_MODEL_PATH", "")
    if env_path and Path(env_path).exists():
        return env_path

    search_root = Path("/kaggle/input")
    if not search_root.exists():
        return ""

    # Walk one level deep to find likely model directories fast
    candidates: list[Path] = []
    for entry in sorted(search_root.rglob("config.json")):
        try:
            cfg = json.loads(entry.read_text())
        except Exception:
            continue
        # Must look like a language model (has vocab_size or model_type)
        if "vocab_size" in cfg or "model_type" in cfg:
            candidates.append(entry.parent)

    # Prefer paths containing "qwen" or "arc"
    for c in candidates:
        if "qwen" in str(c).lower() or "arc" in str(c).lower():
            print(f"[llm-fast] Found model at {c}", flush=True)
            return str(c)
    if candidates:
        print(f"[llm-fast] Using fallback model at {candidates[0]}", flush=True)
        return str(candidates[0])
    return ""'''

new_discover = '''# Hard-coded Kaggle model mount paths for sorokin/qwen3_4b_grids15_sft139
# These mirror the NVARC arc_solver.py discovery order exactly.
_NVARC_MODEL_PATHS = [
    "/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
    "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
    "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
]


def discover_model_path() -> str:
    """
    Find the sorokin/qwen3_4b_grids15_sft139 model directory.
    Priority:
      1. ARC_MODEL_PATH env var (set by Cell 8 env dict)
      2. Hard-coded Kaggle mount paths (mirrors NVARC arc_solver.py)
      3. Glob scan /kaggle/input for any config.json with qwen/arc in path
    """
    env_path = os.getenv("ARC_MODEL_PATH", "")
    if env_path and Path(env_path).exists():
        print(f"[llm-fast] Model via ARC_MODEL_PATH: {env_path}", flush=True)
        return env_path

    # Hard-coded paths first (fastest, most reliable)
    for p in _NVARC_MODEL_PATHS:
        if Path(p).is_dir() and Path(p, "config.json").exists():
            print(f"[llm-fast] Found model at {p}", flush=True)
            return p

    # Fallback: scan /kaggle/input
    search_root = Path("/kaggle/input")
    if not search_root.exists():
        return ""

    candidates: list[Path] = []
    for entry in sorted(search_root.rglob("config.json")):
        try:
            cfg = json.loads(entry.read_text())
        except Exception:
            continue
        if "vocab_size" in cfg or "model_type" in cfg:
            candidates.append(entry.parent)

    for c in candidates:
        if "qwen" in str(c).lower() or "arc" in str(c).lower():
            print(f"[llm-fast] Found model (scan) at {c}", flush=True)
            return str(c)
    if candidates:
        print(f"[llm-fast] Using fallback model at {candidates[0]}", flush=True)
        return str(candidates[0])
    return ""'''

src = src.replace(old_discover, new_discover)

with open("/home/sandbox/arc-jeni-2026/jeni_llm_fast.py", "w") as f:
    f.write(src)

print("Patch applied.")
# Verify key changes
checks = [
    ("nvarc style added", 'style == "nvarc"'),
    ("convert_grid_to_string added", "def convert_grid_to_string"),
    ("hard-coded paths added", "_NVARC_MODEL_PATHS"),
    ("nvarc is primary strategy", 'make_prompt(task_view, "nvarc")'),
    ("fallback is nvarc", 'return make_prompt(task_view, "nvarc")'),
]
for label, check in checks:
    found = check in src
    print(f"  {'✓' if found else '✗'} {label}")
