"""
Prompt templates for ARC-AGI-2 grid reasoning.

"Chakra" ensemble: 8 prompt styles derived from different cognitive frames.
Each yields a different inductive bias, improving coverage when ensembled.
"""

from __future__ import annotations

from ..core.grid_codec import Grid, grid_to_text, grid_dims


def _fmt_pair(inp: Grid, out: Grid, idx: int, mode: str = "numeric") -> str:
    return (
        f"Example {idx + 1}:\n"
        f"  Input  [{len(inp)}×{len(inp[0])}]: {grid_to_text(inp, mode)}\n"
        f"  Output [{len(out)}×{len(out[0])}]: {grid_to_text(out, mode)}\n"
    )


# ── 8 Chakra Prompt Templates ─────────────────────────────────────────────────

def chakra_1_numeric(train_pairs: list[dict], test_input: Grid) -> str:
    """Root chakra: plain numeric encoding, minimal instruction."""
    examples = "".join(_fmt_pair(p["input"], p["output"], i) for i, p in enumerate(train_pairs))
    h, w = grid_dims(test_input)
    return (
        "You are solving an ARC-AGI grid puzzle. "
        "Study the input→output transformations below and apply the same rule.\n\n"
        f"{examples}"
        f"Test input [{h}×{w}]: {grid_to_text(test_input, 'numeric')}\n"
        "Test output (use '|' between rows, space between cells, integers 0-9):"
    )


def chakra_2_symbol(train_pairs: list[dict], test_input: Grid) -> str:
    """Sacral chakra: symbol encoding — more compact tokens."""
    examples = "".join(_fmt_pair(p["input"], p["output"], i, "symbol") for i, p in enumerate(train_pairs))
    h, w = grid_dims(test_input)
    return (
        "Grid puzzle. Symbols: .=0 a=1 b=2 c=3 d=4 e=5 f=6 g=7 h=8 i=9\n"
        "Observe pattern:\n\n"
        f"{examples}"
        f"Test [{h}×{w}]: {grid_to_text(test_input, 'symbol')}\n"
        "Answer (same symbol format, | separates rows):"
    )


def chakra_3_cot(train_pairs: list[dict], test_input: Grid) -> str:
    """Solar plexus: chain-of-thought, reason then answer."""
    examples = "".join(_fmt_pair(p["input"], p["output"], i) for i, p in enumerate(train_pairs))
    h, w = grid_dims(test_input)
    return (
        "Solve this ARC-AGI puzzle step by step.\n\n"
        "--- Training examples ---\n"
        f"{examples}"
        "--- Your reasoning ---\n"
        "Step 1: Describe the transformation rule in plain English.\n"
        "Step 2: Verify the rule against all examples.\n"
        "Step 3: Apply the rule to the test input.\n\n"
        f"Test input [{h}×{w}]: {grid_to_text(test_input)}\n\n"
        "Answer: (numeric grid, | between rows)"
    )


def chakra_4_prog_synth(train_pairs: list[dict], test_input: Grid) -> str:
    """Heart chakra: program synthesis — LLM writes Python code."""
    import json
    pairs_json = json.dumps([
        {"input": p["input"], "output": p["output"]} for p in train_pairs
    ], separators=(",", ":"))
    test_json = json.dumps(test_input, separators=(",", ":"))
    return (
        "Write a Python function `solve(grid)` that produces the correct output "
        "for this ARC-AGI task. The function must work for all examples below.\n\n"
        f"Training pairs (JSON): {pairs_json}\n\n"
        "Requirements:\n"
        "- `solve` takes a 2-D list of ints (0-9) and returns a 2-D list of ints.\n"
        "- No imports beyond standard library.\n"
        "- Keep it concise.\n\n"
        "After the function, on a new line write exactly:\n"
        f"RESULT: {{json of solve({test_json})}}\n\n"
        "```python\n"
    )


def chakra_5_size_hint(train_pairs: list[dict], test_input: Grid) -> str:
    """Throat chakra: include explicit output-size hints."""
    pairs_text = ""
    out_sizes = []
    for i, p in enumerate(train_pairs):
        ih, iw = grid_dims(p["input"])
        oh, ow = grid_dims(p["output"])
        out_sizes.append((oh, ow))
        pairs_text += (
            f"Pair {i+1}: in=[{ih}×{iw}] {grid_to_text(p['input'])} "
            f"→ out=[{oh}×{ow}] {grid_to_text(p['output'])}\n"
        )
    ih, iw = grid_dims(test_input)
    size_hint = ""
    if len(set(out_sizes)) == 1:
        size_hint = f"(expected output size: {out_sizes[0][0]}×{out_sizes[0][1]})"
    return (
        "ARC puzzle. Learn the rule from these pairs:\n\n"
        f"{pairs_text}\n"
        f"Test: in=[{ih}×{iw}] {grid_to_text(test_input)} {size_hint}\n"
        "Output grid (| = row separator):"
    )


def chakra_6_delta(train_pairs: list[dict], test_input: Grid) -> str:
    """Third eye: focus on changed cells (delta between input and output)."""
    delta_lines = []
    for i, p in enumerate(train_pairs):
        inp, out = p["input"], p["output"]
        if grid_dims(inp) == grid_dims(out):
            changes = []
            for r in range(len(inp)):
                for c in range(len(inp[0])):
                    if inp[r][c] != out[r][c]:
                        changes.append(f"({r},{c}): {inp[r][c]}→{out[r][c]}")
            delta_lines.append(f"Pair {i+1} changes: {', '.join(changes[:20]) or 'none'}")
        else:
            delta_lines.append(f"Pair {i+1}: size change {grid_dims(inp)}→{grid_dims(out)}")
    delta_text = "\n".join(delta_lines)
    return (
        "ARC-AGI task. Here are the cell-level changes per training pair:\n\n"
        f"{delta_text}\n\n"
        "Training pairs:\n"
        + "".join(_fmt_pair(p["input"], p["output"], i) for i, p in enumerate(train_pairs))
        + f"\nTest input: {grid_to_text(test_input)}\nOutput:"
    )


def chakra_7_role(train_pairs: list[dict], test_input: Grid) -> str:
    """Crown chakra: role-playing as an expert ARC solver."""
    examples = "".join(_fmt_pair(p["input"], p["output"], i) for i, p in enumerate(train_pairs))
    return (
        "You are the world's best ARC-AGI puzzle solver. "
        "You always find the minimal, elegant transformation rule.\n\n"
        f"Study these {len(train_pairs)} examples carefully:\n{examples}\n"
        f"Now predict the output for: {grid_to_text(test_input)}\n"
        "Output (numeric, | between rows):"
    )


def chakra_8_structured(train_pairs: list[dict], test_input: Grid) -> str:
    """Integration: structured XML-like tags for maximum clarity."""
    pairs_xml = ""
    for i, p in enumerate(train_pairs):
        pairs_xml += (
            f"<pair id='{i}'>"
            f"<input>{grid_to_text(p['input'])}</input>"
            f"<output>{grid_to_text(p['output'])}</output>"
            f"</pair>\n"
        )
    return (
        "<task>\n"
        "<instruction>Find the transformation rule and apply it to the test input.</instruction>\n"
        f"<examples>\n{pairs_xml}</examples>\n"
        f"<test><input>{grid_to_text(test_input)}</input></test>\n"
        "</task>\n"
        "<answer>"
    )


ALL_CHAKRA_TEMPLATES = [
    chakra_1_numeric,
    chakra_2_symbol,
    chakra_3_cot,
    chakra_4_prog_synth,
    chakra_5_size_hint,
    chakra_6_delta,
    chakra_7_role,
    chakra_8_structured,
]
