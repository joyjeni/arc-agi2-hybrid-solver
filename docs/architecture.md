# Architecture

## Overview

arc-jeni-2026 implements a **3-Tier Hybrid Pipeline** for ARC-AGI-2.  
Each tier runs in sequence; a task is skipped in later tiers once a
confident prediction is produced.

```
┌──────────────────────────────────────────────────┐
│              ARC-AGI-2 Test Tasks (240)           │
└───────────────────┬──────────────────────────────┘
                    │
        ┌───────────▼───────────┐
        │   TIER 1 — DSA Solvers │  ~60 s  →  ~5–10 %
        │  (deterministic rules) │
        └───────────┬───────────┘
        unsolved    │
        ┌───────────▼────────────────┐
        │  TIER 2 — Program Synthesis │  ~30 min  →  ~15–25 %
        │  (LLM writes & runs Python) │
        └───────────┬────────────────┘
        still unsolved
        ┌───────────▼─────────────────────┐
        │  TIER 3 — Chakra-Ensemble LLM   │  remaining budget  →  ~10–20 %
        │  (8 prompt templates, majority   │
        │   vote, ThreadPoolExecutor)      │
        └───────────┬─────────────────────┘
                    │
        ┌───────────▼──────────────┐
        │   submission.json        │  exact competition format
        └──────────────────────────┘
```

**Target LB: 35–45 %**

---

## Tier 1 — DSA Solvers

Located in `src/solvers/`.

### Solver registry

`SolverRegistry` (singleton) maps solver names → classes.  
`@register_solver` decorator auto-registers on import.

### Solver families

| Family | Module | Solvers |
|--------|--------|---------|
| DSA    | `dsa_solvers.py` | Identity, ColorFrequency, GravityFall, ReflectH/V, Rotate90, Transpose, CCRecolorBySize, RemoveSmallObjects, ColorSwap, RowSort |
| Vedic / Chinese | `vedic_solvers.py` | Anurupya (scale), Paravartya (symmetry), Vyashti (tile), Lopana (bbox), BaGua (octagram transform), ZhongLi (directed gravity), LoShu (periodic fill) |

Each solver:
1. Checks `can_solve(task_context)` — returns confidence 0–1
2. Calls `solve(task_context)` — returns `SolverResult`

---

## Tier 2 — Program Synthesis

`src/llm/program_synth.py`

1. Encode train pairs as JSON (grid_codec verbose mode)
2. Prompt LLM to write a Python `transform(grid)` function
3. Execute in subprocess with timeout
4. Validate output against all train outputs (exact match)
5. Apply to test inputs if valid; retry up to 3× on failure

---

## Tier 3 — Chakra-Ensemble

`src/llm/chakra_ensemble.py`

8 Chakra prompt templates (spiritual/geometric framing of ARC patterns):

| # | Chakra | Focus |
|---|--------|-------|
| 1 | Muladhara | Grid structure & boundaries |
| 2 | Svadhisthana | Color flow & symmetry |
| 3 | Manipura | Pattern energy & replication |
| 4 | Anahata | Object relationships |
| 5 | Vishuddha | Sequence & transformation |
| 6 | Ajna | Meta-pattern reasoning |
| 7 | Sahasrara | Abstract rule extraction |
| 8 | Integration | Cross-chakra synthesis |

ThreadPoolExecutor runs all 8 in parallel.  
Majority vote (≥ 2 matching grids) selects `attempt_1`.  
Runner-up (or attempt_1 slightly perturbed) used for `attempt_2`.

---

## Submission Format

```json
{
  "task_id_1": [
    {"attempt_1": [[0,1],[2,3]], "attempt_2": [[0,1],[2,3]]}
  ],
  "task_id_2": [
    {"attempt_1": [[…]], "attempt_2": [[…]]},
    {"attempt_1": [[…]], "attempt_2": [[…]]}
  ]
}
```

One list entry per test input.  `attempt_1` and `attempt_2` are independent guesses.
Exact-match scoring: +1 if either attempt matches the ground truth.

---

## LLM Backend

`src/llm/engine.py` provides a unified `QwenEngine` interface:

| Backend | Use case |
|---------|----------|
| `HuggingFaceLLMBackend` | Kaggle notebook (GPU, 4-bit) |
| `ServerLLMBackend` | SGLang / vLLM server |
| `MockLLMBackend` | Unit tests, CI |

Model: **Qwen3-4B** (`qwen3_4b_grids15_sft139`) fine-tuned on ARC grid patterns.  
Loaded with 4-bit BitsAndBytes quantisation on Kaggle T4/P100.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Own approach vs NVARC TTT | No test-time training → faster, lower GPU memory, more time for program synthesis |
| Vedic DSA pre-filter | Reduces LLM token budget; handles structured pattern tasks (tiling, symmetry) deterministically |
| Program synthesis loop | Verifiable against training examples; more reliable than pure generation |
| Chakra multi-prompt | Diverse framings reduce mode collapse; majority vote improves reliability |
| Fallback to input grid | Ensures valid submission even for completely unsolved tasks |
