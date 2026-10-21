# arc-jeni-2026

[![CI](https://github.com/joyjeni/arc-agi2-hybrid-solver/actions/workflows/ci.yml/badge.svg)](https://github.com/joyjeni/arc-agi2-hybrid-solver/actions/workflows/ci.yml)
[![Benchmark](https://github.com/joyjeni/arc-agi2-hybrid-solver/actions/workflows/benchmark.yml/badge.svg)](https://github.com/joyjeni/arc-agi2-hybrid-solver/actions/workflows/benchmark.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**3-Tier Hybrid ARC-AGI-2 Solver**  
Competition: [ARC-Prize-2026 / ARC-AGI-2](https://kaggle.com/competitions/arc-prize-2026-arc-agi-2)  
Target LB: **35–45 %**

---

## Architecture

```
ARC-AGI-2 Test Tasks (240)
        │
  ┌─────▼──────────────┐
  │  Tier 1 — DSA      │  ~60 s   →  ~5–10 %   (deterministic rule solvers)
  └─────┬──────────────┘
        │ unsolved
  ┌─────▼────────────────────┐
  │  Tier 2 — Prog. Synth.   │  ~30 min →  ~15–25 %  (LLM writes Python)
  └─────┬────────────────────┘
        │ still unsolved
  ┌─────▼───────────────────────┐
  │  Tier 3 — Chakra Ensemble   │  rest    →  ~10–20 %  (8 prompts + majority vote)
  └─────┬───────────────────────┘
        │
  submission.json  (exact competition format)
```

See [docs/architecture.md](docs/architecture.md) for full detail.

---

## Approach Highlights

| Component | Design |
|-----------|--------|
| **Tier-1 DSA** | 18 deterministic solvers: ColorFrequency, GravityFall, ReflectH/V, Rotate, Transpose, CCRecolorBySize, RemoveSmallObjects, ColorSwap, RowSort + 7 Vedic/Chinese solvers (Anurupya scale, Paravartya symmetry, BaGua octagram, LoShu periodic fill …) |
| **Tier-2 Program Synthesis** | LLM (Qwen3-4B, fine-tuned on ARC grids) writes a `transform(grid)` function, executed in subprocess, validated against all training examples — retries up to 3× |
| **Tier-3 Chakra Ensemble** | 8 spiritually-framed prompt templates (Muladhara → Sahasrara) covering structure, color, symmetry, object relations, sequence, meta-reasoning; majority vote selects `attempt_1` |
| **Model** | `qwen3_4b_grids15_sft139` (fine-tuned on 15 000 ARC grid pairs, 139 steps SFT); loaded 4-bit BnB on Kaggle GPU |
| **No TTT** | Unlike NVARC (test-time training), this pipeline skips LoRA adaptation — saving GPU budget for program synthesis and broader ensemble coverage |

---

## Repository Structure

```
arc-jeni-2026/
├── configs/
│   └── pipeline.yaml          # all hyperparameters
├── docs/
│   └── architecture.md        # detailed architecture & design decisions
├── notebooks/
│   └── arc_jeni_submission.ipynb   # Kaggle submission notebook
├── scripts/
│   ├── benchmark.py           # local eval benchmark CLI
│   ├── validate_submission.py # submission.json format validator
│   └── create_kaggle_model.py # upload model to Kaggle Models
├── src/
│   ├── core/
│   │   ├── interfaces.py      # ABCs: BaseSolver, EnsembleSolver, Grid, Prediction …
│   │   ├── grid_codec.py      # encode/decode grids, transforms, histogram
│   │   ├── augmentation.py    # 8-view dihedral augmentation + color permutations
│   │   └── submission.py      # build / validate / write / score submission
│   ├── solvers/
│   │   ├── registry.py        # SolverRegistry singleton + @register_solver
│   │   ├── dsa_solvers.py     # 11 deterministic DSA solvers
│   │   └── vedic_solvers.py   # 7 Vedic / Chinese / BaGua solvers
│   ├── llm/
│   │   ├── engine.py          # QwenEngine, HF/Server/Mock backends
│   │   ├── prompts.py         # 8 Chakra prompt templates
│   │   ├── program_synth.py   # Tier-2 program synthesis solver
│   │   └── chakra_ensemble.py # Tier-3 ensemble with majority vote
│   └── pipeline.py            # full 3-tier pipeline orchestrator
└── tests/
    └── test_submission.py     # 9 unit tests (format, codec, augmentation, scoring)
```

---

## Quick Start

### Local setup

```bash
git clone https://github.com/joyjeni/arc-agi2-hybrid-solver.git
cd arc-jeni-2026
pip install -r requirements.txt
```

### Run unit tests

```bash
pytest tests/ -v
```

### Local benchmark (Tier-1 DSA only, no GPU needed)

```bash
python scripts/benchmark.py \
    --split eval \
    --tier 1 \
    --challenges /path/to/arc-agi_evaluation-challenges.json \
    --solutions  /path/to/arc-agi_evaluation-solutions.json
```

### Validate a submission file

```bash
python scripts/validate_submission.py --file submission.json --tasks 240
```

---

## Kaggle Submission

1. Open `notebooks/arc_jeni_submission.ipynb` on Kaggle.
2. Attach dataset: **arc-prize-2026-arc-agi-2**
3. Attach model: **joyjeni/qwen3-4b-grids15-sft139/transformers/sft139/1**
4. Enable GPU accelerator (T4 × 2 or P100)
5. Set Internet = OFF (competition rule)
6. Run all cells → `/kaggle/working/submission.json` is generated automatically.
7. Submit via *Submit Predictions*.

### Submission format (exact)

```json
{
  "task_id": [
    {"attempt_1": [[0,1,2],[3,4,5]], "attempt_2": [[0,1,2],[3,4,5]]}
  ]
}
```

One list entry per test input; `attempt_1` and `attempt_2` are independent grid guesses.

---

## CI / Benchmarks

GitHub Actions runs on every push:

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `ci.yml` | every push | ruff lint + pytest (CPU) |
| `benchmark.yml` | push to `main` | Tier-1 benchmark on eval split, upload `submission.json` artefact |

Set `KAGGLE_USERNAME` and `KAGGLE_KEY` in *Settings → Secrets* to enable data download in CI.

---

## Competition Context

| Metric | Value |
|--------|-------|
| Test tasks | 240 |
| Test inputs per task | 1–3 |
| Attempts per input | 2 |
| Scoring | exact-match grid; +1 if either attempt correct |
| Time limit | ~9 h GPU |
| Top competitor (NVARC, Oct 2026) | ~35.9 % LB (Qwen3-4B TTT + LoRA r=256 + 128-view DFS) |
| Community reports | Qwen3.8-27B ensemble reaches 45 %+; top private scores 55–83 % |

---

## License

MIT — see [LICENSE](LICENSE).
