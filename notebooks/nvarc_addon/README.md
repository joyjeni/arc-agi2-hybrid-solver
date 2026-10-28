# Verified symbolic add-on for an NVARC fork

Two cells that add answers from this repo's solvers to a public NVARC notebook's
`submission.json` — only when those answers are proven on the task's own examples.

## Why it is safe

An answer is used only if the rule that produced it

1. reproduces **every** training output exactly, and
2. passes **leave-one-out**: re-fitted without one training pair, it still predicts
   that pair exactly — for every pair.

Merge rule: if the verified answer differs from both NVARC attempts, it replaces
`attempt_2`; NVARC's top pick stays in `attempt_1`. Any error leaves `submission.json`
untouched.

## Measured (public data, `src/addon/verified.py`)

| Split | Correct outputs | Wrong outputs | Time (1 CPU core) |
|---|---|---|---|
| Evaluation, 120 tasks (unseen) | 3 (2 tasks) | 0 | 22 s |
| Training, 1000 tasks | 330 | 0 | 95 s |

Sources: this repo's 399 solvers (326 train / 0 eval), generic program search
(3 / 0), and `Arc2SymbolicSolver.solve_verified_geometry` (1 / 3; third-party,
Apache-2.0, see `src/addon/third_party/arc2_symbolic.py`).

**Expected effect:** small — roughly 0–2 points on top of NVARC, depending on how many
hidden tasks the rules fit that NVARC misses. The model does nearly all the solving.

## How to use it

1. On Kaggle, open **"ARC2 NVARC Full-Cost SOTA Triple Pass"** (woguoat) → **Copy & Edit**.
2. Insert a code cell **directly after the first cell** (the one that sets `RERUN`, `T0`,
   `global_end_time`) and paste all of `cell_A_start.py`.
3. Insert a code cell **directly after the cell that writes `submission.json`** (it ends with
   `print("*** Reload score:", ...)`) and paste all of `cell_B_merge.py`.
4. Settings: accelerator **GPU L4 ×4**, Internet **off**. **Save Version → Save & Run All.**
5. In the commit log, check for `[jeni] verifier started …` and
   `[jeni] eval outputs correct after merge: …`.
6. **Submit Prediction** with that version.

Rebuild the cells after changing any solver:

```bash
python scripts/build_nvarc_addon.py
```
