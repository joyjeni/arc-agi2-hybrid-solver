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
| Evaluation, 120 tasks | 8 (6 tasks) | 0 | 36 s |
| Training, 1000 tasks | 339 | 0 | 138 s |

Sources, in the order they are tried:

- this repo's 399 solvers
- `Arc2SymbolicSolver.solve_verified_geometry` (third-party, Apache-2.0, see `src/addon/third_party/arc2_symbolic.py`)
- a generic program search
- general rule families in `src/addon/families.py`: symmetry completion (mirrors, rotations, translations), periodic completion (including staggered and diagonal repeats), object selection, grid partition (pick a cell, or a boolean overlay of cells), motif recolouring, a local 3x3 neighbourhood rule, reflection across a marker line or point, rays from seed cells, alignment to a reference marker, and making shapes symmetric by deleting unmatched cells

Families that fit their parameters by choosing among several rules answer a test input only when **every** fitted rule that answers it agrees; otherwise they abstain on that input. A source may answer some test inputs and abstain on others.

Development used the training tasks and a fixed half of the evaluation tasks (by task-id hash); the other half was never inspected. All 6 solved evaluation tasks are in the development half — **0 of the held-out half is solved**, so there is no evidence yet that these rules transfer to unseen tasks.

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
