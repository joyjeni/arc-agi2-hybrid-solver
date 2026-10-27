"""
Qwen3-4B Test-Time Training (TTT) Solver.

Pipeline position: Stage 3 — runs FIRST, highest LLM budget.

Per-task workflow
─────────────────
1. Build an augmented training set:
     - Original pairs               (1×)
     - Colour-permuted pairs        (N_COLOR_PERMS random palette shuffles)
     - Horizontally flipped pairs   (1×)
     - Vertically flipped pairs     (1×)
   Total: (1 + N_COLOR_PERMS + 2) × len(train_pairs) training examples.

2. Test-Time Training (TTT):
     - Snapshot trainable parameters (last transformer block + lm_head).
     - AdamW, lr=TTT_LR, TTT_STEPS gradient steps on the augmented pairs.
     - AMP fp16 when a CUDA device is available.

3. Inference:
     - attempt_1: greedy decoding (temperature → 0).
     - attempt_2: nucleus sampling (temperature=0.7, top_p=0.9).

4. Weight restore:
     Trainable parameter snapshot is restored so every task starts fresh
     from the base model checkpoint.

Stage-3 prediction cache
────────────────────────
After each successful solve the solver writes its best-guess grids into
``ctx.stage3_hints`` (list[Grid], one per test input). The pipeline runner
stores this on the TaskContext so Stage-2 solvers can use the hints as a
fall-back attempt_2 when their own pattern match has low confidence.

Graceful degradation
────────────────────
If ``transformers`` / PyTorch are not installed, or no model path is found,
the solver logs a warning and returns None (no crash). The pipeline
continues with Stage 2 and Stage 1 as usual.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid heavy imports at module load time
    pass

from ..core.interfaces import BaseSolver, Prediction, SolverResult, TaskContext
from ..core.grid_codec import grid_to_text, text_to_grid

logger = logging.getLogger(__name__)

# ── Hyper-parameters ──────────────────────────────────────────────────────────

TTT_STEPS       = 20          # gradient updates per task
TTT_LR          = 5e-5        # AdamW learning rate
N_COLOR_PERMS   = 2           # random colour-permutation augmentations
MAX_SEQ_LEN     = 1024        # tokenizer max length (truncate if needed)
MAX_NEW_TOKENS  = 512         # max tokens generated during inference
TEMP_ATTEMPT1   = 1e-5        # greedy (near-zero temperature)
TEMP_ATTEMPT2   = 0.70        # stochastic for attempt_2
TOP_P_ATTEMPT2  = 0.90

# ── Model paths (Kaggle model hub → HF fallback) ──────────────────────────────

KAGGLE_MODEL_PATHS = [
    "/kaggle/input/qwen3-4b/",
    "/kaggle/input/qwen3/transformers/4b/1/",
    "/kaggle/input/qwen3-4b-instruct/",
]
HF_MODEL_ID = "Qwen/Qwen3-4B"


# ── Singleton model cache ─────────────────────────────────────────────────────

class _ModelCache:
    """Load model + tokenizer exactly once; reuse across all tasks."""

    _model      = None
    _tokenizer  = None
    _device     = None
    _failed     = False  # set True on first load failure to skip retries

    @classmethod
    def get(cls):
        """Return (model, tokenizer, device) or (None, None, None) on failure."""
        if cls._failed:
            return None, None, None
        if cls._model is not None:
            return cls._model, cls._tokenizer, cls._device

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_path = _find_model_path()
            if model_path is None:
                logger.warning("Qwen3TTT: no model found — solver disabled")
                cls._failed = True
                return None, None, None

            logger.info("Qwen3TTT: loading model from %s …", model_path)
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype  = torch.float16 if device == "cuda" else torch.float32
            model  = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
            model.eval()
            if device == "cpu":
                model = model.to(device)

            cls._model, cls._tokenizer, cls._device = model, tokenizer, device
            logger.info("Qwen3TTT: model loaded on %s", device)
            return model, tokenizer, device

        except Exception as exc:
            logger.warning("Qwen3TTT: model load failed — %s", exc)
            cls._failed = True
            return None, None, None


def _find_model_path() -> str | None:
    """Return first existing Kaggle path, else HF model ID."""
    for p in KAGGLE_MODEL_PATHS:
        if os.path.isdir(p):
            return p
    # Check env override
    env = os.getenv("QWEN3_MODEL_PATH")
    if env:
        return env
    # Fall back to HF hub ID (will trigger download on first run)
    return HF_MODEL_ID


# ── Grid ↔ text helpers ───────────────────────────────────────────────────────

def _grid_to_str(grid) -> str:
    """Convert grid to newline-separated rows of space-separated integers."""
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def _str_to_grid(text: str):
    """Parse LLM output back to a Grid. Returns None on parse failure."""
    # Try JSON array format first
    m = re.search(r"\[\s*\[[\d,\s\[\]]+\]\s*\]", text)
    if m:
        try:
            g = json.loads(m.group())
            if isinstance(g, list) and g and all(isinstance(r, list) for r in g):
                return [[int(c) for c in row] for row in g]
        except Exception:
            pass

    # Try pipe-separated format from grid_codec
    g = text_to_grid(text.strip(), mode="numeric")
    if g:
        return g

    # Try plain newline / space separated rows
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    rows = []
    for line in lines:
        tokens = re.findall(r"\d+", line)
        if tokens:
            rows.append([int(t) for t in tokens])
    if rows and all(len(r) == len(rows[0]) for r in rows):
        return rows

    return None


# ── Augmentation helpers ──────────────────────────────────────────────────────

def _colour_permute(grid):
    """Randomly permute the colour palette (excluding background 0)."""
    colors = list({c for row in grid for c in row})
    shuffled = colors[:]
    random.shuffle(shuffled)
    mapping = dict(zip(colors, shuffled))
    return [[mapping[c] for c in row] for row in grid]


def _flip_h(grid):
    return [list(reversed(row)) for row in grid]


def _flip_v(grid):
    return list(reversed(grid))


def _augment_pairs(pairs, n_color_perms: int = N_COLOR_PERMS):
    """
    Return augmented list of training pairs:
      original + n_color_perms colour permutations + h-flip + v-flip
    """
    aug = list(pairs)  # originals

    for _ in range(n_color_perms):
        for p in pairs:
            aug.append({
                "input":  _colour_permute(p["input"]),
                "output": _colour_permute(p["output"]),
            })

    for p in pairs:
        aug.append({"input": _flip_h(p["input"]), "output": _flip_h(p["output"])})
        aug.append({"input": _flip_v(p["input"]), "output": _flip_v(p["output"])})

    return aug


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(train_pairs, test_input) -> str:
    """Format ARC task as a text prompt for Qwen3-4B."""
    lines = [
        "You are solving an ARC (Abstraction and Reasoning Corpus) puzzle.",
        "Study the input→output examples and apply the same rule to the Test input.",
        "Reply with ONLY the output grid: rows of space-separated integers, one row per line.",
        "",
    ]
    for i, pair in enumerate(train_pairs, 1):
        lines.append(f"Example {i} Input:")
        lines.append(_grid_to_str(pair["input"]))
        lines.append(f"Example {i} Output:")
        lines.append(_grid_to_str(pair["output"]))
        lines.append("")

    lines.append("Test Input:")
    lines.append(_grid_to_str(test_input))
    lines.append("Test Output:")
    return "\n".join(lines)


# ── TTT core ──────────────────────────────────────────────────────────────────

def _trainable_layers(model):
    """
    Identify last transformer block + lm_head parameter names.
    Works for Qwen2/Qwen3 architectures; falls back to lm_head only.
    """
    names = []
    try:
        # Qwen3: model.model.layers is the transformer stack
        n_layers = len(model.model.layers)
        last_block = str(n_layers - 1)
        for name, _ in model.named_parameters():
            if f"layers.{last_block}." in name or "lm_head" in name:
                names.append(name)
    except AttributeError:
        for name, _ in model.named_parameters():
            if "lm_head" in name:
                names.append(name)
    return names


def _snapshot_trainable(model, trainable_names: list[str]) -> dict:
    """Save a CPU copy of trainable parameters."""
    sd = model.state_dict()
    return {k: sd[k].clone().cpu() for k in trainable_names}


def _restore_trainable(model, snapshot: dict) -> None:
    """Restore trainable parameters from snapshot (in-place)."""
    sd = model.state_dict()
    for k, v in snapshot.items():
        sd[k].copy_(v.to(sd[k].device))


def _run_ttt(model, tokenizer, aug_pairs, device):
    """
    Fine-tune the model on augmented training pairs for TTT_STEPS steps.
    Uses AdamW on trainable layers only.  AMP fp16 on CUDA.
    """
    import torch
    from torch.optim import AdamW

    trainable_names = _trainable_layers(model)

    # Freeze all params, then unfreeze trainable subset
    for p in model.parameters():
        p.requires_grad = False
    sd = {k: v for k, v in model.named_parameters()}
    for name in trainable_names:
        if name in sd:
            sd[name].requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        logger.warning("Qwen3TTT: no trainable params found — skipping TTT")
        return

    optimizer = AdamW(trainable_params, lr=TTT_LR)
    use_amp = (device == "cuda")

    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    model.train()

    for step in range(TTT_STEPS):
        pair = aug_pairs[step % len(aug_pairs)]
        prompt = _build_prompt([pair], [0])  # dummy test; we only need loss
        # Format as "input → output" training example
        full_text = (
            _build_prompt([], pair["input"]).split("Test Input:\n", 1)[-1].strip()
            + "\nTest Output:\n"
            + _grid_to_str(pair["output"])
        )
        enc = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=False,
        )
        input_ids = enc["input_ids"].to(device)

        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                out = model(input_ids=input_ids, labels=input_ids)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(input_ids=input_ids, labels=input_ids)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

    model.eval()
    # Re-enable gradients for all params to avoid side effects
    for p in model.parameters():
        p.requires_grad = False


# ── Inference helper ──────────────────────────────────────────────────────────

def _infer_grid(model, tokenizer, train_pairs, test_input, device, temperature: float):
    """Run one forward pass and parse the output as a Grid."""
    import torch

    prompt = _build_prompt(train_pairs, test_input)
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    input_ids = enc["input_ids"].to(device)

    do_sample = temperature > 0.01
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = TOP_P_ATTEMPT2

    with torch.no_grad():
        out = model.generate(input_ids=input_ids, **gen_kwargs)

    gen_ids = out[0][input_ids.shape[1]:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return _str_to_grid(raw)


# ── Main Solver class ─────────────────────────────────────────────────────────

class Qwen3TTTSolver(BaseSolver):
    """
    Qwen3-4B Test-Time Training solver.

    Runs FIRST in the new Stage-3 → Stage-2 → Stage-1 pipeline.
    Implements the full TTT loop per task and returns two attempts.
    """

    @property
    def name(self) -> str:
        return "Qwen3TTTSolver"

    @property
    def priority(self) -> int:
        return -10  # highest priority: runs before any registered solver

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        model, tokenizer, device = _ModelCache.get()
        if model is None:
            return None  # graceful no-op when model unavailable

        train_pairs = ctx.train_pairs
        if not train_pairs:
            return None

        # 1. Build augmented training set
        aug_pairs = _augment_pairs(train_pairs, n_color_perms=N_COLOR_PERMS)

        # 2. Snapshot trainable params before TTT
        trainable_names = _trainable_layers(model)
        snapshot = _snapshot_trainable(model, trainable_names)

        # 3. TTT
        try:
            _run_ttt(model, tokenizer, aug_pairs, device)
        except Exception as exc:
            logger.warning("Qwen3TTT %s: TTT failed — %s", ctx.task_id, exc)
            _restore_trainable(model, snapshot)
            return None

        # 4. Inference for every test input
        predictions: list[Prediction] = []
        hints: list = []
        solved_any = False

        for test_input in ctx.test_inputs:
            # attempt_1: greedy
            grid1 = None
            try:
                grid1 = _infer_grid(
                    model, tokenizer, train_pairs, test_input, device, TEMP_ATTEMPT1
                )
            except Exception as exc:
                logger.debug("Qwen3TTT %s: attempt_1 failed — %s", ctx.task_id, exc)

            # attempt_2: nucleus sampling
            grid2 = None
            try:
                grid2 = _infer_grid(
                    model, tokenizer, train_pairs, test_input, device, TEMP_ATTEMPT2
                )
            except Exception as exc:
                logger.debug("Qwen3TTT %s: attempt_2 failed — %s", ctx.task_id, exc)

            # Fall back to input if nothing parsed
            fallback = test_input
            a1 = grid1 if grid1 is not None else fallback
            a2 = grid2 if grid2 is not None else a1

            predictions.append(Prediction(
                attempt_1=a1,
                attempt_2=a2,
                solver_name=self.name,
                confidence=0.6 if grid1 is not None else 0.1,
            ))
            hints.append(a1)
            if grid1 is not None:
                solved_any = True

        # 5. Restore model weights
        _restore_trainable(model, snapshot)

        # 6. Store hints in context so Stage 2 can use them as fallback attempt_2
        ctx.stage3_hints = hints

        if not solved_any:
            return None

        return SolverResult(
            task_id=ctx.task_id,
            predictions=predictions,
            solved=True,
            tier=3,
        )
