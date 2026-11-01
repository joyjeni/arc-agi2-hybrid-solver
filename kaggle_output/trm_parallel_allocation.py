#!/usr/bin/env python3
"""
trm_parallel_allocation.py  —  Dual-model parallel GPU worker for ARC-AGI-2
=============================================================================
Session 61 — VRAM co-allocation edition.

Architecture (per L4 GPU):
  SFT  model  (sorokin/qwen3_4b_grids15_sft139, bf16, ~8 GB)
    → Strategies 1-4 (grid-formatting inference)
    → D4 spatial-transform retry
  Coder model (Qwen2.5-Coder-7B-Instruct, bf16/8-bit, ~7.5-14.5 GB)
    → Strategy 5 TRM (Python program synthesis via Duck Harness)
    → Fallback: uses model_sft when coder unavailable

VRAM budget (L4, 24 GB, 0.90 fraction → 21.6 GB usable):
  Coder bf16  (~14.5 GB) + SFT bf16 (~8 GB)  = ~22.5 GB  — tight / may OOM
  Coder 8-bit (~7.5 GB)  + SFT bf16 (~8 GB)  = ~15.5 GB  — safe
  Mitigation: try bf16 → catch OOM → retry 8-bit → fallback coder=None

Worker output: {output_dir}/preds_rank_{rank}.json
Final output : output_path (JSON), format same as trm_orchestrator.py
  {task_id: [{attempt_1: [[int,...]], attempt_2: [[int,...]]}, ...]}

Self-contained at spawn time: all CUDA-aware imports are inside
run_isolated_gpu_worker() so CUDA_VISIBLE_DEVICES is honoured correctly.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch.multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

_CODER_FALLBACK_PATHS: list = [
    "/kaggle/input/qwen2.5-coder-7b-instruct/transformers/default/1",
    "/kaggle/input/models/qwen/qwen2.5-coder-7b-instruct/transformers/default/1",
]

_SFT_FALLBACK_PATHS: list = [
    "/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
    "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
    "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
]


def discover_competition_models() -> Tuple[str, str]:
    """
    Scan /kaggle/input for both the Coder and SFT model directories.

    Returns:
        (coder_path, sft_path)  — either or both may be '' if not found.

    Priority per model:
      1. Glob scan /kaggle/input rglob("config.json") — prefer exact name matches.
      2. Hardcoded fallback paths.
    """
    coder_path: str = ""
    sft_path:   str = ""

    search_root = Path("/kaggle/input")
    if search_root.exists():
        for cfg_file in sorted(search_root.rglob("config.json")):
            parent_str = str(cfg_file.parent).lower()
            if not coder_path and (
                "coder" in parent_str or "qwen2.5-coder" in parent_str
            ):
                try:
                    cfg = json.loads(cfg_file.read_text())
                    if "model_type" in cfg or "vocab_size" in cfg:
                        coder_path = str(cfg_file.parent)
                        print(f"[discovery] Coder model found: {coder_path}", flush=True)
                except Exception:
                    pass
            if not sft_path and (
                "grids15" in parent_str or "sorokin" in parent_str
            ):
                try:
                    cfg = json.loads(cfg_file.read_text())
                    if "model_type" in cfg or "vocab_size" in cfg:
                        sft_path = str(cfg_file.parent)
                        print(f"[discovery] SFT model found:   {sft_path}", flush=True)
                except Exception:
                    pass
            if coder_path and sft_path:
                break

    if not coder_path:
        for p in _CODER_FALLBACK_PATHS:
            if Path(p).is_dir() and Path(p, "config.json").exists():
                coder_path = p
                print(f"[discovery] Coder model (fallback): {coder_path}", flush=True)
                break

    if not sft_path:
        for p in _SFT_FALLBACK_PATHS:
            if Path(p).is_dir() and Path(p, "config.json").exists():
                sft_path = p
                print(f"[discovery] SFT model (fallback):   {sft_path}", flush=True)
                break

    if not coder_path:
        print("[discovery] WARNING: Coder model path not found.", flush=True)
    if not sft_path:
        print("[discovery] WARNING: SFT model path not found.", flush=True)

    return coder_path, sft_path


# ══════════════════════════════════════════════════════════════════════════════
# DUAL-MODEL GPU WORKER
# ══════════════════════════════════════════════════════════════════════════════

def run_isolated_gpu_worker(
    rank: int,
    task_chunk: list,
    coder_path: str,
    sft_path: str,
    output_dir: str,
    time_per_task: int = 60,
) -> None:
    """
    Isolated subprocess running exclusively on CUDA device `rank`.

    Loads two models on the same GPU:
      model_sft   — sorokin/qwen3_4b_grids15_sft139  (bf16, ~8 GB)
      model_coder — Qwen2.5-Coder-7B-Instruct        (bf16 → 8-bit fallback)

    5-strategy pipeline per panel:
      S1 — QwenFormatter SFT greedy          (model_sft)  → attempt_1
      S2 — QwenFormatter SFT sampled T=0.8   (model_sft)  → attempt_2
      S3 — compact T=0.4 fallback            (model_sft)
      S4 — colored T=0.4 second fallback     (model_sft)
      D4   spatial-transform retry            (model_sft)
      S5 — TRM Duck Harness code generation   (model_coder or model_sft)

    Args:
        rank:         CUDA device index for this worker.
        task_chunk:   List of (task_id, task_data) tuples.
        coder_path:   Path to Qwen2.5-Coder-7B model dir (may be '').
        sft_path:     Path to sorokin/qwen3_4b model dir.
        output_dir:   Directory for incremental JSON saves.
        time_per_task: Soft per-task time budget in seconds.
    """
    # ── CUDA isolation (must happen before any CUDA import) ──────────────────
    os.environ["CUDA_VISIBLE_DEVICES"]   = str(rank)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"]         = "1"
    os.environ["TRANSFORMERS_OFFLINE"]   = "1"

    # ── Spawn-safe heavy imports ─────────────────────────────────────────────
    import gc as _gc                                          # noqa: PLC0415
    import numpy as np                                        # noqa: PLC0415
    import torch                                              # noqa: PLC0415
    from transformers import (                                # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        LogitsProcessorList,
    )

    # ── Limit VRAM to 90% of device capacity ─────────────────────────────────
    torch.cuda.set_per_process_memory_fraction(0.90, 0)

    # ── Import Duck Harness components from trm_orchestrator ─────────────────
    _here = Path(__file__).parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    try:
        from trm_orchestrator import (                        # noqa: PLC0415
            ARCDSL,
            PythonREPL,
            REPL_GLOBALS,
            TimeoutException,
            _timeout_handler,
            check_demo_success,
            _extract_world_model,
            _TRM_MAX_INPUT_TOKENS,
            _TRM_DEPTH_BUDGET_SEC,
            _PANEL_BUDGET_SEC,
            recursive_trm_loop,
            make_prompt,
            parse_grid,
            _apply_chat,
            _transform_task_view,
            _is_valid_result,
            _TRANSFORM_OPS,
            grid_to_digits,
            grid_to_compact,
            grid_to_colored,
            convert_grid_to_string,
            _pixel_majority_vote,
            _demo_consistency_score,
            ARCLogitsProcessor,
        )
        print(f"[GPU {rank}] Imported Duck Harness from trm_orchestrator.", flush=True)
    except ImportError as _ie:
        print(
            f"[GPU {rank}] WARNING: trm_orchestrator import failed ({_ie}). "
            "TRM strategy (S5) will be unavailable.",
            flush=True,
        )
        # Minimal stubs so the file still runs without trm_orchestrator
        _PANEL_BUDGET_SEC = 180
        _TRM_MAX_INPUT_TOKENS = 28_000
        _TRM_DEPTH_BUDGET_SEC = 90.0
        TimeoutException = Exception

        def _timeout_handler(s, f):  # type: ignore[override]
            raise TimeoutException("alarm")

        def check_demo_success(*a, **kw):  # type: ignore[misc]
            return False, "stub"

        def _extract_world_model(t):  # type: ignore[misc]
            return None

        def recursive_trm_loop(*a, **kw):  # type: ignore[misc]
            return {"attempt_1": None, "attempt_2": None}

        def make_prompt(task_view, style):  # type: ignore[misc]
            msgs = [
                {"role": "system", "content": "You are an ARC puzzle solver."},
                {"role": "user",   "content": str(task_view)},
            ]
            return "", msgs

        def parse_grid(text, ref_shape=None):  # type: ignore[misc]
            import re as _re
            rows = _re.findall(r"\d+(?:\s+\d+)+", text)
            grid = [[int(x) for x in r.split()] for r in rows]
            return grid if grid else None

        def _apply_chat(tokenizer, messages, enable_thinking=False):  # type: ignore[misc]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return tokenizer(text, return_tensors="pt").input_ids

        def _transform_task_view(tv, fn):  # type: ignore[misc]
            return tv

        def _is_valid_result(grid, inp):  # type: ignore[misc]
            return (
                isinstance(grid, list)
                and len(grid) > 0
                and isinstance(grid[0], list)
            )

        _TRANSFORM_OPS = []

        def grid_to_digits(g):  # type: ignore[misc]
            return str(g)

        def grid_to_compact(g):  # type: ignore[misc]
            return str(g)

        def grid_to_colored(g):  # type: ignore[misc]
            return str(g)

        def convert_grid_to_string(g):  # type: ignore[misc]
            return str(g)

    # ── Load SFT model (primary, always required) ─────────────────────────────
    print(f"[GPU {rank}] Loading SFT model: {sft_path}", flush=True)
    tokenizer_sft = AutoTokenizer.from_pretrained(sft_path, trust_remote_code=True)
    model_sft = AutoModelForCausalLM.from_pretrained(
        sft_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model_sft.eval()
    print(
        f"[GPU {rank}] SFT model ready. "
        f"VRAM used: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB",
        flush=True,
    )

    # ── Load Coder model (secondary, VRAM-constrained) ────────────────────────
    model_coder = None
    tokenizer_coder = None

    if coder_path:
        print(f"[GPU {rank}] Attempting Coder model bf16: {coder_path}", flush=True)
        try:
            tokenizer_coder = AutoTokenizer.from_pretrained(
                coder_path, trust_remote_code=True
            )
            model_coder = AutoModelForCausalLM.from_pretrained(
                coder_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0",
                trust_remote_code=True,
            )
            model_coder.eval()
            print(
                f"[GPU {rank}] Coder model bf16 ready. "
                f"VRAM used: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB",
                flush=True,
            )
        except torch.cuda.OutOfMemoryError:
            print(
                f"[GPU {rank}] Coder bf16 OOM — retrying with 8-bit quantisation.",
                flush=True,
            )
            _gc.collect()
            torch.cuda.empty_cache()
            try:
                _bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
                model_coder = AutoModelForCausalLM.from_pretrained(
                    coder_path,
                    quantization_config=_bnb_cfg,
                    device_map="cuda:0",
                    trust_remote_code=True,
                )
                model_coder.eval()
                print(
                    f"[GPU {rank}] Coder model 8-bit ready. "
                    f"VRAM used: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB",
                    flush=True,
                )
            except (torch.cuda.OutOfMemoryError, Exception) as _oom2:
                print(
                    f"[GPU {rank}] Coder 8-bit also failed ({_oom2}). "
                    "TRM will use SFT fallback.",
                    flush=True,
                )
                model_coder    = None
                tokenizer_coder = None
                _gc.collect()
                torch.cuda.empty_cache()
        except Exception as _coder_exc:
            print(
                f"[GPU {rank}] Coder load error: {_coder_exc}. "
                "TRM will use SFT fallback.",
                flush=True,
            )
            model_coder    = None
            tokenizer_coder = None
    else:
        print(f"[GPU {rank}] No coder_path supplied — TRM will use SFT.", flush=True)

    # Resolve effective TRM model/tokenizer
    _trm_model     = model_coder    if model_coder    is not None else model_sft
    _trm_tokenizer = tokenizer_coder if tokenizer_coder is not None else tokenizer_sft
    _trm_label     = "Coder" if model_coder is not None else "SFT-fallback"
    print(f"[GPU {rank}] TRM engine: {_trm_label}", flush=True)

    # ── ARCLogitsProcessor: compute allowed token IDs and build logits mask ──
    try:
        _arc_ids_sft: list = []
        for _ch in "0123456789":
            for _tok in tokenizer_sft.encode(_ch, add_special_tokens=False):
                _arc_ids_sft.append(_tok)
        for _extra in ("\n", " "):
            for _tok in tokenizer_sft.encode(_extra, add_special_tokens=False):
                _arc_ids_sft.append(_tok)
        if tokenizer_sft.eos_token_id is not None:
            _arc_ids_sft.append(tokenizer_sft.eos_token_id)
        _arc_ids_sft = list(set(_arc_ids_sft))
        _arc_lp_list = LogitsProcessorList([ARCLogitsProcessor(_arc_ids_sft)])
        print(
            f"[GPU {rank}] ARCLogitsProcessor ready ({len(_arc_ids_sft)} allowed tokens).",
            flush=True,
        )
    except Exception as _lp_exc:
        _arc_lp_list = LogitsProcessorList([])
        print(
            f"[GPU {rank}] ARCLogitsProcessor disabled ({_lp_exc}) — no token masking.",
            flush=True,
        )

    # ── Per-panel inference loop ──────────────────────────────────────────────
    signal.signal(signal.SIGALRM, _timeout_handler)

    worker_preds: dict = {}
    out_path = Path(output_dir) / f"preds_rank_{rank}.json"
    t0 = time.time()

    for step, (task_id, task_data) in enumerate(task_chunk):
        # Memory hygiene before each task
        _gc.collect()
        torch.cuda.empty_cache()

        test_panels = task_data.get("test", [])
        if not test_panels:
            worker_preds[task_id] = []
            continue

        task_preds: list = []

        for panel in test_panels:
            inp       = panel.get("input", [[0]])
            ref_shape = (
                (len(inp), len(inp[0])) if inp and inp[0] else None
            )
            task_view    = {"train": task_data.get("train", []), "test": [panel]}
            _panel_start = time.time()

            attempt_1: Optional[list] = None
            attempt_2: Optional[list] = None
            _candidates: list = []  # collects valid grids for majority vote

            # ── S1: QwenFormatter SFT greedy → attempt_1 ────────────────────
            try:
                try:
                    from arc_loader import QwenFormatter as _QF1  # noqa: PLC0415
                except ImportError:
                    _QF1 = None  # type: ignore[assignment]

                if _QF1 is not None:
                    _qf1  = _QF1(tokenizer_sft)
                    _p1   = _qf1.fmt_train(task_view["train"]) + _qf1.fmt_query([panel])
                    _ids1 = tokenizer_sft(_p1, return_tensors="pt").input_ids.to("cuda:0")
                else:
                    _, _msgs1 = make_prompt(task_view, "nvarc")
                    _ids1 = _apply_chat(
                        tokenizer_sft, _msgs1, enable_thinking=False
                    ).to("cuda:0")

                with torch.no_grad():
                    _out1 = model_sft.generate(
                        _ids1,
                        max_new_tokens=32 * 31,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=tokenizer_sft.eos_token_id,
                        logits_processor=_arc_lp_list,
                    )
                _gen1     = tokenizer_sft.decode(
                    _out1[0][_ids1.shape[-1]:], skip_special_tokens=True
                )
                attempt_1 = parse_grid(_gen1, ref_shape)
                if _is_valid_result(attempt_1, inp):
                    _candidates.append(attempt_1)
            except Exception as exc:
                print(f"[GPU {rank}] {task_id} S1 err: {exc}", flush=True)

            # ── S2: QwenFormatter SFT sampled T=0.8 → attempt_2 ─────────────
            try:
                try:
                    from arc_loader import QwenFormatter as _QF2  # noqa: PLC0415
                except ImportError:
                    _QF2 = None  # type: ignore[assignment]

                if _QF2 is not None:
                    _qf2  = _QF2(tokenizer_sft)
                    _p2   = _qf2.fmt_train(task_view["train"]) + _qf2.fmt_query([panel])
                    _ids2 = tokenizer_sft(_p2, return_tensors="pt").input_ids.to("cuda:0")
                else:
                    _, _msgs2 = make_prompt(task_view, "nvarc")
                    _ids2 = _apply_chat(
                        tokenizer_sft, _msgs2, enable_thinking=False
                    ).to("cuda:0")

                with torch.no_grad():
                    _out2 = model_sft.generate(
                        _ids2,
                        max_new_tokens=32 * 31,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        pad_token_id=tokenizer_sft.eos_token_id,
                        logits_processor=_arc_lp_list,
                    )
                _gen2     = tokenizer_sft.decode(
                    _out2[0][_ids2.shape[-1]:], skip_special_tokens=True
                )
                attempt_2 = parse_grid(_gen2, ref_shape)
                if _is_valid_result(attempt_2, inp):
                    _candidates.append(attempt_2)
            except Exception as exc:
                print(f"[GPU {rank}] {task_id} S2 err: {exc}", flush=True)

            # ── S3: compact T=0.4 fallback (SFT) ────────────────────────────
            if attempt_1 is None or attempt_2 is None:
                try:
                    _sys3, _usr3 = make_prompt(task_view, "compact")
                    _msgs3 = [
                        {"role": "system", "content": _sys3},
                        {"role": "user",   "content": _usr3},
                    ]
                    _ids3 = _apply_chat(
                        tokenizer_sft, _msgs3, enable_thinking=False
                    ).to("cuda:0")
                    with torch.no_grad():
                        _out3 = model_sft.generate(
                            _ids3,
                            max_new_tokens=32 * 31,
                            do_sample=True,
                            temperature=0.4,
                            top_p=0.9,
                            pad_token_id=tokenizer_sft.eos_token_id,
                            logits_processor=_arc_lp_list,
                        )
                    _gen3 = tokenizer_sft.decode(
                        _out3[0][_ids3.shape[-1]:], skip_special_tokens=True
                    )
                    _fb3 = parse_grid(_gen3, ref_shape)
                    if _is_valid_result(_fb3, inp):
                        _candidates.append(_fb3)
                    if attempt_1 is None:
                        attempt_1 = _fb3
                    if attempt_2 is None:
                        attempt_2 = _fb3
                except Exception as exc:
                    print(f"[GPU {rank}] {task_id} S3 err: {exc}", flush=True)

            # ── S4: colored T=0.4 second fallback (SFT) ─────────────────────
            if attempt_1 is None or attempt_2 is None:
                try:
                    _sys4, _usr4 = make_prompt(task_view, "colored")
                    _msgs4 = [
                        {"role": "system", "content": _sys4},
                        {"role": "user",   "content": _usr4},
                    ]
                    _ids4 = _apply_chat(
                        tokenizer_sft, _msgs4, enable_thinking=False
                    ).to("cuda:0")
                    with torch.no_grad():
                        _out4 = model_sft.generate(
                            _ids4,
                            max_new_tokens=32 * 31,
                            do_sample=True,
                            temperature=0.4,
                            top_p=0.9,
                            pad_token_id=tokenizer_sft.eos_token_id,
                            logits_processor=_arc_lp_list,
                        )
                    _gen4 = tokenizer_sft.decode(
                        _out4[0][_ids4.shape[-1]:], skip_special_tokens=True
                    )
                    _fb4 = parse_grid(_gen4, ref_shape)
                    if _is_valid_result(_fb4, inp):
                        _candidates.append(_fb4)
                    if attempt_1 is None:
                        attempt_1 = _fb4
                    if attempt_2 is None:
                        attempt_2 = _fb4
                except Exception as exc:
                    print(f"[GPU {rank}] {task_id} S4 err: {exc}", flush=True)

            # ── D4 spatial-transform retry (SFT) ────────────────────────────
            if (
                not _is_valid_result(attempt_1, inp)
                or not _is_valid_result(attempt_2, inp)
            ):
                for _tname, _fwd, _inv in _TRANSFORM_OPS:
                    if (
                        _is_valid_result(attempt_1, inp)
                        and _is_valid_result(attempt_2, inp)
                    ):
                        break
                    try:
                        _tv_aug   = _transform_task_view(task_view, _fwd)
                        _, _aug_msgs = make_prompt(_tv_aug, "nvarc")
                        _aug_ids  = _apply_chat(
                            tokenizer_sft, _aug_msgs, enable_thinking=False
                        ).to("cuda:0")
                        with torch.no_grad():
                            _aug_out = model_sft.generate(
                                _aug_ids,
                                max_new_tokens=32 * 31,
                                do_sample=False,
                                temperature=None,
                                top_p=None,
                                pad_token_id=tokenizer_sft.eos_token_id,
                                logits_processor=_arc_lp_list,
                            )
                        _aug_gen  = tokenizer_sft.decode(
                            _aug_out[0][_aug_ids.shape[-1]:], skip_special_tokens=True
                        )
                        _aug_grid = parse_grid(_aug_gen, ref_shape)
                        if _aug_grid is not None:
                            _restored = _inv(_aug_grid)
                            if _is_valid_result(_restored, inp):
                                _candidates.append(_restored)
                            if not _is_valid_result(attempt_1, inp):
                                attempt_1 = _restored
                            elif not _is_valid_result(attempt_2, inp):
                                attempt_2 = _restored
                    except Exception as _tex:
                        print(
                            f"[GPU {rank}] {task_id} D4/{_tname} err: {_tex}",
                            flush=True,
                        )

            # ── Majority vote + demo consistency (TbV) selection ─────────────
            if len(_candidates) >= 2:
                _mv = _pixel_majority_vote(_candidates)
                if _mv is not None and _is_valid_result(_mv, inp):
                    _train_pairs = task_data.get("train", [])
                    _scored = sorted(
                        [c for c in _candidates if c is not None],
                        key=lambda g: _demo_consistency_score(g, _train_pairs),
                        reverse=True,
                    )
                    _best_consistent = _scored[0] if _scored else _mv
                    attempt_1 = _mv
                    if _is_valid_result(_best_consistent, inp) and _best_consistent != _mv:
                        attempt_2 = _best_consistent
                    print(
                        f"[GPU {rank}] {task_id} MV({len(_candidates)} cands) "
                        f"a1_ok={_is_valid_result(attempt_1, inp)} "
                        f"a2_ok={_is_valid_result(attempt_2, inp)}",
                        flush=True,
                    )

            # ── S5: TRM Duck Harness (Coder or SFT fallback) ─────────────────
            _elapsed     = time.time() - _panel_start
            _budget_left = _PANEL_BUDGET_SEC - int(_elapsed)
            if (
                (
                    not _is_valid_result(attempt_1, inp)
                    or not _is_valid_result(attempt_2, inp)
                )
                and _budget_left > 30
            ):
                signal.alarm(max(30, min(_budget_left, time_per_task + 10)))
                try:
                    _trm_data = {
                        "train": task_data.get("train", []),
                        "test":  panel,
                    }
                    _trm_res = recursive_trm_loop(
                        task_id,
                        _trm_data,
                        _trm_model,
                        _trm_tokenizer,
                        time_allowance=_budget_left,
                        max_recursion_depth=4,
                    )
                    if not _is_valid_result(attempt_1, inp):
                        _c1 = _trm_res.get("attempt_1")
                        if _is_valid_result(_c1, inp):
                            attempt_1 = _c1
                            print(
                                f"[GPU {rank}] {task_id} TRM({_trm_label}) -> attempt_1 OK",
                                flush=True,
                            )
                    if not _is_valid_result(attempt_2, inp):
                        _c2 = _trm_res.get("attempt_2")
                        if _is_valid_result(_c2, inp):
                            attempt_2 = _c2
                            print(
                                f"[GPU {rank}] {task_id} TRM({_trm_label}) -> attempt_2 OK",
                                flush=True,
                            )
                except (TimeoutException, Exception) as _trm_exc:
                    print(
                        f"[GPU {rank}] {task_id} TRM err "
                        f"({type(_trm_exc).__name__}): {_trm_exc}",
                        flush=True,
                    )
                finally:
                    signal.alarm(0)

            # ── Diversity guard: force attempt_2 ≠ attempt_1 ─────────────────
            if (
                _is_valid_result(attempt_1, inp)
                and _is_valid_result(attempt_2, inp)
                and attempt_1 == attempt_2
            ):
                try:
                    _sys_dv, _usr_dv = make_prompt(task_view, "colored")
                    _msgs_dv = [
                        {"role": "system", "content": _sys_dv},
                        {"role": "user",   "content": _usr_dv},
                    ]
                    _ids_dv = _apply_chat(
                        tokenizer_sft, _msgs_dv, enable_thinking=False
                    ).to("cuda:0")
                    with torch.no_grad():
                        _out_dv = model_sft.generate(
                            _ids_dv,
                            max_new_tokens=32 * 31,
                            do_sample=True,
                            temperature=1.2,
                            top_p=0.95,
                            pad_token_id=tokenizer_sft.eos_token_id,
                        )
                    _gen_dv = tokenizer_sft.decode(
                        _out_dv[0][_ids_dv.shape[-1]:], skip_special_tokens=True
                    )
                    _dv_grid = parse_grid(_gen_dv, ref_shape)
                    if _is_valid_result(_dv_grid, inp) and _dv_grid != attempt_1:
                        attempt_2 = _dv_grid
                        print(
                            f"[GPU {rank}] {task_id} diversity guard → attempt_2 diversified",
                            flush=True,
                        )
                except Exception as _dv_exc:
                    print(
                        f"[GPU {rank}] {task_id} diversity guard err: {_dv_exc}",
                        flush=True,
                    )

            # ── Hard fallback: echo test input ───────────────────────────────
            if attempt_1 is None:
                attempt_1 = inp
            if attempt_2 is None:
                attempt_2 = attempt_1

            task_preds.append({"attempt_1": attempt_1, "attempt_2": attempt_2})

        worker_preds[task_id] = task_preds

        # Incremental save — crash recovery
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(worker_preds))
        except Exception:
            pass

        if (step + 1) % 5 == 0 or (step + 1) == len(task_chunk):
            _elapsed_total = time.time() - t0
            _rate          = (step + 1) / max(_elapsed_total, 1e-9)
            _eta           = (len(task_chunk) - step - 1) / max(_rate, 1e-9)
            print(
                f"[GPU {rank}] {step+1}/{len(task_chunk)} | "
                f"{_elapsed_total:.0f}s elapsed | ETA {_eta:.0f}s",
                flush=True,
            )

    # Final write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(worker_preds))
    print(
        f"[GPU {rank}] Done — {len(worker_preds)} predictions saved to {out_path}",
        flush=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_dual_model_pass(
    tasks: dict,
    coder_path: str,
    sft_path: str,
    output_path: str,
    output_dir: str = "/kaggle/working",
    n_workers: int = 4,
    time_per_task: int = 60,
    deadline: float = 0.0,
) -> int:
    """
    Distribute tasks across n_workers GPU processes (one per L4).
    Each worker runs run_isolated_gpu_worker() with both model paths.

    Args:
        tasks:        {task_id: task_data} dict.
        coder_path:   Path to Qwen2.5-Coder model (may be '').
        sft_path:     Path to sorokin/qwen3_4b model.
        output_path:  Merged JSON output path.
        output_dir:   Working directory for per-rank incremental JSONs.
        n_workers:    Number of GPU workers (default 4 = all L4s).
        time_per_task: Soft per-task budget in seconds.
        deadline:     Unix timestamp wall-clock deadline (0 = no deadline).

    Returns:
        Number of task predictions written to output_path.
    """
    task_items = list(tasks.items())
    n = len(task_items)
    print(
        f"[dual] {n} tasks | {n_workers} GPU workers | "
        f"SFT={sft_path} | Coder={coder_path or 'SFT-fallback'}",
        flush=True,
    )

    # Round-robin distribution across workers
    chunks: list = [[] for _ in range(n_workers)]
    for i, item in enumerate(task_items):
        chunks[i % n_workers].append(item)

    ctx = mp.get_context("spawn")
    processes = []
    for rank in range(n_workers):
        if not chunks[rank]:
            continue
        p = ctx.Process(
            target=run_isolated_gpu_worker,
            args=(rank, chunks[rank], coder_path, sft_path, output_dir, time_per_task),
            daemon=False,
        )
        p.start()
        processes.append((rank, p))
        print(f"[dual] Spawned GPU{rank} with {len(chunks[rank])} tasks.", flush=True)

    for rank, p in processes:
        if deadline > 0.0:
            timeout = max(1.0, deadline - time.time())
            p.join(timeout=timeout)
            if p.is_alive():
                print(f"[dual] GPU{rank} deadline exceeded — terminating.", flush=True)
                p.terminate()
                p.join(timeout=30)
        else:
            p.join()
        print(f"[dual] GPU{rank} joined (exit={p.exitcode})", flush=True)

    # Aggregate per-rank results into a single output file
    all_results: dict = {}
    for rank in range(n_workers):
        rank_file = Path(output_dir) / f"preds_rank_{rank}.json"
        if rank_file.exists():
            try:
                chunk_results = json.loads(rank_file.read_text())
                all_results.update(chunk_results)
                print(
                    f"[dual] Loaded rank {rank}: {len(chunk_results)} predictions.",
                    flush=True,
                )
            except Exception as exc:
                print(f"[dual] Could not load {rank_file}: {exc}", flush=True)
        else:
            print(f"[dual] Rank {rank} output missing: {rank_file}", flush=True)

    # ── Two-pass retry: re-run tasks where attempt_1 echoes the test input ──
    try:
        _echo_tasks = {
            tid: tasks[tid]
            for tid, preds in all_results.items()
            if tid in tasks
            and preds
            and preds[0].get("attempt_1") == tasks[tid].get("test", [{}])[0].get("input")
        }
        if _echo_tasks:
            print(
                f"[dual] Two-pass retry: {len(_echo_tasks)} echo tasks → re-running.",
                flush=True,
            )
            _retry_dir = str(Path(output_dir) / "dual_retry")
            Path(_retry_dir).mkdir(parents=True, exist_ok=True)
            _chunks2: list = [[] for _ in range(n_workers)]
            for _i2, _item2 in enumerate(_echo_tasks.items()):
                _chunks2[_i2 % n_workers].append(_item2)
            _ctx2 = mp.get_context("spawn")
            _procs2 = []
            for _rk2 in range(n_workers):
                if not _chunks2[_rk2]:
                    continue
                _p2 = _ctx2.Process(
                    target=run_isolated_gpu_worker,
                    args=(_rk2, _chunks2[_rk2], coder_path, sft_path,
                          _retry_dir, max(time_per_task, 90)),
                    daemon=False,
                )
                _p2.start()
                _procs2.append((_rk2, _p2))
            for _rk2, _p2 in _procs2:
                if deadline > 0.0:
                    _t2 = max(1.0, deadline - time.time())
                    _p2.join(timeout=_t2)
                    if _p2.is_alive():
                        _p2.terminate()
                        _p2.join(timeout=30)
                else:
                    _p2.join()
            for _rk2 in range(n_workers):
                _rf2 = Path(_retry_dir) / f"preds_rank_{_rk2}.json"
                if _rf2.exists():
                    try:
                        _rr = json.loads(_rf2.read_text())
                        for _tid2, _rp in _rr.items():
                            _orig_inp = tasks.get(_tid2, {}).get("test", [{}])[0].get("input")
                            if _rp and _rp[0].get("attempt_1") != _orig_inp:
                                all_results[_tid2] = _rp
                                print(f"[dual] Retry improved {_tid2}.", flush=True)
                    except Exception as _re2:
                        print(f"[dual] Retry load err rank {_rk2}: {_re2}", flush=True)
        else:
            print("[dual] Two-pass retry: no echo tasks — skipping.", flush=True)
    except Exception as _tp_exc:
        print(f"[dual] Two-pass retry error: {_tp_exc}", flush=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(all_results))
    print(
        f"[dual] Saved {len(all_results)} predictions -> {output_path}",
        flush=True,
    )
    return len(all_results)


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dual-model parallel ARC-AGI-2 inference (Coder-7B + SFT-4B)."
    )
    parser.add_argument(
        "--challenge",
        default="",
        help="Path to challenge JSON. Auto-discovered if omitted.",
    )
    parser.add_argument(
        "--output",
        default="/kaggle/working/dual_preds.json",
        help="Output predictions JSON path.",
    )
    parser.add_argument(
        "--output_dir",
        default="/kaggle/working",
        help="Working directory for per-rank intermediate files.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=0,
        help="Number of GPU workers. 0 = auto-detect from CUDA device count.",
    )
    parser.add_argument(
        "--time_per_task",
        type=int,
        default=60,
        help="Soft time budget per task in seconds.",
    )
    parser.add_argument(
        "--deadline_minutes",
        type=float,
        default=0.0,
        help="Wall-clock deadline in minutes from now (0 = no deadline).",
    )
    parser.add_argument(
        "--coder_path",
        default="",
        help="Override Coder model path (default: auto-discover).",
    )
    parser.add_argument(
        "--sft_path",
        default="",
        help="Override SFT model path (default: auto-discover).",
    )
    args = parser.parse_args()

    # Model path resolution
    if args.coder_path and args.sft_path:
        coder_path = args.coder_path
        sft_path   = args.sft_path
    else:
        _disc_coder, _disc_sft = discover_competition_models()
        coder_path = args.coder_path or _disc_coder
        sft_path   = args.sft_path   or _disc_sft

    if not sft_path:
        print("[main] ERROR: SFT model path not resolved. Aborting.", flush=True)
        sys.exit(1)

    # Challenge file resolution
    challenge_path = args.challenge
    if not challenge_path:
        for _candidate in [
            "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json",
            "/kaggle/input/arc-prize-2026/arc-agi_test_challenges.json",
            "/kaggle/working/arc-agi_test_challenges.json",
        ]:
            if Path(_candidate).exists():
                challenge_path = _candidate
                break
        if not challenge_path:
            _found = list(Path("/kaggle/input").rglob("*test_challenges*.json"))
            if _found:
                challenge_path = str(_found[0])

    if not challenge_path or not Path(challenge_path).exists():
        print(
            f"[main] ERROR: Challenge file not found (tried: {challenge_path}). "
            "Pass --challenge.",
            flush=True,
        )
        sys.exit(1)

    print(f"[main] Challenge file: {challenge_path}", flush=True)
    tasks = json.loads(Path(challenge_path).read_text())
    print(f"[main] Loaded {len(tasks)} tasks.", flush=True)

    # GPU worker count
    import torch  # noqa: PLC0415
    n_gpus    = torch.cuda.device_count()
    n_workers = args.n_workers if args.n_workers > 0 else max(1, n_gpus)
    print(f"[main] CUDA devices detected: {n_gpus} | workers: {n_workers}", flush=True)

    # Deadline
    deadline = (
        time.time() + args.deadline_minutes * 60.0
        if args.deadline_minutes > 0
        else 0.0
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    n_saved = run_dual_model_pass(
        tasks=tasks,
        coder_path=coder_path,
        sft_path=sft_path,
        output_path=args.output,
        output_dir=args.output_dir,
        n_workers=n_workers,
        time_per_task=args.time_per_task,
        deadline=deadline,
    )
    print(f"[main] Complete — {n_saved} task predictions written.", flush=True)


if __name__ == "__main__":
    main()
