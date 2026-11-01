#!/usr/bin/env python3
"""
jeni_llm_fast.py  —  LLM fast-inference pass for ARC-AGI-2
=========================================================
Uses up to 4x L4 GPUs (or however many CUDA devices are visible).
Each worker process loads Qwen3-4B on its own GPU and processes a slice of tasks.

Output: /kaggle/working/jeni_llm_preds.json
  Format: {task_id: [{attempt_1: [[int,...]], attempt_2: [[int,...]]}, ...]}

Design:
  • attempt_1 ← greedy direct-digit prompt  (temperature=0)
  • attempt_2 ← chain-of-thought prompt      (temperature=0.7, sampled)
  • fallback   ← compact-digit prompt        (temperature=0.4) when either failed
  • hard fallback ← copy of test input       when all parsing fails

Merge priority in the notebook's merge cell:
  1. NVARC submission.json  → attempt_1   (highest trust)
  2. DSA  dsa_solved.json   → attempt_2   (deterministic)
  3. LLM  jeni_llm_preds.json → attempt_2 diversity (fill gaps)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
import torch.multiprocessing as mp


# ── Color map ─────────────────────────────────────────────────────────────────
COLOR_NAMES = {
    0: "black", 1: "blue",  2: "red",    3: "green",  4: "yellow",
    5: "grey",  6: "magenta", 7: "orange", 8: "azure", 9: "maroon",
}

# ── Grid utilities ─────────────────────────────────────────────────────────────

def grid_to_digits(grid: list) -> str:
    """Space-separated rows: '0 1 2\\n3 4 5'"""
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def grid_to_compact(grid: list) -> str:
    """No-space rows: '012\\n345'"""
    return "\n".join("".join(str(c) for c in row) for row in grid)


def grid_to_colored(grid: list) -> str:
    """'black blue\\nred green' label rows"""
    return "\n".join(
        " ".join(COLOR_NAMES.get(c, str(c)) for c in row) for row in grid
    )


def parse_grid(text: str, ref_shape: Optional[tuple] = None) -> Optional[list]:
    """
    Parse LLM output into a 2-D integer grid.
    Strips <think>...</think> blocks and markdown fences first.
    Returns None if no valid grid can be extracted.
    """
    # Strip thinking blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip markdown fences
    text = re.sub(r"```[^\n]*\n?", "", text)
    text = text.strip()

    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    rows: list[list[int]] = []
    for line in lines:
        # Space-separated single digits
        parts = line.split()
        if parts and all(p.isdigit() and len(p) == 1 for p in parts):
            rows.append([int(p) for p in parts])
            continue
        # Compact: no spaces, all digits
        compact = line.replace(" ", "")
        if compact and all(c.isdigit() for c in compact):
            rows.append([int(c) for c in compact])
            continue
        # Comma-separated
        parts_c = [p.strip() for p in line.split(",")]
        if parts_c and all(p.isdigit() and len(p) == 1 for p in parts_c):
            rows.append([int(p) for p in parts_c])
            continue
        # Bracket / mixed: extract all single digits on this line
        digits = re.findall(r"\b\d\b", line)
        if digits and len(digits) >= 2:
            rows.append([int(d) for d in digits])
            continue

    if not rows:
        return None

    # Use most-common row length
    most_common_len = Counter(len(r) for r in rows).most_common(1)[0][0]
    rows = [r for r in rows if len(r) == most_common_len]

    # Validate values 0-9
    for row in rows:
        if any(v < 0 or v > 9 for v in row):
            return None

    # Clip / pad to ref_shape
    if ref_shape is not None:
        h, w = ref_shape
        rows = rows[:h]
        rows = [r[:w] + [0] * max(0, w - len(r)) for r in rows]
        while len(rows) < h:
            rows.append([0] * w)

    return rows if rows else None


# ── System prompts ─────────────────────────────────────────────────────────────

_SYS_DIRECT = (
    "You are an expert at ARC-AGI puzzles. "
    "Study the input/output training examples, find the transformation rule, "
    "and apply it to the test input. "
    "Output ONLY the answer grid: rows of space-separated single digits. "
    "No explanation, no code, no markdown."
)

_SYS_COT = (
    "You are an expert at ARC-AGI puzzles. "
    "Study the input/output training examples and find the transformation rule. "
    "Reason step by step inside <think>...</think> tags. "
    "After </think>, output ONLY the final answer grid: "
    "rows of space-separated single digits."
)

_SYS_COMPACT = (
    "You solve ARC-AGI puzzles. "
    "Find the rule from the examples and apply it to the test input. "
    "Output ONLY the answer grid: each row on its own line, digits with no spaces."
)

_SYS_COLORED = (
    "You are an ARC-AGI expert. "
    "0=black 1=blue 2=red 3=green 4=yellow 5=grey 6=magenta 7=orange 8=azure 9=maroon. "
    "Study the examples and output ONLY the answer: "
    "rows of space-separated single digits."
)


# ── Prompt builder ─────────────────────────────────────────────────────────────

def make_prompt(task_view: dict, style: str) -> tuple[str, str]:
    """
    Build (system_msg, user_msg) for one test panel.
    task_view has 'train' list and 'test' list with exactly one panel.
    """
    train = task_view.get("train", [])
    test_panels = task_view.get("test", [])
    test_input = test_panels[0]["input"] if test_panels else [[0]]

    if style == "direct":
        sys_msg = _SYS_DIRECT
        parts = ["Training examples:"]
        for i, ex in enumerate(train):
            parts.append(f"\nExample {i+1} input:\n{grid_to_digits(ex['input'])}")
            parts.append(f"Example {i+1} output:\n{grid_to_digits(ex['output'])}")
        parts.append(f"\nTest input:\n{grid_to_digits(test_input)}")
        parts.append("\nOutput:")

    elif style == "cot":
        sys_msg = _SYS_COT
        parts = ["Study these ARC examples:"]
        for i, ex in enumerate(train):
            parts.append(f"\nExample {i+1} input:\n{grid_to_digits(ex['input'])}")
            parts.append(f"Example {i+1} output:\n{grid_to_digits(ex['output'])}")
        parts.append(f"\nTest input:\n{grid_to_digits(test_input)}")
        parts.append("\nFind the pattern and give the answer grid.")

    elif style == "compact":
        sys_msg = _SYS_COMPACT
        parts = ["Examples (compact, no spaces):"]
        for i, ex in enumerate(train):
            parts.append(f"\nEx{i+1} in:\n{grid_to_compact(ex['input'])}")
            parts.append(f"Ex{i+1} out:\n{grid_to_compact(ex['output'])}")
        parts.append(f"\nTest:\n{grid_to_compact(test_input)}")
        parts.append("\nAnswer:")

    elif style == "colored":
        sys_msg = _SYS_COLORED
        parts = ["Examples with color names:"]
        for i, ex in enumerate(train):
            parts.append(f"\nExample {i+1} input:\n{grid_to_colored(ex['input'])}")
            parts.append(f"Example {i+1} output:\n{grid_to_digits(ex['output'])}")
        parts.append(f"\nTest input:\n{grid_to_colored(test_input)}")
        parts.append("\nOutput (digits only):")

    else:
        return make_prompt(task_view, "direct")

    return sys_msg, "\n".join(parts)


# ── Per-GPU worker ─────────────────────────────────────────────────────────────

def _apply_chat(tokenizer, messages: list, enable_thinking: bool = False):
    """Apply chat template, gracefully ignoring unknown kwargs."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )


def worker_inference(
    rank: int,
    task_items: list,
    model_path: str,
    worker_out_file: str,
) -> None:
    """
    Spawned child process: loads model on GPU `rank`, processes task_items,
    writes JSON results to worker_out_file.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    results: dict = {}

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        print(f"[GPU{rank}] Loading model …", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        model.eval()
        print(
            f"[GPU{rank}] Ready — {len(task_items)} tasks to process.", flush=True
        )

        t0 = time.time()
        for step, (task_id, task) in enumerate(task_items):
            test_panels = task.get("test", [])
            if not test_panels:
                results[task_id] = []
                continue

            task_preds = []
            for panel in test_panels:
                inp = panel.get("input", [[0]])
                ref_shape = (len(inp), len(inp[0])) if inp and inp[0] else None
                task_view = {"train": task.get("train", []), "test": [panel]}

                attempt_1: Optional[list] = None
                attempt_2: Optional[list] = None

                # ── Strategy 1: greedy direct → attempt_1 ────────────────────
                try:
                    sys_m, usr_m = make_prompt(task_view, "direct")
                    msgs = [
                        {"role": "system",  "content": sys_m},
                        {"role": "user",    "content": usr_m},
                    ]
                    ids = _apply_chat(tokenizer, msgs, enable_thinking=False).to("cuda:0")
                    with torch.no_grad():
                        out = model.generate(
                            ids,
                            max_new_tokens=512,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    gen = tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
                    attempt_1 = parse_grid(gen, ref_shape)
                except Exception as exc:
                    print(f"[GPU{rank}] {task_id} direct err: {exc}", flush=True)

                # ── Strategy 2: CoT sampled T=0.7 → attempt_2 ────────────────
                try:
                    sys_m2, usr_m2 = make_prompt(task_view, "cot")
                    msgs2 = [
                        {"role": "system",  "content": sys_m2},
                        {"role": "user",    "content": usr_m2},
                    ]
                    ids2 = _apply_chat(tokenizer, msgs2, enable_thinking=True).to("cuda:0")
                    with torch.no_grad():
                        out2 = model.generate(
                            ids2,
                            max_new_tokens=1024,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    gen2 = tokenizer.decode(out2[0][ids2.shape[-1]:], skip_special_tokens=True)
                    attempt_2 = parse_grid(gen2, ref_shape)
                except Exception as exc:
                    print(f"[GPU{rank}] {task_id} cot err: {exc}", flush=True)

                # ── Strategy 3: compact T=0.4 (fallback if either failed) ────
                if attempt_1 is None or attempt_2 is None:
                    try:
                        sys_m3, usr_m3 = make_prompt(task_view, "compact")
                        msgs3 = [
                            {"role": "system",  "content": sys_m3},
                            {"role": "user",    "content": usr_m3},
                        ]
                        ids3 = _apply_chat(tokenizer, msgs3, enable_thinking=False).to("cuda:0")
                        with torch.no_grad():
                            out3 = model.generate(
                                ids3,
                                max_new_tokens=512,
                                do_sample=True,
                                temperature=0.4,
                                top_p=0.9,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        gen3 = tokenizer.decode(out3[0][ids3.shape[-1]:], skip_special_tokens=True)
                        fb = parse_grid(gen3, ref_shape)
                        if attempt_1 is None:
                            attempt_1 = fb
                        if attempt_2 is None:
                            attempt_2 = fb
                    except Exception as exc:
                        print(f"[GPU{rank}] {task_id} compact err: {exc}", flush=True)

                # ── Strategy 4: colored T=0.4 (second fallback) ─────────────
                if attempt_1 is None or attempt_2 is None:
                    try:
                        sys_m4, usr_m4 = make_prompt(task_view, "colored")
                        msgs4 = [
                            {"role": "system",  "content": sys_m4},
                            {"role": "user",    "content": usr_m4},
                        ]
                        ids4 = _apply_chat(tokenizer, msgs4, enable_thinking=False).to("cuda:0")
                        with torch.no_grad():
                            out4 = model.generate(
                                ids4,
                                max_new_tokens=512,
                                do_sample=True,
                                temperature=0.4,
                                top_p=0.9,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        gen4 = tokenizer.decode(out4[0][ids4.shape[-1]:], skip_special_tokens=True)
                        fb4 = parse_grid(gen4, ref_shape)
                        if attempt_1 is None:
                            attempt_1 = fb4
                        if attempt_2 is None:
                            attempt_2 = fb4
                    except Exception as exc:
                        print(f"[GPU{rank}] {task_id} colored err: {exc}", flush=True)

                # ── Hard fallback: test input ─────────────────────────────────
                if attempt_1 is None:
                    attempt_1 = inp
                if attempt_2 is None:
                    attempt_2 = attempt_1

                task_preds.append(
                    {"attempt_1": attempt_1, "attempt_2": attempt_2}
                )

            results[task_id] = task_preds

            if (step + 1) % 10 == 0 or (step + 1) == len(task_items):
                elapsed = time.time() - t0
                rate = (step + 1) / max(elapsed, 1e-9)
                eta = (len(task_items) - step - 1) / rate
                print(
                    f"[GPU{rank}] {step+1}/{len(task_items)} | "
                    f"{elapsed:.0f}s | ETA {eta:.0f}s",
                    flush=True,
                )

        print(f"[GPU{rank}] Done — {len(results)} predictions.", flush=True)

    except Exception as exc:  # noqa: BLE001
        print(f"[GPU{rank}] Worker crashed: {exc}", flush=True)
        traceback.print_exc()

    Path(worker_out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(worker_out_file, "w") as fh:
        json.dump(results, fh)


# ── Main orchestrator ──────────────────────────────────────────────────────────

def run_llm_fast_pass(
    tasks: dict,
    model_path: str,
    output_path: str,
    n_workers: int = 4,
) -> int:
    """
    Distribute tasks across n_workers GPU processes.
    Returns count of task predictions saved.
    """
    task_items = list(tasks.items())
    n = len(task_items)
    print(
        f"[llm-fast] {n} tasks | {n_workers} GPU workers | model={model_path}",
        flush=True,
    )

    # Even distribution: round-robin
    chunks: list[list] = [[] for _ in range(n_workers)]
    for i, item in enumerate(task_items):
        chunks[i % n_workers].append(item)

    worker_files = [
        f"/kaggle/working/jeni_llm_worker_{r}.json" for r in range(n_workers)
    ]

    ctx = mp.get_context("spawn")
    processes: list[tuple[int, mp.Process]] = []
    for rank in range(n_workers):
        if not chunks[rank]:
            continue
        p = ctx.Process(
            target=worker_inference,
            args=(rank, chunks[rank], model_path, worker_files[rank]),
            daemon=False,
        )
        p.start()
        processes.append((rank, p))

    for rank, p in processes:
        p.join()
        print(
            f"[llm-fast] GPU{rank} joined (exit={p.exitcode})", flush=True
        )

    # Aggregate results
    all_results: dict = {}
    for rank in range(n_workers):
        wf = worker_files[rank]
        if Path(wf).exists():
            try:
                with open(wf) as fh:
                    all_results.update(json.load(fh))
                print(f"[llm-fast] Loaded worker {rank}: {wf}", flush=True)
            except Exception as exc:
                print(f"[llm-fast] Could not load {wf}: {exc}", flush=True)
        else:
            print(f"[llm-fast] Worker {rank} output missing: {wf}", flush=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(all_results, fh)

    print(
        f"[llm-fast] Saved {len(all_results)} task predictions → {output_path}",
        flush=True,
    )
    return len(all_results)


# ── Model discovery ────────────────────────────────────────────────────────────

def discover_model_path() -> str:
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
    return ""


# ── CLI entry ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM fast-inference pass for ARC-AGI-2")
    parser.add_argument("--model-path",     default=None,
                        help="Path to Qwen3-4B model dir. Defaults to ARC_MODEL_PATH env.")
    parser.add_argument("--challenge-file", default=None,
                        help="Path to ARC challenge JSON. Auto-discovered if omitted.")
    parser.add_argument("--output",         default="/kaggle/working/jeni_llm_preds.json",
                        help="Output path for LLM predictions.")
    parser.add_argument("--n-workers",      type=int, default=4,
                        help="Number of GPU workers (default 4).")
    parser.add_argument("--end-time",       type=float, default=None,
                        help="Unix timestamp deadline. Script exits early if < 30 min remain.")
    args = parser.parse_args()

    # ── Time-budget check ─────────────────────────────────────────────────────
    if args.end_time is not None:
        remaining = args.end_time - time.time()
        if remaining < 1800:  # < 30 min
            print(
                f"[llm-fast] Only {remaining:.0f}s remaining — skipping LLM pass.",
                flush=True,
            )
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as fh:
                json.dump({}, fh)
            return

    # ── Model path ────────────────────────────────────────────────────────────
    model_path = args.model_path or discover_model_path()
    if not model_path or not Path(model_path).exists():
        print("[llm-fast] ERROR: model not found. Set --model-path or ARC_MODEL_PATH.",
              flush=True)
        sys.exit(1)

    # ── Challenge file ────────────────────────────────────────────────────────
    challenge_file = args.challenge_file
    if not challenge_file:
        for d in [
            "/kaggle/input/competitions/arc-prize-2026-arc-agi-2",
            "/kaggle/input/arc-prize-2026-arc-agi-2",
        ]:
            is_rerun = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
            fname = (
                "arc-agi_test_challenges.json"
                if is_rerun
                else "arc-agi_evaluation_challenges.json"
            )
            candidate = Path(d) / fname
            if candidate.exists():
                challenge_file = str(candidate)
                break

    if not challenge_file or not Path(challenge_file).exists():
        print(f"[llm-fast] ERROR: challenge file not found: {challenge_file}", flush=True)
        sys.exit(1)

    print(f"[llm-fast] Challenge file: {challenge_file}", flush=True)
    with open(challenge_file) as fh:
        tasks = json.load(fh)

    # ── GPU count ─────────────────────────────────────────────────────────────
    n_visible = torch.cuda.device_count()
    n_workers = min(args.n_workers, max(1, n_visible))
    print(f"[llm-fast] CUDA devices visible: {n_visible}, using {n_workers} workers",
          flush=True)

    t0 = time.time()
    count = run_llm_fast_pass(tasks, model_path, args.output, n_workers=n_workers)
    elapsed = time.time() - t0
    print(f"[llm-fast] Finished: {count} tasks in {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
