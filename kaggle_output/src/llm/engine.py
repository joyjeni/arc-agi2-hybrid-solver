"""
Qwen3-4B LLM Engine — inference wrapper with augmentation.

Supports:
  - vLLM / SGLang server mode  (KAGGLE_RERUN=1)
  - HuggingFace transformers   (local development / GPU)
  - Dry-run / mock mode        (unit tests, no GPU)
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod

from ..core.augmentation import eight_views, inverse_view
from ..core.grid_codec import Grid, text_to_grid

logger = logging.getLogger(__name__)

# ── Abstract LLM Backend ────────────────────────────────────────────────────

class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str: ...


# ── HuggingFace Backend ─────────────────────────────────────────────────────

class HuggingFaceLLMBackend(LLMBackend):
    """Loads Qwen3-4B (or any HF model) for local inference."""

    def __init__(self, model_path: str, device: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info("Loading model from %s ...", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map=device, trust_remote_code=True,
        )
        self.model.eval()
        logger.info("Model loaded.")

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 1e-5),
                do_sample=(temperature > 0.0),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)


# ── SGLang / vLLM HTTP Backend ──────────────────────────────────────────────

class ServerLLMBackend(LLMBackend):
    """Calls a running vLLM / SGLang OpenAI-compatible server."""

    def __init__(self, base_url: str = "http://localhost:30000/v1", model: str = "qwen3-4b"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        import urllib.request
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["text"]


# ── Mock Backend (tests / no GPU) ──────────────────────────────────────────

class MockLLMBackend(LLMBackend):
    """Returns deterministic stub output — for unit testing."""

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        return "0 | 0\n"


# ── High-Level Engine ───────────────────────────────────────────────────────

class QwenEngine:
    """
    Main inference engine.

    Wraps a backend and adds:
    - 8-view augmentation for direct grid prediction
    - Confidence-weighted voting across views
    - Per-task error isolation
    """

    def __init__(self, backend: LLMBackend):
        self._backend = backend

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        return self._backend.generate(prompt, max_tokens=max_tokens, temperature=temperature)

    def predict_with_augmentation(
        self,
        train_pairs: list[dict],
        test_input: Grid,
        prompt_fn,
        n_views: int = 4,
        max_tokens: int = 256,
    ) -> Grid | None:
        """
        Generate predictions across `n_views` augmented views and vote.
        Returns the most-voted grid, or None if no consistent answer.
        """
        views = eight_views(test_input)[:n_views]
        aug_pairs_views = [
            [{
                "input": eight_views(p["input"])[v_idx],
                "output": eight_views(p["output"])[v_idx],
            } for p in train_pairs]
            for v_idx in range(n_views)
        ]
        votes: dict[str, list[tuple[int, Grid]]] = {}  # key → [(view_idx, grid)]

        for v_idx, (view, aug_pairs) in enumerate(zip(views, aug_pairs_views)):
            try:
                prompt = prompt_fn(aug_pairs, view)
                raw = self._backend.generate(prompt, max_tokens=max_tokens)
                grid = _parse_grid_from_response(raw)
                if grid is None:
                    continue
                # Invert augmentation
                canonical = inverse_view(v_idx, grid)
                key = json.dumps(canonical, separators=(",", ":"))
                if key not in votes:
                    votes[key] = []
                votes[key].append((v_idx, canonical))
            except Exception as e:
                logger.debug("View %d failed: %s", v_idx, e)

        if not votes:
            return None
        # Return most-voted grid
        best_key = max(votes, key=lambda k: len(votes[k]))
        return votes[best_key][0][1]


def _parse_grid_from_response(text: str) -> Grid | None:
    """Extract grid from LLM output (numeric | separator or JSON array)."""
    # Try JSON first
    m = re.search(r"\[\s*\[[\d,\s\[\]]+\]\s*\]", text)
    if m:
        try:
            g = json.loads(m.group())
            if isinstance(g, list) and all(isinstance(r, list) for r in g):
                return g
        except Exception:
            pass
    # Try | separator format
    grid = text_to_grid(text.strip(), mode="numeric")
    if grid:
        return grid
    # Try RESULT: prefix
    m = re.search(r"RESULT:\s*(\[.*?\])", text, re.DOTALL)
    if m:
        try:
            g = json.loads(m.group(1))
            if isinstance(g, list):
                return g
        except Exception:
            pass
    return None


# ── Factory ─────────────────────────────────────────────────────────────────

def build_engine(model_path: str | None = None, server_url: str | None = None) -> QwenEngine:
    """
    Auto-select backend:
      1. server_url provided → ServerLLMBackend
      2. MOCK_LLM=1 env → MockLLMBackend  (testing)
      3. model_path provided → HuggingFaceLLMBackend
      4. fallback → MockLLMBackend with warning
    """
    if os.getenv("MOCK_LLM") == "1":
        logger.warning("MOCK_LLM=1 — using stub backend")
        return QwenEngine(MockLLMBackend())
    if server_url:
        return QwenEngine(ServerLLMBackend(server_url))
    if model_path:
        return QwenEngine(HuggingFaceLLMBackend(model_path))
    logger.warning("No model_path or server_url — using MockLLMBackend")
    return QwenEngine(MockLLMBackend())
