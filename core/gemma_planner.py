#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.llm_runner import LLMRunner


def _extract_json_list(text: str) -> Optional[Any]:
    if not text:
        return None
    # Prefer first [...] block.
    i = text.find("[")
    j = text.rfind("]")
    if i == -1 or j == -1 or j <= i:
        return None
    blob = text[i : j + 1]
    try:
        return json.loads(blob)
    except Exception:
        return None


def _validate_words(words: Any, *, min_len: int = 3, max_len: int = 8, max_power: int = 6) -> List[List[int]]:
    out: List[List[int]] = []
    if not isinstance(words, list):
        return out
    for w in words:
        if not isinstance(w, list):
            continue
        if len(w) < int(min_len) or len(w) > int(max_len):
            continue
        ww: List[int] = []
        ok = True
        last: Optional[int] = None
        for a in w:
            try:
                ai = int(a)
            except Exception:
                ok = False
                break
            if ai == 0 or abs(ai) > int(max_power):
                ok = False
                break
            # avoid immediate repetition
            if last is not None and ai == last:
                ok = False
                break
            ww.append(ai)
            last = ai
        if ok:
            out.append(ww)
    return out


class GemmaPlanner:
    def __init__(
        self,
        model_path: str = "/Users/machome/models/gemma/gemma-3-1b-it-Q4_K_M.gguf",
        llama_cli: str = "/Users/machome/llama.cpp/llama-cli",
        backend: str = "llama_cpp",
        max_length: int = 8,
        max_power: int = 6,
    ) -> None:
        self.backend = str(backend)
        self.max_length = int(max_length)
        self.max_power = int(max_power)
        self.runner = LLMRunner(
            model_path=str(model_path),
            llama_cli=str(llama_cli),
            n_ctx=2048,
            n_threads=6,
            n_gpu_layers=0,
            timeout_s=20.0,
        )

    def suggest_words(self, context_dict: Dict[str, Any]) -> List[List[int]]:
        if str(self.backend).lower() not in ("llama_cpp", "llama.cpp", "llama"):
            return []

        best_words = context_dict.get("best_words", [])
        recent_losses = context_dict.get("recent_losses", [])

        prompt = (
            "You are optimizing Artin words.\n\n"
            f"Best words:\n{best_words}\n\n"
            f"Recent losses:\n{recent_losses}\n\n"
            "Constraints:\n"
            "- length 3-8\n"
            "- avoid repetition\n"
            "- prefer small |a|\n\n"
            "Return ONLY JSON list:\n"
            "[[1,-1,2], [2,-2,1], ...]\n"
        )

        try:
            text = self.runner.generate(prompt, max_tokens=128, temperature=0.7)
            parsed = _extract_json_list(text)
            words = _validate_words(
                parsed,
                min_len=3,
                max_len=int(self.max_length),
                max_power=int(self.max_power),
            )
            if words:
                return words
        except Exception:
            return []

        return []


def main() -> None:
    ap = argparse.ArgumentParser(description="Gemma Planner (llama.cpp GGUF via llama-cli)")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--llama_cli", type=str, default="/Users/machome/llama.cpp/llama-cli")
    ap.add_argument("--model_path", type=str, default="/Users/machome/models/gemma/gemma-3-1b-it-Q4_K_M.gguf")
    ap.add_argument("--backend", type=str, default="llama_cpp")
    ap.add_argument("--max_length", type=int, default=8)
    ap.add_argument("--max_power", type=int, default=6)
    args = ap.parse_args()
    if args.test:
        planner = GemmaPlanner(
            model_path=str(args.model_path),
            llama_cli=str(args.llama_cli),
            backend=str(args.backend),
            max_length=int(args.max_length),
            max_power=int(args.max_power),
        )
        ctx = {"best_words": [[-2, 1, -1, 3]], "recent_losses": [1.0, 0.8, 0.7]}
        print(json.dumps(planner.suggest_words(ctx), indent=2))
        return


if __name__ == "__main__":
    main()

