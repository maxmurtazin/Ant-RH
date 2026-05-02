#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.llm_runner import LLMRunner


def clean_llm_json(text: str) -> Optional[Any]:
    if not text:
        return None
    cleaned = str(text).replace("```json", "").replace("```", "").strip()
    i = cleaned.find("[")
    j = cleaned.rfind("]")
    if i == -1:
        return None
    if j == -1 or j < i:
        blob = cleaned[i:].strip() + "]"
    else:
        blob = cleaned[i : j + 1].strip()
    if blob.count("[") > blob.count("]"):
        blob = blob + ("]" * (blob.count("[") - blob.count("]")))
    try:
        return json.loads(blob)
    except Exception:
        pass

    matches = re.findall(r"\[\s*-?\d+(?:\s*,\s*-?\d+)+\s*\]", cleaned)
    if not matches:
        return None
    out: List[List[int]] = []
    for m in matches:
        try:
            vals = [int(x) for x in re.findall(r"-?\d+", m)]
        except Exception:
            continue
        if vals:
            out.append(vals)
    return out or None


def validate_word(word: Any, max_length: int, max_power: int) -> bool:
    if not isinstance(word, list):
        return False
    if len(word) < 3 or len(word) > int(max_length):
        return False
    vals: List[int] = []
    for a in word:
        if not isinstance(a, int):
            return False
        if a == 0 or abs(a) > int(max_power):
            return False
        vals.append(int(a))
    for i in range(1, len(vals)):
        if vals[i] == vals[i - 1]:
            return False
    if len(vals) >= 4:
        for size in range(1, (len(vals) // 2) + 1):
            if vals[-size:] == vals[-2 * size : -size]:
                return False
    return True


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
        self.last_raw_response: str = ""
        self.last_parsed_words: List[List[int]] = []
        self.last_valid_words: List[List[int]] = []
        self.last_rejected_count: int = 0
        self.last_candidate_braid_words: List[Any] = []
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
        self.last_raw_response = ""
        self.last_parsed_words = []
        self.last_valid_words = []
        self.last_rejected_count = 0

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
            "[[1,-1,2], [2,-2,1], ...]\n\n"
            "Optional (for braid/NCG): include candidate_words as braid tokens, e.g.\n"
            '{"words": [[1,-1,2]], "candidate_words": [["sigma1","sigma2^-1","sigma1"]]}\n'
        )

        fallback_words = [[1, -1, 2], [2, -1, 1], [-1, 2, -2]]
        try:
            text = self.runner.generate(prompt, max_tokens=128, temperature=0.7)
            self.last_raw_response = str(text or "")
            self.last_candidate_braid_words = []
            try:
                cleaned_full = str(text or "").replace("```json", "").replace("```", "").strip()
                obj_try = json.loads(cleaned_full)
                if isinstance(obj_try, dict):
                    cw = obj_try.get("candidate_words")
                    if isinstance(cw, list):
                        self.last_candidate_braid_words = cw
            except Exception:
                pass
            parsed = clean_llm_json(text)
            parsed_words = parsed if isinstance(parsed, list) else []
            self.last_parsed_words = [list(w) for w in parsed_words if isinstance(w, list)]
            valid_words: List[List[int]] = []
            seen = set()
            for w in parsed_words:
                if not isinstance(w, list):
                    continue
                cleaned = []
                ok = True
                for a in w:
                    try:
                        ai = int(a)
                    except Exception:
                        ok = False
                        break
                    if ai == 0:
                        ok = False
                        break
                    cleaned.append(ai)
                if not ok:
                    continue
                if not validate_word(cleaned, self.max_length, self.max_power):
                    continue
                key = tuple(cleaned)
                if key in seen:
                    continue
                seen.add(key)
                valid_words.append(cleaned)
            self.last_valid_words = [list(w) for w in valid_words]
            self.last_rejected_count = max(0, len(self.last_parsed_words) - len(self.last_valid_words))
            if valid_words:
                return valid_words
        except Exception:
            pass

        deduped_fallback: List[List[int]] = []
        seen = set()
        for w in fallback_words:
            if validate_word(w, self.max_length, self.max_power):
                key = tuple(w)
                if key not in seen:
                    seen.add(key)
                    deduped_fallback.append(list(w))
        self.last_valid_words = [list(w) for w in deduped_fallback]
        return deduped_fallback


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

