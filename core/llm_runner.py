#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@dataclass
class LLMRunner:
    model_path: str
    llama_cli: str = "llama-cli"
    n_ctx: int = 2048
    n_threads: int = 6
    n_gpu_layers: int = 0
    timeout_s: float = 60.0

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        prompt = str(prompt)
        if not prompt:
            return ""

        llama_cli_str = str(self.llama_cli)
        resolved_llama_cli = None
        if os.sep in llama_cli_str or (os.altsep and os.altsep in llama_cli_str):
            if os.path.exists(llama_cli_str):
                resolved_llama_cli = llama_cli_str
        else:
            resolved_llama_cli = shutil.which(llama_cli_str)

        if not resolved_llama_cli:
            raise FileNotFoundError(
                f"llama.cpp binary not found on PATH: {self.llama_cli}. "
                "Install llama.cpp (brew install llama.cpp) or set --llama_cli to a full path."
            )

        cmd = [
            str(resolved_llama_cli),
            "-m",
            str(self.model_path),
            "-p",
            prompt,
            "-n",
            str(int(max_tokens)),
        ]
        # best-effort perf knobs (supported by most llama.cpp builds)
        cmd += [
            "--temp",
            str(float(temperature)),
            "--ctx-size",
            str(int(self.n_ctx)),
            "--threads",
            str(int(self.n_threads)),
            "--n-gpu-layers",
            str(int(self.n_gpu_layers)),
        ]

        try:
            p = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=float(self.timeout_s),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ""
        except Exception:
            return ""

        if p.returncode != 0:
            return ""

        out = (p.stdout or "").strip()
        if not out:
            return ""

        # Many llama-cli builds echo the prompt; remove it if present.
        if out.startswith(prompt):
            out = out[len(prompt) :].lstrip()

        # Some builds echo "prompt:" lines; remove the first exact prompt occurrence.
        pos = out.find(prompt)
        if 0 <= pos < 20:
            out = out[pos + len(prompt) :].lstrip()

        return out.strip()

