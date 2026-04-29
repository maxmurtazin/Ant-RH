#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Iterator


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

    def _resolve_llama_cli(self) -> str:
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
        return str(resolved_llama_cli)

    def _build_cmd(self, prompt: str, max_tokens: int, temperature: float) -> list[str]:
        return [
            self._resolve_llama_cli(),
            "-m",
            str(self.model_path),
            "-p",
            str(prompt),
            "-n",
            str(int(max_tokens)),
            "--temp",
            str(float(temperature)),
            "--ctx-size",
            str(int(self.n_ctx)),
            "--threads",
            str(int(self.n_threads)),
            "--n-gpu-layers",
            str(int(self.n_gpu_layers)),
        ]

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        prompt = str(prompt)
        if not prompt:
            return ""
        try:
            result = subprocess.run(
                [
                    self._resolve_llama_cli(),
                    "-m",
                    str(self.model_path),
                    "-p",
                    prompt,
                    "-n",
                    str(int(max_tokens)),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return ""
        except Exception:
            return ""
        return (result.stdout or "").strip()

    def generate_stream(self, prompt, max_tokens: int = 256, temperature: float = 0.3) -> Iterator[str]:
        prompt = str(prompt)
        if not prompt:
            return

        cmd = self._build_cmd(prompt, max_tokens=max_tokens, temperature=temperature)
        try:
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception:
            return

        start = time.time()
        prompt_removed = False
        stdout = p.stdout
        if stdout is None:
            try:
                p.kill()
            except Exception:
                pass
            return

        try:
            prompt_idx = 0
            prompt_removed = False
            while True:
                if float(self.timeout_s) > 0 and (time.time() - start) > float(self.timeout_s):
                    try:
                        p.kill()
                    except Exception:
                        pass
                    break

                ch = stdout.read(1)
                if ch == "":
                    if p.poll() is not None:
                        break
                    continue

                frag = ch
                if not prompt_removed:
                    if prompt_idx < len(prompt) and frag == prompt[prompt_idx]:
                        prompt_idx += 1
                        continue
                    prompt_removed = True
                yield frag
        finally:
            try:
                if p.poll() is None:
                    p.wait(timeout=1.0)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass

