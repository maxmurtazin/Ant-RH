#!/usr/bin/env python3
from __future__ import annotations

import re
import time
from typing import Any, List


class LocalTTS:
    def __init__(
        self,
        voice_hint: str = "female",
        rate: int = 155,
        volume: float = 0.85,
        pause_short: float = 0.25,
        pause_long: float = 0.7,
        soft_mode: bool = True,
    ) -> None:
        try:
            import pyttsx3  # type: ignore
        except Exception as e:
            raise RuntimeError("python3 -m pip install pyttsx3") from e

        self.pyttsx3 = pyttsx3
        self.engine = pyttsx3.init()
        self.voice_hint = str(voice_hint)
        self.rate = int(rate)
        self.volume = float(volume)
        self.pause_short = float(pause_short)
        self.pause_long = float(pause_long)
        self.soft_mode = bool(soft_mode)

        if self.soft_mode:
            self.rate = max(120, self.rate - 5)
            self.pause_short = max(self.pause_short, 0.28)
            self.pause_long = max(self.pause_long, 0.75)

        self.engine.setProperty("rate", self.rate)
        self.engine.setProperty("volume", max(0.0, min(1.0, self.volume)))
        self._select_voice()

    def _select_voice(self) -> None:
        voices = self.engine.getProperty("voices") or []
        prefer = ["samantha", "ava", "allison", "serena"]
        chosen = None
        for v in voices:
            vid = str(getattr(v, "id", "")).lower()
            vname = str(getattr(v, "name", "")).lower()
            blob = f"{vid} {vname}"
            if any(p in blob for p in prefer):
                chosen = v
                break
        if chosen is None and self.voice_hint.lower() == "female":
            for v in voices:
                blob = f"{str(getattr(v, 'id', '')).lower()} {str(getattr(v, 'name', '')).lower()}"
                if any(k in blob for k in ["female", "woman", "samantha", "ava", "allison", "serena"]):
                    chosen = v
                    break
        if chosen is not None:
            self.engine.setProperty("voice", getattr(chosen, "id", ""))

    def list_voices(self) -> None:
        voices = self.engine.getProperty("voices") or []
        for v in voices:
            vid = str(getattr(v, "id", ""))
            name = str(getattr(v, "name", ""))
            langs_raw: Any = getattr(v, "languages", [])
            langs: List[str] = []
            if isinstance(langs_raw, (list, tuple)):
                for x in langs_raw:
                    try:
                        if isinstance(x, bytes):
                            langs.append(x.decode("utf-8", errors="ignore"))
                        else:
                            langs.append(str(x))
                    except Exception:
                        langs.append(str(x))
            print(f"id={vid} | name={name} | languages={langs}")

    def prepare_text(self, text: str) -> List[str]:
        t = re.sub(r"\s+", " ", str(text or "")).strip()
        if not t:
            return []

        replacements = {
            "ACO": "A C O",
            "RL": "R L",
            "PDE": "P D E",
            "DTES": "D T E S",
            "JSON": "J SON",
            "CSV": "C S V",
        }
        for k, v in replacements.items():
            t = re.sub(rf"\b{k}\b", v, t)

        # Avoid speaking huge JSON-like blocks literally
        t = re.sub(r"\{[^{}]{180,}\}", " structured data omitted ", t)
        t = re.sub(r"\[[^\[\]]{180,}\]", " list data omitted ", t)

        sentences = re.split(r"(?<=[.!?])\s+", t)
        chunks: List[str] = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            while len(s) > 120:
                cut = s.rfind(" ", 0, 120)
                if cut <= 0:
                    cut = 120
                chunks.append(s[:cut].strip())
                s = s[cut:].strip()
            if s:
                chunks.append(s)
        return chunks

    def speak(self, text: str) -> None:
        chunks = self.prepare_text(text)
        for chunk in chunks:
            self.engine.say(chunk)
            self.engine.runAndWait()
            low = chunk.lower()
            if any(k in low for k in ["warnings", "important", "critical", "not learning", "failed"]):
                time.sleep(self.pause_long)
            else:
                time.sleep(self.pause_short)

