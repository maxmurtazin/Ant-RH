#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
from typing import Tuple


def chunk_text(text, max_chars=220):
    import re
    import textwrap

    text = re.sub(r"\s+", " ", str(text or "").strip())
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) + 1 <= max_chars:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            if len(s) > max_chars:
                chunks.extend(textwrap.wrap(s, width=max_chars))
                current = ""
            else:
                current = s

    if current:
        chunks.append(current)

    return chunks


def speak_with_say(
    text,
    voice="Samantha",
    rate=150,
    max_chars=220,
    pause=0.35,
):
    import subprocess
    import time

    for chunk in chunk_text(text, max_chars=max_chars):
        subprocess.run(
            [
                "say",
                "-v",
                voice,
                "-r",
                str(rate),
                chunk,
            ],
            check=False,
        )
        time.sleep(pause)


def speak_text(text: str, voice_name: str = "Samantha") -> Tuple[bool, str]:
    t = (text or "").strip()
    if not t:
        return False, "empty text"
    say_bin = shutil.which("say")
    if not say_bin:
        return False, "say command not found"
    try:
        speak_with_say(t, voice=str(voice_name))
        return True, ""
    except Exception as e:
        return False, str(e)

