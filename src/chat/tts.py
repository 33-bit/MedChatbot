from __future__ import annotations

import html
import io
import os
import re
import shutil
import subprocess
import threading
import wave
from functools import lru_cache

import numpy as np

from src.config import set_hf_offline


_SYNTHESIS_LOCK = threading.Lock()
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)]\([^)]*\)")
_CITATION_RE = re.compile(r"\[(?:\d+[\s,;-]*)+]")
_URL_RE = re.compile(r"https?://\S+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"[ \t]+")


def _speech_text(text: str) -> str:
    cleaned = _MARKDOWN_LINK_RE.sub(r"\1", text)
    cleaned = _CITATION_RE.sub("", cleaned)
    cleaned = _URL_RE.sub("", cleaned)
    cleaned = _HTML_TAG_RE.sub("", cleaned)
    cleaned = cleaned.translate(str.maketrans("", "", "*_`#"))
    cleaned = html.unescape(cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return "\n".join(line.strip(" -\t") for line in cleaned.splitlines() if line.strip())


@lru_cache(maxsize=1)
def _get_engine():
    # Retrieval runs offline after preload. VieNeu needs one online pass to
    # populate the shared Hugging Face cache on first use, then runs locally.
    restore_offline = os.environ.get("HF_HUB_OFFLINE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    set_hf_offline(False)
    try:
        from vieneu import Vieneu

        return Vieneu()
    finally:
        set_hf_offline(restore_offline)


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise RuntimeError("VieNeu-TTS returned empty audio")
    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()


def _ffmpeg_executable() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("ffmpeg is required to encode Telegram voice messages") from exc


def _encode_ogg_opus(wav_audio: bytes) -> bytes:
    result = subprocess.run(
        [
            _ffmpeg_executable(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "wav",
            "-i",
            "pipe:0",
            "-vn",
            "-ac",
            "1",
            "-c:a",
            "libopus",
            "-b:a",
            "32k",
            "-application",
            "voip",
            "-f",
            "ogg",
            "pipe:1",
        ],
        input=wav_audio,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to encode TTS audio: {detail}")
    return result.stdout


def synthesize_speech(text: str) -> bytes:
    spoken_text = _speech_text(text)
    if not spoken_text:
        raise ValueError("TTS text is empty after normalization")
    with _SYNTHESIS_LOCK:
        engine = _get_engine()
        audio = engine.infer(spoken_text)
        wav_audio = _wav_bytes(audio, engine.sample_rate)
    return _encode_ogg_opus(wav_audio)
