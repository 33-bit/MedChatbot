from __future__ import annotations

import numpy as np

from src.chat import tts
from src.chat.storage.telegram_tts import is_tts_enabled, set_tts_enabled


def test_speech_text_removes_markup_urls_and_citations():
    assert tts._speech_text(
        "## **Ho** [1]\n[Xem nguồn](https://example.com) https://example.com/a"
    ) == "Ho\nXem nguồn"


def test_synthesize_speech_uses_vieneu_and_encodes_telegram_audio(monkeypatch):
    seen: list[str] = []

    class Engine:
        sample_rate = 48_000

        def infer(self, text: str):
            seen.append(text)
            return np.array([0.0, 0.25, -0.25], dtype=np.float32)

    monkeypatch.setattr(tts, "_get_engine", lambda: Engine())
    monkeypatch.setattr(tts, "_encode_ogg_opus", lambda wav: b"ogg:" + wav[:4])

    result = tts.synthesize_speech("**Xin chào** [1]")

    assert seen == ["Xin chào"]
    assert result.startswith(b"ogg:RIFF")


def test_telegram_tts_preference_is_persistent():
    assert is_tts_enabled(123) is False

    set_tts_enabled(123, True)
    assert is_tts_enabled("123") is True

    set_tts_enabled("123", False)
    assert is_tts_enabled(123) is False
