from __future__ import annotations


def answer(*args, **kwargs):
    from src.chat.pipeline import answer as _answer

    return _answer(*args, **kwargs)

__all__ = ["answer"]
