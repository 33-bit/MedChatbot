from __future__ import annotations


def answer(*args, **kwargs):
    from src.chat.pipeline import answer as _answer

    return _answer(*args, **kwargs)


def answer_with_meta(*args, **kwargs):
    from src.chat.pipeline import answer_with_meta as _answer_with_meta

    return _answer_with_meta(*args, **kwargs)


__all__ = ["answer", "answer_with_meta"]
