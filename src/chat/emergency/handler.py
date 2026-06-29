"""High-level emergency reply builder.

Public API:
  - emergency_fast_reply(subject)         -> short, deterministic lead block
  - emergency_first_aid_reply(...)        -> corpus-checked aid + safety block
  - build_emergency_reply(...)            -> combined reply (used by pipeline)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Mapping

from src.chat.emergency.generation import generate_emergency_aid
from src.chat.emergency.retrieval import retrieve_emergency_aid
from src.chat.emergency.safety import apply_safety_post_check

log = logging.getLogger(__name__)


def _emit_timing(
    callback: Callable[[str, float], None] | None,
    stage: str,
    started_at: float,
) -> None:
    if callback is None:
        return
    try:
        callback(stage, (time.perf_counter() - started_at) * 1000)
    except Exception:
        log.warning("Emergency timing callback failed for stage %s", stage)


def emergency_fast_reply(
    subject_address: str = "bạn",
    red_flags: Iterable[str] | str | None = None,
    question: str = "",
    context: Mapping[str, object] | None = None,
) -> str:
    """Deterministic fast reply. Never waits for retrieval.

    The user must see the 115 instruction immediately; first-aid details are
    sent by the aid block.
    """
    return "Đây có thể là tình trạng cấp cứu. Hãy gọi 115 ngay."


def emergency_first_aid_reply(
    question: str,
    red_flags: Iterable[str] | str | None = None,
    subject_address: str = "bạn",
    *,
    timing_callback: Callable[[str, float], None] | None = None,
) -> str:
    """Emergency-corpus checked first-aid block.

    With EMERGENCY_AID_USE_LLM=1, confident retrieved Cấp cứu chunks drive the
    aid details. Deterministic templates are fallback-only, and every result
    passes through the safety post-check.
    """
    retrieval_started = time.perf_counter()
    try:
        hits = retrieve_emergency_aid(question, red_flags=red_flags)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Emergency retrieval failed: %s", exc)
        hits = []
    _emit_timing(timing_callback, "retrieval", retrieval_started)
    generation_started = time.perf_counter()
    try:
        bullets = generate_emergency_aid(
            question,
            hits,
            red_flags=red_flags,
            subject_address=subject_address,
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Emergency generation failed: %s", exc)
        bullets = [
            "Theo các đoạn tài liệu cấp cứu hiện có, hãy gọi 115 và làm theo "
            "hướng dẫn trực tiếp của điều phối viên.",
            "Để bệnh nhân ở tư thế an toàn, dễ thở và có người theo dõi liên tục.",
        ]
    _emit_timing(timing_callback, "generator", generation_started)
    bullets = apply_safety_post_check(
        bullets, question=question, red_flags=red_flags
    )
    if not bullets:
        bullets = [
            "Trong lúc chờ 115, để bệnh nhân ở tư thế an toàn, dễ thở và có người theo dõi.",
        ]
    subject = (subject_address or "bạn").strip() or "bạn"
    heading = "Hướng dẫn sơ cứu ban đầu (theo tài liệu Bạch Mai):"
    action_text = "\n".join(f"- {b}" for b in bullets[:5])
    return f"{heading}\n{action_text}"


def build_emergency_reply(
    subject_address: str = "bạn",
    red_flags: Iterable[str] | str | None = None,
    question: str = "",
    context: Mapping[str, object] | None = None,
    *,
    timing_callback: Callable[[str, float], None] | None = None,
) -> str:
    """Combine the fast reply with the corpus-checked first-aid block.

    The combined format keeps the public contract (single string) stable for
    eval + tests, while exposing a second paragraph the channels can use
    to push a delayed "second message" once retrieval finishes.
    """
    fast = emergency_fast_reply(
        subject_address, red_flags=red_flags, question=question, context=context
    )
    aid = emergency_first_aid_reply(
        question,
        red_flags=red_flags,
        subject_address=subject_address,
        timing_callback=timing_callback,
    )
    return f"{fast}\n\n{aid}"
