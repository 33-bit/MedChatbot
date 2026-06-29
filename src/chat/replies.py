"""Shared user-facing replies."""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatReply:
    text: str
    choices: tuple[str, ...] = ()
    selection_mode: str = "single"
    suggest_mode: str | None = None
    retry_question: str | None = None
    doctor_offer: bool = False
    doctor_specialty: str | None = None


_CHILD_TERMS = (
    "be",
    "be trai",
    "be gai",
    "chau",
    "con toi",
    "con ban",
    "tre",
    "tre em",
)


def _normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFD", str(value or ""))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D").casefold()


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def _is_child_context(question: str, context: Mapping[str, object] | None) -> bool:
    pieces = [question]
    if context:
        subject = context.get("subject")
        if isinstance(subject, Mapping):
            pieces.extend(
                str(subject.get(key) or "")
                for key in ("relationship", "display_name", "id")
            )
    signal = _normalize_text(" ".join(pieces))
    return _contains_any(signal, _CHILD_TERMS)


def _add_action(actions: list[str], action: str) -> None:
    if action not in actions:
        actions.append(action)


def _red_flag_text(red_flags: Iterable[str] | str | None) -> str:
    if red_flags is None:
        return ""
    if isinstance(red_flags, str):
        return red_flags
    return " ".join(str(flag) for flag in red_flags)


def _avoid_food_drink_and_meds_action(subject: str) -> str:
    if subject == "bạn":
        return "Ghi lại thời điểm khởi phát triệu chứng; không ăn uống hoặc tự dùng thuốc."
    return (
        f"Ghi lại thời điểm khởi phát triệu chứng; không cho {subject} ăn uống "
        "hoặc tự dùng thuốc."
    )


def _avoid_abdominal_meds_action(subject: str) -> str:
    if subject == "bạn":
        return "Không tự dùng thuốc giảm đau hoặc ăn uống trong lúc chờ được đánh giá."
    return (
        f"Không tự ý cho {subject} dùng thuốc giảm đau hoặc ăn uống "
        "trong lúc chờ được đánh giá."
    )


def _emergency_actions(
    subject: str,
    red_flags: Iterable[str] | str | None,
    question: str,
    context: Mapping[str, object] | None,
) -> list[str]:
    flags = _red_flag_text(red_flags)
    signal = _normalize_text(f"{question} {flags}")
    child_context = _is_child_context(question, context)
    actions: list[str] = []

    if _contains_any(
        signal,
        (
            "dot than",
            "than suoi",
            "phong kin",
            "khi doc",
            "ngo doc khi co",
            "ngo doc co",
            "carbon monoxide",
        ),
    ):
        _add_action(
            actions,
            (
                f"Nếu làm được an toàn, đưa {subject} ra nơi thoáng khí "
                "và tránh để người hỗ trợ tiếp tục hít khí độc."
            ),
        )

    if _contains_any(
        signal,
        (
            "khong tho",
            "ngung tho",
            "khong tho binh thuong",
            "ngung tim",
            "ngung tuan hoan",
            "khong bat duoc mach",
        ),
    ):
        _add_action(
            actions,
            (
                f"Nếu {subject} không thở bình thường, thực hiện ép tim ngoài "
                "lồng ngực nếu đã được hướng dẫn và dùng AED nếu có sẵn."
            ),
        )

    if _contains_any(
        signal,
        (
            "dot quy",
            "meo mieng",
            "noi ngong",
            "noi kho",
            "yeu liet",
            "liet nua nguoi",
            "yeu nua nguoi",
            "te liet nua nguoi",
        ),
    ):
        _add_action(actions, _avoid_food_drink_and_meds_action(subject))

    if _contains_any(signal, ("co giat", "dong kinh", "giat toan than")):
        _add_action(
            actions,
            (
                f"Đặt {subject} nằm nghiêng an toàn, bảo vệ khỏi va đập, "
                "không nhét bất cứ thứ gì vào miệng và theo dõi thời gian co giật."
            ),
        )

    has_allergy = _contains_any(
        signal,
        ("phan ve", "di ung", "noi me day", "sung moi", "sung luoi", "hai san"),
    )
    has_breathing_problem = _contains_any(
        signal,
        ("kho tho", "kho khe", "tim tai", "nghet tho"),
    )
    if _contains_any(signal, ("phan ve",)) or (has_allergy and has_breathing_problem):
        _add_action(
            actions,
            f"Nếu {subject} có bút tiêm epinephrine đã được kê sẵn, dùng theo hướng dẫn đi kèm.",
        )
        _add_action(
            actions,
            f"Để {subject} ở tư thế dễ thở và theo dõi nhịp thở, ý thức.",
        )

    if _contains_any(
        signal,
        (
            "dau nguc",
            "that nguc",
            "lan len ham",
            "lan tay trai",
            "nhoi mau co tim",
            "hoi chung vanh",
        ),
    ):
        if child_context:
            _add_action(
                actions,
                f"Để {subject} nghỉ, tránh gắng sức trong lúc chờ trợ giúp.",
            )
        elif subject == "bạn":
            _add_action(actions, "Hãy nghỉ ngơi, tránh gắng sức và đừng tự lái xe.")
        else:
            _add_action(
                actions,
                f"Để {subject} nghỉ, tránh gắng sức và không để {subject} tự lái xe.",
            )

    if (
        _contains_any(
            signal,
            ("kho tho", "tim tai", "moi tim", "khong nam duoc", "tho rit"),
        )
        and not has_allergy
    ):
        _add_action(
            actions,
            f"Để {subject} ở tư thế dễ thở, nới lỏng quần áo và theo dõi nhịp thở, ý thức.",
        )

    if _contains_any(signal, ("dau bung", "bung cung", "bung cung nhu go")):
        _add_action(actions, _avoid_abdominal_meds_action(subject))

    if not actions:
        _add_action(
            actions,
            f"Để {subject} nghỉ ở tư thế an toàn, dễ thở và luôn có người theo dõi.",
        )
    return actions[:3]


def emergency_fast_reply(
    subject_address: str = "bạn",
    red_flags: Iterable[str] | str | None = None,
    question: str = "",
    context: Mapping[str, object] | None = None,
) -> str:
    """Deterministic emergency lead. Never waits for retrieval.

    This is the message that must reach the user immediately while the
    emergency aid handler is still building the first-aid block.
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

    When EMERGENCY_AID_USE_LLM=1, confident retrieved Cấp cứu chunks drive the
    aid details. Deterministic templates are fallback-only, and every result is
    guarded by the safety post-check.
    """
    try:
        from src.chat.emergency.handler import emergency_first_aid_reply as _impl
        return _impl(
            question,
            red_flags=red_flags,
            subject_address=subject_address,
            timing_callback=timing_callback,
        )
    except Exception:
        return (
            "Hướng dẫn sơ cứu ban đầu (theo tài liệu Bạch Mai):\n"
            "- Trong lúc chờ 115, để bệnh nhân ở tư thế an toàn, dễ thở và có người theo dõi."
        )


def emergency_reply(
    subject_address: str = "bạn",
    red_flags: Iterable[str] | str | None = None,
    question: str = "",
    context: Mapping[str, object] | None = None,
    *,
    use_rag: bool = False,
    timing_callback: Callable[[str, float], None] | None = None,
) -> str:
    """Backward-compatible emergency reply.

    Default behaviour (use_rag=False) preserves the legacy deterministic
    output for tests/eval. When use_rag=True the reply becomes the combined
    fast + emergency-corpus checked first-aid format used by the production
    pipeline.
    """
    if not use_rag:
        return emergency_fast_reply(subject_address, red_flags, question, context)
    try:
        from src.chat.emergency.handler import build_emergency_reply as _impl
        return _impl(
            subject_address,
            red_flags=red_flags,
            question=question,
            context=context,
            timing_callback=timing_callback,
        )
    except Exception:
        return emergency_fast_reply(subject_address, red_flags, question, context)


TECHNICAL_ERROR_REPLY = (
    "Hiện hệ thống đang gặp sự cố kỹ thuật nên tôi chưa thể trả lời chính xác lúc này. "
    "Bạn vui lòng thử lại sau ít phút. Nếu có triệu chứng nặng như khó thở, đau ngực dữ dội, "
    "lơ mơ, yếu liệt, co giật hoặc chảy máu nhiều, hãy gọi cấp cứu 115 hoặc đến cơ sở y tế gần nhất ngay."
)
