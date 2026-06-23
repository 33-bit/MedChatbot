"""Shared user-facing replies."""

from __future__ import annotations

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


EMERGENCY_REPLY = (
    "Triệu chứng bạn đang gặp có thể là dấu hiệu của tình trạng cấp cứu. "
    "Bạn hãy gọi 115 ngay. Nếu không thể chờ xe cấp cứu, hãy nhờ người bên cạnh đưa bạn đến khoa Cấp cứu gần nhất. "
    "Đừng tự lái xe.\n\n"
    "Trong lúc chờ trợ giúp, hãy ngừng vận động, ở tư thế dễ thở và nhờ một người ở bên. "
    "Nếu bạn bất tỉnh hoặc không thở bình thường, người bên cạnh cần gọi 115 và thực hiện "
    "hồi sức tim phổi nếu đã được hướng dẫn."
)


def emergency_reply(subject_address: str = "bạn") -> str:
    subject = subject_address.strip() or "bạn"
    if subject == "bạn":
        return EMERGENCY_REPLY
    return (
        f"Những triệu chứng {subject} đang gặp có thể là dấu hiệu của tình trạng cấp cứu. "
        f"Hãy gọi 115 ngay. Nếu không thể chờ xe cấp cứu, hãy đưa {subject} đến khoa Cấp cứu gần nhất. "
        f"Không để {subject} tự lái xe.\n\n"
        f"Trong lúc chờ trợ giúp, hãy giúp {subject} ngừng vận động, ở tư thế dễ thở và luôn có người ở bên. "
        f"Nếu {subject} bất tỉnh hoặc không thở bình thường, hãy gọi 115 và thực hiện hồi sức tim phổi "
        "nếu đã được hướng dẫn."
    )


TECHNICAL_ERROR_REPLY = (
    "Hiện hệ thống đang gặp sự cố kỹ thuật nên tôi chưa thể trả lời chính xác lúc này. "
    "Bạn vui lòng thử lại sau ít phút. Nếu có triệu chứng nặng như khó thở, đau ngực dữ dội, "
    "lơ mơ, yếu liệt, co giật hoặc chảy máu nhiều, hãy gọi cấp cứu 115 hoặc đến cơ sở y tế gần nhất ngay."
)
