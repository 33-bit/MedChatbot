"""Shared user-facing replies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatReply:
    text: str
    choices: tuple[str, ...] = ()

TECHNICAL_ERROR_REPLY = (
    "Hiện hệ thống đang gặp sự cố kỹ thuật nên tôi chưa thể trả lời chính xác lúc này. "
    "Bạn vui lòng thử lại sau ít phút. Nếu có triệu chứng nặng như khó thở, đau ngực dữ dội, "
    "lơ mơ, yếu liệt, co giật hoặc chảy máu nhiều, hãy gọi cấp cứu 115 hoặc đến cơ sở y tế gần nhất ngay."
)
