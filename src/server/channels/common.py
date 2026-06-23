from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from src.chat import answer_with_choices
from src.chat.replies import ChatReply, TECHNICAL_ERROR_REPLY


async def answer_and_send(
    text: str,
    recipient_id: str | int,
    session_id: str,
    send_text: Callable[[str | int, str, tuple[str, ...]], Awaitable[None]],
    *,
    owner_id: str | None = None,
    logger: logging.Logger | None = None,
    channel: str = "Channel",
) -> None:
    try:
        answer_kwargs = {"session_id": session_id}
        if owner_id:
            answer_kwargs["owner_id"] = owner_id
        reply = await asyncio.to_thread(
            answer_with_choices,
            text,
            **answer_kwargs,
        )
    except Exception:
        if logger:
            logger.exception("%s answer failed", channel)
        reply = ChatReply(TECHNICAL_ERROR_REPLY)

    try:
        await send_text(recipient_id, reply.text, reply.choices)
    except Exception:
        if logger:
            logger.exception("%s send failed", channel)
