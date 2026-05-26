from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from src.chat import answer
from src.chat.replies import TECHNICAL_ERROR_REPLY


async def answer_and_send(
    text: str,
    recipient_id: str | int,
    session_id: str,
    send_text: Callable[[str | int, str], Awaitable[None]],
    *,
    logger: logging.Logger | None = None,
    channel: str = "Channel",
) -> None:
    try:
        reply = await asyncio.to_thread(answer, text, session_id=session_id)
    except Exception:
        if logger:
            logger.exception("%s answer failed", channel)
        reply = TECHNICAL_ERROR_REPLY

    try:
        await send_text(recipient_id, reply)
    except Exception:
        if logger:
            logger.exception("%s send failed", channel)
