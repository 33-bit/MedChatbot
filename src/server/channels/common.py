from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable

from src.chat import answer_with_choices
from src.chat.replies import ChatReply, TECHNICAL_ERROR_REPLY


def _supports_preliminary_reply(fn: Callable) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return (
        "on_preliminary_reply" in sig.parameters
        or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    )


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
    loop = asyncio.get_running_loop()

    def send_preliminary_reply(message: str) -> None:
        future = asyncio.run_coroutine_threadsafe(
            send_text(recipient_id, message, ()),
            loop,
        )
        try:
            future.result()
        except Exception:
            if logger:
                logger.exception("%s preliminary send failed", channel)

    try:
        answer_kwargs = {"session_id": session_id}
        if owner_id:
            answer_kwargs["owner_id"] = owner_id
        if _supports_preliminary_reply(answer_with_choices):
            answer_kwargs["on_preliminary_reply"] = send_preliminary_reply
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
