from __future__ import annotations

from collections.abc import Awaitable, Callable

from src.chat import answer


async def answer_and_send(
    text: str,
    recipient_id: str | int,
    session_id: str,
    send_text: Callable[[str | int, str], Awaitable[None]],
) -> None:
    reply = answer(text, session_id=session_id)
    await send_text(recipient_id, reply)
