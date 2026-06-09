"""
router.py
---------
PayOS webhook endpoint.

Money-crediting endpoint: the HMAC signature check and idempotent crediting
are hard safety boundaries. A forged or replayed event must never credit a
balance.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, Response

from src.chat.storage.session import reserve_webhook_update
from src.chat.storage.wallet import apply_payment, get_order, mark_order_paid
from src.server.channels import telegram
from src.server.payments.payos import verify_webhook_signature

log = logging.getLogger(__name__)
router = APIRouter()


def _txn_id(data: dict) -> str:
    """Stable per-transaction id for dedupe."""
    for key in ("reference", "id", "paymentLinkId"):
        value = data.get(key)
        if value:
            return str(value)
    return f"order:{data.get('orderCode')}"


async def _notify_paid(account_id: str, amount: int, balance: int) -> None:
    if not account_id.startswith("tg:"):
        return
    chat_id = account_id[len("tg:"):]
    text = (
        f"✅ Đã nhận thanh toán {amount:,} VND.\n"
        f"Số dư hiện tại: {balance:,} VND."
    )
    await telegram.send_text(chat_id, text)


@router.post("/webhook/payos")
async def payos_webhook(request: Request) -> Response:
    payload = await request.json()

    if not verify_webhook_signature(payload):
        log.warning("PayOS webhook signature mismatch")
        return Response(status_code=403)

    data = payload.get("data") or {}
    order_code = data.get("orderCode")
    if order_code is None:
        return Response(status_code=200)

    # Dedupe replays before touching the balance.
    if not reserve_webhook_update("payos", _txn_id(data)):
        log.info("Duplicate PayOS webhook ignored: order=%s", order_code)
        return Response(status_code=200)

    order = get_order(int(order_code))
    if order is None:
        log.warning("PayOS webhook for unknown order %s", order_code)
        return Response(status_code=200)

    if not mark_order_paid(int(order_code)):
        log.info("PayOS order %s already paid; skipping credit", order_code)
        return Response(status_code=200)

    balance = apply_payment(order["account_id"], order["amount"])
    try:
        await _notify_paid(order["account_id"], order["amount"], balance)
    except Exception:
        log.exception("PayOS paid notification failed for order %s", order_code)

    # Remove the QR message so the user can't scan and pay it again. Uses the
    # message id persisted on the order, so it works across server restarts.
    await telegram._delete_order_qr(order)

    return Response(status_code=200)
