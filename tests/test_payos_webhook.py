from __future__ import annotations

import hashlib
import hmac

import pytest

from src.chat.storage import wallet
from src.server.payments import payos


CHECKSUM = "secret-key"


def _sign(data: dict) -> str:
    signed = "&".join(f"{k}={payos._sig_value(v)}" for k, v in sorted(data.items()))
    return hmac.new(CHECKSUM.encode(), signed.encode(), hashlib.sha256).hexdigest()


def _payload(order_code: int, amount: int, ref: str) -> dict:
    data = {
        "orderCode": order_code,
        "amount": amount,
        "reference": ref,
        "code": "00",
        "desc": "success",
    }
    return {"code": "00", "desc": "success", "success": True,
            "data": data, "signature": _sign(data)}


@pytest.fixture
def notify(monkeypatch):
    from src.server.channels import telegram

    sent: list[tuple] = []

    async def fake_send_text(chat_id, text, *args, **kwargs):
        sent.append((chat_id, text))

    monkeypatch.setattr(telegram, "send_text", fake_send_text)
    return sent


@pytest.fixture(autouse=True)
def _checksum(monkeypatch):
    monkeypatch.setattr(payos, "PAYOS_CHECKSUM_KEY", CHECKSUM)


def test_valid_webhook_credits_balance(app_client, notify, monkeypatch):
    client, _ = app_client
    monkeypatch.setattr(
        "src.server.payments.router.verify_webhook_signature",
        lambda p: payos.verify_webhook_signature(p),
    )
    wallet.create_order(1001, "tg:777", 50_000, "plink-1")

    resp = client.post("/webhook/payos", json=_payload(1001, 50_000, "FT001"))

    assert resp.status_code == 200
    assert wallet.get_balance("tg:777") == 50_000
    assert wallet.get_order(1001)["status"] == "paid"
    assert notify and notify[0][0] == "777"


def test_valid_webhook_deletes_tracked_qr_message(app_client, notify, monkeypatch):
    client, _ = app_client
    from src.server.channels import telegram

    deleted: list[tuple] = []

    async def fake_delete(chat_id, message_id):
        deleted.append((chat_id, message_id))

    monkeypatch.setattr(
        "src.server.payments.router.verify_webhook_signature",
        lambda p: payos.verify_webhook_signature(p),
    )
    monkeypatch.setattr(telegram, "_delete_message", fake_delete, raising=False)
    wallet.create_order(1010, "tg:777", 50_000, "plink-x")
    wallet.set_order_qr_message(1010, 4321)

    resp = client.post("/webhook/payos", json=_payload(1010, 50_000, "FT010"))

    assert resp.status_code == 200
    assert wallet.get_balance("tg:777") == 50_000
    # QR message removed using the persisted id (chat derived from account_id).
    assert deleted == [("777", 4321)]


def test_invalid_signature_rejected_no_credit(app_client, notify, monkeypatch):
    client, _ = app_client
    monkeypatch.setattr(
        "src.server.payments.router.verify_webhook_signature",
        lambda p: payos.verify_webhook_signature(p),
    )
    wallet.create_order(1002, "tg:777", 50_000, "plink-2")
    payload = _payload(1002, 50_000, "FT002")
    payload["signature"] = "deadbeef"

    resp = client.post("/webhook/payos", json=payload)

    assert resp.status_code == 403
    assert wallet.get_balance("tg:777") == 0
    assert wallet.get_order(1002)["status"] == "pending"
    assert notify == []


def test_duplicate_webhook_credits_once(app_client, notify, monkeypatch):
    client, _ = app_client
    monkeypatch.setattr(
        "src.server.payments.router.verify_webhook_signature",
        lambda p: payos.verify_webhook_signature(p),
    )
    wallet.create_order(1003, "tg:777", 30_000, "plink-3")
    payload = _payload(1003, 30_000, "FT003")

    first = client.post("/webhook/payos", json=payload)
    second = client.post("/webhook/payos", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert wallet.get_balance("tg:777") == 30_000


def test_unknown_order_acked_no_credit(app_client, notify, monkeypatch):
    client, _ = app_client
    monkeypatch.setattr(
        "src.server.payments.router.verify_webhook_signature",
        lambda p: payos.verify_webhook_signature(p),
    )

    resp = client.post("/webhook/payos", json=_payload(999999, 10_000, "FT999"))

    assert resp.status_code == 200
    assert notify == []
