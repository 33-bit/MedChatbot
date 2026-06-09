from __future__ import annotations

import hashlib
import hmac

import pytest

from src.server.payments import payos


def _expected_sig(data: dict, key: str) -> str:
    items = sorted(data.items(), key=lambda kv: kv[0])
    signed = "&".join(f"{k}={payos._sig_value(v)}" for k, v in items)
    return hmac.new(key.encode(), signed.encode(), hashlib.sha256).hexdigest()


def test_sig_value_normalizes_none_and_scalars():
    assert payos._sig_value(None) == ""
    assert payos._sig_value(123) == "123"
    assert payos._sig_value("abc") == "abc"


def test_extract_transfer_content_reads_emv_tag_62():
    # Real PayOS qr_code shape: tag 62 -> sub-tag 08 holds the full memo,
    # including PayOS's unique reconciliation reference.
    qr = (
        "00020101021238570010A00000072701270006970422011355555230220050208QRIBFTTA"
        "53037045405100005802VN62270823CSAEY62NIP0 NAPTIENTEST6304C629"
    )
    assert payos.extract_transfer_content(qr) == "CSAEY62NIP0 NAPTIENTEST"


def test_extract_transfer_content_returns_none_when_absent():
    assert payos.extract_transfer_content("not-an-emv-string") is None
    assert payos.extract_transfer_content("") is None


def test_create_signature_uses_sorted_fields(monkeypatch):
    monkeypatch.setattr(payos, "PAYOS_CHECKSUM_KEY", "secret-key")
    sig = payos._create_payment_signature(
        order_code=123,
        amount=50_000,
        description="NAPTIEN123",
        cancel_url="https://x/cancel",
        return_url="https://x/return",
    )
    expected = _expected_sig(
        {
            "amount": 50_000,
            "cancelUrl": "https://x/cancel",
            "description": "NAPTIEN123",
            "orderCode": 123,
            "returnUrl": "https://x/return",
        },
        "secret-key",
    )
    assert sig == expected


def test_verify_webhook_signature_accepts_valid(monkeypatch):
    monkeypatch.setattr(payos, "PAYOS_CHECKSUM_KEY", "secret-key")
    data = {"orderCode": 123, "amount": 50_000, "reference": "FT123", "code": "00"}
    payload = {"data": data, "signature": _expected_sig(data, "secret-key")}
    assert payos.verify_webhook_signature(payload) is True


def test_verify_webhook_signature_rejects_tampered(monkeypatch):
    monkeypatch.setattr(payos, "PAYOS_CHECKSUM_KEY", "secret-key")
    data = {"orderCode": 123, "amount": 50_000}
    payload = {"data": data, "signature": _expected_sig(data, "secret-key")}
    payload["data"]["amount"] = 999_999  # tamper after signing
    assert payos.verify_webhook_signature(payload) is False


def test_verify_webhook_signature_rejects_missing_signature(monkeypatch):
    monkeypatch.setattr(payos, "PAYOS_CHECKSUM_KEY", "secret-key")
    assert payos.verify_webhook_signature({"data": {"orderCode": 1}}) is False


def test_create_payment_posts_and_parses(monkeypatch):
    monkeypatch.setattr(payos, "PAYOS_CHECKSUM_KEY", "secret-key")
    monkeypatch.setattr(payos, "PAYOS_CLIENT_ID", "cid")
    monkeypatch.setattr(payos, "PAYOS_API_KEY", "akey")
    monkeypatch.setattr(payos, "PAYOS_BASE_URL", "https://api.test")

    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "code": "00",
                "desc": "success",
                "data": {
                    "checkoutUrl": "https://pay.test/web/abc",
                    "qrCode": "00020101021238...",
                    "paymentLinkId": "plink-xyz",
                    "status": "PENDING",
                    "bin": "970415",
                    "accountNumber": "113366668888",
                    "accountName": "CHATBOT MEDICAL",
                },
            }

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr(payos.httpx, "post", fake_post)

    result = payos.create_payment(
        order_code=123,
        amount=50_000,
        description="NAPTIEN123",
        return_url="https://x/return",
        cancel_url="https://x/cancel",
    )

    assert result == {
        "qr_code": "00020101021238...",
        "checkout_url": "https://pay.test/web/abc",
        "payment_link_id": "plink-xyz",
        "bin": "970415",
        "account_number": "113366668888",
        "account_name": "CHATBOT MEDICAL",
    }
    assert captured["url"] == "https://api.test/v2/payment-requests"
    assert captured["headers"]["x-client-id"] == "cid"
    assert captured["headers"]["x-api-key"] == "akey"
    assert captured["json"]["signature"] == _expected_sig(
        {
            "amount": 50_000,
            "cancelUrl": "https://x/cancel",
            "description": "NAPTIEN123",
            "orderCode": 123,
            "returnUrl": "https://x/return",
        },
        "secret-key",
    )


def test_create_payment_raises_on_error_code(monkeypatch):
    monkeypatch.setattr(payos, "PAYOS_CHECKSUM_KEY", "secret-key")
    monkeypatch.setattr(payos, "PAYOS_CLIENT_ID", "cid")
    monkeypatch.setattr(payos, "PAYOS_API_KEY", "akey")

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"code": "20", "desc": "invalid", "data": None}

    monkeypatch.setattr(payos.httpx, "post", lambda *a, **k: FakeResponse())

    with pytest.raises(payos.PayOSError):
        payos.create_payment(
            order_code=1,
            amount=10_000,
            description="x",
            return_url="https://x/r",
            cancel_url="https://x/c",
        )
