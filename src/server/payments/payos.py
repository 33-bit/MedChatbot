"""
payos.py
--------
Thin PayOS client: create a VietQR payment and verify webhook signatures.

Signature scheme (both directions): take the relevant fields, sort keys
alphabetically, join as ``key=value&key2=value2`` (None -> empty string),
then HMAC-SHA256 with the channel checksum key.
"""

from __future__ import annotations

import hashlib
import hmac
import logging

import httpx

from src.config import (
    PAYOS_API_KEY,
    PAYOS_BASE_URL,
    PAYOS_CHECKSUM_KEY,
    PAYOS_CLIENT_ID,
)

log = logging.getLogger(__name__)

# Fields the create-payment signature is computed over, in PayOS's order.
_CREATE_SIGNED_FIELDS = ("amount", "cancelUrl", "description", "orderCode", "returnUrl")


class PayOSError(RuntimeError):
    """Raised when PayOS rejects a request or returns a non-success code."""


def _sig_value(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _parse_emv(s: str) -> dict[str, str]:
    """Parse an EMVCo TLV string into a {tag: value} dict (one level)."""
    out: dict[str, str] = {}
    i = 0
    while i + 4 <= len(s):
        tag = s[i:i + 2]
        try:
            length = int(s[i + 2:i + 4])
        except ValueError:
            break
        value = s[i + 4:i + 4 + length]
        if len(value) != length:
            break
        out[tag] = value
        i += 4 + length
    return out


def extract_transfer_content(qr_code: str) -> str | None:
    """Pull the full transfer memo from a PayOS VietQR string.

    PayOS stores the memo — including the unique reconciliation reference it
    uses to match an incoming bank transfer — in EMV tag 62, sub-tag 08.
    Returns None if the string is not a parseable EMV QR with that field.
    """
    if not qr_code:
        return None
    additional = _parse_emv(qr_code).get("62")
    if not additional:
        return None
    return _parse_emv(additional).get("08")


def _sign(data: dict, key: str) -> str:
    signed = "&".join(f"{k}={_sig_value(v)}" for k, v in sorted(data.items()))
    return hmac.new(key.encode(), signed.encode(), hashlib.sha256).hexdigest()


def _create_payment_signature(
    *,
    order_code: int,
    amount: int,
    description: str,
    cancel_url: str,
    return_url: str,
) -> str:
    data = {
        "amount": amount,
        "cancelUrl": cancel_url,
        "description": description,
        "orderCode": order_code,
        "returnUrl": return_url,
    }
    return _sign(data, PAYOS_CHECKSUM_KEY)


def verify_webhook_signature(payload: dict) -> bool:
    """Return True only if the webhook signature matches its data object."""
    signature = payload.get("signature")
    data = payload.get("data")
    if not signature or not isinstance(data, dict):
        return False
    expected = _sign(data, PAYOS_CHECKSUM_KEY)
    return hmac.compare_digest(expected, str(signature))


def create_payment(
    *,
    order_code: int,
    amount: int,
    description: str,
    return_url: str,
    cancel_url: str,
) -> dict:
    """Create a PayOS payment request and return QR + checkout details."""
    signature = _create_payment_signature(
        order_code=order_code,
        amount=amount,
        description=description,
        cancel_url=cancel_url,
        return_url=return_url,
    )
    body = {
        "orderCode": order_code,
        "amount": amount,
        "description": description,
        "cancelUrl": cancel_url,
        "returnUrl": return_url,
        "signature": signature,
    }
    headers = {
        "x-client-id": PAYOS_CLIENT_ID,
        "x-api-key": PAYOS_API_KEY,
        "Content-Type": "application/json",
    }
    response = httpx.post(
        f"{PAYOS_BASE_URL}/v2/payment-requests",
        json=body,
        headers=headers,
        timeout=20.0,
    )
    if response.status_code >= 400:
        raise PayOSError(f"PayOS HTTP {response.status_code}: {response.text[:300]}")
    parsed = response.json()
    if parsed.get("code") != "00":
        raise PayOSError(f"PayOS error {parsed.get('code')}: {parsed.get('desc')}")
    data = parsed.get("data") or {}
    return {
        "qr_code": data.get("qrCode"),
        "checkout_url": data.get("checkoutUrl"),
        "payment_link_id": data.get("paymentLinkId"),
        "bin": data.get("bin"),
        "account_number": data.get("accountNumber"),
        "account_name": data.get("accountName"),
    }
