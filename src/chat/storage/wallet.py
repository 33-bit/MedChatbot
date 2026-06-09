"""
wallet.py
---------
Persistent VND balance and top-up order tracking (SQLite).

Balance is durable: it lives outside the Redis session so it survives
session TTL expiry and /new. Crediting is exactly-once, guarded by the
pending->paid transition in mark_order_paid().
"""

from __future__ import annotations

import math
import time
from threading import RLock

from src.chat.clients import get_sqlite

_SQLITE_LOCK = RLock()

# Negative balances are debt. The user keeps chatbot access for this long after
# first going into debt; after that they are banned bot-wide until they pay the
# debt plus a penalty. Clearing the debt to >= 0 ends the episode.
DEBT_GRACE_SECONDS = 10 * 24 * 60 * 60
DEBT_PENALTY_RATE = 0.10


def get_balance(account_id: str) -> int:
    with _SQLITE_LOCK:
        conn = get_sqlite()
        row = conn.execute(
            "SELECT balance FROM account_balance WHERE account_id = ?",
            (account_id,),
        ).fetchone()
    return int(row[0]) if row else 0


def credit(account_id: str, amount: int) -> int:
    """Adjust a balance by amount (negative to debit) and maintain debt_since.

    debt_since anchors the grace/ban window. It is set the moment a balance
    first crosses from >= 0 into negative, preserved while the balance stays
    negative (even on further debits), and cleared when the balance returns to
    >= 0. This keeps the debt episode's start stable across many small debits.
    """
    with _SQLITE_LOCK:
        conn = get_sqlite()
        now = time.time()
        prev = conn.execute(
            "SELECT balance, debt_since FROM account_balance WHERE account_id = ?",
            (account_id,),
        ).fetchone()
        prev_balance = int(prev[0]) if prev else 0
        prev_debt_since = prev[1] if prev else None
        new_balance = prev_balance + amount

        if new_balance < 0:
            debt_since = prev_debt_since if prev_debt_since is not None else now
        else:
            debt_since = None

        conn.execute(
            "INSERT INTO account_balance (account_id, balance, updated_at, debt_since) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(account_id) DO UPDATE SET "
            "balance = excluded.balance, updated_at = excluded.updated_at, "
            "debt_since = excluded.debt_since",
            (account_id, new_balance, now, debt_since),
        )
        conn.commit()
    return new_balance


def debit(account_id: str, amount: int) -> int:
    """Subtract VND from a balance, allowing it to go negative (debt).

    Paid doctor consultations bill pay-as-you-go; when a balance runs out the
    user keeps chatting and the balance goes negative as debt. The grace/ban
    policy is enforced via debt_status(). Returns the new balance.
    """
    return credit(account_id, -int(amount))


def debt_status(account_id: str, now: float | None = None) -> dict:
    """Report debt/grace/ban state for an account.

    Returns:
      in_debt:        balance < 0
      debt:           abs(balance) when in debt, else 0
      debt_since:     timestamp the current debt episode began (or None)
      banned:         in debt AND past the grace window
      grace_ends_at:  debt_since + DEBT_GRACE_SECONDS (or None)
      payoff_amount:  debt + 10% penalty, rounded up (or 0)
    """
    if now is None:
        now = time.time()
    with _SQLITE_LOCK:
        conn = get_sqlite()
        row = conn.execute(
            "SELECT balance, debt_since FROM account_balance WHERE account_id = ?",
            (account_id,),
        ).fetchone()
    balance = int(row[0]) if row else 0
    debt_since = row[1] if row else None
    if balance >= 0:
        return {
            "in_debt": False,
            "debt": 0,
            "debt_since": None,
            "banned": False,
            "grace_ends_at": None,
            "payoff_amount": 0,
        }
    debt = -balance
    grace_ends_at = (debt_since + DEBT_GRACE_SECONDS) if debt_since is not None else None
    banned = debt_since is not None and now > grace_ends_at
    payoff_amount = int(math.ceil(debt * (1 + DEBT_PENALTY_RATE)))
    return {
        "in_debt": True,
        "debt": debt,
        "debt_since": debt_since,
        "banned": banned,
        "grace_ends_at": grace_ends_at,
        "payoff_amount": payoff_amount,
    }


def settle_debt(account_id: str, payment: int) -> bool:
    """Clear a debt if payment covers debt + 10% penalty.

    The penalty is burned as a fee: a successful settlement zeroes the balance
    (it does not leave positive change). Returns False if not in debt or the
    payment is insufficient.
    """
    with _SQLITE_LOCK:
        conn = get_sqlite()
        status = debt_status(account_id)
        if not status["in_debt"]:
            return False
        if int(payment) < status["payoff_amount"]:
            return False
        conn.execute(
            "UPDATE account_balance SET balance = 0, updated_at = ?, debt_since = NULL "
            "WHERE account_id = ?",
            (time.time(), account_id),
        )
        conn.commit()
    return True


def apply_payment(account_id: str, amount: int) -> int:
    """Apply an incoming payment (top-up or debt payoff).

    If the account is in debt and the payment covers the full payoff (debt +
    10% penalty), the debt is settled and the balance returns to exactly 0
    (penalty burned). Otherwise the payment is credited normally — which, for a
    smaller payment against a debt, simply reduces the debt. Returns the
    resulting balance.
    """
    with _SQLITE_LOCK:
        status = debt_status(account_id)
        if status["in_debt"] and int(amount) >= status["payoff_amount"]:
            settle_debt(account_id, amount)
            return 0
        return credit(account_id, int(amount))


def create_order(
    order_code: int,
    account_id: str,
    amount: int,
    payment_link_id: str | None,
) -> None:
    with _SQLITE_LOCK:
        conn = get_sqlite()
        conn.execute(
            "INSERT INTO topup_order "
            "(order_code, account_id, amount, status, payment_link_id, created_at) "
            "VALUES (?, ?, ?, 'pending', ?, ?)",
            (order_code, account_id, amount, payment_link_id, time.time()),
        )
        conn.commit()


def get_order(order_code: int) -> dict | None:
    with _SQLITE_LOCK:
        conn = get_sqlite()
        row = conn.execute(
            "SELECT order_code, account_id, amount, status, payment_link_id, "
            "created_at, paid_at, qr_message_id FROM topup_order WHERE order_code = ?",
            (order_code,),
        ).fetchone()
    if row is None:
        return None
    return {
        "order_code": row[0],
        "account_id": row[1],
        "amount": row[2],
        "status": row[3],
        "payment_link_id": row[4],
        "created_at": row[5],
        "paid_at": row[6],
        "qr_message_id": row[7],
    }


def set_order_qr_message(order_code: int, qr_message_id: int) -> None:
    """Persist the Telegram message_id of the QR so it can be deleted later,
    even across server restarts (the in-memory map does not survive those)."""
    with _SQLITE_LOCK:
        conn = get_sqlite()
        conn.execute(
            "UPDATE topup_order SET qr_message_id = ? WHERE order_code = ?",
            (qr_message_id, order_code),
        )
        conn.commit()


def mark_order_paid(order_code: int) -> bool:
    """Flip a pending order to paid. Return True only on the transition.

    Returns False if the order is unknown or already paid, making this the
    idempotency guard for crediting.
    """
    with _SQLITE_LOCK:
        conn = get_sqlite()
        cursor = conn.execute(
            "UPDATE topup_order SET status = 'paid', paid_at = ? "
            "WHERE order_code = ? AND status = 'pending'",
            (time.time(), order_code),
        )
        conn.commit()
    return cursor.rowcount > 0


def log_admin_credit(
    *,
    order_code: int,
    admin_user_id: int,
    account_id: str,
    amount: int,
) -> None:
    """Record that an admin manually credited an order (audit trail)."""
    with _SQLITE_LOCK:
        conn = get_sqlite()
        conn.execute(
            "INSERT INTO admin_credit "
            "(order_code, admin_user_id, account_id, amount, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (order_code, admin_user_id, account_id, amount, time.time()),
        )
        conn.commit()


def get_admin_credits(order_code: int | None = None) -> list[dict]:
    """Return admin credit log rows, newest first; optionally filtered by order."""
    with _SQLITE_LOCK:
        conn = get_sqlite()
        if order_code is None:
            rows = conn.execute(
                "SELECT order_code, admin_user_id, account_id, amount, created_at "
                "FROM admin_credit ORDER BY created_at DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT order_code, admin_user_id, account_id, amount, created_at "
                "FROM admin_credit WHERE order_code = ? ORDER BY created_at DESC",
                (order_code,),
            ).fetchall()
    return [
        {
            "order_code": r[0],
            "admin_user_id": r[1],
            "account_id": r[2],
            "amount": r[3],
            "created_at": r[4],
        }
        for r in rows
    ]
