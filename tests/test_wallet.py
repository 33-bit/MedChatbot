from __future__ import annotations

from src.chat.storage import wallet


def test_balance_starts_at_zero():
    assert wallet.get_balance("tg:1") == 0


def test_credit_accumulates():
    assert wallet.credit("tg:1", 50_000) == 50_000
    assert wallet.credit("tg:1", 20_000) == 70_000
    assert wallet.get_balance("tg:1") == 70_000


def test_credit_is_per_account():
    wallet.credit("tg:1", 50_000)
    wallet.credit("tg:2", 10_000)
    assert wallet.get_balance("tg:1") == 50_000
    assert wallet.get_balance("tg:2") == 10_000


def test_debit_reduces_balance():
    wallet.credit("tg:3", 10_000)
    assert wallet.debit("tg:3", 4_000) == 6_000
    assert wallet.get_balance("tg:3") == 6_000


def test_debit_allows_negative_balance_for_debt():
    wallet.credit("tg:4", 1_000)
    new_balance = wallet.debit("tg:4", 3_000)
    assert new_balance == -2_000
    assert wallet.get_balance("tg:4") == -2_000


def test_debit_on_fresh_account_goes_negative():
    assert wallet.debit("tg:debt", 5_000) == -5_000


def test_debt_status_clear_when_positive():
    wallet.credit("tg:d1", 10_000)
    status = wallet.debt_status("tg:d1")
    assert status["in_debt"] is False
    assert status["debt"] == 0
    assert status["banned"] is False
    assert status["payoff_amount"] == 0


def test_debt_status_marks_debt_and_sets_debt_since():
    wallet.debit("tg:d2", 8_000)
    status = wallet.debt_status("tg:d2")
    assert status["in_debt"] is True
    assert status["debt"] == 8_000
    assert status["debt_since"] is not None
    assert status["banned"] is False
    # Payoff = debt + 10% penalty.
    assert status["payoff_amount"] == 8_800


def test_debt_since_clears_when_repaid_positive():
    wallet.debit("tg:d3", 5_000)
    assert wallet.debt_status("tg:d3")["in_debt"] is True
    wallet.credit("tg:d3", 5_000)
    status = wallet.debt_status("tg:d3")
    assert status["in_debt"] is False
    assert status["debt_since"] is None


def test_debt_since_anchored_to_first_crossing_not_later_debits():
    import time as _t
    from src.chat.clients import get_sqlite

    wallet.debit("tg:d4", 1_000)
    first_since = wallet.debt_status("tg:d4")["debt_since"]
    assert first_since is not None
    # Backdate so we can detect if a later debit wrongly resets it.
    conn = get_sqlite()
    conn.execute("UPDATE account_balance SET debt_since = ? WHERE account_id = ?", (first_since - 1000, "tg:d4"))
    conn.commit()
    anchored = wallet.debt_status("tg:d4")["debt_since"]
    wallet.debit("tg:d4", 2_000)  # deeper into debt, still same episode
    assert wallet.debt_status("tg:d4")["debt_since"] == anchored


def test_banned_after_grace_window():
    from src.chat.clients import get_sqlite

    wallet.debit("tg:d5", 9_000)
    conn = get_sqlite()
    # Backdate debt_since beyond the grace window.
    conn.execute(
        "UPDATE account_balance SET debt_since = ? WHERE account_id = ?",
        (wallet.debt_status("tg:d5")["debt_since"] - wallet.DEBT_GRACE_SECONDS - 10, "tg:d5"),
    )
    conn.commit()
    status = wallet.debt_status("tg:d5")
    assert status["in_debt"] is True
    assert status["banned"] is True


def test_settle_debt_requires_payoff_and_zeroes_balance():
    wallet.debit("tg:d6", 10_000)
    # Underpayment is rejected.
    assert wallet.settle_debt("tg:d6", 10_000) is False
    assert wallet.debt_status("tg:d6")["in_debt"] is True
    # Exact payoff (debt + 10%) clears to zero; penalty is burned as a fee.
    assert wallet.settle_debt("tg:d6", 11_000) is True
    assert wallet.get_balance("tg:d6") == 0
    assert wallet.debt_status("tg:d6")["in_debt"] is False
    assert wallet.debt_status("tg:d6")["debt_since"] is None


def test_apply_payment_settles_debt_when_covering_payoff():
    wallet.debit("tg:p1", 10_000)  # payoff = 11,000
    balance = wallet.apply_payment("tg:p1", 11_000)
    assert balance == 0
    assert wallet.debt_status("tg:p1")["in_debt"] is False


def test_apply_payment_credits_normally_when_not_in_debt():
    wallet.credit("tg:p2", 5_000)
    balance = wallet.apply_payment("tg:p2", 20_000)
    assert balance == 25_000


def test_apply_payment_partial_reduces_debt_without_clearing():
    wallet.debit("tg:p3", 10_000)  # payoff = 11,000
    # Pays less than payoff: just reduces the debt, episode continues.
    balance = wallet.apply_payment("tg:p3", 4_000)
    assert balance == -6_000
    assert wallet.debt_status("tg:p3")["in_debt"] is True





def test_mark_order_paid_is_idempotent():
    wallet.create_order(1001, "tg:1", 50_000, "plink-1")
    assert wallet.get_order(1001)["status"] == "pending"

    assert wallet.mark_order_paid(1001) is True
    assert wallet.get_order(1001)["status"] == "paid"

    # second call must not re-credit
    assert wallet.mark_order_paid(1001) is False


def test_mark_order_paid_unknown_order():
    assert wallet.mark_order_paid(999999) is False


def test_qr_message_id_persists_on_order():
    wallet.create_order(8001, "tg:9", 10_000, "plink-q")
    assert wallet.get_order(8001)["qr_message_id"] is None
    wallet.set_order_qr_message(8001, 4242)
    assert wallet.get_order(8001)["qr_message_id"] == 4242


def test_paid_order_credits_once_in_realistic_flow():
    wallet.create_order(2002, "tg:5", 100_000, "plink-2")
    # Simulate webhook: only credit when the order actually flips to paid.
    if wallet.mark_order_paid(2002):
        wallet.credit("tg:5", wallet.get_order(2002)["amount"])
    # Duplicate webhook delivery
    if wallet.mark_order_paid(2002):
        wallet.credit("tg:5", wallet.get_order(2002)["amount"])
    assert wallet.get_balance("tg:5") == 100_000


def test_admin_credit_log_records_who_and_what():
    wallet.log_admin_credit(
        order_code=7001,
        admin_user_id=6866285714,
        account_id="tg:42",
        amount=30_000,
    )
    rows = wallet.get_admin_credits()
    assert len(rows) == 1
    row = rows[0]
    assert row["order_code"] == 7001
    assert row["admin_user_id"] == 6866285714
    assert row["account_id"] == "tg:42"
    assert row["amount"] == 30_000
    assert row["created_at"] > 0


def test_admin_credit_log_filter_by_order():
    wallet.log_admin_credit(order_code=7100, admin_user_id=1, account_id="tg:1", amount=10_000)
    wallet.log_admin_credit(order_code=7200, admin_user_id=2, account_id="tg:2", amount=20_000)
    rows = wallet.get_admin_credits(order_code=7200)
    assert len(rows) == 1
    assert rows[0]["admin_user_id"] == 2
