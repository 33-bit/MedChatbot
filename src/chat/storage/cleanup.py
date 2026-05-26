"""
cleanup.py
----------
Daily maintenance:
  - Delete consultations older than CONSULT_RETENTION_DAYS
  - Delete patient_profile rows not updated in PROFILE_RETENTION_DAYS
  - Purge rate_limit rows older than 2 minutes (safety net)

Run via cron:
    0 3 * * *  cd /path/to/Chatbot && python -m src.chat.storage.cleanup

Run VACUUM only during API downtime:
    python -m src.chat.storage.cleanup --vacuum
"""

from __future__ import annotations

import argparse
import os
import time

from src.chat.clients import get_sqlite

CONSULT_RETENTION_DAYS = int(os.getenv("CONSULT_RETENTION_DAYS", "30"))
PROFILE_RETENTION_DAYS = int(os.getenv("PROFILE_RETENTION_DAYS", "30"))


def run(verbose: bool = True, vacuum: bool = False) -> dict:
    now = time.time()
    conn = get_sqlite()
    stats = {}

    cutoff = now - CONSULT_RETENTION_DAYS * 86400
    cur = conn.execute("DELETE FROM consultations WHERE created_at < ?", (cutoff,))
    stats["consultations_deleted"] = cur.rowcount

    cutoff = now - PROFILE_RETENTION_DAYS * 86400
    cur = conn.execute("DELETE FROM patient_profile WHERE updated_at < ?", (cutoff,))
    stats["profiles_deleted"] = cur.rowcount

    cur = conn.execute("DELETE FROM rate_limit WHERE ts < ?", (now - 120,))
    stats["rate_limit_purged"] = cur.rowcount

    conn.commit()
    if vacuum:
        conn.execute("VACUUM")

    if verbose:
        for k, v in stats.items():
            print(f"  {k}: {v}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--vacuum",
        action="store_true",
        help="Run SQLite VACUUM. Use only when the API is stopped.",
    )
    args = parser.parse_args()
    run(verbose=not args.quiet, vacuum=args.vacuum)


if __name__ == "__main__":
    main()
