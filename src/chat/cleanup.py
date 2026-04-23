"""
cleanup.py
----------
Daily maintenance:
  - Delete consultations older than CONSULT_RETENTION_DAYS
  - Delete patient_profile rows not updated in PROFILE_RETENTION_DAYS
  - Purge rate_limit rows older than 2 minutes (safety net)
  - VACUUM to reclaim space

Run via cron:
    0 3 * * *  cd /path/to/Chatbot && python -m src.chat.cleanup
"""

from __future__ import annotations

import argparse
import os
import time

from src.chat.session import _db

CONSULT_RETENTION_DAYS = int(os.getenv("CONSULT_RETENTION_DAYS", "30"))
PROFILE_RETENTION_DAYS = int(os.getenv("PROFILE_RETENTION_DAYS", "30"))


def run(verbose: bool = True) -> dict:
    now = time.time()
    conn = _db()
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
    conn.execute("VACUUM")

    if verbose:
        for k, v in stats.items():
            print(f"  {k}: {v}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run(verbose=not args.quiet)


if __name__ == "__main__":
    main()
