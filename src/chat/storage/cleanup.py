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
from src.chat.storage.doctors import sweep_idle

CONSULT_RETENTION_DAYS = int(os.getenv("CONSULT_RETENTION_DAYS", "30"))
PROFILE_RETENTION_DAYS = int(os.getenv("PROFILE_RETENTION_DAYS", "30"))
TRACE_RETENTION_DAYS = int(os.getenv("TRACE_RETENTION_DAYS", "30"))
FEEDBACK_RETENTION_DAYS = int(os.getenv("FEEDBACK_RETENTION_DAYS", "30"))
SUPERSEDED_PROFILE_RETENTION_DAYS = int(
    os.getenv("SUPERSEDED_PROFILE_RETENTION_DAYS", "30")
)


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

    cutoff = now - TRACE_RETENTION_DAYS * 86400
    cur = conn.execute("DELETE FROM chat_trace WHERE created_at < ?", (cutoff,))
    stats["traces_deleted"] = cur.rowcount

    cutoff = now - FEEDBACK_RETENTION_DAYS * 86400
    cur = conn.execute("DELETE FROM response_feedback WHERE created_at < ?", (cutoff,))
    stats["feedback_deleted"] = cur.rowcount

    cur = conn.execute(
        "DELETE FROM medical_profile_fact "
        "WHERE valid_until IS NOT NULL AND valid_until < ?",
        (now,),
    )
    stats["expired_profile_facts_deleted"] = cur.rowcount

    cutoff = now - SUPERSEDED_PROFILE_RETENTION_DAYS * 86400
    cur = conn.execute(
        "DELETE FROM medical_profile_fact "
        "WHERE superseded_by IS NOT NULL AND updated_at < ?",
        (cutoff,),
    )
    stats["superseded_profile_facts_deleted"] = cur.rowcount

    cur = conn.execute(
        "DELETE FROM medical_profile_subject WHERE NOT EXISTS ("
        "SELECT 1 FROM medical_profile_fact "
        "WHERE medical_profile_fact.owner_id = medical_profile_subject.owner_id "
        "AND medical_profile_fact.subject_id = medical_profile_subject.subject_id)"
    )
    stats["orphaned_profile_subjects_deleted"] = cur.rowcount

    cur = conn.execute("DELETE FROM rate_limit WHERE ts < ?", (now - 120,))
    stats["rate_limit_purged"] = cur.rowcount

    ended = sweep_idle()
    stats["doctor_consultations_ended"] = len(ended)

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
