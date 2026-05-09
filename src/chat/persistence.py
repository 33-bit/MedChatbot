from __future__ import annotations

import logging
import time

from src.chat.storage.session import PatientSession, log_consultation, save_profile, save_session

log = logging.getLogger(__name__)


def persist_final_turn(
    session: PatientSession,
    session_id: str,
    question: str,
    reply: str,
    trace_id: str,
    log_timing,
) -> None:
    stage_start = time.perf_counter()
    try:
        save_session(session)
        save_profile(session)
        log_consultation(session_id, question, reply)
    except Exception:
        log.exception("Pipeline persist failed trace=%s session=%s", trace_id, session_id)
        log_timing(trace_id, "persist", stage_start, failed=True)
        return
    log_timing(trace_id, "persist", stage_start)
