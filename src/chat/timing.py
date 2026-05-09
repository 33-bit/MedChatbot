"""
timing.py
---------
Small helpers for consistent stage timing logs.
"""

from __future__ import annotations

import logging
import time


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _fields_suffix(fields: dict) -> str:
    if not fields:
        return ""
    return " " + " ".join(f"{key}={value}" for key, value in fields.items())


def log_stage_timing(
    logger: logging.Logger,
    namespace: str,
    stage: str,
    start: float,
    **fields,
) -> None:
    logger.info(
        "%s timing stage=%s ms=%.1f%s",
        namespace,
        stage,
        elapsed_ms(start),
        _fields_suffix(fields),
    )


def log_trace_timing(
    logger: logging.Logger,
    namespace: str,
    trace_id: str,
    stage: str,
    start: float,
    **fields,
) -> None:
    logger.info(
        "%s timing trace=%s stage=%s ms=%.1f%s",
        namespace,
        trace_id,
        stage,
        elapsed_ms(start),
        _fields_suffix(fields),
    )
