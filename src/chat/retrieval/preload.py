"""
preload.py
----------
Startup preloading for local Hugging Face retrieval models.
"""

from __future__ import annotations

import logging

from src.config import (
    EMBED_MODEL,
    HF_OFFLINE_AFTER_PRELOAD,
    HF_PRELOAD_REQUIRED,
    HF_PRELOAD_RETRIEVAL_MODELS,
    RERANKER_MODEL,
    set_hf_offline,
)

log = logging.getLogger(__name__)


def preload_retrieval_models() -> None:
    """Download/load retrieval models at startup, then restore offline mode."""
    if not HF_PRELOAD_RETRIEVAL_MODELS:
        if HF_OFFLINE_AFTER_PRELOAD:
            set_hf_offline(True)
        log.info("HF retrieval model preload disabled.")
        return

    log.info(
        "HF retrieval model preload starting; network allowed temporarily "
        "embed_model=%s reranker_model=%s",
        EMBED_MODEL,
        RERANKER_MODEL,
    )
    set_hf_offline(False)
    try:
        from src.chat.retrieval.dense import preload_models

        preload_models()
        log.info("HF retrieval model preload completed.")
    except Exception as e:
        log.exception("HF retrieval model preload failed: %s", e)
        if HF_PRELOAD_REQUIRED:
            raise
    finally:
        if HF_OFFLINE_AFTER_PRELOAD:
            set_hf_offline(True)
            log.info("HF offline mode enabled after retrieval model preload.")
