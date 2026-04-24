"""
types.py
--------
Shared types for the retrieval subpackage (keeps dense/sparse/rerank/kg
from depending on each other).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Hit:
    text: str
    score: float
    source_type: str        # "disease" | "drug"
    source_name: str
    heading_path: str
    source_slug: str = ""
    chunk_id: str = ""
    metadata: dict | None = None
