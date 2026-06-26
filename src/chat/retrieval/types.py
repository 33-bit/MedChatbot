"""
types.py
--------
Shared types for the retrieval subpackage (keeps dense/sparse/rerank/kg
from depending on each other).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


RetrievalScope = Literal["medical", "health_insurance"]


@dataclass
class Hit:
    """Retrieved chunk.

    score is stage-local: dense, sparse, fusion, and rerank each use different
    scales, so consumers should treat it only as ordering metadata from the
    current retrieval stage.
    """

    text: str
    score: float
    source_type: str        # "disease" | "drug" | "health_insurance"
    source_name: str
    heading_path: str
    source_slug: str = ""
    chunk_id: str = ""
    metadata: dict | None = None
    id: str = ""
