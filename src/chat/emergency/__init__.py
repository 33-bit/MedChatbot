"""Emergency-specific retrieval + first-aid generation.

The emergency route is split into a deterministic fast reply (call 115) and a
slower emergency-corpus checked first-aid block. When EMERGENCY_AID_USE_LLM=1,
retrieved Cấp cứu chunks are the normal source of aid details; deterministic
templates are fallback-only, and the safety post-check injects mandatory actions.
"""

from src.chat.emergency.corpus import (
    EMERGENCY_HEADING_SIGNALS,
    EMERGENCY_TEXT_SIGNALS,
    is_emergency_chunk,
    load_emergency_corpus,
)
from src.chat.emergency.generation import (
    EMERGENCY_AID_SYSTEM_PROMPT,
    build_emergency_aid_prompt,
    generate_emergency_aid,
)
from src.chat.emergency.handler import (
    build_emergency_reply,
    emergency_first_aid_reply,
    emergency_fast_reply,
)
from src.chat.emergency.intents import classify_emergency_intent
from src.chat.emergency.retrieval import (
    EmergencyHit,
    retrieve_emergency_aid,
)
from src.chat.emergency.safety import apply_safety_post_check

__all__ = [
    "EMERGENCY_AID_SYSTEM_PROMPT",
    "EMERGENCY_HEADING_SIGNALS",
    "EMERGENCY_TEXT_SIGNALS",
    "EmergencyHit",
    "apply_safety_post_check",
    "build_emergency_aid_prompt",
    "build_emergency_reply",
    "classify_emergency_intent",
    "emergency_fast_reply",
    "emergency_first_aid_reply",
    "generate_emergency_aid",
    "is_emergency_chunk",
    "load_emergency_corpus",
    "retrieve_emergency_aid",
]
