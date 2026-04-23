"""
Configuration and shared clients.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    from xai_sdk import Client
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Client = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ENTITY_DIR = OUTPUT_DIR / "entities"
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
ENTITY_DIR.mkdir(parents=True, exist_ok=True)

XAI_API_KEY = os.getenv("XAI_API_KEY", "")


def make_xai_client(api_key: str | None = None):
    if Client is None:
        raise ModuleNotFoundError("Thiếu thư viện `xai_sdk`. Cài bằng `pip install xai-sdk`.")
    return Client(api_key=api_key or XAI_API_KEY)


client = make_xai_client() if Client is not None else None

MODEL = os.getenv("XAI_MODEL", "grok-4.20-0309-reasoning")
FAST_MODEL = os.getenv("XAI_FAST_MODEL", "grok-4-1-fast-reasoning")
GUARDRAIL_MODEL = os.getenv("XAI_GUARDRAIL_MODEL", "grok-3-mini")
VISION_MODEL = os.getenv("XAI_VISION_MODEL", MODEL)

# --- Abuse protection ---
GLOBAL_LLM_QUOTA_PER_MINUTE  = int(os.getenv("GLOBAL_LLM_QUOTA_PER_MINUTE", "500"))
SESSION_LLM_QUOTA_PER_DAY    = int(os.getenv("SESSION_LLM_QUOTA_PER_DAY", "100"))
CHAT_API_KEY                 = os.getenv("CHAT_API_KEY", "")

# --- Session & cache ---
REDIS_URL                = os.getenv("REDIS_URL", "")
SESSION_TTL_SECONDS      = int(os.getenv("SESSION_TTL_SECONDS", "86400"))  # 24h
RATE_LIMIT_PER_MINUTE    = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
SQLITE_PATH              = os.getenv("SQLITE_PATH", str(PROJECT_ROOT / "outputs" / "chatbot.db"))

QDRANT_URL     = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# --- Neo4j ---
NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# --- Embedding / RAG ---
EMBED_MODEL              = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-large")
DISEASES_COLLECTION      = os.getenv("DISEASES_COLLECTION", "medical_guidelines")
DRUGS_COLLECTION         = os.getenv("DRUGS_COLLECTION", "otc_drugs")
RAG_TOP_K                = int(os.getenv("RAG_TOP_K", "6"))
RERANKER_MODEL           = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_TOP_K             = int(os.getenv("RERANK_TOP_K", "6"))
HYBRID_CANDIDATE_K       = int(os.getenv("HYBRID_CANDIDATE_K", "20"))

# --- Messaging channels ---
ZALO_OA_ACCESS_TOKEN     = os.getenv("ZALO_OA_ACCESS_TOKEN", "")
ZALO_APP_SECRET          = os.getenv("ZALO_APP_SECRET", "")

TELEGRAM_BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_WEBHOOK_SECRET  = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")

MESSENGER_PAGE_TOKEN     = os.getenv("MESSENGER_PAGE_TOKEN", "")
MESSENGER_VERIFY_TOKEN   = os.getenv("MESSENGER_VERIFY_TOKEN", "")
MESSENGER_APP_SECRET     = os.getenv("MESSENGER_APP_SECRET", "")
