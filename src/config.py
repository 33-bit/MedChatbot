"""
Configuration and shared clients.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Keep Hugging Face / Transformers on the local model cache during normal
# server runs. This avoids per-process metadata checks to huggingface.co.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
_HF_OFFLINE_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def set_hf_offline(enabled: bool) -> None:
    """Toggle HF offline flags for this process, including already-imported modules."""
    value = "1" if enabled else "0"
    for var in _HF_OFFLINE_VARS:
        os.environ[var] = value

    try:
        import huggingface_hub.constants as hf_constants

        hf_constants.HF_HUB_OFFLINE = enabled
    except Exception:
        pass

    try:
        import transformers.utils.hub as transformers_hub

        transformers_hub._is_offline_mode = enabled
    except Exception:
        pass

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ENTITY_DIR = OUTPUT_DIR / "entities"
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
ENTITY_DIR.mkdir(parents=True, exist_ok=True)

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "")

def make_openai_client(api_key: str | None = None):
    if OpenAI is None:
        raise ModuleNotFoundError("Thiếu thư viện `openai`. Cài bằng `pip install openai`.")
    resolved_key = api_key or LLM_API_KEY
    if not resolved_key:
        raise RuntimeError("LLM_API_KEY not configured")
    return OpenAI(api_key=resolved_key, base_url=BASE_URL)


client = make_openai_client() if OpenAI is not None and LLM_API_KEY else None

MODEL = os.getenv("MODEL", "gpt-4.1")
FAST_MODEL = os.getenv("FAST_MODEL", "gpt-4.1-mini")
GUARDRAIL_MODEL = os.getenv("GUARDRAIL_MODEL", FAST_MODEL)
VISION_MODEL = os.getenv("VISION_MODEL", MODEL)
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "4096"))
FAST_MODEL_MAX_TOKENS = int(os.getenv("FAST_MODEL_MAX_TOKENS", "1024"))
GUARDRAIL_MAX_TOKENS = int(os.getenv("GUARDRAIL_MAX_TOKENS", "1024"))
BATCH_MAX_TOKENS = int(os.getenv("BATCH_MAX_TOKENS", "16000"))

# --- Abuse protection ---
GLOBAL_LLM_QUOTA_PER_MINUTE  = int(os.getenv("GLOBAL_LLM_QUOTA_PER_MINUTE", "500"))
SESSION_LLM_QUOTA_PER_DAY    = int(os.getenv("SESSION_LLM_QUOTA_PER_DAY", "100"))
CHAT_API_KEY                 = os.getenv("CHAT_API_KEY", "")

# --- Session & persistence ---
REDIS_URL                = os.getenv("REDIS_URL", "")
SESSION_TTL_SECONDS      = int(os.getenv("SESSION_TTL_SECONDS", "86400"))  # 24h
RATE_LIMIT_PER_MINUTE    = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
SQLITE_PATH              = os.getenv("SQLITE_PATH", str(PROJECT_ROOT / "outputs" / "chatbot.db"))
CONVERSATION_CONTEXT_ENABLED = _env_bool("CONVERSATION_CONTEXT_ENABLED", "0")
PROFILE_READ_ENABLED = _env_bool("PROFILE_READ_ENABLED", "0")
PROFILE_WRITE_ENABLED = _env_bool("PROFILE_WRITE_ENABLED", "0")
PROFILE_REQUIRE_CONSENT = _env_bool(
    "PROFILE_REQUIRE_CONSENT", "1"
)
PROFILE_IDENTITY_ACTIVE_VERSION = os.getenv(
    "PROFILE_IDENTITY_ACTIVE_VERSION",
    os.getenv("PROFILE_IDENTITY_KEY_VERSION", "v1"),
).strip()
PROFILE_IDENTITY_PREVIOUS_VERSIONS = tuple(
    version.strip()
    for version in os.getenv("PROFILE_IDENTITY_PREVIOUS_VERSIONS", "").split(",")
    if version.strip()
)
PROFILE_IDENTITY_HMAC_KEY = os.getenv("PROFILE_IDENTITY_HMAC_KEY", "").strip()
PROFILE_DEFAULT_TENANT_ID = os.getenv("PROFILE_DEFAULT_TENANT_ID", "default").strip()
CHAT_API_TENANT_ID = os.getenv("CHAT_API_TENANT_ID", "").strip()
PROFILE_THIRD_PARTY_TTL_SECONDS = int(
    os.getenv("PROFILE_THIRD_PARTY_TTL_SECONDS", "7776000")
)
PROFILE_SUPERSEDED_RETENTION_SECONDS = int(
    os.getenv("PROFILE_SUPERSEDED_RETENTION_SECONDS", "2592000")
)

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
HEALTH_INSURANCE_COLLECTION = os.getenv(
    "HEALTH_INSURANCE_COLLECTION",
    "health_insurance_law",
)
RAG_TOP_K                = int(os.getenv("RAG_TOP_K", "6"))
RERANKER_MODEL           = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_DEVICE          = os.getenv("RERANKER_DEVICE", "cpu")
RERANK_BATCH_SIZE        = int(os.getenv("RERANK_BATCH_SIZE", "4"))
RERANK_TOP_K             = int(os.getenv("RERANK_TOP_K", "6"))
HYBRID_CANDIDATE_K       = int(os.getenv("HYBRID_CANDIDATE_K", "20"))
RRF_K                    = int(os.getenv("RRF_K", "60"))
E5_QUERY_PREFIX          = "query: "
E5_PASSAGE_PREFIX        = "passage: "
HF_PRELOAD_RETRIEVAL_MODELS = _env_bool("HF_PRELOAD_RETRIEVAL_MODELS", "1")
HF_OFFLINE_AFTER_PRELOAD    = _env_bool("HF_OFFLINE_AFTER_PRELOAD", "1")
HF_PRELOAD_REQUIRED         = _env_bool("HF_PRELOAD_REQUIRED", "0")

# --- Diagnostic narrowing ---
CLARIFICATION_BATCH_SIZE = int(os.getenv("CLARIFICATION_BATCH_SIZE", "4"))
MIN_CANDIDATES_TO_STOP   = int(os.getenv("MIN_CANDIDATES_TO_STOP", "2"))

# --- Guardrail ---
GUARDRAIL_MIN_LEN        = int(os.getenv("GUARDRAIL_MIN_LEN", "3"))
GUARDRAIL_MAX_LEN        = int(os.getenv("GUARDRAIL_MAX_LEN", "2000"))

# --- Messaging channels ---
ZALO_BOT_TOKEN           = os.getenv("ZALO_BOT_TOKEN", os.getenv("ZALO_OA_ACCESS_TOKEN", ""))
ZALO_WEBHOOK_SECRET      = os.getenv(
    "ZALO_WEBHOOK_SECRET",
    os.getenv("ZALO_OA_SECRET_KEY", os.getenv("ZALO_APP_SECRET", "")),
)
ZALO_OA_ACCESS_TOKEN     = os.getenv("ZALO_OA_ACCESS_TOKEN", "")
ZALO_OA_SECRET_KEY       = os.getenv("ZALO_OA_SECRET_KEY", "")
ZALO_APP_SECRET          = os.getenv("ZALO_APP_SECRET", "")

TELEGRAM_BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_WEBHOOK_SECRET  = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")


def _parse_id_set(raw: str) -> set[int]:
    ids: set[int] = set()
    for part in re.split(r"[,\s]+", raw.strip()):
        if part:
            try:
                ids.add(int(part))
            except ValueError:
                pass
    return ids


# Numeric Telegram user ids allowed to run admin commands (e.g. /admin_paid).
# Numeric ids are immutable, unlike usernames.
TELEGRAM_ADMIN_IDS       = _parse_id_set(os.getenv("TELEGRAM_ADMIN_IDS", ""))

MESSENGER_PAGE_TOKEN     = os.getenv("MESSENGER_PAGE_TOKEN", "")
MESSENGER_VERIFY_TOKEN   = os.getenv("MESSENGER_VERIFY_TOKEN", "")
MESSENGER_APP_SECRET     = os.getenv("MESSENGER_APP_SECRET", "")

# --- Payments (PayOS / VietQR) ---
PAYOS_CLIENT_ID          = os.getenv("PAYOS_CLIENT_ID", "")
PAYOS_API_KEY            = os.getenv("PAYOS_API_KEY", "")
PAYOS_CHECKSUM_KEY       = os.getenv("PAYOS_CHECKSUM_KEY", "")
PAYOS_BASE_URL           = os.getenv("PAYOS_BASE_URL", "https://api-merchant.payos.vn")
PUBLIC_BASE_URL          = os.getenv("PUBLIC_BASE_URL", "")
