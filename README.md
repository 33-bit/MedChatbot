# Medical RAG Chatbot

Vietnamese medical chatbot built with FastAPI, retrieval-augmented generation (RAG), diagnostic narrowing, and multi-channel webhook support.

This project is intended for medical information support and experimentation. It is not a replacement for professional diagnosis, emergency care, or treatment decisions.

## Features

- Vietnamese medical Q&A over disease and drug knowledge sources.
- RAG retrieval with dense Qdrant search, local BM25 sparse search, reranking, and citations.
- Neo4j knowledge graph context for medical entity relationships and diagnostic narrowing.
- One-shot LLM analyzer for guardrails, routing, query rewriting, and entity extraction.
- Multi-turn symptom collection before answering diagnostic questions.
- Session, profile, consultation, quota, and webhook dedupe storage using Redis and SQLite.
- FastAPI debug API plus Telegram, Zalo, and Messenger webhook adapters.
- Test suite for pipeline behavior, API auth, channels, retrieval contracts, preload behavior, and storage.

## Architecture

Main request flow:

```text
Client / channel webhook
  -> FastAPI route in src/server/app.py or src/server/channels/
  -> src.chat.answer()
  -> src/chat/pipeline.py
  -> analyzer, diagnosis, retrieval, generator, storage
  -> channel/API response
```

Core modules:

| Path | Purpose |
|---|---|
| `src/server/app.py` | FastAPI app, health check, `/chat`, channel routers, startup preload |
| `src/server/channels/` | Telegram, Zalo, and Messenger webhook adapters |
| `src/chat/pipeline.py` | Main conversation orchestration |
| `src/chat/llm/analyzer.py` | Fast-model guardrail, route, rewrite, and entity analysis |
| `src/chat/diagnosis/` | Symptom normalization and diagnostic narrowing |
| `src/chat/retrieval/` | Dense, sparse, KG, rerank, and retrieval service code |
| `src/chat/llm/generator.py` | Final answer generation with patient context, KG context, and citations |
| `src/chat/storage/` | Redis session state and SQLite persistence |
| `src/processing/` | Offline document, drug, and symptom processing scripts |
| `src/rag/` | Qdrant collection and Neo4j graph build scripts |
| `eval/` | Evaluation dataset and benchmark helpers |
| `tests/` | Automated pytest suite |

## Requirements

- Python 3.11+
- FastAPI and Uvicorn
- Redis for sessions and quotas
- SQLite for profiles, consultations, rate limits, and webhook dedupe rows
- Qdrant for vector retrieval
- Neo4j for knowledge graph context
- OpenAI-compatible LLM endpoint for analyzer and generator calls

Python dependencies live in `requirements.txt`.

## Setup

Create and activate your Python environment, then install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Create a `.env` file with the services you need. Common variables:

```bash
# LLM
LLM_API_KEY=...
BASE_URL=...
MODEL=...
FAST_MODEL=...
GUARDRAIL_MODEL=...

# Redis 
REDIS_URL=...

# Abuse protection
CHAT_API_KEY=...
GLOBAL_LLM_QUOTA_PER_MINUTE=...
SESSION_LLM_QUOTA_PER_DAY=...

# Qdrant
QDRANT_URL=...
QDRANT_API_KEY=...

# Neo4j
NEO4J_URI=...
NEO4J_USER=...
NEO4J_PASSWORD=...

# Telegram
TELEGRAM_BOT_TOKEN=...
TELEGRAM_WEBHOOK_SECRET=...

# Zalo 
ZALO_BOT_TOKEN=...
ZALO_WEBHOOK_SECRET=...

# Messenger 
MESSENGER_PAGE_TOKEN=...
MESSENGER_VERIFY_TOKEN=...
MESSENGER_APP_SECRET=...
```

Hugging Face offline flags default to enabled in `src/config.py`. Retrieval model preload behavior is controlled by:

- `HF_PRELOAD_RETRIEVAL_MODELS`
- `HF_OFFLINE_AFTER_PRELOAD`
- `HF_PRELOAD_REQUIRED`

## Run locally

Start the FastAPI app:

```bash
uvicorn src.server.app:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Debug chat endpoint:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $CHAT_API_KEY" \
  -d '{"question":"Tôi bị ho và sốt nên làm gì?","session_id":"demo"}'
```

`session_id` is required for `/chat`; the server hashes it together with the API key before calling the chat pipeline so users sharing one API key do not share patient context.

## Run with Docker

The Docker Compose setup starts the FastAPI API, exposes it through ngrok, and
runs one-shot Telegram and Zalo `setWebhook` registration containers.

Fill `.env` with your real values. `.env.docker.example` shows the Docker-specific
variables:

```bash
NGROK_AUTHTOKEN=...
NGROK_URL=https://your-static-domain.ngrok-free.dev
TELEGRAM_BOT_TOKEN=...
TELEGRAM_WEBHOOK_SECRET=...
TELEGRAM_DROP_PENDING_UPDATES=true
ZALO_BOT_TOKEN=...
ZALO_WEBHOOK_SECRET=...
```

Then start everything:

```bash
docker compose up --build
```

The public endpoints are:

```text
${NGROK_URL}/health
${NGROK_URL}/chat
${NGROK_URL}/webhook/telegram
${NGROK_URL}/webhook/zalo
${NGROK_URL}/webhook/messenger
```

The `telegram-webhook` and `zalo-webhook` containers exit after registering
webhooks; this is expected. Keep the `api` and `ngrok` containers running.


## Data and RAG pipeline

Source documents and data live under `documents/`. Offline processing scripts live under `src/processing/` and `src/rag/`.

Typical responsibilities:

- `src/processing/bachmai/` extracts and finalizes Bạch Mai disease guideline documents.
- `src/processing/drugs/` scrapes, parses, and extracts OTC drug entities.
- `src/processing/symptom_canon.py` builds canonical symptom data.
- `src/rag/build_qdrant.py` builds Qdrant vector collections.
- `src/rag/kg_builder.py` builds the Neo4j graph.

Prefer dry-run, prepare, or status modes for processing workflows. Do not run live scraping, batch submission, webhook setup, or external service mutation unless you explicitly intend to affect external systems.

## API and channels

The server exposes:

- `GET /health`
- `POST /chat` protected by `CHAT_API_KEY`, with required per-user `session_id`
- Telegram webhook routes
- Zalo webhook routes
- Messenger webhook routes

Channel adapters normalize inbound messages and call the same `src.chat.answer()` pipeline used by the debug API.


## Safety notes

- Answers should provide medical information, not definitive diagnosis.
- Emergency symptoms should trigger urgent-care guidance.
- Prescription drug advice must stay within project guardrails.
- Retrieval dependency failures should not silently become unsafe degraded medical answers.

## Repository layout

```text
.
├── documents/                 # Source medical documents and OTC whitelist
├── eval/                      # Benchmark and evaluation helpers
├── outputs/                   # Local runtime outputs such as SQLite DB
├── src/
│   ├── chat/                  # Core chatbot pipeline
│   ├── processing/            # Offline extraction and data preparation
│   ├── rag/                   # Vector and KG build scripts
│   └── server/                # FastAPI app and channel webhooks
├── tests/                     # pytest suite
├── requirements.txt           # Python dependencies
└── TEST_PLAN.md               # Manual and automated test plan
```
