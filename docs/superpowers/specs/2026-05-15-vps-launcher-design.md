# VPS Launcher Design

## Goal

Provide one command for a VPS without container support:

```bash
python3 run_vps.py
```

The launcher should start the chatbot stack, bootstrap missing local data, expose
the API through ngrok, and register the Telegram webhook.

## Scope

The launcher manages:

- local Redis
- local Neo4j
- local Qdrant
- FastAPI
- ngrok
- Telegram webhook registration

It does not replace the existing application modules, duplicate chatbot logic,
or bundle source data into one file. The project files, `.env`, Python
dependencies, and generated `outputs/` assets remain separate.

## Configuration

The launcher reads `.env` through the existing project configuration. Required
local service defaults:

```env
REDIS_URL=redis://127.0.0.1:6379/0
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
QDRANT_URL=http://127.0.0.1:6333
QDRANT_API_KEY=
NGROK_AUTHTOKEN=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_WEBHOOK_SECRET=...
```

The VPS must already have Python dependencies, Redis, Neo4j, Qdrant, and ngrok
installed. The launcher is an orchestrator, not a package installer.

## Startup Flow

1. Load `.env` and validate required settings.
2. Detect whether Redis is already reachable.
3. If Redis is not reachable, start local Redis and remember that the launcher
   owns that process.
4. Detect whether Neo4j is already reachable.
5. If Neo4j is not reachable, start local Neo4j and remember ownership.
6. Detect whether Qdrant is already reachable.
7. If Qdrant is not reachable, start local Qdrant with a persistent storage
   directory and remember ownership.
8. Check whether the Neo4j graph contains KG nodes.
9. If the KG is empty, run:

   ```bash
   python3 -m src.rag.kg_builder --clear
   ```

10. Check both configured Qdrant collections:
    - `medical_guidelines`
    - `otc_drugs`
11. If either collection is missing or has zero points, build only the missing
    side:

    ```bash
    python3 -m src.rag.build_qdrant --diseases
    python3 -m src.rag.build_qdrant --drugs
    ```

12. Start FastAPI on `0.0.0.0:8000`.
13. Wait until `/health` returns success.
14. Start ngrok for port `8000`.
15. Read the public HTTPS URL from ngrok's local API.
16. Register the Telegram webhook with that URL and the configured secret.
17. Keep the launcher running until interrupted.

## Shutdown Flow

On `Ctrl+C`, the launcher stops:

- ngrok if it started ngrok
- FastAPI if it started FastAPI
- Qdrant if it started Qdrant
- Neo4j if it started Neo4j
- Redis if it started Redis

If a service was already running before launch, the launcher leaves it running.
Shutdown should happen in reverse startup order.

## Failure Behavior

- Missing required configuration stops startup with a clear message.
- A service that fails readiness checks stops startup before later services run.
- Missing source files for KG or Qdrant builders stop startup with the builder's
  existing error message.
- Telegram webhook registration failure stops startup because the one-command
  VPS workflow is not complete without a reachable webhook.
- If startup fails after the launcher created child processes, it cleans up only
  the processes it owns.

## Data Persistence

- Redis may be ephemeral unless the VPS Redis configuration enables persistence.
- Neo4j stores data in its normal local database directory.
- Qdrant uses a local persistent storage directory so collections survive
  launcher restarts.
- SQLite remains at the configured `SQLITE_PATH`.

## Testing Strategy

- Unit tests cover service ownership decisions, empty/non-empty KG detection,
  missing/non-empty Qdrant collection decisions, webhook URL construction, and
  cleanup behavior.
- Process-launching tests use mocks; they do not start real services.
- Manual verification on a VPS runs:

  ```bash
  python3 run_vps.py
  curl http://127.0.0.1:8000/health
  ```

  Then verifies that Telegram receives a message and that a second run skips KG
  and Qdrant rebuilds when data already exists.
