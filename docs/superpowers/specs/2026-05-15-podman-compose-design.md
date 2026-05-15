# Podman Compose Design

## Goal

Add a Podman deployment path that mirrors the existing Docker Compose stack
while keeping the two runtime-specific files separate.

## Approach

Create a new `podman-compose.yml` instead of reusing `docker-compose.yml`.
Keeping a dedicated file makes the Podman workflow explicit and leaves room for
Podman-specific changes later without risking the Docker path.

## Services

The Podman stack contains the same services as Docker Compose:

- `redis`
- `qdrant`
- `neo4j`
- `api`
- `ngrok`
- `telegram-webhook`

Service-to-service URLs remain:

```env
REDIS_URL=redis://redis:6379/0
QDRANT_URL=http://qdrant:6333
NEO4J_URI=bolt://neo4j:7687
```

## Data

The stack uses named volumes for:

- `redis-data`
- `qdrant-data`
- `neo4j-data`
- `neo4j-logs`
- `hf-cache`

The first-run data build commands remain the same except for the Podman Compose
prefix:

```bash
podman compose -f podman-compose.yml run --rm api python -m src.rag.kg_builder --clear
podman compose -f podman-compose.yml run --rm api python -m src.rag.build_qdrant
```

## Documentation

`SERVER_SETUP.md` should gain a dedicated Podman section with:

- prerequisite check commands
- the `podman compose -f podman-compose.yml ...` start/stop commands
- first-run KG/Qdrant build commands
- a note that `podman compose` depends on an installed Compose provider

## Out Of Scope

- Replacing Docker Compose
- Adding raw `podman run` commands
- Adding generated systemd units
- Changing application source code
