"""
One-command launcher for VPS deployments without container support.

Usage:
    python3 run_server.py [--reload]
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import httpx
import redis
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

from src.chat.retrieval.kg import ensure_fulltext_indexes
from src.config import (
    DISEASES_COLLECTION,
    DRUGS_COLLECTION,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    PROJECT_ROOT,
    QDRANT_API_KEY,
    QDRANT_URL,
    REDIS_URL,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_WEBHOOK_SECRET,
    ZALO_BOT_TOKEN,
    ZALO_WEBHOOK_SECRET,
)

log = logging.getLogger(__name__)

FASTAPI_HOST = os.getenv("VPS_HOST", "0.0.0.0")
FASTAPI_PORT = int(os.getenv("VPS_PORT", "8000"))
NGROK_API_URL = os.getenv("NGROK_API_URL", "http://127.0.0.1:4040/api/tunnels")
NGROK_URL = os.getenv("NGROK_URL", "")
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN", "")
TELEGRAM_DROP_PENDING_UPDATES = os.getenv("TELEGRAM_DROP_PENDING_UPDATES", "true")
QDRANT_STORAGE_PATH = Path(
    os.getenv("QDRANT_STORAGE_PATH", str(PROJECT_ROOT / "outputs" / "qdrant-storage"))
)
STARTUP_TIMEOUT_SECONDS = float(os.getenv("VPS_STARTUP_TIMEOUT_SECONDS", "120"))
PROCESS_STOP_TIMEOUT_SECONDS = float(os.getenv("VPS_STOP_TIMEOUT_SECONDS", "10"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the chatbot server and supporting services for VPS deployments."
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Start FastAPI with code reload enabled.",
    )
    return parser.parse_args(argv)


def _command_from_env(name: str, default: str) -> list[str]:
    return shlex.split(os.getenv(name, default))


def validate_config() -> None:
    required = {
        "REDIS_URL": REDIS_URL,
        "NEO4J_URI": NEO4J_URI,
        "NEO4J_PASSWORD": NEO4J_PASSWORD,
        "QDRANT_URL": QDRANT_URL,
        "NGROK_AUTHTOKEN": NGROK_AUTHTOKEN,
    }
    if telegram_configured():
        required.update(
            {
                "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
                "TELEGRAM_WEBHOOK_SECRET": TELEGRAM_WEBHOOK_SECRET,
            }
        )
    if zalo_configured():
        required.update(
            {
                "ZALO_BOT_TOKEN": ZALO_BOT_TOKEN,
                "ZALO_WEBHOOK_SECRET": ZALO_WEBHOOK_SECRET,
            }
        )
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError("Missing required .env values: " + ", ".join(missing))
    if not telegram_configured() and not zalo_configured():
        raise RuntimeError("At least one messaging channel must be configured: Telegram or Zalo")


def telegram_configured() -> bool:
    return bool(TELEGRAM_BOT_TOKEN or TELEGRAM_WEBHOOK_SECRET)


def zalo_configured() -> bool:
    return bool(ZALO_BOT_TOKEN or ZALO_WEBHOOK_SECRET)


def wait_until_ready(name: str, is_ready: Callable[[], bool]) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if is_ready():
            return
        time.sleep(1)
    raise RuntimeError(f"{name} did not become ready within {STARTUP_TIMEOUT_SECONDS:.0f}s")


def ensure_service(
    name: str,
    is_ready: Callable[[], bool],
    start_process: Callable[[], subprocess.Popen],
    owned_processes: list[subprocess.Popen],
) -> None:
    if is_ready():
        log.info("%s already running.", name)
        return
    process = start_process()
    owned_processes.append(process)
    wait_until_ready(name, is_ready)
    log.info("%s started by launcher.", name)


def start_owned_process(
    start_process: Callable[[], subprocess.Popen],
    owned_processes: list[subprocess.Popen],
) -> subprocess.Popen:
    process = start_process()
    owned_processes.append(process)
    return process


def stop_owned_processes(processes: list[subprocess.Popen]) -> None:
    for process in reversed(processes):
        if process.poll() is not None:
            continue
        process.terminate()
        try:
            process.wait(timeout=PROCESS_STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=PROCESS_STOP_TIMEOUT_SECONDS)


def _start_process(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.Popen:
    log.info("Starting process: %s", " ".join(command))
    return subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
    )


def redis_ready() -> bool:
    try:
        client = redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        return bool(client.ping())
    except (redis.RedisError, OSError, ValueError):
        return False


def start_redis() -> subprocess.Popen:
    return _start_process(
        _command_from_env("REDIS_START_COMMAND", "redis-server --appendonly yes")
    )


def neo4j_ready() -> bool:
    try:
        with GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            connection_timeout=2,
        ) as driver:
            driver.verify_connectivity()
        return True
    except Exception:
        return False


def start_neo4j() -> subprocess.Popen:
    return _start_process(_command_from_env("NEO4J_START_COMMAND", "neo4j console"))


def qdrant_ready() -> bool:
    try:
        QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None, timeout=2).get_collections()
        return True
    except Exception:
        return False


def start_qdrant() -> subprocess.Popen:
    QDRANT_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["QDRANT__STORAGE__STORAGE_PATH"] = str(QDRANT_STORAGE_PATH)
    return _start_process(_command_from_env("QDRANT_START_COMMAND", "qdrant"), env=env)


def _kg_node_count() -> int:
    query = (
        "MATCH (n) "
        "WHERE n:Disease OR n:Drug OR n:Symptom OR n:Chapter "
        "RETURN count(n) AS count"
    )
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        with driver.session() as session:
            row = session.run(query).single()
            return int(row["count"]) if row else 0


def kg_needs_build() -> bool:
    return _kg_node_count() == 0


def _qdrant_collection_points(collection_name: str) -> int:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
    if not client.collection_exists(collection_name):
        return 0
    info = client.get_collection(collection_name)
    return int(info.points_count or 0)


def qdrant_build_modes() -> list[str]:
    modes: list[str] = []
    if _qdrant_collection_points(DISEASES_COLLECTION) == 0:
        modes.append("--diseases")
    if _qdrant_collection_points(DRUGS_COLLECTION) == 0:
        modes.append("--drugs")
    return modes


def _run_module(module: str, *args: str) -> None:
    command = [sys.executable, "-m", module, *args]
    log.info("Running bootstrap command: %s", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def ensure_bootstrap_data() -> None:
    if kg_needs_build():
        _run_module("src.rag.kg_builder", "--clear")
    ensure_fulltext_indexes()

    for mode in qdrant_build_modes():
        _run_module("src.rag.build_qdrant", mode)


def fastapi_ready() -> bool:
    try:
        response = httpx.get(f"http://127.0.0.1:{FASTAPI_PORT}/health", timeout=2)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def start_fastapi(*, reload: bool = False) -> subprocess.Popen:
    command = [
        sys.executable,
        "run.py",
        "--host",
        FASTAPI_HOST,
        "--port",
        str(FASTAPI_PORT),
    ]
    if reload:
        command.append("--reload")
    return _start_process(command)


def start_ngrok() -> subprocess.Popen:
    command = ["ngrok", "http"]
    if NGROK_URL:
        command.extend(["--url", NGROK_URL])
    command.append(str(FASTAPI_PORT))
    env = os.environ.copy()
    env["NGROK_AUTHTOKEN"] = NGROK_AUTHTOKEN
    return _start_process(command, env=env)


def ngrok_public_url() -> str:
    response = httpx.get(NGROK_API_URL, timeout=5)
    response.raise_for_status()
    tunnels = response.json().get("tunnels", [])
    for tunnel in tunnels:
        public_url = tunnel.get("public_url", "")
        if public_url.startswith("https://"):
            return public_url
    raise RuntimeError("ngrok did not expose an HTTPS tunnel")


def wait_for_ngrok_public_url() -> str:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            return ngrok_public_url()
        except (httpx.HTTPError, RuntimeError, ValueError):
            time.sleep(1)
    raise RuntimeError(f"ngrok did not expose a public URL within {STARTUP_TIMEOUT_SECONDS:.0f}s")


def telegram_webhook_url(public_url: str) -> str:
    return public_url.rstrip("/") + "/webhook/telegram"


def zalo_webhook_url(public_url: str) -> str:
    return public_url.rstrip("/") + "/webhook/zalo"


def register_telegram_webhook(public_url: str) -> None:
    webhook_url = telegram_webhook_url(public_url)
    response = httpx.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook",
        data={
            "url": webhook_url,
            "secret_token": TELEGRAM_WEBHOOK_SECRET,
            "drop_pending_updates": TELEGRAM_DROP_PENDING_UPDATES,
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError(f"Telegram setWebhook failed: {payload}")
    log.info("Telegram webhook registered: %s", webhook_url)


def register_zalo_webhook(public_url: str) -> None:
    webhook_url = zalo_webhook_url(public_url)
    response = httpx.post(
        f"https://bot-api.zaloplatforms.com/bot{ZALO_BOT_TOKEN}/setWebhook",
        data={
            "url": webhook_url,
            "secret_token": ZALO_WEBHOOK_SECRET,
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError(f"Zalo setWebhook failed: {payload}")
    log.info("Zalo webhook registered: %s", webhook_url)


def wait_forever() -> None:
    while True:
        time.sleep(3600)


def main(*, reload: bool = False) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    validate_config()
    owned_processes: list[subprocess.Popen] = []
    try:
        ensure_service("Redis", redis_ready, start_redis, owned_processes)
        ensure_service("Neo4j", neo4j_ready, start_neo4j, owned_processes)
        ensure_service("Qdrant", qdrant_ready, start_qdrant, owned_processes)
        ensure_bootstrap_data()
        ensure_service(
            "FastAPI",
            fastapi_ready,
            lambda: start_fastapi(reload=reload),
            owned_processes,
        )
        start_owned_process(start_ngrok, owned_processes)
        public_url = wait_for_ngrok_public_url()
        if telegram_configured():
            register_telegram_webhook(public_url)
        if zalo_configured():
            register_zalo_webhook(public_url)
        log.info("Server ready: %s", public_url)
        wait_forever()
    except KeyboardInterrupt:
        log.info("Shutdown requested.")
    finally:
        stop_owned_processes(owned_processes)


if __name__ == "__main__":
    args = parse_args()
    main(reload=args.reload)
