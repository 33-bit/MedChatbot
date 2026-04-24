"""
build_qdrant.py
---------------
Đọc chunks JSONL → embed (multilingual-e5-large) → upsert Qdrant.

Cách dùng:
    python -m src.rag.build_qdrant                  # index cả bệnh lẫn thuốc
    python -m src.rag.build_qdrant --reset           # xóa collection cũ
    python -m src.rag.build_qdrant --drugs           # chỉ index thuốc OTC
    python -m src.rag.build_qdrant --diseases        # chỉ index bệnh

Yêu cầu:
    pip install qdrant-client sentence-transformers
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DISEASES_COLLECTION,
    DRUGS_COLLECTION,
    EMBED_MODEL,
    OUTPUT_DIR,
    QDRANT_API_KEY,
    QDRANT_URL,
)

CHUNKS_DIR = OUTPUT_DIR / "chunks"
DISEASE_CHUNKS = CHUNKS_DIR / "disease_chunks.jsonl"
DRUG_CHUNKS = CHUNKS_DIR / "drug_chunks.jsonl"

EMBED_BATCH = 64
E5_PASSAGE_PREFIX = "passage: "


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if "id" not in c:
                c["id"] = str(uuid.uuid4())
            chunks.append(c)
    return chunks


def embed_chunks(
    model: SentenceTransformer,
    chunks: list[dict],
    model_name: str,
) -> list[list[float]]:
    is_e5 = "e5" in model_name.lower()
    texts = [
        (E5_PASSAGE_PREFIX + c["text"] if is_e5 else c["text"])
        for c in chunks
    ]
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def create_collection(client: QdrantClient, name: str, dim: int, reset: bool) -> None:
    if reset and client.collection_exists(name):
        client.delete_collection(name)
        print(f"  Đã xóa collection '{name}'")

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"  Tạo collection '{name}' (dim={dim})")
    else:
        print(f"  Collection '{name}' đã tồn tại, upsert thêm")


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    BATCH = 100
    for start in range(0, len(chunks), BATCH):
        batch_chunks = chunks[start : start + BATCH]
        batch_embeds = embeddings[start : start + BATCH]
        points = [
            PointStruct(
                id=chunk["id"],
                vector=emb,
                payload={k: v for k, v in chunk.items() if k != "id"},
            )
            for chunk, emb in zip(batch_chunks, batch_embeds)
        ]
        client.upsert(collection_name=collection, points=points)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=EMBED_MODEL)
    parser.add_argument("--reset", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--drugs", action="store_true")
    mode.add_argument("--diseases", action="store_true")
    args = parser.parse_args()

    do_diseases = args.diseases or (not args.drugs and not args.diseases)
    do_drugs = args.drugs or (not args.drugs and not args.diseases)

    print(f"Kết nối Qdrant Cloud tại {QDRANT_URL} ...")
    qclient = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)

    print(f"Loading model: {args.model} ...")
    st_model = SentenceTransformer(args.model)
    dim = st_model.get_embedding_dimension()
    print(f"Embedding dim: {dim}\n")

    if do_diseases:
        if not DISEASE_CHUNKS.exists():
            raise SystemExit(f"Không tìm thấy {DISEASE_CHUNKS}. Chạy `python -m src.rag.chunker` trước.")
        disease_chunks = load_chunks(DISEASE_CHUNKS)
        print(f"[Diseases] {len(disease_chunks)} chunks")
        print("Đang embed ...")
        disease_embeds = embed_chunks(st_model, disease_chunks, args.model)

        create_collection(qclient, DISEASES_COLLECTION, dim, args.reset)
        upsert_chunks(qclient, DISEASES_COLLECTION, disease_chunks, disease_embeds)
        info = qclient.get_collection(DISEASES_COLLECTION)
        print(f"Collection '{DISEASES_COLLECTION}': {info.points_count} points\n")

    if do_drugs:
        if not DRUG_CHUNKS.exists():
            raise SystemExit(f"Không tìm thấy {DRUG_CHUNKS}. Chạy `python -m src.rag.chunker` trước.")
        drug_chunks = load_chunks(DRUG_CHUNKS)
        print(f"[OTC Drugs] {len(drug_chunks)} chunks")
        print("Đang embed ...")
        drug_embeds = embed_chunks(st_model, drug_chunks, args.model)

        create_collection(qclient, DRUGS_COLLECTION, dim, args.reset)
        upsert_chunks(qclient, DRUGS_COLLECTION, drug_chunks, drug_embeds)
        info = qclient.get_collection(DRUGS_COLLECTION)
        print(f"Collection '{DRUGS_COLLECTION}': {info.points_count} points\n")

    print("Xong.")


if __name__ == "__main__":
    main()
