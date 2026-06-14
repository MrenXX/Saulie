#!/usr/bin/env python3
"""
Index mccauley_products_500k.csv into Qdrant amazon_products_v2 (BGE-M3 hybrid).
SERVER_BATCH_SIZE must match TRT engine max batch (2).
"""
import os
import uuid
import numpy as np
import requests
import pandas as pd
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from tqdm import tqdm

_RAG_ROOT = os.getenv("RAG_ROOT", str(Path(__file__).resolve().parent))
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:1234")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "amazon_products_v2")
CSV_PATH = os.getenv("MCCAULEY_CSV", f"{_RAG_ROOT}/mccauley_products_500k.csv")
SERVER_URL = os.getenv("EMBED_URL", "http://localhost:8888/embed")
TEXT_FIELD = "embed_text"

SERVER_BATCH_SIZE = 2  # must match serve.py / TRT engine
UPSERT_BATCH_SIZE = 128
MIN_DENSE_NORM = 0.9


def get_embeddings_from_server(texts: list[str]) -> dict:
    response = requests.post(SERVER_URL, json={"text": texts}, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"Embed server {response.status_code}: {response.text[:500]}")
    return response.json()


def validate_dense_batch(dense_batch: list, batch_offset: int):
    for i, vec in enumerate(dense_batch):
        norm = float(np.linalg.norm(vec))
        if norm < MIN_DENSE_NORM or np.isnan(vec).any():
            raise ValueError(
                f"Invalid dense vector at batch row {i} (offset {batch_offset}): norm={norm}"
            )


def prepare_metadata(row) -> dict:
    asin = str(row.get("parent_asin", "")).strip().upper()

    def safe_float(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0

    def safe_int(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return 0

    return {
        "name": str(row.get("name", "")),
        "main_category": str(row.get("main_category", "other")),
        "sub_category": str(row.get("sub_category", "")),
        "parent_asin": asin,
        "store": str(row.get("store", "")),
        "ratings": safe_float(row.get("ratings")),
        "no_of_ratings": safe_int(row.get("no_of_ratings")),
        "discount_price": safe_float(row.get("discount_price")),
        "actual_price": safe_float(row.get("actual_price")),
    }


def create_collection_hybrid(client: QdrantClient, collection_name: str, dense_dim: int = 1024):
    if client.collection_exists(collection_name):
        print(f"Deleting existing collection '{collection_name}'...")
        client.delete_collection(collection_name)

    print(f"Creating collection '{collection_name}'...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": rest.VectorParams(
                size=dense_dim,
                distance=rest.Distance.COSINE,
                on_disk=False,
                hnsw_config=rest.HnswConfigDiff(m=32, ef_construct=256),
            )
        },
        sparse_vectors_config={
            "sparse": rest.SparseVectorParams(
                index=rest.SparseIndexParams(on_disk=False),
                modifier=None,
            )
        },
        on_disk_payload=False,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="main_category",
        field_schema=rest.PayloadSchemaType.KEYWORD,
        wait=True,
    )
    print("Collection ready.")


def main():
    print("--- McAuley hybrid indexing ---")
    print(f"CSV: {CSV_PATH}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embed batch: {SERVER_BATCH_SIZE}")

    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df[df[TEXT_FIELD].notna() & (df[TEXT_FIELD].astype(str).str.strip() != "")]
    print(f"Rows to index: {len(df):,}")

    client = QdrantClient(url=QDRANT_URL, check_compatibility=False)
    create_collection_hybrid(client, COLLECTION_NAME)

    points_buffer = []
    total = len(df)

    with tqdm(total=total, desc="Indexing") as pbar:
        for i in range(0, total, SERVER_BATCH_SIZE):
            batch_df = df.iloc[i : i + SERVER_BATCH_SIZE]
            texts = batch_df[TEXT_FIELD].astype(str).tolist()

            api_result = get_embeddings_from_server(texts)
            dense_batch = api_result["dense"]
            sparse_batch = api_result["sparse"]
            validate_dense_batch(dense_batch, i)

            for dense, sparse, (_, row) in zip(dense_batch, sparse_batch, batch_df.iterrows()):
                points_buffer.append(
                    rest.PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "dense": dense,
                            "sparse": rest.SparseVector(
                                indices=sparse["indices"],
                                values=sparse["values"],
                            ),
                        },
                        payload=prepare_metadata(row),
                    )
                )

            if len(points_buffer) >= UPSERT_BATCH_SIZE or (i + SERVER_BATCH_SIZE >= total):
                client.upsert(collection_name=COLLECTION_NAME, points=points_buffer)
                points_buffer = []

            pbar.update(len(batch_df))

    info = client.get_collection(COLLECTION_NAME)
    print(f"\n[ok] Indexed {info.points_count:,} points into '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
