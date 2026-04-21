"""Vector database retrieval logic for financial documents using Milvus."""

import os
import logging
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)
from src.rag.embeddings import generate_embeddings, get_embedding_dimension

logger = logging.getLogger(__name__)

COLLECTION_NAME = "financial_docs"

_connected = False


def get_milvus_connection():
    """Connect to Milvus running in Docker."""
    global _connected
    if _connected:
        return

    host = os.getenv("MILVUS_HOST", "localhost")
    port = os.getenv("MILVUS_PORT", "19530")
    connections.connect("default", host=host, port=port)
    logger.info(f"Connected to Milvus at {host}:{port}")
    _connected = True


def create_collection(index_type: str = "HNSW") -> Collection:
    """Create the financial_docs collection."""
    get_milvus_connection()
    dim = get_embedding_dimension()

    if utility.has_collection(COLLECTION_NAME):
        collection = Collection(COLLECTION_NAME)
        collection.load()
        return collection

    fields = [
        FieldSchema(name="id",          dtype=DataType.VARCHAR, is_primary=True, max_length=128),
        FieldSchema(name="text",        dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="doc_title",   dtype=DataType.VARCHAR, max_length=512),
        # doc_type: firm_policy | client_profile | market_data | malicious
        FieldSchema(name="doc_type",    dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="doc_id",      dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="embedding",   dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]

    schema = CollectionSchema(fields=fields, description="Financial advisor documents")
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 256},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()
    logger.info(f"Created collection '{COLLECTION_NAME}'")
    return collection


def insert_chunks(chunks: list[dict]) -> None:
    """Insert document chunks into Milvus."""
    get_milvus_connection()

    if not utility.has_collection(COLLECTION_NAME):
        create_collection()

    collection = Collection(COLLECTION_NAME)

    texts = [c["text"] for c in chunks]
    embeddings = generate_embeddings(texts)

    data = [
        [c["chunk_id"]  for c in chunks],
        texts,
        [c["doc_title"] for c in chunks],
        [c["doc_type"]  for c in chunks],
        [c["doc_id"]    for c in chunks],
        embeddings,
    ]

    collection.insert(data)
    collection.flush()
    logger.info(f"Inserted {len(chunks)} chunks into '{COLLECTION_NAME}'")


def search_documents(
    query: str,
    doc_type: str | None = None,
    top_k: int = 6,
) -> list[dict]:
    """Search for relevant financial document chunks."""
    get_milvus_connection()

    if not utility.has_collection(COLLECTION_NAME):
        return []

    collection = Collection(COLLECTION_NAME)
    collection.load()

    query_embedding = generate_embeddings([query])
    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

    expr = None
    if doc_type and doc_type != "All":
        expr = f'doc_type == "{doc_type}"'

    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=["text", "doc_title", "doc_type", "doc_id"],
    )

    search_results = []
    for hits in results:
        for hit in hits:
            search_results.append(
                {
                    "text":      hit.entity.get("text"),
                    "doc_title": hit.entity.get("doc_title"),
                    "doc_type":  hit.entity.get("doc_type"),
                    "doc_id":    hit.entity.get("doc_id"),
                    "score":     hit.score,
                }
            )

    return search_results


def get_collection_stats() -> dict:
    """Get statistics about the financial_docs collection."""
    get_milvus_connection()

    if not utility.has_collection(COLLECTION_NAME):
        return {"exists": False, "count": 0}

    collection = Collection(COLLECTION_NAME)
    collection.flush()
    return {"exists": True, "count": collection.num_entities}


def drop_collection() -> None:
    """Drop the collection."""
    get_milvus_connection()
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        logger.info(f"Dropped collection '{COLLECTION_NAME}'")
