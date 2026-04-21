"""Seed the Milvus vector store with synthetic financial documents.

Run this once after the stack is up:
    python seed_data.py

By default seeds only firm policies (clean baseline).
Pass --with-injections to also seed the malicious documents (for ASR testing).
"""

import os
import sys
import json
import uuid
import argparse
import httpx

AGENT_URL = os.getenv("AGENT_URL", "http://localhost:8000")


def index_document(text: str, doc_title: str, doc_type: str) -> dict:
    response = httpx.post(
        f"{AGENT_URL}/index-document",
        json={"text": text, "doc_title": doc_title, "doc_type": doc_type},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="Seed financial advisor vector store")
    parser.add_argument(
        "--with-injections",
        action="store_true",
        help="Also index the malicious injection documents (for ASR testing)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the index before seeding",
    )
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(__file__), "data", "synthetic_documents.json")
    with open(data_path) as f:
        data = json.load(f)

    if args.reset:
        print("Resetting index...")
        r = httpx.delete(f"{AGENT_URL}/reset-index", timeout=30.0)
        r.raise_for_status()
        print("  Index reset.")

    print("\n=== Seeding firm policies (clean documents) ===")
    for doc in data["firm_policies"]:
        result = index_document(
            text=doc["text"],
            doc_title=doc["doc_title"],
            doc_type=doc["doc_type"],
        )
        print(f"  ✓ '{doc['doc_title']}' — {result['num_chunks']} chunks")

    if args.with_injections:
        print("\n=== Seeding malicious injection documents (ASR test mode) ===")
        for doc in data["malicious_injection_documents"]:
            result = index_document(
                text=doc["text"],
                doc_title=doc["doc_title"],
                doc_type=doc["doc_type"],
            )
            print(
                f"  ⚠ '{doc['doc_title']}' [{doc['injection_type']}] — {result['num_chunks']} chunks"
            )

    print("\nDone. Run `python seed_data.py --help` for options.")
    stats = httpx.get(f"{AGENT_URL}/stats", timeout=10.0).json()
    print(f"Total chunks in index: {stats.get('count', 'unknown')}")


if __name__ == "__main__":
    main()
