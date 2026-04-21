"""Text chunking and indexing pipeline for financial documents."""

import uuid


def chunk_text(text: str, chunk_size: int = 256, overlap: int = 30) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def process_document(
    text: str,
    doc_title: str,
    doc_type: str,
    doc_id: str | None = None,
) -> list[dict]:
    """Process a text document into chunks with financial metadata.

    Args:
        text: Raw document text.
        doc_title: Human-readable title.
        doc_type: One of firm_policy | client_profile | market_data | malicious
        doc_id: Optional stable ID; generated if not provided.

    Returns:
        List of chunk dicts ready for insert_chunks().
    """
    if doc_id is None:
        doc_id = str(uuid.uuid4())

    chunks = chunk_text(text)

    return [
        {
            "chunk_id": f"{doc_id}_{i}",
            "text": chunk,
            "doc_title": doc_title,
            "doc_type": doc_type,
            "doc_id": doc_id,
        }
        for i, chunk in enumerate(chunks)
    ]
