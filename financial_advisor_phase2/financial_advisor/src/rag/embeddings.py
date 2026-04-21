"""Embedding generation using Sentence Transformers."""

import os
from sentence_transformers import SentenceTransformer

_model = None


def get_embedding_model() -> SentenceTransformer:
    """Return a singleton embedding model instance."""
    global _model
    if _model is None:
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        _model = SentenceTransformer(model_name)
    return _model


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts."""
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def get_embedding_dimension() -> int:
    """Return the dimension of the embedding model."""
    model = get_embedding_model()
    return model.get_sentence_embedding_dimension()
