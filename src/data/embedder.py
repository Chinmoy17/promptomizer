"""Embedding generation module.

Uses OpenAI text-embedding-3-small by default.
"""

import numpy as np
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

from src.config import get_settings
from src.data.chunker import Chunk


def get_embedding_client() -> OpenAI:
    """Create OpenAI client for embeddings."""
    settings = get_settings()
    kwargs = {"api_key": settings.openai_api_key}
    if settings.openai_api_base:
        kwargs["base_url"] = settings.openai_api_base
    return OpenAI(**kwargs)


def embed_texts(
    texts: list[str],
    model: str = "text-embedding-3-small",
    dimensions: int = 1536,
    batch_size: int = 100,
) -> np.ndarray:
    """Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed.
        model: Embedding model name.
        dimensions: Output embedding dimensions.
        batch_size: Number of texts per API call.

    Returns:
        numpy array of shape (len(texts), dimensions).
    """
    client = get_embedding_client()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model=model,
            dimensions=dimensions,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    embeddings = np.array(all_embeddings, dtype=np.float32)
    logger.info(f"Generated {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")
    return embeddings


def embed_chunks(
    chunks: list[Chunk],
    model: str = "text-embedding-3-small",
    dimensions: int = 1536,
    batch_size: int = 100,
) -> np.ndarray:
    """Generate embeddings for a list of Chunk objects.

    Args:
        chunks: List of Chunk objects.
        model: Embedding model name.
        dimensions: Output dimensions.
        batch_size: Batch size for API calls.

    Returns:
        numpy array of shape (len(chunks), dimensions).
    """
    texts = [chunk.text for chunk in chunks]
    return embed_texts(texts, model=model, dimensions=dimensions, batch_size=batch_size)


def embed_query(
    query: str,
    model: str = "text-embedding-3-small",
    dimensions: int = 1536,
) -> np.ndarray:
    """Embed a single query string.

    Returns:
        numpy array of shape (1, dimensions).
    """
    client = get_embedding_client()
    response = client.embeddings.create(
        input=[query],
        model=model,
        dimensions=dimensions,
    )
    return np.array([response.data[0].embedding], dtype=np.float32)
