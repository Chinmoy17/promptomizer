"""FAISS-based retriever shared across all RAG systems.

Ensures fair comparison: all systems use the same retrieval results.
"""

from pathlib import Path

import faiss
import numpy as np
from loguru import logger

from src.data.embedder import embed_query
from src.data.indexer import load_index


class FAISSRetriever:
    """Shared FAISS retriever for all RAG system variants.

    Caches query results to guarantee identical retrieval across systems.
    """

    def __init__(
        self,
        index_path: str | Path,
        embedding_model: str = "text-embedding-3-small",
        embedding_dims: int = 1536,
        top_k: int = 5,
    ):
        self.embedding_model = embedding_model
        self.embedding_dims = embedding_dims
        self.top_k = top_k
        self._cache: dict[str, list[dict]] = {}

        # Load index and metadata
        self.index, self.metadata = load_index(index_path)
        logger.info(
            f"FAISSRetriever initialized: {self.index.ntotal} vectors, top_k={top_k}"
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Retrieve top-k relevant chunks for a query.

        Args:
            query: The search query.
            top_k: Override default top_k.

        Returns:
            List of dicts with 'text', 'source', 'score', 'chunk_index'.
        """
        k = top_k or self.top_k

        # Check cache
        cache_key = f"{query}::{k}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Embed query
        query_embedding = embed_query(
            query,
            model=self.embedding_model,
            dimensions=self.embedding_dims,
        )

        # Search
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append(
                {
                    "text": meta["text"],
                    "source": meta["source"],
                    "chunk_index": meta["chunk_index"],
                    "score": float(dist),
                }
            )

        # Cache results
        self._cache[cache_key] = results
        return results

    def retrieve_texts(self, query: str, top_k: int | None = None) -> list[str]:
        """Retrieve only the text content (convenience method).

        Returns:
            List of chunk text strings.
        """
        results = self.retrieve(query, top_k=top_k)
        return [r["text"] for r in results]

    def clear_cache(self) -> None:
        """Clear the query cache."""
        self._cache.clear()

    @property
    def num_vectors(self) -> int:
        """Number of vectors in the index."""
        return self.index.ntotal
