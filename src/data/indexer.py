"""FAISS index construction and persistence."""

import json
from pathlib import Path

import faiss
import numpy as np
from loguru import logger

from src.data.chunker import Chunk


def build_faiss_index(
    embeddings: np.ndarray,
    index_type: str = "flat_l2",
) -> faiss.Index:
    """Build a FAISS index from embeddings.

    Args:
        embeddings: numpy array of shape (n, dim).
        index_type: Type of FAISS index ('flat_l2', 'flat_ip', 'ivf').

    Returns:
        FAISS index.
    """
    n, dim = embeddings.shape

    if index_type == "flat_l2":
        index = faiss.IndexFlatL2(dim)
    elif index_type == "flat_ip":
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dim)
    elif index_type == "ivf":
        nlist = min(int(np.sqrt(n)), 100)
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.train(embeddings)
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    index.add(embeddings)
    logger.info(f"Built FAISS index: type={index_type}, vectors={index.ntotal}, dim={dim}")
    return index


def save_index(
    index: faiss.Index,
    chunks: list[Chunk],
    persist_path: str | Path,
) -> None:
    """Save FAISS index and chunk metadata to disk.

    Creates:
      - {persist_path}.faiss  (the index)
      - {persist_path}_meta.json  (chunk metadata)
    """
    persist_path = Path(persist_path)
    persist_path.parent.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, str(persist_path.with_suffix(".faiss")))

    # Save metadata
    metadata = [
        {
            "text": chunk.text,
            "source": chunk.source,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
        }
        for chunk in chunks
    ]
    meta_path = persist_path.parent / f"{persist_path.stem}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved index to {persist_path.with_suffix('.faiss')}")
    logger.info(f"Saved metadata to {meta_path}")


def load_index(persist_path: str | Path) -> tuple[faiss.Index, list[dict]]:
    """Load FAISS index and chunk metadata from disk.

    Returns:
        Tuple of (faiss_index, metadata_list).
    """
    persist_path = Path(persist_path)

    index = faiss.read_index(str(persist_path.with_suffix(".faiss")))

    meta_path = persist_path.parent / f"{persist_path.stem}_meta.json"
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info(f"Loaded index with {index.ntotal} vectors")
    return index, metadata
