"""Phase 1: Data Pipeline Orchestrator.

Runs the full data pipeline:
1. Ingest documents from data/raw/
2. Chunk documents
3. Generate embeddings
4. Build and save FAISS index
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.config import load_config, PROJECT_ROOT
from src.data.chunker import chunk_documents
from src.data.embedder import embed_chunks
from src.data.indexer import build_faiss_index, save_index
from src.data.ingest import ingest_directory


def run_pipeline(domain: str | None = None):
    """Execute the full data pipeline.

    Args:
        domain: Domain name to process (e.g., 'cuad'). If None, uses base.yaml default.
    """
    config = load_config(domain=domain)
    base = config["base"]
    active_domain = base.get("project", {}).get("domain", "default")
    logger.info(f"Running pipeline for domain: {active_domain}")

    # Paths
    raw_dir = PROJECT_ROOT / base["data"]["raw_dir"]
    processed_dir = PROJECT_ROOT / base["data"]["processed_dir"]
    index_dir = PROJECT_ROOT / base["data"]["index_dir"]

    processed_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Ingest
    logger.info("=" * 60)
    logger.info("STEP 1: Document Ingestion")
    logger.info("=" * 60)
    documents = ingest_directory(
        raw_dir,
        supported_formats=base["data"]["supported_formats"],
    )
    if not documents:
        logger.error(f"No documents found in {raw_dir}. Add documents and retry.")
        return

    # Save processed documents
    docs_path = processed_dir / "documents.json"
    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(documents)} documents to {docs_path}")

    # Step 2: Chunk
    logger.info("=" * 60)
    logger.info("STEP 2: Chunking")
    logger.info("=" * 60)
    chunks = chunk_documents(
        documents,
        chunk_size=base["chunking"]["chunk_size"],
        chunk_overlap=base["chunking"]["chunk_overlap"],
        encoding_name=base["chunking"]["tokenizer"],
        strategy=base["chunking"]["strategy"],
    )

    # Save chunks
    chunks_data = [
        {
            "text": c.text,
            "source": c.source,
            "chunk_index": c.chunk_index,
            "token_count": c.token_count,
        }
        for c in chunks
    ]
    chunks_path = processed_dir / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(chunks)} chunks to {chunks_path}")

    # Step 3: Embed
    logger.info("=" * 60)
    logger.info("STEP 3: Embedding Generation")
    logger.info("=" * 60)
    embeddings = embed_chunks(
        chunks,
        model=base["embedding"]["model"],
        dimensions=base["embedding"]["dimensions"],
        batch_size=base["embedding"]["batch_size"],
    )

    # Step 4: Build Index
    logger.info("=" * 60)
    logger.info("STEP 4: FAISS Index Construction")
    logger.info("=" * 60)
    index = build_faiss_index(
        embeddings,
        index_type=base["vector_store"]["index_type"],
    )
    save_index(
        index,
        chunks,
        persist_path=PROJECT_ROOT / base["vector_store"]["persist_path"],
    )

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Documents: {len(documents)}")
    logger.info(f"  Chunks: {len(chunks)}")
    logger.info(f"  Embeddings: {embeddings.shape}")
    logger.info(f"  Index vectors: {index.ntotal}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
