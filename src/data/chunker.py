"""Text chunking strategies.

Implements configurable chunking with token-based sizing.
"""

from dataclasses import dataclass

import tiktoken
from loguru import logger


@dataclass
class Chunk:
    """A text chunk with metadata."""

    text: str
    source: str
    chunk_index: int
    token_count: int


def get_tokenizer(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """Get tiktoken encoding."""
    return tiktoken.get_encoding(encoding_name)


def chunk_text_recursive(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    encoding_name: str = "cl100k_base",
    separators: list[str] | None = None,
) -> list[str]:
    """Split text into chunks using recursive character splitting.

    Tries to split on larger separators first, falling back to smaller ones.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    enc = get_tokenizer(encoding_name)

    def token_len(s: str) -> int:
        return len(enc.encode(s))

    def split_recursive(text: str, seps: list[str]) -> list[str]:
        if not text:
            return []

        # If text fits in one chunk, return it
        if token_len(text) <= chunk_size:
            return [text]

        # Find the best separator
        separator = seps[-1]
        for sep in seps:
            if sep in text:
                separator = sep
                break

        # Split on separator
        parts = text.split(separator) if separator else list(text)

        chunks = []
        current = ""

        for part in parts:
            candidate = current + separator + part if current else part
            if token_len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If single part exceeds chunk_size, recurse with next separator
                if token_len(part) > chunk_size:
                    remaining_seps = seps[seps.index(separator) + 1 :] if separator in seps else seps[1:]
                    chunks.extend(split_recursive(part, remaining_seps or [""]))
                else:
                    current = part

        if current:
            chunks.append(current)

        return chunks

    raw_chunks = split_recursive(text, separators)

    # Apply overlap
    if chunk_overlap > 0 and len(raw_chunks) > 1:
        overlapped = [raw_chunks[0]]
        for i in range(1, len(raw_chunks)):
            prev_tokens = enc.encode(raw_chunks[i - 1])
            overlap_tokens = prev_tokens[-chunk_overlap:]
            overlap_text = enc.decode(overlap_tokens)
            overlapped.append(overlap_text + raw_chunks[i])
        return overlapped

    return raw_chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    encoding_name: str = "cl100k_base",
    strategy: str = "recursive",
) -> list[Chunk]:
    """Chunk a list of documents.

    Args:
        documents: List of dicts with 'text' and 'source' keys.
        chunk_size: Max tokens per chunk.
        chunk_overlap: Overlap tokens between adjacent chunks.
        encoding_name: Tiktoken encoding name.
        strategy: Chunking strategy ('recursive' or 'fixed').

    Returns:
        List of Chunk objects.
    """
    enc = get_tokenizer(encoding_name)
    all_chunks = []

    for doc in documents:
        text = doc["text"]
        source = doc["source"]

        if strategy == "recursive":
            text_chunks = chunk_text_recursive(
                text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                encoding_name=encoding_name,
            )
        elif strategy == "fixed":
            # Simple fixed-size chunking by tokens
            tokens = enc.encode(text)
            text_chunks = []
            step = chunk_size - chunk_overlap
            for i in range(0, len(tokens), step):
                chunk_tokens = tokens[i : i + chunk_size]
                text_chunks.append(enc.decode(chunk_tokens))
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        for idx, chunk_text in enumerate(text_chunks):
            token_count = len(enc.encode(chunk_text))
            all_chunks.append(
                Chunk(
                    text=chunk_text,
                    source=source,
                    chunk_index=idx,
                    token_count=token_count,
                )
            )

    logger.info(
        f"Chunked {len(documents)} documents into {len(all_chunks)} chunks "
        f"(strategy={strategy}, size={chunk_size}, overlap={chunk_overlap})"
    )
    return all_chunks
