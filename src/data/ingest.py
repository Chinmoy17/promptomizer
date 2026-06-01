"""Document ingestion module.

Supports: PDF, DOCX, TXT, JSON, Markdown.
"""

import json
from pathlib import Path

from loguru import logger


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from a PDF file."""
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def extract_text_from_docx(file_path: Path) -> str:
    """Extract text from a DOCX file."""
    from docx import Document

    doc = Document(str(file_path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def extract_text_from_txt(file_path: Path) -> str:
    """Extract text from a plain text or markdown file."""
    return file_path.read_text(encoding="utf-8")


def extract_text_from_json(file_path: Path) -> str:
    """Extract text from a JSON file.

    Expects either:
    - A list of objects with a 'text' or 'content' field
    - A single object with a 'text' or 'content' field
    """
    data = json.loads(file_path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        texts = []
        for item in data:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or str(item)
                texts.append(text)
            else:
                texts.append(str(item))
        return "\n\n".join(texts)
    elif isinstance(data, dict):
        return data.get("text") or data.get("content") or json.dumps(data)
    return str(data)


EXTRACTORS = {
    ".pdf": extract_text_from_pdf,
    ".docx": extract_text_from_docx,
    ".txt": extract_text_from_txt,
    ".md": extract_text_from_txt,
    ".json": extract_text_from_json,
}


def ingest_document(file_path: Path) -> dict:
    """Ingest a single document and return structured data.

    Returns:
        dict with keys: 'source', 'text', 'format', 'char_count'
    """
    suffix = file_path.suffix.lower()
    if suffix not in EXTRACTORS:
        logger.warning(f"Unsupported format: {suffix} for {file_path.name}")
        return None

    extractor = EXTRACTORS[suffix]
    text = extractor(file_path)

    return {
        "source": str(file_path.name),
        "text": text,
        "format": suffix,
        "char_count": len(text),
    }


def ingest_directory(
    directory: str | Path,
    supported_formats: list[str] | None = None,
) -> list[dict]:
    """Ingest all supported documents from a directory.

    Args:
        directory: Path to directory containing documents.
        supported_formats: List of extensions to process (e.g., ['.pdf', '.txt']).

    Returns:
        List of document dicts with 'source', 'text', 'format', 'char_count'.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if supported_formats is None:
        supported_formats = list(EXTRACTORS.keys())

    documents = []
    files = sorted(directory.iterdir())

    for file_path in files:
        if file_path.is_file() and file_path.suffix.lower() in supported_formats:
            logger.info(f"Ingesting: {file_path.name}")
            doc = ingest_document(file_path)
            if doc and doc["text"].strip():
                documents.append(doc)

    logger.info(f"Ingested {len(documents)} documents from {directory}")
    return documents
