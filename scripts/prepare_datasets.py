"""Dataset preparation script.

Converts benchmark JSONs from data/data_source/ into:
  1. data/raw/{dataset}/          — one .txt file per unique source document
  2. data/ground_truth/{dataset}_qa.json — Q&A pairs for evaluation

Source format (each JSON):
    {
        "tests": [
            {
                "query": "...",
                "snippets": [
                    {"file_path": "dataset/doc.txt", "span": [...], "answer": "..."},
                    ...
                ]
            }
        ]
    }

Run:
    python scripts/prepare_datasets.py
    python scripts/prepare_datasets.py --dataset cuad   # single dataset
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent

DATASETS = ["contractnli", "cuad", "maud", "privacy_qa"]


def sanitize_filename(name: str) -> str:
    """Convert a file path string to a safe filename stem."""
    # Take the basename, strip extension, replace unsafe chars
    stem = Path(name).stem
    stem = re.sub(r"[^\w\-]", "_", stem)
    return stem[:120]  # cap length


def prepare_dataset(name: str) -> dict:
    """Prepare a single dataset.

    Args:
        name: Dataset name (e.g., 'cuad').

    Returns:
        Stats dict.
    """
    source_path = PROJECT_ROOT / "data" / "data_source" / f"{name}.json"
    if not source_path.exists():
        logger.error(f"Source file not found: {source_path}")
        return {}

    with open(source_path, encoding="utf-8") as f:
        data = json.load(f)

    tests = data["tests"]
    logger.info(f"[{name}] Loaded {len(tests)} test items")

    # --- Step 1: Build document corpus ---
    # Group all snippets by their source file_path
    doc_snippets: dict[str, list[str]] = defaultdict(list)
    for item in tests:
        for snippet in item["snippets"]:
            file_path = snippet["file_path"]
            answer_text = snippet.get("answer", "").strip()
            if answer_text and answer_text not in doc_snippets[file_path]:
                doc_snippets[file_path].append(answer_text)

    # Write one .txt per unique source document
    raw_dir = PROJECT_ROOT / "data" / "raw" / name
    raw_dir.mkdir(parents=True, exist_ok=True)

    for file_path, passages in doc_snippets.items():
        safe_name = sanitize_filename(file_path) + ".txt"
        out_path = raw_dir / safe_name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(passages))

    logger.info(f"[{name}] Wrote {len(doc_snippets)} document files to {raw_dir}")

    # --- Step 2: Build ground truth Q&A pairs ---
    qa_pairs = []
    for item in tests:
        question = item["query"].strip()
        # Combine all snippet answers as the ground truth answer
        answers = [s.get("answer", "").strip() for s in item["snippets"] if s.get("answer", "").strip()]
        if not answers:
            continue

        combined_answer = " ".join(answers)
        source_docs = list({s["file_path"] for s in item["snippets"]})

        qa_pairs.append({
            "question": question,
            "answer": combined_answer,
            "source_docs": source_docs,
        })

    gt_dir = PROJECT_ROOT / "data" / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_path = gt_dir / f"{name}_qa.json"

    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    logger.info(f"[{name}] Wrote {len(qa_pairs)} Q&A pairs to {gt_path}")

    return {
        "dataset": name,
        "tests": len(tests),
        "unique_docs": len(doc_snippets),
        "qa_pairs": len(qa_pairs),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark datasets for PromptoMizer")
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default=None,
        help="Prepare a single dataset (default: all)",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")

    targets = [args.dataset] if args.dataset else DATASETS

    logger.info("=" * 60)
    logger.info("Dataset Preparation")
    logger.info("=" * 60)

    all_stats = []
    for name in targets:
        stats = prepare_dataset(name)
        if stats:
            all_stats.append(stats)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    total_qa = 0
    total_docs = 0
    for s in all_stats:
        logger.info(
            f"  {s['dataset']:15s} | {s['unique_docs']:4d} docs | {s['qa_pairs']:5d} Q&A pairs"
        )
        total_qa += s["qa_pairs"]
        total_docs += s["unique_docs"]
    logger.info(f"  {'TOTAL':15s} | {total_docs:4d} docs | {total_qa:5d} Q&A pairs")
    logger.info("=" * 60)
    logger.info("Next: python main.py pipeline --domain <dataset_name>")


if __name__ == "__main__":
    main()
