"""Dataset preparation script.

Generates ground truth Q&A files from benchmark JSONs in data/data_source/.
The actual corpus documents (data/raw/{dataset}/) must already be present.

Only Q&A pairs whose source document exists in data/raw/{dataset}/ are included.
Use --limit to cap the number of pairs (default: 25 for initial testing).

Source JSON format:
    {
        "tests": [
            {
                "query": "...",
                "snippets": [
                    {"file_path": "dataset/doc.txt", "span": [...], "answer": "..."},
                ]
            }
        ]
    }

Run:
    python scripts/prepare_datasets.py                          # all datasets, 25 pairs each
    python scripts/prepare_datasets.py --dataset cuad           # single dataset
    python scripts/prepare_datasets.py --limit 0                # no limit (all pairs)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent

DATASETS = ["contractnli", "cuad", "maud", "privacy_qa"]


def prepare_dataset(name: str, limit: int = 25) -> dict:
    """Build Q&A ground truth for a single dataset.

    Matches each test item's source file_path against files actually present
    in data/raw/{name}/. Only pairs with an existing corpus file are included.

    Args:
        name: Dataset name (e.g., 'cuad').
        limit: Max number of Q&A pairs to emit. 0 = no limit.

    Returns:
        Stats dict.
    """
    source_path = PROJECT_ROOT / "data" / "data_source" / f"{name}.json"
    raw_dir = PROJECT_ROOT / "data" / "raw" / name

    if not source_path.exists():
        logger.error(f"Source file not found: {source_path}")
        return {}

    if not raw_dir.exists():
        logger.warning(f"Corpus directory missing: {raw_dir} — skipping {name}")
        return {}

    # Build a lookup of filenames actually present in the corpus
    existing_files = {f.name for f in raw_dir.iterdir() if f.is_file()}
    logger.info(f"[{name}] Found {len(existing_files)} corpus files in {raw_dir}")

    with open(source_path, encoding="utf-8") as f:
        data = json.load(f)

    tests = data["tests"]
    logger.info(f"[{name}] Loaded {len(tests)} test items from benchmark JSON")

    qa_pairs = []
    skipped_missing = 0

    for item in tests:
        if limit and len(qa_pairs) >= limit:
            break

        question = item["query"].strip()
        if not question:
            continue

        # Check that at least one source document exists in the corpus
        resolved_sources = []
        for snippet in item["snippets"]:
            doc_name = Path(snippet["file_path"]).name
            if doc_name in existing_files:
                resolved_sources.append(doc_name)

        if not resolved_sources:
            skipped_missing += 1
            continue

        # Ground truth answer: concatenate all snippet answers
        answers = [
            s.get("answer", "").strip()
            for s in item["snippets"]
            if s.get("answer", "").strip()
        ]
        combined_answer = " ".join(answers) if answers else ""

        qa_pairs.append({
            "question": question,
            "answer": combined_answer,
            "source_docs": list(dict.fromkeys(resolved_sources)),  # deduplicated, ordered
        })

    if skipped_missing:
        logger.warning(f"[{name}] Skipped {skipped_missing} items (source doc not in corpus)")

    # Save
    gt_dir = PROJECT_ROOT / "data" / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_path = gt_dir / f"{name}_qa.json"

    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    label = f"{len(qa_pairs)}" + (" (limited)" if limit and len(qa_pairs) == limit else "")
    logger.info(f"[{name}] Wrote {label} Q&A pairs → {gt_path}")

    return {
        "dataset": name,
        "corpus_files": len(existing_files),
        "benchmark_tests": len(tests),
        "qa_pairs": len(qa_pairs),
        "skipped_missing": skipped_missing,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark Q&A ground truth for PromptoMizer")
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default=None,
        help="Prepare a single dataset (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Max Q&A pairs per dataset. 0 = no limit (default: 25)",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")

    targets = [args.dataset] if args.dataset else DATASETS
    limit_label = str(args.limit) if args.limit else "unlimited"

    logger.info("=" * 60)
    logger.info(f"Dataset Preparation  (limit={limit_label} pairs per dataset)")
    logger.info("=" * 60)

    all_stats = []
    for name in targets:
        stats = prepare_dataset(name, limit=args.limit)
        if stats:
            all_stats.append(stats)

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for s in all_stats:
        logger.info(
            f"  {s['dataset']:15s} | {s['corpus_files']:4d} corpus docs "
            f"| {s['qa_pairs']:4d} Q&A pairs"
            + (f" | {s['skipped_missing']} skipped" if s["skipped_missing"] else "")
        )
    logger.info("=" * 60)

    if args.limit:
        logger.info(f"Tip: re-run with --limit 0 to use all available pairs")
    logger.info("Next: python main.py pipeline --domain <dataset_name>")


if __name__ == "__main__":
    main()

