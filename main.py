"""PromptoMizer - DSPy RAG Optimization & Evaluation Framework.

Main entry point for running the full pipeline.
Usage:
    python main.py prepare                        # Prepare all datasets (Phase 0)
    python main.py prepare --dataset cuad         # Prepare single dataset
    python main.py pipeline [--domain DOMAIN]     # Run data pipeline (Phase 1)
    python main.py evaluate [--domain DOMAIN]     # Run evaluation (Phase 4)
    python main.py visualize                      # Generate figures (Phase 5)
    python main.py all [--domain DOMAIN]          # Run everything end-to-end

Domains: contractnli | cuad | maud | privacy_qa
"""

import argparse
import sys

from loguru import logger


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")

    parser = argparse.ArgumentParser(description="PromptoMizer RAG Framework", add_help=True)
    parser.add_argument("command", choices=["prepare", "pipeline", "evaluate", "visualize", "all"])
    parser.add_argument("--domain", default=None, help="Domain to use (contractnli|cuad|maud|privacy_qa)")
    parser.add_argument("--dataset", default=None, help="Dataset for prepare command")
    args = parser.parse_args()

    if args.command == "prepare":
        from scripts.prepare_datasets import main as prepare_main
        sys.argv = ["prepare_datasets.py"]
        if args.dataset:
            sys.argv += ["--dataset", args.dataset]
        prepare_main()

    elif args.command == "pipeline":
        from scripts.run_pipeline import run_pipeline
        run_pipeline(domain=args.domain)

    elif args.command == "evaluate":
        from scripts.run_evaluation import run_evaluation
        run_evaluation(domain=args.domain)

    elif args.command == "visualize":
        from scripts.run_viz import run_viz
        run_viz()

    elif args.command == "all":
        from scripts.run_pipeline import run_pipeline
        from scripts.run_evaluation import run_evaluation
        from scripts.run_viz import run_viz
        run_pipeline(domain=args.domain)
        run_evaluation(domain=args.domain)
        run_viz()


if __name__ == "__main__":
    main()

