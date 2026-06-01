"""PromptoMizer - DSPy RAG Optimization & Evaluation Framework.

Main entry point for running the full pipeline.
Usage:
    python main.py pipeline    # Run data pipeline (Phase 1)
    python main.py evaluate    # Run evaluation (Phase 4)
    python main.py visualize   # Generate figures (Phase 5)
    python main.py all         # Run everything end-to-end
"""

import sys

from loguru import logger


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")

    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1].lower()

    if command == "pipeline":
        from scripts.run_pipeline import run_pipeline
        run_pipeline()
    elif command == "evaluate":
        from scripts.run_evaluation import run_evaluation
        run_evaluation()
    elif command == "visualize":
        from scripts.run_viz import run_viz
        run_viz()
    elif command == "all":
        from scripts.run_pipeline import run_pipeline
        from scripts.run_evaluation import run_evaluation
        from scripts.run_viz import run_viz
        run_pipeline()
        run_evaluation()
        run_viz()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()

