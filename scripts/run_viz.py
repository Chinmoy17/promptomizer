"""Phase 5: Visualization Generator.

Loads evaluation results and generates all publication-quality figures.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.config import load_config, PROJECT_ROOT
from src.visualization.plots import generate_all_figures


def run_viz():
    """Generate all visualization figures from evaluation results."""
    config = load_config()
    base = config["base"]

    results_dir = PROJECT_ROOT / "results" / "evaluations"
    output_dir = PROJECT_ROOT / base["visualization"]["output_dir"]

    # Load evaluation results
    results_path = results_dir / "full_results.json"
    if not results_path.exists():
        logger.error(f"No results found at {results_path}. Run evaluation first.")
        return

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    results_df = pd.DataFrame(results)
    logger.info(f"Loaded {len(results_df)} evaluation results")

    # Load efficiency data
    efficiency_path = results_dir / "efficiency_summary.json"
    efficiency_data = []
    if efficiency_path.exists():
        with open(efficiency_path, encoding="utf-8") as f:
            efficiency_data = json.load(f)

    # Generate all figures
    generate_all_figures(
        results_df=results_df,
        efficiency_data=efficiency_data,
        output_dir=output_dir,
    )

    logger.info(f"All figures saved to {output_dir}")


if __name__ == "__main__":
    run_viz()
