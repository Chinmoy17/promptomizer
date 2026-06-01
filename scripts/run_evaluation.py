"""Phase 4: Evaluation Runner.

Runs all 4 RAG systems on the test set and evaluates with RAGAS metrics.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from loguru import logger

from src.config import load_config, get_settings, PROJECT_ROOT
from src.evaluation.efficiency import EfficiencyTracker, estimate_cost
from src.evaluation.judge import LLMJudge
from src.evaluation.ragas_eval import EvaluationCache
from src.retriever.faiss_retriever import FAISSRetriever
from src.systems.dspy_baseline import DSPyBaselineRAG
from src.systems.dspy_bootstrap import DSPyBootstrapRAG, optimize_bootstrap
from src.systems.dspy_mipro import DSPyMIPRORAG, optimize_mipro
from src.systems.langchain_rag import LangChainRAG


def load_ground_truth(gt_dir: Path) -> list[dict]:
    """Load ground truth Q&A pairs.

    Expects a JSON file with format:
    [{"question": "...", "answer": "..."}, ...]
    """
    # Look for any JSON file in ground_truth dir
    json_files = list(gt_dir.glob("*.json"))
    if not json_files:
        logger.error(f"No ground truth JSON files found in {gt_dir}")
        return []

    gt_file = json_files[0]
    with open(gt_file, encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} Q&A pairs from {gt_file.name}")
    return data


def split_data(data: list[dict], train_ratio: float = 0.7, seed: int = 42) -> tuple:
    """Split data into train/test sets."""
    import random

    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def setup_dspy(config: dict):
    """Configure DSPy with the active generation model."""
    models_config = config["models"]
    active_model = models_config["active_generation_model"]
    model_cfg = models_config["generation_models"][active_model]

    settings = get_settings()

    lm = dspy.LM(
        model=f"openai/{model_cfg['model_name']}",
        api_key=settings.openai_api_key,
        temperature=model_cfg["temperature"],
        max_tokens=model_cfg["max_tokens"],
    )
    dspy.configure(lm=lm)
    logger.info(f"DSPy configured with model: {model_cfg['model_name']}")


def run_evaluation():
    """Run full evaluation pipeline."""
    config = load_config()
    base = config["base"]

    # Setup
    setup_dspy(config)

    # Load retriever
    index_path = PROJECT_ROOT / base["vector_store"]["persist_path"]
    retriever = FAISSRetriever(
        index_path=index_path,
        embedding_model=base["embedding"]["model"],
        embedding_dims=base["embedding"]["dimensions"],
        top_k=base["retrieval"]["top_k"],
    )

    # Load ground truth
    gt_dir = PROJECT_ROOT / base["data"]["ground_truth_dir"]
    gt_data = load_ground_truth(gt_dir)
    if not gt_data:
        logger.error("No ground truth data. Create Q&A pairs first.")
        return

    # Split
    train_data, test_data = split_data(
        gt_data,
        train_ratio=base["split"]["train_ratio"],
        seed=base["project"]["seed"],
    )
    logger.info(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Initialize systems
    systems = {}

    # System 1: LangChain
    active_model = config["models"]["active_generation_model"]
    model_name = config["models"]["generation_models"][active_model]["model_name"]

    systems["langchain_rag"] = LangChainRAG(
        retriever=retriever,
        model_name=model_name,
    )

    # System 2: DSPy Baseline
    systems["dspy_baseline"] = DSPyBaselineRAG(retriever=retriever)

    # System 3: DSPy Bootstrap (requires optimization)
    logger.info("Optimizing DSPy Bootstrap...")
    train_examples = [
        dspy.Example(question=d["question"], answer=d["answer"]).with_inputs("question")
        for d in train_data
    ]

    def simple_metric(example, pred, trace=None):
        """Simple metric for optimization: check if answer is non-empty."""
        return len(pred.answer.strip()) > 0

    opt_config = config["models"]["optimization"]["bootstrap_few_shot"]
    systems["dspy_bootstrap"] = optimize_bootstrap(
        retriever=retriever,
        trainset=train_examples,
        metric_fn=simple_metric,
        max_bootstrapped_demos=opt_config["max_bootstrapped_demos"],
        max_labeled_demos=opt_config["max_labeled_demos"],
        max_rounds=opt_config["max_rounds"],
    )

    # System 4: DSPy MIPRO (requires optimization)
    logger.info("Optimizing DSPy MIPROv2...")
    mipro_config = config["models"]["optimization"]["mipro_v2"]
    systems["dspy_mipro"] = optimize_mipro(
        retriever=retriever,
        trainset=train_examples,
        metric_fn=simple_metric,
        num_candidates=mipro_config["num_candidates"],
        num_trials=mipro_config["num_trials"],
        max_bootstrapped_demos=mipro_config["max_bootstrapped_demos"],
        max_labeled_demos=mipro_config["max_labeled_demos"],
    )

    # Run evaluation on test set
    logger.info("=" * 60)
    logger.info("RUNNING EVALUATION ON TEST SET")
    logger.info("=" * 60)

    tracker = EfficiencyTracker()
    cache = EvaluationCache(cache_dir=base["evaluation"]["cache_dir"])
    judge = LLMJudge(
        model_name=base["evaluation"]["judge_model"],
        temperature=base["evaluation"]["judge_temperature"],
    )

    all_results = []

    for system_name, system in systems.items():
        logger.info(f"\nEvaluating: {system_name}")

        for item in test_data:
            question = item["question"]
            reference = item["answer"]

            # Check cache
            if cache.has(question, system_name, base["project"]["domain"]):
                cached = cache.get(question, system_name, base["project"]["domain"])
                all_results.append(cached)
                continue

            # Generate answer with timing
            start = time.perf_counter()
            result = system.answer(question)
            latency_ms = (time.perf_counter() - start) * 1000

            # Track efficiency
            tracker.record(
                system=system_name,
                question=question,
                latency_ms=latency_ms,
                model=model_name,
            )

            # Judge scoring
            scores = judge.evaluate_sample(
                question=question,
                answer=result["answer"],
                contexts=result["contexts"],
                reference=reference,
            )

            # Compile result
            eval_result = {
                "system": system_name,
                "question": question,
                "answer": result["answer"],
                "reference": reference,
                "latency_ms": latency_ms,
                **scores,
            }

            all_results.append(eval_result)
            cache.put(question, system_name, eval_result, base["project"]["domain"])

    # Save results
    results_path = PROJECT_ROOT / "results" / "evaluations" / "full_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Save efficiency summary
    efficiency_path = PROJECT_ROOT / "results" / "evaluations" / "efficiency_summary.json"
    with open(efficiency_path, "w", encoding="utf-8") as f:
        json.dump(tracker.summary(), f, indent=2)

    logger.info(f"\nResults saved to {results_path}")
    logger.info(f"Efficiency saved to {efficiency_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    for summary in tracker.summary():
        logger.info(f"  {summary['system']}: avg_latency={summary['avg_latency_ms']:.0f}ms")


if __name__ == "__main__":
    run_evaluation()
