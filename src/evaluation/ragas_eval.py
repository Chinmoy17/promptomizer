"""RAGAS evaluation metrics.

Uses RAGAS v0.2+ for Answer Accuracy, Context Relevance, Faithfulness.
"""

import json
from pathlib import Path

from loguru import logger
from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
)
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI


def create_judge_llm(
    model_name: str = "gpt-4o",
    temperature: float = 0.0,
) -> LangchainLLMWrapper:
    """Create the LLM-as-judge wrapper for RAGAS."""
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    return LangchainLLMWrapper(llm)


def build_evaluation_dataset(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] | None = None,
) -> EvaluationDataset:
    """Build a RAGAS EvaluationDataset from system outputs.

    Args:
        questions: List of questions.
        answers: List of generated answers.
        contexts: List of context lists (one per question).
        ground_truths: Optional list of reference answers.

    Returns:
        RAGAS EvaluationDataset.
    """
    samples = []
    for i in range(len(questions)):
        sample = SingleTurnSample(
            user_input=questions[i],
            response=answers[i],
            retrieved_contexts=contexts[i],
        )
        if ground_truths:
            sample.reference = ground_truths[i]
        samples.append(sample)

    return EvaluationDataset(samples=samples)


def run_ragas_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] | None = None,
    judge_model: str = "gpt-4o",
    metrics: list[str] | None = None,
) -> dict:
    """Run RAGAS evaluation on system outputs.

    Args:
        questions: List of questions.
        answers: Generated answers.
        contexts: Retrieved contexts per question.
        ground_truths: Reference answers (optional).
        judge_model: Model to use as judge.
        metrics: List of metric names to compute.

    Returns:
        Dict with metric scores (per-sample and aggregated).
    """
    if metrics is None:
        metrics = ["faithfulness", "context_relevance", "response_relevancy"]

    judge_llm = create_judge_llm(model_name=judge_model)

    # Map metric names to RAGAS metric objects
    metric_map = {
        "faithfulness": Faithfulness(llm=judge_llm),
        "context_relevance": LLMContextPrecisionWithoutReference(llm=judge_llm),
        "response_relevancy": ResponseRelevancy(llm=judge_llm),
    }

    active_metrics = [metric_map[m] for m in metrics if m in metric_map]

    dataset = build_evaluation_dataset(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths,
    )

    logger.info(f"Running RAGAS evaluation: {len(questions)} samples, metrics={metrics}")
    results = evaluate(dataset=dataset, metrics=active_metrics)

    return results.to_pandas().to_dict(orient="records")


class EvaluationCache:
    """Cache evaluation results to avoid redundant API calls.

    Caches by (question, system_name, domain) tuple.
    """

    def __init__(self, cache_dir: str | Path = "results/evaluations"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / "eval_cache.json"
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        if self._cache_file.exists():
            with open(self._cache_file, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self) -> None:
        with open(self._cache_file, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    def get(self, question: str, system: str, domain: str = "default") -> dict | None:
        """Get cached evaluation result."""
        key = f"{domain}::{system}::{question}"
        return self._cache.get(key)

    def put(self, question: str, system: str, result: dict, domain: str = "default") -> None:
        """Cache an evaluation result."""
        key = f"{domain}::{system}::{question}"
        self._cache[key] = result
        self._save_cache()

    def has(self, question: str, system: str, domain: str = "default") -> bool:
        """Check if result is cached."""
        key = f"{domain}::{system}::{question}"
        return key in self._cache
