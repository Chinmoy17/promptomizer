"""System 4: DSPy + MIPROv2 Optimization.

Instruction optimization using Bayesian search over prompt candidates.
"""

import dspy
from dspy.teleprompt import MIPROv2
from loguru import logger

from src.retriever.faiss_retriever import FAISSRetriever
from src.systems.dspy_baseline import AnswerQuestion


class DSPyMIPRORAG(dspy.Module):
    """DSPy RAG optimized with MIPROv2.

    Uses Bayesian optimization to find optimal instruction wording.
    """

    def __init__(self, retriever: FAISSRetriever):
        super().__init__()
        self.retriever = retriever
        self.generate_answer = dspy.ChainOfThought(AnswerQuestion)

    def forward(self, question: str) -> dspy.Prediction:
        """Run the MIPRO-optimized RAG pipeline."""
        contexts = self.retriever.retrieve_texts(question)
        prediction = self.generate_answer(context=contexts, question=question)

        return dspy.Prediction(
            answer=prediction.answer,
            contexts=contexts,
            system="dspy_mipro",
        )

    def answer(self, question: str) -> dict:
        """Convenience method matching the common interface."""
        pred = self.forward(question)
        return {
            "answer": pred.answer,
            "contexts": pred.contexts,
            "system": "dspy_mipro",
        }

    @property
    def system_name(self) -> str:
        return "DSPy + MIPROv2"


def optimize_mipro(
    retriever: FAISSRetriever,
    trainset: list[dspy.Example],
    metric_fn,
    num_candidates: int = 10,
    num_trials: int = 20,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 4,
    eval_kwargs: dict | None = None,
) -> DSPyMIPRORAG:
    """Optimize the DSPy RAG module using MIPROv2.

    Args:
        retriever: Shared FAISS retriever.
        trainset: Training examples (dspy.Example with 'question' and 'answer').
        metric_fn: Evaluation metric function (RAGAS-based).
        num_candidates: Number of instruction candidates to generate.
        num_trials: Number of Bayesian optimization trials.
        max_bootstrapped_demos: Max bootstrapped demonstrations.
        max_labeled_demos: Max labeled demonstrations.
        eval_kwargs: Additional kwargs for evaluation.

    Returns:
        Optimized DSPyMIPRORAG module.
    """
    if eval_kwargs is None:
        eval_kwargs = {"num_threads": 4}

    # Create unoptimized module
    module = DSPyMIPRORAG(retriever=retriever)

    # Configure MIPROv2 optimizer
    optimizer = MIPROv2(
        metric=metric_fn,
        num_candidates=num_candidates,
        num_threads=eval_kwargs.get("num_threads", 4),
    )

    # Run optimization
    logger.info(
        f"Starting MIPROv2 optimization: "
        f"trainset={len(trainset)}, candidates={num_candidates}, trials={num_trials}"
    )
    optimized = optimizer.compile(
        module,
        trainset=trainset,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        num_trials=num_trials,
    )

    logger.info("MIPROv2 optimization complete")
    return optimized
