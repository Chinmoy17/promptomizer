"""System 3: DSPy + BootstrapFewShot Optimization.

Few-shot optimization that auto-selects demonstrations from training data.
"""

import dspy
from dspy.teleprompt import BootstrapFewShot
from loguru import logger

from src.retriever.faiss_retriever import FAISSRetriever
from src.systems.dspy_baseline import AnswerQuestion


class DSPyBootstrapRAG(dspy.Module):
    """DSPy RAG optimized with BootstrapFewShot.

    Selects top-k demonstrations from training examples
    to include as few-shot examples in the prompt.
    """

    def __init__(self, retriever: FAISSRetriever):
        super().__init__()
        self.retriever = retriever
        self.generate_answer = dspy.ChainOfThought(AnswerQuestion)

    def forward(self, question: str) -> dspy.Prediction:
        """Run the optimized RAG pipeline."""
        contexts = self.retriever.retrieve_texts(question)
        prediction = self.generate_answer(context=contexts, question=question)

        return dspy.Prediction(
            answer=prediction.answer,
            contexts=contexts,
            system="dspy_bootstrap",
        )

    def answer(self, question: str) -> dict:
        """Convenience method matching the common interface."""
        pred = self.forward(question)
        return {
            "answer": pred.answer,
            "contexts": pred.contexts,
            "system": "dspy_bootstrap",
        }

    @property
    def system_name(self) -> str:
        return "DSPy + BootstrapFewShot"


def optimize_bootstrap(
    retriever: FAISSRetriever,
    trainset: list[dspy.Example],
    metric_fn,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 8,
    max_rounds: int = 1,
    metric_threshold: float | None = None,
) -> DSPyBootstrapRAG:
    """Optimize the DSPy RAG module using BootstrapFewShot.

    Args:
        retriever: Shared FAISS retriever.
        trainset: Training examples (dspy.Example with 'question' and 'answer').
        metric_fn: Evaluation metric function.
        max_bootstrapped_demos: Max bootstrapped demonstrations.
        max_labeled_demos: Max labeled demonstrations.
        max_rounds: Number of bootstrap rounds.
        metric_threshold: Optional threshold for demo selection.

    Returns:
        Optimized DSPyBootstrapRAG module.
    """
    # Create unoptimized module
    module = DSPyBootstrapRAG(retriever=retriever)

    # Configure optimizer
    optimizer = BootstrapFewShot(
        metric=metric_fn,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        max_rounds=max_rounds,
    )

    # Run optimization
    logger.info(
        f"Starting BootstrapFewShot optimization: "
        f"trainset={len(trainset)}, max_demos={max_bootstrapped_demos}"
    )
    optimized = optimizer.compile(module, trainset=trainset)

    logger.info("BootstrapFewShot optimization complete")
    return optimized
