"""System 2: DSPy Baseline RAG.

Signature-based RAG with Chain-of-Thought using DSPy.
"""

import dspy

from src.retriever.faiss_retriever import FAISSRetriever


class AnswerQuestion(dspy.Signature):
    """Answer a question based on the provided context passages."""

    context: list[str] = dspy.InputField(desc="Retrieved context passages")
    question: str = dspy.InputField(desc="The question to answer")
    answer: str = dspy.OutputField(desc="A concise, factual answer based on the context")


class DSPyBaselineRAG(dspy.Module):
    """DSPy baseline RAG with Chain-of-Thought reasoning.

    Uses DSPy signatures for structured generation.
    """

    def __init__(self, retriever: FAISSRetriever):
        super().__init__()
        self.retriever = retriever
        self.generate_answer = dspy.ChainOfThought(AnswerQuestion)

    def forward(self, question: str) -> dspy.Prediction:
        """Run the RAG pipeline.

        Args:
            question: The user's question.

        Returns:
            dspy.Prediction with 'answer' and 'contexts'.
        """
        # Retrieve
        contexts = self.retriever.retrieve_texts(question)

        # Generate with CoT
        prediction = self.generate_answer(context=contexts, question=question)

        return dspy.Prediction(
            answer=prediction.answer,
            contexts=contexts,
            system="dspy_baseline",
        )

    def answer(self, question: str) -> dict:
        """Convenience method matching the LangChain interface.

        Returns:
            dict with 'answer', 'contexts', 'system'.
        """
        pred = self.forward(question)
        return {
            "answer": pred.answer,
            "contexts": pred.contexts,
            "system": "dspy_baseline",
        }

    @property
    def system_name(self) -> str:
        return "DSPy Baseline (CoT)"
