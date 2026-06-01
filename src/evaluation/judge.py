"""LLM-as-judge scoring module.

Provides discrete scoring (0.00, 0.25, 0.50, 0.75, 1.00) using a judge LLM.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger


JUDGE_PROMPT = """You are an expert evaluator. Score the following answer on a scale.

Question: {question}
Reference Answer: {reference}
Generated Answer: {answer}
Retrieved Context: {context}

Evaluate the generated answer on this criterion: {criterion}

Score MUST be exactly one of: 0.00, 0.25, 0.50, 0.75, 1.00

Scoring guide:
- 1.00: Perfect - completely correct and well-supported
- 0.75: Good - mostly correct with minor gaps
- 0.50: Partial - some correct elements but significant gaps
- 0.25: Poor - mostly incorrect or irrelevant
- 0.00: Wrong - completely incorrect or no answer

Respond with ONLY the numeric score (e.g., 0.75). No explanation."""


CRITERIA = {
    "answer_accuracy": "How accurately does the generated answer match the reference answer?",
    "context_relevance": "How relevant is the retrieved context to answering the question?",
    "faithfulness": "Is the generated answer faithful to (supported by) the retrieved context?",
}


class LLMJudge:
    """LLM-as-judge for discrete scoring."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
    ):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def score(
        self,
        question: str,
        answer: str,
        context: str,
        reference: str = "",
        criterion: str = "answer_accuracy",
    ) -> float:
        """Score a single answer on a given criterion.

        Args:
            question: The original question.
            answer: The generated answer.
            context: The retrieved context (joined).
            reference: The ground truth reference answer.
            criterion: Which criterion to evaluate.

        Returns:
            Float score in {0.00, 0.25, 0.50, 0.75, 1.00}.
        """
        criterion_desc = CRITERIA.get(criterion, criterion)

        result = self.chain.invoke({
            "question": question,
            "reference": reference,
            "answer": answer,
            "context": context,
            "criterion": criterion_desc,
        })

        try:
            score = float(result.strip())
            # Snap to nearest valid level
            valid_scores = [0.00, 0.25, 0.50, 0.75, 1.00]
            score = min(valid_scores, key=lambda x: abs(x - score))
            return score
        except ValueError:
            logger.warning(f"Judge returned non-numeric score: {result}")
            return 0.0

    def evaluate_sample(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        reference: str = "",
        criteria: list[str] | None = None,
    ) -> dict[str, float]:
        """Evaluate a sample across multiple criteria.

        Returns:
            Dict mapping criterion name to score.
        """
        if criteria is None:
            criteria = list(CRITERIA.keys())

        context_str = "\n\n---\n\n".join(contexts)
        scores = {}

        for criterion in criteria:
            scores[criterion] = self.score(
                question=question,
                answer=answer,
                context=context_str,
                reference=reference,
                criterion=criterion,
            )

        return scores
