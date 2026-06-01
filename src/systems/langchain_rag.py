"""System 1: LangChain RAG Baseline.

Traditional template-based RAG using LangChain with the shared FAISS retriever.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.retriever.faiss_retriever import FAISSRetriever

RAG_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the provided context.
- If the context does not contain enough information, say so clearly.
- Be concise and accurate.

Answer:"""


class LangChainRAG:
    """LangChain-based RAG system (baseline).

    Uses a standard prompt template with the shared retriever.
    """

    def __init__(
        self,
        retriever: FAISSRetriever,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.retriever = retriever
        self.model_name = model_name

        # Build chain
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def answer(self, question: str) -> dict:
        """Generate an answer for a question.

        Args:
            question: The user's question.

        Returns:
            dict with 'answer', 'contexts', 'system'.
        """
        # Retrieve relevant contexts
        contexts = self.retriever.retrieve_texts(question)
        context_str = "\n\n---\n\n".join(contexts)

        # Generate answer
        answer = self.chain.invoke({
            "context": context_str,
            "question": question,
        })

        return {
            "answer": answer,
            "contexts": contexts,
            "system": "langchain_rag",
        }

    @property
    def system_name(self) -> str:
        return "LangChain RAG (Baseline)"
