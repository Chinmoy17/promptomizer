"""Configuration loader for PromptoMizer."""

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings


# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class ChunkingConfig(BaseModel):
    strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 50
    tokenizer: str = "cl100k_base"


class EmbeddingConfig(BaseModel):
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 100
    provider: str = "openai"


class VectorStoreConfig(BaseModel):
    type: str = "faiss"
    index_type: str = "flat_l2"
    persist_path: str = "data/indices/faiss_index"


class RetrievalConfig(BaseModel):
    top_k: int = 5
    score_threshold: float | None = None


class GenerationConfig(BaseModel):
    temperature: float = 0.0
    max_tokens: int = 1024


class EvaluationConfig(BaseModel):
    metrics: list[str] = ["answer_accuracy", "context_relevance", "faithfulness"]
    judge_model: str = "gpt-4o"
    judge_temperature: float = 0.0
    score_levels: list[float] = [0.00, 0.25, 0.50, 0.75, 1.00]
    cache_results: bool = True
    cache_dir: str = "results/evaluations"


class SplitConfig(BaseModel):
    train_ratio: float = 0.7
    test_ratio: float = 0.3
    stratify: bool = False


class Settings(BaseSettings):
    """Global settings loaded from environment variables."""

    openai_api_key: str = ""
    openai_judge_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    openai_api_base: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path) as f:
        return yaml.safe_load(f)


def load_config() -> dict[str, Any]:
    """Load and merge all configuration files."""
    base = load_yaml_config("configs/base.yaml")
    models = load_yaml_config("configs/models.yaml")
    return {"base": base, "models": models}


def get_settings() -> Settings:
    """Get application settings from environment."""
    return Settings()
