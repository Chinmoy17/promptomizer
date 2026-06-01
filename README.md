# PromptoMizer

**DSPy RAG Optimization & Evaluation Framework**

A reproducible pipeline that compares multiple retrieval-augmented generation systems on a custom domain dataset using LLM-as-judge evaluation (RAGAS).

## Architecture

Compares 4 RAG system variants using the **same retriever** for fair comparison:

| System | Approach | Optimization |
|--------|----------|-------------|
| LangChain RAG | Template-based | None (baseline) |
| DSPy Baseline | Signature + CoT | None |
| DSPy + BootstrapFewShot | Few-shot | Auto-selected demonstrations |
| DSPy + MIPROv2 | Instruction optimization | Bayesian prompt search |

## Project Structure

```
PromptoMizer/
├── configs/
│   ├── base.yaml              # Main configuration
│   └── models.yaml            # Model & optimization settings
├── data/
│   ├── raw/                   # Source documents (PDF/DOCX/TXT/JSON)
│   ├── processed/             # Chunks & embeddings
│   ├── indices/               # FAISS vector index
│   └── ground_truth/          # Q&A pairs (JSON)
├── src/
│   ├── config.py              # Configuration loader
│   ├── data/
│   │   ├── ingest.py          # Document ingestion (multi-format)
│   │   ├── chunker.py         # Token-based recursive chunking
│   │   ├── embedder.py        # OpenAI embedding generation
│   │   └── indexer.py         # FAISS index build/save/load
│   ├── retriever/
│   │   └── faiss_retriever.py # Shared cached retriever
│   ├── systems/
│   │   ├── langchain_rag.py   # System 1: LangChain baseline
│   │   ├── dspy_baseline.py   # System 2: DSPy + CoT
│   │   ├── dspy_bootstrap.py  # System 3: BootstrapFewShot
│   │   └── dspy_mipro.py      # System 4: MIPROv2
│   ├── evaluation/
│   │   ├── ragas_eval.py      # RAGAS metrics + caching
│   │   ├── judge.py           # LLM-as-judge discrete scoring
│   │   └── efficiency.py      # Tokens/latency/cost tracking
│   └── visualization/
│       └── plots.py           # 6 publication-quality figure types
├── scripts/
│   ├── run_pipeline.py        # Phase 1: Data pipeline
│   ├── run_evaluation.py      # Phase 4: Full evaluation
│   └── run_viz.py             # Phase 5: Generate figures
├── results/
│   ├── evaluations/           # Cached scores & results
│   └── figures/               # Generated plots
├── main.py                    # CLI entry point
└── pyproject.toml             # Dependencies (uv)
```

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

### 3. Add source documents

Place your documents in `data/raw/` (supports PDF, DOCX, TXT, JSON, MD).

### 4. Add ground truth Q&A pairs

Create `data/ground_truth/qa_pairs.json`:

```json
[
  {"question": "What is X?", "answer": "X is ..."},
  {"question": "How does Y work?", "answer": "Y works by ..."}
]
```

### 5. Run the pipeline

```bash
# Phase 1: Ingest → Chunk → Embed → Index
python main.py pipeline

# Phase 4: Evaluate all systems
python main.py evaluate

# Phase 5: Generate figures
python main.py visualize

# Or run everything:
python main.py all
```

## Configuration

All settings are in `configs/base.yaml` and `configs/models.yaml`. Key parameters:

- **Chunking**: 512 tokens, 50 overlap (recursive strategy)
- **Embedding**: text-embedding-3-small (1536 dims)
- **Retrieval**: Top-5 FAISS
- **Generation**: Temperature 0.0 (deterministic)
- **Judge**: GPT-4o with discrete scoring (0.00–1.00)

## Metrics

| Category | Metrics |
|----------|---------|
| Quality | Answer Accuracy, Context Relevance, Faithfulness |
| Efficiency | Tokens/query, Cost/query, Latency (avg + p95) |
| Training | Optimization time, samples used, improvement delta |

## Design Decisions

- All systems share the **same retriever + index** (fair comparison)
- Temperature = 0.0 everywhere (deterministic outputs)
- Judge LLM is separate from generation model (GPT-4o)
- Results are cached by (question, system, domain) to avoid redundant API calls
- Config-driven: switch domains/models by editing YAML only
