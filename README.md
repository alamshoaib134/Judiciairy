Developed with 🧠 by **[Shoaib Alam](https://alamshoaib134.github.io/)** (AI Engineer at JPMC | NLP Researcher @ IIT Gandhinagar | Hybrid RAG Pioneer)
---
# JudicAIry ⚖️

A Retrieval-Augmented Generation (RAG) system for U.S. Supreme Court judgments. Ask questions about constitutional law, find relevant precedents, and explore centuries of Supreme Court jurisprudence.

## Features

- **Semantic Search** — Find relevant cases by meaning, not just keywords
- **Hybrid Retrieval** — Combines vector search with BM25 for precision
- **Cross-Encoder Reranking** — Improves result relevance
- **LLM-Powered Answers** — Get synthesized answers with citations
- **Beautiful UI** — Streamlit interface with a sophisticated legal aesthetic
- **REST API** — FastAPI backend for integration

## Quick Start

### 1. Installation

```bash
# Clone and navigate to project
cd JudicAIry

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Ingest Data

Download and index Supreme Court opinions:

```bash
# Full dataset (25K+ cases) - takes ~30 minutes
python scripts/ingest_data.py

# Or start with a sample for testing
python scripts/ingest_data.py --limit 100
```

### 3. Run the Application

**Option A: Start both API and UI**

```bash
# Terminal 1: Start API
python scripts/run_api.py

# Terminal 2: Start UI  
python scripts/run_ui.py
```

**Option B: Use Python directly**

```python
from src.rag.pipeline import RAGPipeline

pipeline = RAGPipeline()
result = pipeline.query("What is the standard for free speech?")
print(result.answer)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     JudicAIry Architecture                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   CaseSumm   │    │     SCDB     │    │ supremecourt │  │
│  │   Dataset    │    │   Metadata   │    │    .gov      │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │          │
│         └───────────────────┼───────────────────┘          │
│                             ▼                               │
│                    ┌────────────────┐                       │
│                    │  Data Loader   │                       │
│                    └────────┬───────┘                       │
│                             ▼                               │
│                    ┌────────────────┐                       │
│                    │    Chunker     │                       │
│                    │ (Semantic)     │                       │
│                    └────────┬───────┘                       │
│                             ▼                               │
│  ┌────────────────────────────────────────────────────────┐│
│  │                    Embeddings                          ││
│  │   BGE-Large / Legal-BERT → ChromaDB Vector Store       ││
│  └────────────────────────────────────────────────────────┘│
│                             │                               │
│                             ▼                               │
│  ┌────────────────────────────────────────────────────────┐│
│  │                   RAG Pipeline                         ││
│  │  ┌─────────┐   ┌──────────┐   ┌────────────────────┐  ││
│  │  │ Hybrid  │ → │ Reranker │ → │   LLM Generator    │  ││
│  │  │Retriever│   │(CrossEnc)│   │(Mistral/GPT-4)     │  ││
│  │  └─────────┘   └──────────┘   └────────────────────┘  ││
│  └────────────────────────────────────────────────────────┘│
│                             │                               │
│         ┌───────────────────┼───────────────────┐          │
│         ▼                   ▼                   ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  FastAPI     │    │  Streamlit   │    │   Python     │  │
│  │   Backend    │    │     UI       │    │    SDK       │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Data Sources

| Source | Description | Size |
|--------|-------------|------|
| [CaseSumm](https://huggingface.co/datasets/ChicagoHAI/CaseSumm) | SCOTUS opinions with official syllabuses | 25.6K cases |
| [SCDB](http://scdb.wustl.edu/) | Case metadata (1791-2024) | 200+ fields/case |
| [supremecourt.gov](https://supremecourt.gov/opinions/opinions.aspx) | Official opinions | Current term |

## Project Structure

```
JudicAIry/
├── src/
│   ├── config.py           # Configuration settings
│   ├── data/
│   │   ├── loader.py       # Dataset loading (CaseSumm, SCDB)
│   │   └── chunker.py      # Document chunking strategies
│   ├── embeddings/
│   │   ├── embedder.py     # Embedding model wrapper
│   │   └── vectorstore.py  # ChromaDB vector store
│   ├── rag/
│   │   ├── retriever.py    # Hybrid retrieval (vector + BM25)
│   │   ├── reranker.py     # Cross-encoder reranking
│   │   ├── generator.py    # LLM answer generation
│   │   └── pipeline.py     # Complete RAG pipeline
│   ├── api/
│   │   └── main.py         # FastAPI backend
│   └── ui/
│       └── app.py          # Streamlit frontend
├── scripts/
│   ├── ingest_data.py      # Data ingestion script
│   ├── run_api.py          # Start API server
│   ├── run_ui.py           # Start Streamlit UI
│   └── demo.py             # Interactive demo
├── data/
│   ├── raw/                # Downloaded datasets
│   ├── processed/          # Processed JSON files
│   └── chroma_db/          # Vector database
└── requirements.txt
```

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/api/query` | Ask a legal question |
| POST | `/api/search` | Search for cases |
| GET | `/api/cases` | List indexed cases |
| GET | `/api/cases/{id}` | Get case details |
| GET | `/api/stats` | Collection statistics |

### Example Query

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the standard for free speech?", "top_k": 5}'
```

## Configuration

Set environment variables or create a `.env` file:

```bash
# Optional: HuggingFace token for gated models
HUGGINGFACE_TOKEN=your_token

# Optional: OpenAI for GPT-4 generation
OPENAI_API_KEY=your_key

# Model configuration
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```

## Performance Tips

1. **GPU Acceleration** — For faster embedding and generation, use a CUDA-capable GPU
2. **Limit Initial Load** — Use `--limit 100` for testing before full ingestion
3. **OpenAI API** — For best generation quality without local GPU resources
4. **Simple Reranker** — Disable cross-encoder reranking for faster responses

## Legal Disclaimer

This tool is for research and educational purposes. It should not be used as a substitute for professional legal advice. Always consult a qualified attorney for legal matters.

## Contributing

Contributions welcome! Please read our contributing guidelines and submit PRs.

## License

MIT License - see LICENSE file for details.

---

Built with ❤️ for legal research | Powered by LangChain, ChromaDB, and HuggingFace
