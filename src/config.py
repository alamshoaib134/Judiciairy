"""
Configuration settings for JudicAIry
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Application configuration"""
    
    # Base paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    CHROMA_PERSIST_DIR: Path = DATA_DIR / "chroma_db"
    
    # HuggingFace settings
    HUGGINGFACE_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")
    CASESUMM_DATASET: str = "ChicagoHAI/CaseSumm"
    
    # Embedding model
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    EMBEDDING_DIMENSION: int = 1024  # BGE-large dimension
    
    # LLM settings
    LLM_MODEL: str = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Chunking settings
    CHUNK_SIZE: int = 1000  # tokens
    CHUNK_OVERLAP: int = 200  # tokens
    
    # Retrieval settings
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 5
    
    # ChromaDB collection name
    COLLECTION_NAME: str = "scotus_opinions"
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()
