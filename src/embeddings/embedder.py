"""
Embedding model wrapper for legal documents
Supports BGE, Legal-BERT, and other sentence transformer models
"""

from typing import List, Optional
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import config


class Embedder:
    """Wrapper for embedding models"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        normalize_embeddings: bool = True
    ):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        
        print(f"Loading embedding model: {self.model_name}")
        
        # Use SentenceTransformer directly for more control
        self.model = SentenceTransformer(
            self.model_name,
            device=device
        )
        
        # Also create LangChain-compatible embeddings
        self.langchain_embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": normalize_embeddings}
        )
        
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.dimension}")
    
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed a list of texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings (N x dimension)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query
        
        For asymmetric models (like BGE), queries should be
        prefixed differently than documents
        """
        # BGE models expect query prefix
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"
            
        embedding = self.model.encode(
            query,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )
        return embedding
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Embed documents (LangChain compatible interface)
        """
        embeddings = self.embed_texts(documents, show_progress=False)
        return embeddings.tolist()
    
    def get_langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """Get LangChain-compatible embedding function"""
        return self.langchain_embeddings


def create_embedder(model_name: Optional[str] = None) -> Embedder:
    """Factory function to create embedder"""
    return Embedder(model_name=model_name)
