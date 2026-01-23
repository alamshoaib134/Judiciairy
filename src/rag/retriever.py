"""
Hybrid retriever combining vector search with BM25 keyword search
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
import numpy as np

from src.embeddings.vectorstore import VectorStore
from src.config import config


@dataclass
class RetrievalResult:
    """A single retrieval result"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str  # "vector", "bm25", or "hybrid"


class HybridRetriever:
    """
    Hybrid retriever combining:
    - Dense vector search (semantic similarity)
    - Sparse BM25 search (keyword matching)
    
    Fusion using Reciprocal Rank Fusion (RRF)
    """
    
    def __init__(
        self,
        vectorstore: VectorStore,
        alpha: float = 0.5,  # Weight for vector vs BM25
        top_k: int = None
    ):
        self.vectorstore = vectorstore
        self.alpha = alpha  # 0 = only BM25, 1 = only vector
        self.top_k = top_k or config.TOP_K_RETRIEVAL
        
        # BM25 index (built lazily)
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_corpus: List[Dict[str, Any]] = []
    
    def build_bm25_index(self, documents: List[Dict[str, Any]] = None):
        """
        Build BM25 index from documents
        
        Args:
            documents: List of docs with 'id', 'content', 'metadata'
                      If None, fetch from vectorstore
        """
        if documents is None:
            # Fetch all documents from vectorstore
            print("Fetching documents from vectorstore for BM25 indexing...")
            all_docs = self.vectorstore.collection.get(
                include=["documents", "metadatas"]
            )
            
            documents = []
            for i, doc_id in enumerate(all_docs["ids"]):
                documents.append({
                    "id": doc_id,
                    "content": all_docs["documents"][i],
                    "metadata": all_docs["metadatas"][i]
                })
        
        print(f"Building BM25 index with {len(documents)} documents...")
        
        self.bm25_corpus = documents
        
        # Tokenize documents for BM25
        tokenized_corpus = [
            self._tokenize(doc["content"]) 
            for doc in documents
        ]
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 index built successfully")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        # Lowercase and split on whitespace/punctuation
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents using hybrid search
        
        Args:
            query: Search query
            top_k: Number of results
            filter_dict: Metadata filters
            use_hybrid: Whether to use hybrid (vector + BM25) or just vector
            
        Returns:
            List of RetrievalResult objects
        """
        top_k = top_k or self.top_k
        
        # Vector search
        vector_results = self.vectorstore.search(
            query=query,
            top_k=top_k * 2,  # Fetch more for fusion
            filter_dict=filter_dict
        )
        
        # If not using hybrid or BM25 not built, return vector results
        if not use_hybrid or self.bm25 is None:
            return [
                RetrievalResult(
                    id=r["id"],
                    content=r["document"],
                    score=r["score"],
                    metadata=r["metadata"],
                    source="vector"
                )
                for r in vector_results[:top_k]
            ]
        
        # BM25 search
        bm25_results = self._bm25_search(query, top_k * 2)
        
        # Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            top_k=top_k
        )
        
        return fused_results
    
    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search using BM25"""
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if there's a match
                doc = self.bm25_corpus[idx]
                results.append({
                    "id": doc["id"],
                    "document": doc["content"],
                    "metadata": doc["metadata"],
                    "score": float(scores[idx])
                })
                
        return results
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int,
        k: int = 60  # RRF parameter
    ) -> List[RetrievalResult]:
        """
        Combine results using Reciprocal Rank Fusion
        
        RRF score = sum(1 / (k + rank)) for each result list
        """
        # Build score dictionaries
        scores = {}
        doc_info = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results):
            doc_id = result["id"]
            rrf_score = self.alpha / (k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            doc_info[doc_id] = {
                "content": result["document"],
                "metadata": result["metadata"]
            }
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results):
            doc_id = result["id"]
            rrf_score = (1 - self.alpha) / (k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            if doc_id not in doc_info:
                doc_info[doc_id] = {
                    "content": result["document"],
                    "metadata": result["metadata"]
                }
        
        # Sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Build results
        results = []
        for doc_id in sorted_ids[:top_k]:
            results.append(RetrievalResult(
                id=doc_id,
                content=doc_info[doc_id]["content"],
                score=scores[doc_id],
                metadata=doc_info[doc_id]["metadata"],
                source="hybrid"
            ))
            
        return results
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: int = None,
        include_adjacent: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieve with surrounding context chunks
        
        For legal documents, it's often helpful to include
        adjacent chunks for better context
        """
        results = self.retrieve(query, top_k=top_k)
        
        if not include_adjacent:
            return results
        
        # Collect adjacent chunk IDs
        adjacent_ids = set()
        for result in results:
            case_id = result.metadata.get("case_id", "")
            chunk_index = result.metadata.get("chunk_index", 0)
            
            # Generate adjacent chunk IDs
            for offset in [-1, 1]:
                adj_index = chunk_index + offset
                if adj_index >= 0:
                    # Attempt to construct adjacent ID
                    section_type = result.metadata.get("section_type", "body")
                    adj_id = f"{case_id}_{section_type}_{adj_index}"
                    adjacent_ids.add(adj_id)
        
        # Filter out already retrieved
        adjacent_ids -= {r.id for r in results}
        
        # Fetch adjacent chunks
        if adjacent_ids:
            try:
                adjacent_docs = self.vectorstore.collection.get(
                    ids=list(adjacent_ids),
                    include=["documents", "metadatas"]
                )
                
                for i, doc_id in enumerate(adjacent_docs["ids"]):
                    results.append(RetrievalResult(
                        id=doc_id,
                        content=adjacent_docs["documents"][i],
                        score=0.5,  # Lower score for adjacent
                        metadata=adjacent_docs["metadatas"][i],
                        source="adjacent"
                    ))
            except Exception:
                pass  # Adjacent chunks might not exist
        
        return results
