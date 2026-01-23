"""
Cross-encoder reranker for improving retrieval precision
"""

from typing import List, Optional
from dataclasses import dataclass

from sentence_transformers import CrossEncoder
import numpy as np

from src.rag.retriever import RetrievalResult
from src.config import config


@dataclass 
class RerankResult:
    """Result after reranking"""
    id: str
    content: str
    original_score: float
    rerank_score: float
    combined_score: float
    metadata: dict
    

class Reranker:
    """
    Cross-encoder reranker for precision improvement
    
    Uses a cross-encoder model to compute query-document relevance
    scores more accurately than bi-encoder similarity
    """
    
    # Popular cross-encoder models for reranking
    MODELS = {
        "bge-reranker": "BAAI/bge-reranker-large",
        "ms-marco": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "msmacro-large": "cross-encoder/ms-marco-electra-base"
    }
    
    def __init__(
        self,
        model_name: str = "bge-reranker",
        device: str = "cpu",
        top_k: int = None
    ):
        # Resolve model name
        if model_name in self.MODELS:
            model_name = self.MODELS[model_name]
            
        self.model_name = model_name
        self.device = device
        self.top_k = top_k or config.TOP_K_RERANK
        
        print(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=512
        )
        print("Reranker loaded successfully")
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = None,
        score_weight: float = 0.3
    ) -> List[RerankResult]:
        """
        Rerank retrieval results using cross-encoder
        
        Args:
            query: Original query
            results: List of retrieval results
            top_k: Number of results to return after reranking
            score_weight: Weight for original score in combined score
            
        Returns:
            Reranked results with combined scores
        """
        top_k = top_k or self.top_k
        
        if not results:
            return []
        
        # Prepare query-document pairs
        pairs = [(query, result.content) for result in results]
        
        # Get cross-encoder scores
        rerank_scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Normalize rerank scores to [0, 1]
        if len(rerank_scores) > 1:
            min_score = min(rerank_scores)
            max_score = max(rerank_scores)
            if max_score > min_score:
                rerank_scores = (rerank_scores - min_score) / (max_score - min_score)
            else:
                rerank_scores = np.ones_like(rerank_scores) * 0.5
        else:
            rerank_scores = np.array([0.5])
        
        # Build reranked results
        reranked = []
        for i, result in enumerate(results):
            combined_score = (
                score_weight * result.score + 
                (1 - score_weight) * rerank_scores[i]
            )
            
            reranked.append(RerankResult(
                id=result.id,
                content=result.content,
                original_score=result.score,
                rerank_score=float(rerank_scores[i]),
                combined_score=float(combined_score),
                metadata=result.metadata
            ))
        
        # Sort by combined score
        reranked.sort(key=lambda x: x.combined_score, reverse=True)
        
        return reranked[:top_k]
    
    def rerank_with_diversity(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = None,
        diversity_weight: float = 0.2
    ) -> List[RerankResult]:
        """
        Rerank with diversity consideration
        
        Tries to include results from different cases/sections
        to provide a more comprehensive answer
        """
        reranked = self.rerank(query, results, top_k=len(results))
        
        if len(reranked) <= top_k:
            return reranked
        
        # Greedy selection with diversity
        selected = []
        seen_cases = set()
        seen_sections = set()
        
        for result in reranked:
            case_id = result.metadata.get("case_id", "")
            section_type = result.metadata.get("section_type", "")
            
            # Calculate diversity bonus
            diversity_bonus = 0
            if case_id not in seen_cases:
                diversity_bonus += 0.5
            if section_type not in seen_sections:
                diversity_bonus += 0.3
            
            # Adjust score
            adjusted_score = (
                (1 - diversity_weight) * result.combined_score +
                diversity_weight * diversity_bonus
            )
            
            result.combined_score = adjusted_score
            selected.append(result)
            
            seen_cases.add(case_id)
            seen_sections.add(section_type)
            
            if len(selected) >= top_k:
                break
        
        return selected


class SimpleReranker:
    """
    Simple reranker without cross-encoder
    
    Uses heuristics like keyword matching and metadata
    for environments where loading a cross-encoder is too heavy
    """
    
    def __init__(self, top_k: int = None):
        self.top_k = top_k or config.TOP_K_RERANK
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = None
    ) -> List[RerankResult]:
        """
        Simple reranking using keyword overlap and heuristics
        """
        top_k = top_k or self.top_k
        
        query_terms = set(query.lower().split())
        
        scored = []
        for result in results:
            content_terms = set(result.content.lower().split())
            
            # Keyword overlap score
            overlap = len(query_terms & content_terms) / max(len(query_terms), 1)
            
            # Section type bonus (syllabus and holding are often most relevant)
            section_bonus = {
                "syllabus": 0.2,
                "holding": 0.15,
                "reasoning": 0.1,
                "body": 0.05,
                "dissent": 0.0
            }.get(result.metadata.get("section_type", "body"), 0.05)
            
            combined = 0.5 * result.score + 0.3 * overlap + 0.2 * section_bonus
            
            scored.append(RerankResult(
                id=result.id,
                content=result.content,
                original_score=result.score,
                rerank_score=overlap + section_bonus,
                combined_score=combined,
                metadata=result.metadata
            ))
        
        scored.sort(key=lambda x: x.combined_score, reverse=True)
        return scored[:top_k]
