"""
Complete RAG pipeline combining retrieval, reranking, and generation
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.embeddings.vectorstore import VectorStore
from src.rag.retriever import HybridRetriever, RetrievalResult
from src.rag.reranker import Reranker, SimpleReranker, RerankResult
from src.rag.generator import LegalGenerator, GenerationResult
from src.config import config


@dataclass
class QueryResult:
    """Complete result from RAG query"""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_results: List[Dict[str, Any]] = field(default_factory=list)
    rerank_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": self.sources,
            "retrieval_count": len(self.retrieval_results),
            "rerank_count": len(self.rerank_results),
            "metadata": self.metadata
        }


class RAGPipeline:
    """
    Complete RAG pipeline for SCOTUS legal Q&A
    
    Pipeline stages:
    1. Query processing
    2. Hybrid retrieval (vector + BM25)
    3. Cross-encoder reranking
    4. LLM generation with citations
    """
    
    def __init__(
        self,
        vectorstore: Optional[VectorStore] = None,
        use_reranker: bool = True,
        use_openai: bool = False,
        retrieval_top_k: int = None,
        rerank_top_k: int = None
    ):
        # Initialize components
        print("Initializing RAG pipeline...")
        
        # Vector store
        self.vectorstore = vectorstore or VectorStore()
        
        # Retriever
        self.retriever = HybridRetriever(
            vectorstore=self.vectorstore,
            top_k=retrieval_top_k or config.TOP_K_RETRIEVAL
        )
        
        # Reranker (optional - can be resource intensive)
        if use_reranker:
            try:
                self.reranker = Reranker(top_k=rerank_top_k or config.TOP_K_RERANK)
            except Exception as e:
                print(f"Could not load cross-encoder reranker: {e}")
                print("Falling back to simple reranker")
                self.reranker = SimpleReranker(top_k=rerank_top_k or config.TOP_K_RERANK)
        else:
            self.reranker = SimpleReranker(top_k=rerank_top_k or config.TOP_K_RERANK)
        
        # Generator
        self.generator = LegalGenerator(use_openai=use_openai)
        
        print("RAG pipeline initialized")
    
    def query(
        self,
        question: str,
        top_k: int = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_sources: bool = True
    ) -> QueryResult:
        """
        Process a legal question through the RAG pipeline
        
        Args:
            question: User's legal question
            top_k: Number of sources to retrieve
            filter_dict: Metadata filters (e.g., by year, issue area)
            include_sources: Whether to include source details
            
        Returns:
            QueryResult with answer and sources
        """
        # Stage 1: Retrieval
        retrieval_results = self.retriever.retrieve(
            query=question,
            top_k=top_k or config.TOP_K_RETRIEVAL,
            filter_dict=filter_dict
        )
        
        if not retrieval_results:
            return QueryResult(
                query=question,
                answer="I couldn't find any relevant Supreme Court cases for your question. Please try rephrasing or broadening your query.",
                sources=[],
                metadata={"status": "no_results"}
            )
        
        # Stage 2: Reranking
        rerank_results = self.reranker.rerank(
            query=question,
            results=retrieval_results
        )
        
        # Stage 3: Generation
        generation_result = self.generator.generate(
            question=question,
            context_results=rerank_results
        )
        
        # Build response
        return QueryResult(
            query=question,
            answer=generation_result.answer,
            sources=generation_result.sources if include_sources else [],
            retrieval_results=[
                {"id": r.id, "score": r.score, "source": r.source}
                for r in retrieval_results
            ],
            rerank_results=[
                {"id": r.id, "original_score": r.original_score, "rerank_score": r.rerank_score}
                for r in rerank_results
            ],
            metadata={
                "model": generation_result.model_used,
                "retrieval_count": len(retrieval_results),
                "rerank_count": len(rerank_results)
            }
        )
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Simple search without generation
        
        Useful for browsing and exploring cases
        """
        results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        return [
            {
                "id": r.id,
                "content": r.content,
                "case_name": r.metadata.get("case_name", "Unknown"),
                "citation": r.metadata.get("citation", ""),
                "section": r.metadata.get("section_type", ""),
                "score": r.score
            }
            for r in results
        ]
    
    def get_case(self, case_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific case
        """
        # Search by case_id filter
        all_docs = self.vectorstore.collection.get(
            where={"case_id": case_id},
            include=["documents", "metadatas"]
        )
        
        results = []
        for i, doc_id in enumerate(all_docs["ids"]):
            results.append({
                "id": doc_id,
                "content": all_docs["documents"][i],
                "metadata": all_docs["metadatas"][i]
            })
        
        # Sort by chunk index
        results.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
        
        return results
    
    def list_cases(self, limit: int = 100) -> List[Dict[str, str]]:
        """
        List all indexed cases
        """
        # Get sample of documents
        sample = self.vectorstore.collection.peek(limit=limit)
        
        # Extract unique cases
        cases = {}
        for metadata in sample["metadatas"]:
            case_id = metadata.get("case_id", "")
            if case_id and case_id not in cases:
                cases[case_id] = {
                    "case_id": case_id,
                    "case_name": metadata.get("case_name", "Unknown"),
                    "citation": metadata.get("citation", "")
                }
        
        return list(cases.values())
    
    def build_bm25_index(self):
        """Build BM25 index for hybrid search"""
        self.retriever.build_bm25_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            "vectorstore": self.vectorstore.get_collection_stats(),
            "retrieval_top_k": config.TOP_K_RETRIEVAL,
            "rerank_top_k": config.TOP_K_RERANK
        }


def create_pipeline(
    use_reranker: bool = True,
    use_openai: bool = False
) -> RAGPipeline:
    """Factory function to create RAG pipeline"""
    return RAGPipeline(
        use_reranker=use_reranker,
        use_openai=use_openai
    )
