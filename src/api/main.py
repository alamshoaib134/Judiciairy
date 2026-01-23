"""
FastAPI backend for JudicAIry Legal RAG Assistant
"""

from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.rag.pipeline import RAGPipeline
from src.config import config


# Global pipeline instance
pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app"""
    global pipeline
    
    print("Starting JudicAIry API...")
    
    # Initialize RAG pipeline
    try:
        pipeline = RAGPipeline(
            use_reranker=True,
            use_openai=bool(config.OPENAI_API_KEY)
        )
        
        # Try to build BM25 index if documents exist
        stats = pipeline.get_stats()
        if stats["vectorstore"]["total_documents"] > 0:
            print("Building BM25 index for hybrid search...")
            pipeline.build_bm25_index()
            
    except Exception as e:
        print(f"Warning: Could not fully initialize pipeline: {e}")
        print("API will start but some features may be limited")
    
    yield
    
    print("Shutting down JudicAIry API...")


# Create FastAPI app
app = FastAPI(
    title="JudicAIry",
    description="Legal RAG Assistant for U.S. Supreme Court Opinions",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for legal queries"""
    question: str = Field(..., description="Legal question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of sources to retrieve")
    filter_year: Optional[str] = Field(default=None, description="Filter by decision year")
    filter_issue_area: Optional[str] = Field(default=None, description="Filter by issue area")


class SearchRequest(BaseModel):
    """Request model for search"""
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results")


class SourceInfo(BaseModel):
    """Source information model"""
    case_name: str
    citation: Optional[str] = None
    section: str
    excerpt: str
    score: float


class QueryResponse(BaseModel):
    """Response model for legal queries"""
    query: str
    answer: str
    sources: List[SourceInfo]
    metadata: Dict[str, Any]


class SearchResult(BaseModel):
    """Search result model"""
    id: str
    content: str
    case_name: str
    citation: Optional[str] = None
    section: str
    score: float


class StatsResponse(BaseModel):
    """Statistics response model"""
    total_documents: int
    collection_name: str
    section_distribution_sample: Dict[str, int]


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "JudicAIry",
        "description": "Legal RAG Assistant for U.S. Supreme Court Opinions",
        "version": "0.1.0",
        "endpoints": {
            "query": "/api/query",
            "search": "/api/search",
            "cases": "/api/cases",
            "stats": "/api/stats"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    global pipeline
    
    if pipeline is None:
        return {"status": "degraded", "message": "Pipeline not initialized"}
    
    try:
        stats = pipeline.get_stats()
        return {
            "status": "healthy",
            "documents_indexed": stats["vectorstore"]["total_documents"]
        }
    except Exception as e:
        return {"status": "degraded", "message": str(e)}


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Answer a legal question using RAG
    
    This endpoint:
    1. Retrieves relevant Supreme Court opinion excerpts
    2. Reranks them for relevance
    3. Generates an answer with citations
    """
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Build filter dict
    filter_dict = None
    if request.filter_year or request.filter_issue_area:
        filter_dict = {}
        if request.filter_year:
            filter_dict["term"] = {"$eq": request.filter_year}
        if request.filter_issue_area:
            filter_dict["issue_areas"] = {"$contains": request.filter_issue_area}
    
    try:
        result = pipeline.query(
            question=request.question,
            top_k=request.top_k,
            filter_dict=filter_dict
        )
        
        return QueryResponse(
            query=result.query,
            answer=result.answer,
            sources=[
                SourceInfo(
                    case_name=s["case_name"],
                    citation=s.get("citation"),
                    section=s["section"],
                    excerpt=s["excerpt"],
                    score=s["score"]
                )
                for s in result.sources
            ],
            metadata=result.metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/api/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    """
    Search for relevant Supreme Court opinions
    
    Returns matching document chunks without generating an answer
    """
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        results = pipeline.search(
            query=request.query,
            top_k=request.top_k
        )
        
        return [
            SearchResult(
                id=r["id"],
                content=r["content"],
                case_name=r["case_name"],
                citation=r.get("citation"),
                section=r["section"],
                score=r["score"]
            )
            for r in results
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/api/cases")
async def list_cases(
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of cases to return")
):
    """
    List all indexed cases
    """
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        cases = pipeline.list_cases(limit=limit)
        return {"cases": cases, "count": len(cases)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list cases: {str(e)}")


@app.get("/api/cases/{case_id}")
async def get_case(case_id: str):
    """
    Get all chunks for a specific case
    """
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        chunks = pipeline.get_case(case_id)
        
        if not chunks:
            raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")
        
        return {
            "case_id": case_id,
            "chunks": chunks,
            "chunk_count": len(chunks)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get case: {str(e)}")


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get statistics about the indexed collection
    """
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        stats = pipeline.get_stats()
        vs_stats = stats["vectorstore"]
        
        return StatsResponse(
            total_documents=vs_stats["total_documents"],
            collection_name=vs_stats["collection_name"],
            section_distribution_sample=vs_stats.get("section_distribution_sample", {})
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# Run with: uvicorn src.api.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )
