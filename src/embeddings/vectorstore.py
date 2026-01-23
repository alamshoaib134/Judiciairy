"""
Vector store for SCOTUS document embeddings
Uses ChromaDB for persistence and retrieval
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.config import config
from src.data.chunker import DocumentChunk
from src.embeddings.embedder import Embedder


class VectorStore:
    """ChromaDB-based vector store for legal documents"""
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_dir: Optional[Path] = None,
        embedder: Optional[Embedder] = None
    ):
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.persist_dir = persist_dir or config.CHROMA_PERSIST_DIR
        
        # Initialize embedder
        self.embedder = embedder or Embedder()
        
        # Initialize ChromaDB client with persistence
        print(f"Initializing ChromaDB at {self.persist_dir}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        print(f"Collection '{self.collection_name}' has {self.collection.count()} documents")
    
    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 100
    ):
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of DocumentChunk objects
            batch_size: Batch size for adding documents
        """
        if not chunks:
            print("No chunks to add")
            return
            
        print(f"Adding {len(chunks)} chunks to vector store...")
        
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Extract data for ChromaDB
            ids = [chunk.chunk_id for chunk in batch]
            documents = [chunk.content for chunk in batch]
            metadatas = [self._prepare_metadata(chunk) for chunk in batch]
            
            # Generate embeddings
            embeddings = self.embedder.embed_texts(documents, show_progress=False)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings.tolist(),
                metadatas=metadatas
            )
            
            print(f"  Added batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}")
        
        print(f"Vector store now contains {self.collection.count()} documents")
    
    def _prepare_metadata(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """Prepare metadata for ChromaDB (must be flat)"""
        metadata = {
            "case_id": chunk.case_id,
            "case_name": chunk.case_name,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "section_type": chunk.section_type,
        }
        
        # Add flat metadata fields
        if chunk.metadata:
            if chunk.metadata.get("citation"):
                metadata["citation"] = chunk.metadata["citation"]
            if chunk.metadata.get("decision_date"):
                metadata["decision_date"] = chunk.metadata["decision_date"]
            if chunk.metadata.get("term"):
                metadata["term"] = chunk.metadata["term"]
            if chunk.metadata.get("token_count"):
                metadata["token_count"] = chunk.metadata["token_count"]
            # Convert lists to comma-separated strings
            if chunk.metadata.get("legal_provisions"):
                metadata["legal_provisions"] = ",".join(chunk.metadata["legal_provisions"])
            if chunk.metadata.get("issue_areas"):
                metadata["issue_areas"] = ",".join(chunk.metadata["issue_areas"])
                
        return metadata
    
    def search(
        self,
        query: str,
        top_k: int = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Metadata filters (ChromaDB where clause)
            
        Returns:
            List of search results with documents, metadata, and scores
        """
        top_k = top_k or config.TOP_K_RETRIEVAL
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Build query params
        query_params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        
        if filter_dict:
            query_params["where"] = filter_dict
        
        # Execute search
        results = self.collection.query(**query_params)
        
        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "score": 1 - results["distances"][0][i]  # Convert distance to similarity
            })
            
        return formatted_results
    
    def search_by_case(
        self,
        case_name: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for chunks from a specific case"""
        return self.search(
            query=case_name,
            top_k=top_k,
            filter_dict={"case_name": {"$eq": case_name}}
        )
    
    def search_by_section(
        self,
        query: str,
        section_type: str,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Search within a specific section type"""
        return self.search(
            query=query,
            top_k=top_k,
            filter_dict={"section_type": {"$eq": section_type}}
        )
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        count = self.collection.count()
        
        # Sample some documents to get metadata distribution
        if count > 0:
            sample = self.collection.peek(limit=min(100, count))
            section_types = [m.get("section_type", "unknown") for m in sample["metadatas"]]
            section_dist = {s: section_types.count(s) for s in set(section_types)}
        else:
            section_dist = {}
            
        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "section_distribution_sample": section_dist
        }
    
    def delete_collection(self):
        """Delete the entire collection"""
        self.client.delete_collection(self.collection_name)
        print(f"Deleted collection '{self.collection_name}'")
    
    def reset(self):
        """Reset the collection (delete and recreate)"""
        self.delete_collection()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Reset collection '{self.collection_name}'")


def create_vectorstore(collection_name: Optional[str] = None) -> VectorStore:
    """Factory function to create vector store"""
    return VectorStore(collection_name=collection_name)
