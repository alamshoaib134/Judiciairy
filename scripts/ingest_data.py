#!/usr/bin/env python3
"""
Data ingestion script for JudicAIry
Downloads CaseSumm dataset, processes cases, and indexes them in ChromaDB
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.data.loader import DataLoader, SCOTUSCase
from src.data.chunker import DocumentChunker
from src.embeddings.vectorstore import VectorStore


def main():
    parser = argparse.ArgumentParser(
        description="Ingest SCOTUS data into JudicAIry"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of cases to process (for testing)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (train, validation, test)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the vector store before ingestion"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for indexing"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("JudicAIry Data Ingestion")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n📥 Step 1: Loading CaseSumm dataset...")
    loader = DataLoader()
    
    cases = list(loader.load_casesumm(
        split=args.split,
        limit=args.limit
    ))
    
    if not cases:
        print("❌ No cases loaded! Check your internet connection and HuggingFace access.")
        sys.exit(1)
    
    print(f"✅ Loaded {len(cases)} cases")
    
    # Step 2: Chunk documents
    print("\n📝 Step 2: Chunking documents...")
    chunker = DocumentChunker()
    chunks = chunker.chunk_cases(cases)
    
    print(f"✅ Created {len(chunks)} chunks")
    
    # Step 3: Index in vector store
    print("\n🗄️ Step 3: Indexing in vector store...")
    vectorstore = VectorStore()
    
    if args.reset:
        print("⚠️ Resetting vector store...")
        vectorstore.reset()
    
    vectorstore.add_chunks(chunks, batch_size=args.batch_size)
    
    # Step 4: Save processed cases
    print("\n💾 Step 4: Saving processed cases...")
    loader.save_processed_cases(cases)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Ingestion Complete!")
    print("=" * 60)
    
    stats = vectorstore.get_collection_stats()
    print(f"📊 Total documents indexed: {stats['total_documents']}")
    print(f"📁 Collection name: {stats['collection_name']}")
    print(f"📂 Persist directory: {config.CHROMA_PERSIST_DIR}")
    
    print("\n🚀 Next steps:")
    print("  1. Start the API: uvicorn src.api.main:app --reload")
    print("  2. Start the UI: streamlit run src/ui/app.py")
    print("  3. Or use Python: from src.rag.pipeline import RAGPipeline")


if __name__ == "__main__":
    main()
