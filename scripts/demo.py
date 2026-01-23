#!/usr/bin/env python3
"""
Demo script showing JudicAIry capabilities
Run this after ingesting data to see the RAG system in action
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.pipeline import RAGPipeline


def main():
    print("=" * 60)
    print("JudicAIry Demo")
    print("=" * 60)
    
    # Initialize pipeline
    print("\n🔧 Initializing RAG pipeline...")
    pipeline = RAGPipeline(
        use_reranker=False,  # Faster for demo
        use_openai=False
    )
    
    # Check if we have data
    stats = pipeline.get_stats()
    doc_count = stats["vectorstore"]["total_documents"]
    
    if doc_count == 0:
        print("❌ No documents indexed!")
        print("Run: python scripts/ingest_data.py --limit 100")
        sys.exit(1)
    
    print(f"✅ Found {doc_count} indexed documents")
    
    # Demo queries
    demo_queries = [
        "What is the standard for freedom of speech?",
        "How does the Fourth Amendment apply to searches?",
        "What are the requirements for due process?"
    ]
    
    print("\n" + "=" * 60)
    print("Demo Queries")
    print("=" * 60)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 40)
        
        # Search (fast, no LLM)
        results = pipeline.search(query, top_k=3)
        
        print(f"📚 Found {len(results)} relevant sources:")
        for j, r in enumerate(results, 1):
            print(f"\n  [{j}] {r['case_name']}")
            if r.get('citation'):
                print(f"      Citation: {r['citation']}")
            print(f"      Section: {r['section']}")
            print(f"      Score: {r['score']:.4f}")
            print(f"      Preview: {r['content'][:150]}...")
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Mode")
    print("=" * 60)
    print("Type your legal questions (or 'quit' to exit)")
    
    while True:
        try:
            query = input("\n❓ Your question: ").strip()
            
            if query.lower() in ('quit', 'exit', 'q'):
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print("🔍 Searching...")
            results = pipeline.search(query, top_k=3)
            
            if results:
                print(f"\n📚 Top {len(results)} relevant sources:")
                for j, r in enumerate(results, 1):
                    print(f"\n  [{j}] {r['case_name']}")
                    print(f"      {r['content'][:200]}...")
            else:
                print("No relevant cases found.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
