#!/usr/bin/env python3
"""
Start the JudicAIry API server
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from src.config import config


def main():
    print("🚀 Starting JudicAIry API Server...")
    print(f"📍 Host: {config.API_HOST}")
    print(f"🔌 Port: {config.API_PORT}")
    print(f"📚 Docs: http://localhost:{config.API_PORT}/docs")
    print("-" * 40)
    
    uvicorn.run(
        "src.api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )


if __name__ == "__main__":
    main()
