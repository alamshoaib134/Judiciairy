#!/usr/bin/env python3
"""
Start the JudicAIry Streamlit UI
"""

import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).parent.parent
    app_path = project_root / "src" / "ui" / "app.py"
    
    print("🚀 Starting JudicAIry UI...")
    print(f"📍 App: {app_path}")
    print("-" * 40)
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", "8501",
        "--server.address", "localhost"
    ])


if __name__ == "__main__":
    main()
