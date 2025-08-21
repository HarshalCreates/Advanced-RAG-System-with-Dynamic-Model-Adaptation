#!/usr/bin/env python3
"""
RAG System Launcher - Automatically starts the server from the correct directory
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the RAG server from the correct directory."""
    
    # Get the directory containing this script
    current_dir = Path(__file__).parent
    
    # Look for the advanced_rag directory
    advanced_rag_dir = current_dir / "advanced_rag"
    
    if not advanced_rag_dir.exists():
        print("❌ Error: 'advanced_rag' directory not found!")
        print(f"   Current directory: {current_dir}")
        print("   Make sure this script is in the root directory of your RAG project")
        sys.exit(1)
    
    # Check if start_server.py exists
    start_server_script = advanced_rag_dir / "start_server.py"
    if not start_server_script.exists():
        print("❌ Error: 'start_server.py' not found in advanced_rag directory!")
        sys.exit(1)
    
    print("🚀 Starting Advanced RAG System...")
    print(f"   Working directory: {advanced_rag_dir}")
    
    # Change to the advanced_rag directory and run the server
    try:
        os.chdir(advanced_rag_dir)
        subprocess.run([sys.executable, "start_server.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")

if __name__ == "__main__":
    main()
