#!/usr/bin/env python3
"""
Advanced RAG System - Server Startup Script
A robust and user-friendly way to start your Advanced RAG system
"""

import signal
import sys
import os
import socket
import time
from pathlib import Path

def signal_handler(sig, frame):
    print('\n🛑 Shutting down Advanced RAG server...')
    print('   Cleaning up resources...')
    sys.exit(0)

def check_port_available(port):
    """Check if a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False

def find_available_port(start_port=8000, max_attempts=10):
    """Find an available port starting from start_port."""
    for i in range(max_attempts):
        port = start_port + i
        if check_port_available(port):
            return port
    return None

def check_dependencies():
    """Check if all required dependencies are available."""
    print("🔍 Checking system dependencies...")
    
    missing_deps = []
    
    # Check critical dependencies
    try:
        import uvicorn
        print("   ✅ Uvicorn")
    except ImportError:
        missing_deps.append("uvicorn")
        print("   ❌ Uvicorn")
    
    try:
        import fastapi
        print("   ✅ FastAPI")
    except ImportError:
        missing_deps.append("fastapi")
        print("   ❌ FastAPI")
    
    try:
        import torch
        print("   ✅ PyTorch")
    except ImportError:
        print("   ⚠️  PyTorch (optional)")
    
    try:
        import transformers
        print("   ✅ Transformers")
    except ImportError:
        print("   ⚠️  Transformers (optional)")
    
    try:
        import sympy
        print("   ✅ SymPy (Math processing)")
    except ImportError:
        print("   ⚠️  SymPy (Math processing)")
    
    try:
        import pygments
        print("   ✅ Pygments (Code processing)")
    except ImportError:
        print("   ⚠️  Pygments (Code processing)")
    
    if missing_deps:
        print(f"\n❌ Missing critical dependencies: {', '.join(missing_deps)}")
        print("   Install them with: pip install -r requirements.txt")
        return False
    
    print("   ✅ All critical dependencies available")
    return True

def check_project_structure():
    """Check if the project structure is correct."""
    print("📁 Checking project structure...")
    
    required_files = [
        "app/main.py",
        "app/api/routes.py",
        "app/retrieval/service.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"   ❌ {file_path}")
        else:
            print(f"   ✅ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing project files: {', '.join(missing_files)}")
        print("   Make sure you're in the correct directory")
        return False
    
    print("   ✅ Project structure looks good")
    return True

def start_server(port):
    """Start the RAG server on the specified port."""
    try:
        import uvicorn
        from app.main import app
        
        print(f"🚀 Starting Advanced RAG Server on port {port}")
        print("=" * 50)
        
        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        print("✅ Dependencies loaded successfully")
        
        # Warm up the system
        from app.pipeline.manager import PipelineManager
        manager = PipelineManager.get_instance()
        manager.warm_up()
        
        print("✅ Advanced features initialized:")
        print("   • Cross-encoder reranking with BERT/RoBERTa")
        print("   • Mathematical formula extraction (LaTeX, SymPy)")
        print("   • Code snippet detection (15+ languages)")
        print("   • Intelligent query routing & classification")
        print("   • Multi-modal document processing")
        print("   • Temporal filtering & time-aware search")
        print("   • Real-time streaming & WebSocket support")
        print("   • Enterprise security & monitoring")
        print()
        print(f"🌐 Server starting on: http://localhost:{port}")
        print(f"📊 Admin Dashboard: http://localhost:{port}/api/admin/dashboard")
        print(f"🔍 API Health Check: http://localhost:{port}/api/health")
        print(f"📚 API Documentation: http://localhost:{port}/docs")
        print()
        print("📋 Available Endpoints:")
        print("   • POST /api/ingest - Upload documents")
        print("   • POST /api/ingest/directory - Ingest directory")
        print("   • POST /api/query - Query your documents")
        print("   • GET /api/health - Health check")
        print("   • GET /api/admin/dashboard - Admin interface")
        print("   • WebSocket /ws/query - Real-time streaming")
        print()
        print("⚡ Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Start the server
        uvicorn.run(
            app, 
            host='0.0.0.0', 
            port=port, 
            log_level='info',
            access_log=True,
            reload=False  # Disable reload for production
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    """Main function to run the server startup."""
    print("🚀 Advanced RAG System - Server Startup")
    print("=" * 50)
    
    # Ensure we're in the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check if we're in the right directory
    if not Path("app").exists():
        print("❌ Error: 'app' directory not found!")
        print("   Make sure you're running this script from the 'advanced_rag' directory")
        print("   Current directory:", os.getcwd())
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check project structure
    if not check_project_structure():
        sys.exit(1)
    
    print("\n🎯 System ready! Starting server...")
    
    # Try to find an available port
    preferred_port = 8000
    if check_port_available(preferred_port):
        port = preferred_port
        print(f"✅ Port {port} is available")
    else:
        print(f"⚠️  Port {preferred_port} is in use, searching for available port...")
        port = find_available_port(preferred_port)
        if port:
            print(f"✅ Found available port: {port}")
        else:
            print("❌ No available ports found")
            sys.exit(1)
    
    # Start the server
    start_server(port)

if __name__ == "__main__":
    main()
