#!/usr/bin/env python3
"""
Production-ready RAG server startup script with Ollama integration.
This script ensures all services are running and models are available.
"""

import asyncio
import subprocess
import time
import os
import sys
import httpx
import psutil
from pathlib import Path
from typing import List, Dict, Any
import json

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

class ProductionRAGManager:
    """Manages the complete RAG system in production mode."""
    
    def __init__(self):
        self.ollama_process = None
        self.rag_process = None
        self.chainlit_process = None
        self.ollama_port = 11434
        self.rag_port = 8000
        self.chainlit_port = 8001
        
    def check_port_available(self, port: int) -> bool:
        """Check if a port is available."""
        for conn in psutil.net_connections():
            if conn.laddr.port == port:
                return False
        return True
    
    def wait_for_service(self, url: str, timeout: int = 30) -> bool:
        """Wait for a service to be available."""
        print(f"Waiting for service at {url}...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                with httpx.Client(timeout=5) as client:
                    response = client.get(url)
                    if response.status_code == 200:
                        print(f"✅ Service at {url} is ready!")
                        return True
            except:
                pass
            time.sleep(1)
        
        print(f"❌ Service at {url} failed to start within {timeout} seconds")
        return False
    
    def start_ollama_service(self) -> bool:
        """Start Ollama service if not running."""
        print("🦙 Starting Ollama service...")
        
        # Check if Ollama is already running
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(f"http://127.0.0.1:{self.ollama_port}/api/tags")
                if response.status_code == 200:
                    print("✅ Ollama is already running!")
                    return True
        except:
            pass
        
        # Start Ollama service
        try:
            if os.name == 'nt':  # Windows
                self.ollama_process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:  # Linux/Mac
                self.ollama_process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True
                )
            
            # Wait for Ollama to be ready
            return self.wait_for_service(f"http://127.0.0.1:{self.ollama_port}/api/tags")
            
        except FileNotFoundError:
            print("❌ Ollama not found! Please install Ollama first.")
            print("   Visit: https://ollama.ai/download")
            return False
        except Exception as e:
            print(f"❌ Failed to start Ollama: {e}")
            return False
    
    def get_available_ollama_models(self) -> List[str]:
        """Get list of available Ollama models."""
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(f"http://127.0.0.1:{self.ollama_port}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    return [model["name"] for model in data.get("models", [])]
        except:
            pass
        return []
    
    def pull_recommended_models(self) -> List[str]:
        """Pull recommended Llama models if not available."""
        recommended_models = [
            "llama3.2:1b",   # Lightweight for testing
            "llama3.2:3b",   # Good balance
            "llama3.2",      # Latest version
        ]
        
        available_models = self.get_available_ollama_models()
        pulled_models = []
        
        for model in recommended_models:
            if model not in available_models:
                print(f"🔄 Pulling {model} (this may take a while)...")
                try:
                    result = subprocess.run(
                        ["ollama", "pull", model],
                        capture_output=True,
                        text=True,
                        timeout=600  # 10 minutes timeout
                    )
                    if result.returncode == 0:
                        print(f"✅ Successfully pulled {model}")
                        pulled_models.append(model)
                    else:
                        print(f"❌ Failed to pull {model}: {result.stderr}")
                except subprocess.TimeoutExpired:
                    print(f"⏰ Timeout pulling {model}")
                except Exception as e:
                    print(f"❌ Error pulling {model}: {e}")
            else:
                print(f"✅ {model} already available")
                pulled_models.append(model)
        
        return pulled_models
    
    def setup_environment(self):
        """Setup environment variables and configuration."""
        print("🔧 Setting up environment...")
        
        # Set default environment variables
        env_vars = {
            "PYTHONPATH": str(Path(__file__).parent),
            "RAG_API_BASE": f"http://127.0.0.1:{self.rag_port}",
            "OLLAMA_BASE_URL": f"http://127.0.0.1:{self.ollama_port}",
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
            print(f"  {key}={value}")
    
    def start_rag_server(self) -> bool:
        """Start the RAG API server."""
        print("🚀 Starting RAG API server...")
        
        try:
            # Change to the correct directory
            os.chdir(Path(__file__).parent)
            
            self.rag_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", 
                "app.main:app",
                "--host", "0.0.0.0",
                "--port", str(self.rag_port),
                "--reload"
            ])
            
            # Wait for RAG server to be ready
            return self.wait_for_service(f"http://127.0.0.1:{self.rag_port}/health")
            
        except Exception as e:
            print(f"❌ Failed to start RAG server: {e}")
            return False
    
    def start_chainlit_ui(self) -> bool:
        """Start the Chainlit UI."""
        print("🎨 Starting Chainlit UI...")
        
        try:
            chainlit_app = Path(__file__).parent / "ui" / "chainlit" / "langchain_app.py"
            
            self.chainlit_process = subprocess.Popen([
                sys.executable, "-m", "chainlit", "run", str(chainlit_app),
                "--port", str(self.chainlit_port),
                "--host", "0.0.0.0"
            ])
            
            # Wait for Chainlit to be ready
            return self.wait_for_service(f"http://127.0.0.1:{self.chainlit_port}")
            
        except Exception as e:
            print(f"❌ Failed to start Chainlit UI: {e}")
            return False
    
    def print_status(self):
        """Print system status and available models."""
        print("\n" + "="*60)
        print("🎉 PRODUCTION RAG SYSTEM READY!")
        print("="*60)
        
        print(f"\n📡 Services:")
        print(f"  • RAG API Server: http://127.0.0.1:{self.rag_port}")
        print(f"  • Chainlit UI: http://127.0.0.1:{self.chainlit_port}")
        print(f"  • Ollama API: http://127.0.0.1:{self.ollama_port}")
        
        available_models = self.get_available_ollama_models()
        if available_models:
            print(f"\n🦙 Available Llama Models ({len(available_models)}):")
            for model in available_models:
                print(f"  • {model}")
        else:
            print(f"\n⚠️  No Llama models available. Use the UI to pull models.")
        
        print(f"\n🎮 How to use:")
        print(f"  1. Open http://127.0.0.1:{self.chainlit_port} in your browser")
        print(f"  2. Click the gear icon ⚙️ to change models")
        print(f"  3. Select 'ollama' backend and choose a Llama model")
        print(f"  4. Start chatting with your documents!")
        
        print(f"\n💡 Model Switching:")
        print(f"  • GUI: Use the dropdown menus in settings")
        print(f"  • Command: Type '/model ollama llama3.2' in chat")
        print(f"  • API: POST to /api/admin/hot-swap/generation")
        
        print(f"\n🛑 To stop: Press Ctrl+C")
        print("="*60)
    
    def cleanup(self):
        """Clean up processes on shutdown."""
        print("\n🛑 Shutting down services...")
        
        processes = [
            ("Chainlit", self.chainlit_process),
            ("RAG Server", self.rag_process),
            ("Ollama", self.ollama_process)
        ]
        
        for name, process in processes:
            if process and process.poll() is None:
                print(f"  Stopping {name}...")
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                except:
                    pass
        
        print("✅ All services stopped!")
    
    async def run(self):
        """Run the complete production RAG system."""
        try:
            print("🚀 Starting Production RAG System...")
            print("This may take a few minutes on first run...")
            
            # Setup environment
            self.setup_environment()
            
            # Start Ollama service
            if not self.start_ollama_service():
                print("⚠️  Continuing without Ollama (cloud models only)")
            else:
                # Pull recommended models in background
                print("📥 Checking for recommended models...")
                self.pull_recommended_models()
            
            # Start RAG server
            if not self.start_rag_server():
                print("❌ Failed to start RAG server!")
                return False
            
            # Start Chainlit UI
            if not self.start_chainlit_ui():
                print("❌ Failed to start Chainlit UI!")
                return False
            
            # Print status
            self.print_status()
            
            # Keep running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Received shutdown signal...")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            self.cleanup()

async def main():
    """Main entry point."""
    manager = ProductionRAGManager()
    await manager.run()

if __name__ == "__main__":
    asyncio.run(main())
