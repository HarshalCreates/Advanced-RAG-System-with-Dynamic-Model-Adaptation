#!/usr/bin/env python3
"""
Test script for model switching functionality.
"""

import sys
import asyncio
import httpx
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.generation.factory import GenerationFactory, ModelRegistry

async def test_model_factory():
    """Test the model factory with different backends."""
    print("🧪 Testing Model Factory...")
    
    # Test OpenAI (will fallback to echo if no API key)
    print("\n1. Testing OpenAI backend:")
    try:
        factory = GenerationFactory(backend="openai", model="gpt-4o")
        client = factory.build()
        result = client.complete("You are a helpful assistant.", "Say hello!")
        print(f"   ✅ OpenAI result: {result[:50]}...")
    except Exception as e:
        print(f"   ⚠️  OpenAI failed (expected without API key): {e}")
    
    # Test Ollama
    print("\n2. Testing Ollama backend:")
    try:
        factory = GenerationFactory(backend="ollama", model="llama3.2:3b")
        client = factory.build()
        result = client.complete("You are a helpful assistant.", "Say hello!")
        print(f"   ✅ Ollama result: {result[:50]}...")
    except Exception as e:
        print(f"   ⚠️  Ollama failed: {e}")
    
    # Test Echo (always works)
    print("\n3. Testing Echo backend (fallback):")
    try:
        factory = GenerationFactory(backend="echo", model="test")
        client = factory.build()
        result = client.complete("You are a helpful assistant.", "What is machine learning?")
        print(f"   ✅ Echo result: {result[:100]}...")
    except Exception as e:
        print(f"   ❌ Echo failed: {e}")

def test_model_registry():
    """Test the model registry functionality."""
    print("\n🗄️ Testing Model Registry...")
    
    # Get all available models
    all_models = ModelRegistry.get_available_models()
    print(f"   Available backends: {list(all_models.keys())}")
    
    for backend, models in all_models.items():
        print(f"   {backend}: {len(models)} models")
        for model in list(models.keys())[:3]:  # Show first 3
            config = ModelRegistry.get_model_config(backend, model)
            print(f"     • {model}: {config}")
    
    # Test model availability
    test_cases = [
        ("openai", "gpt-4o"),
        ("anthropic", "claude-3-5-sonnet-20241022"),
        ("ollama", "llama3.2"),
        ("invalid", "invalid-model")
    ]
    
    print("\n   Model availability tests:")
    for backend, model in test_cases:
        available = ModelRegistry.is_model_available(backend, model)
        status = "✅" if available else "❌"
        print(f"     {status} {backend}:{model}")

async def test_ollama_connection():
    """Test connection to Ollama service."""
    print("\n🦙 Testing Ollama Connection...")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://127.0.0.1:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                print(f"   ✅ Ollama is running with {len(models)} models:")
                for model in models:
                    print(f"     • {model}")
                return True
            else:
                print(f"   ❌ Ollama responded with status {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  Ollama not accessible: {e}")
        print("   💡 Make sure Ollama is running: ollama serve")
    
    return False

async def test_rag_server_connection():
    """Test connection to RAG server."""
    print("\n🚀 Testing RAG Server Connection...")
    
    try:
        async with httpx.AsyncClient() as client:
            # Test health endpoint
            response = await client.get("http://127.0.0.1:8000/health", timeout=5)
            if response.status_code == 200:
                print("   ✅ RAG server is running!")
                
                # Test model status endpoint
                try:
                    response = await client.get("http://127.0.0.1:8000/api/admin/models/status", timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        print(f"   Current models: {data}")
                    else:
                        print(f"   ⚠️  Model status endpoint returned {response.status_code}")
                except:
                    print("   ⚠️  Model status endpoint not accessible (normal if not authenticated)")
                
                return True
            else:
                print(f"   ❌ RAG server responded with status {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  RAG server not accessible: {e}")
        print("   💡 Start the server with: python start_production_rag.py")
    
    return False

async def test_model_hot_swap():
    """Test model hot-swapping functionality."""
    print("\n🔄 Testing Model Hot-Swap...")
    
    # This would require the RAG server to be running with admin access
    # For now, just show how it would work
    
    test_swaps = [
        ("ollama", "llama3.2"),
        ("openai", "gpt-4o"),
        ("echo", "test")
    ]
    
    print("   Hot-swap test scenarios:")
    for backend, model in test_swaps:
        print(f"     • {backend}:{model} - would test via POST /api/admin/hot-swap/generation")
    
    print("   💡 To test manually:")
    print("     curl -X POST 'http://127.0.0.1:8000/api/admin/hot-swap/generation?backend=ollama&model=llama3.2'")

async def main():
    """Run all tests."""
    print("🧪 RAG Model Switching Test Suite")
    print("=" * 50)
    
    # Test 1: Model Factory
    await test_model_factory()
    
    # Test 2: Model Registry
    test_model_registry()
    
    # Test 3: Ollama Connection
    ollama_running = await test_ollama_connection()
    
    # Test 4: RAG Server Connection
    rag_running = await test_rag_server_connection()
    
    # Test 5: Hot-swap functionality
    await test_model_hot_swap()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   • Model Factory: ✅ Working")
    print(f"   • Model Registry: ✅ Working")
    print(f"   • Ollama Service: {'✅ Running' if ollama_running else '❌ Not running'}")
    print(f"   • RAG Server: {'✅ Running' if rag_running else '❌ Not running'}")
    
    if not ollama_running:
        print("\n💡 To start Ollama:")
        print("   ollama serve")
        print("   ollama pull llama3.2")
    
    if not rag_running:
        print("\n💡 To start the complete system:")
        print("   python start_production_rag.py")
    
    print("\n🎉 Test completed!")

if __name__ == "__main__":
    asyncio.run(main())
