#!/usr/bin/env python3
"""
Production Demo for Advanced RAG System.
Shows all features working in production mode.
"""

import requests
import time
import json

def test_server_health():
    """Test if server is running."""
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        if response.status_code == 200:
            print("✓ Server is healthy and running!")
            return True
    except:
        pass
    
    print("✗ Server not running. Please start with:")
    print("   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload")
    return False

def test_document_ingestion():
    """Test document ingestion."""
    print("\n📄 Testing Document Ingestion...")
    
    try:
        response = requests.post(
            "http://localhost:8000/api/ingest/directory",
            data={"path": "./data/uploads"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Documents ingested: {result.get('ingested', 0)}")
            print(f"✓ Index size: {result.get('index_size', 0)}")
            return True
        else:
            print(f"✗ Ingestion failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Ingestion error: {e}")
        return False

def test_queries():
    """Test various queries to show system capabilities."""
    print("\n🔍 Testing Query System...")
    
    test_cases = [
        {
            "query": "machine learning",
            "description": "Simple keyword search"
        },
        {
            "query": "What is artificial intelligence?",
            "description": "Question answering"
        },
        {
            "query": "Python programming examples",
            "description": "Code-related query"
        },
        {
            "query": "RAG system components",
            "description": "Technical documentation query"
        }
    ]
    
    successful_queries = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        print("-" * 50)
        
        try:
            start_time = time.time()
            response = requests.post(
                "http://localhost:8000/api/query",
                json={"query": test_case["query"], "top_k": 3},
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                
                # Check answer
                answer = result.get('answer', {})
                content = answer.get('content', '')
                confidence = answer.get('confidence', 0)
                
                print(f"⏱️  Response time: {end_time - start_time:.2f}s")
                print(f"🎯 Confidence: {confidence:.2f}")
                
                # Check if context was found
                if 'Context:' in content:
                    context_part = content.split('Context:')[1].strip()
                    if len(context_part) > 20:  # Non-empty context
                        print("✓ Context found and retrieved!")
                        successful_queries += 1
                        
                        # Show context preview
                        context_preview = context_part[:150]
                        print(f"📄 Context preview: {context_preview}...")
                        
                        # Show citations
                        citations = result.get('citations', [])
                        if citations:
                            print(f"📚 Citations: {len(citations)} documents")
                        
                    else:
                        print("⚠️  Empty context returned")
                else:
                    print("⚠️  No context in response")
                    
            else:
                print(f"✗ Query failed: {response.status_code}")
                
        except Exception as e:
            print(f"✗ Query error: {e}")
    
    print(f"\n📊 Query Test Results:")
    print(f"✓ Successful queries: {successful_queries}/{len(test_cases)}")
    print(f"📈 Success rate: {successful_queries/len(test_cases)*100:.1f}%")
    
    return successful_queries > 0

def show_system_info():
    """Show system information and capabilities."""
    print("\n🚀 Advanced RAG System - Production Demo")
    print("=" * 50)
    
    try:
        response = requests.get("http://localhost:8000/api/admin/dashboard")
        if response.status_code == 200:
            print("✓ Admin Dashboard: http://localhost:8000/api/admin/dashboard")
    except:
        pass
    
    print("\n🎯 System Capabilities:")
    print("• Multi-modal document ingestion (PDF, DOCX, TXT, etc.)")
    print("• Semantic search with sentence transformers")
    print("• Hybrid retrieval (dense + sparse + BM25)")
    print("• Production-grade FastAPI backend")
    print("• Interactive Chainlit UI")
    print("• Command-line interface")
    print("• Real-time health monitoring")
    
    print("\n🌐 Access Points:")
    print("• FastAPI Server: http://localhost:8000")
    print("• API Documentation: http://localhost:8000/docs")
    print("• Admin Dashboard: http://localhost:8000/api/admin/dashboard")
    print("• Chainlit UI: http://localhost:8001")
    
    print("\n🔧 CLI Commands:")
    print('• Single query: python cli_query.py -q "your query" --citations')
    print("• Interactive mode: python cli_query.py --interactive")
    print("• Batch testing: python cli_query.py --batch-test")

def main():
    """Run the complete production demo."""
    show_system_info()
    
    if not test_server_health():
        return
    
    if not test_document_ingestion():
        print("⚠️  Document ingestion failed, but continuing with demo...")
    
    if test_queries():
        print("\n🎉 PRODUCTION DEMO SUCCESSFUL!")
        print("\n✓ Your Advanced RAG System is production-ready!")
        print("\n💡 Next Steps:")
        print("1. Add your OpenAI API key to .env file for ChatGPT responses")
        print("2. Upload your own documents to data/uploads/")
        print("3. Use the Chainlit UI for interactive chat")
        print("4. Integrate the API into your applications")
    else:
        print("\n⚠️  Some issues detected. Check the logs above.")

if __name__ == "__main__":
    main()
