#!/usr/bin/env python3
"""
CLI Query Tool for Advanced RAG System
Simple command-line interface to test queries against the RAG API.
"""

import requests
import json
import sys
import time
from datetime import datetime
import argparse

class RAGQueryCLI:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def check_health(self):
        """Check if the RAG server is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/api/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Server is healthy! Status: {health_data.get('status', 'unknown')}")
                return True
            else:
                print(f"❌ Health check failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot reach server: {e}")
            return False
    
    def query(self, query_text, top_k=5, show_citations=True, show_enhanced=False):
        """Send a query to the RAG system."""
        print(f"\n🔍 Query: '{query_text}'")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            payload = {
                "query": query_text,
                "top_k": top_k,
                "filters": {}
            }
            
            response = self.session.post(
                f"{self.base_url}/api/query",
                json=payload,
                timeout=30
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                self._display_result(result, response_time, show_citations, show_enhanced)
                return True
            else:
                print(f"❌ Query failed: HTTP {response.status_code}")
                print(f"Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error during query: {e}")
            return False
    
    def _display_result(self, result, response_time, show_citations=True, show_enhanced=False):
        """Display the query result in a formatted way."""
        print(f"⏱️  Response time: {response_time:.2f}s")
        print()
        
        # Display answer
        if 'answer' in result:
            answer = result['answer']
            print("📝 ANSWER:")
            print("-" * 20)
            content = answer.get('content', 'No answer provided')
            print(content)
            print()
            
            confidence = answer.get('confidence', 0)
            print(f"🎯 Confidence: {confidence:.2f}")
            print()
        
        # Display citations
        if show_citations and 'citations' in result:
            citations = result['citations']
            if citations:
                print(f"📚 CITATIONS ({len(citations)} documents):")
                print("-" * 30)
                for i, citation in enumerate(citations, 1):
                    source = citation.get('source', 'Unknown source')
                    score = citation.get('score', 0)
                    chunk_text = citation.get('text', '')[:200] + "..." if len(citation.get('text', '')) > 200 else citation.get('text', '')
                    
                    print(f"{i}. 📄 {source} (Score: {score:.3f})")
                    print(f"   📖 {chunk_text}")
                    print()
        
        # Display enhanced features
        if show_enhanced and 'enhanced_features' in result:
            features = result['enhanced_features']
            print("🧠 ENHANCED FEATURES:")
            print("-" * 25)
            
            if 'query_classification' in features:
                query_class = features['query_classification']
                print(f"🏷️  Query Type: {query_class.get('type', 'N/A')}")
                print(f"📊 Intent: {query_class.get('intent', 'N/A')}")
            
            if 'content_analysis' in features:
                content = features['content_analysis']
                print(f"🧮 Math Formulas: {content.get('math_formulas_found', 0)}")
                print(f"💻 Code Snippets: {content.get('code_snippets_found', 0)}")
            
            if 'cross_encoder_reranking' in features:
                print(f"🔄 Cross-encoder Reranking: {features['cross_encoder_reranking']}")
            
            print()
    
    def interactive_mode(self):
        """Run in interactive mode."""
        print("🚀 Advanced RAG System - Interactive Query CLI")
        print("=" * 50)
        print("Type your queries below. Commands:")
        print("  /help     - Show this help")
        print("  /health   - Check server health")
        print("  /quit     - Exit")
        print("  /enhanced - Toggle enhanced features display")
        print("  /citations - Toggle citations display")
        print("=" * 50)
        
        show_citations = True
        show_enhanced = False
        
        while True:
            try:
                query = input("\n💬 Query: ").strip()
                
                if not query:
                    continue
                
                if query == "/quit":
                    print("👋 Goodbye!")
                    break
                elif query == "/help":
                    print("\nCommands:")
                    print("  /help     - Show this help")
                    print("  /health   - Check server health")
                    print("  /quit     - Exit")
                    print("  /enhanced - Toggle enhanced features display")
                    print("  /citations - Toggle citations display")
                elif query == "/health":
                    self.check_health()
                elif query == "/enhanced":
                    show_enhanced = not show_enhanced
                    print(f"Enhanced features display: {'ON' if show_enhanced else 'OFF'}")
                elif query == "/citations":
                    show_citations = not show_citations
                    print(f"Citations display: {'ON' if show_citations else 'OFF'}")
                else:
                    self.query(query, show_citations=show_citations, show_enhanced=show_enhanced)
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
    
    def batch_test(self):
        """Run a batch of test queries."""
        test_queries = [
            "What are the main components of a RAG system?",
            "Show me Python code for calculating area",
            "What is the quadratic formula?",
            "Explain AI developments in 2024",
            "How does hybrid retrieval work?",
            "What are arrow functions in JavaScript?"
        ]
        
        print("🧪 Running Batch Test Queries")
        print("=" * 35)
        
        results = []
        for i, query in enumerate(test_queries, 1):
            print(f"\n[{i}/{len(test_queries)}] Testing: '{query}'")
            success = self.query(query, top_k=3, show_citations=False, show_enhanced=False)
            results.append(success)
            
            if i < len(test_queries):
                print("\n" + "-" * 60)
        
        # Summary
        successful = sum(results)
        print(f"\n📊 BATCH TEST SUMMARY")
        print("=" * 25)
        print(f"✅ Successful: {successful}/{len(test_queries)}")
        print(f"📈 Success rate: {(successful/len(test_queries)*100):.1f}%")

def main():
    parser = argparse.ArgumentParser(description="CLI Query Tool for Advanced RAG System")
    parser.add_argument("--url", default="http://localhost:8000", help="RAG API base URL")
    parser.add_argument("--query", "-q", help="Single query to execute")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of top results")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--batch-test", "-b", action="store_true", help="Run batch test queries")
    parser.add_argument("--citations", action="store_true", help="Show citations")
    parser.add_argument("--enhanced", action="store_true", help="Show enhanced features")
    
    args = parser.parse_args()
    
    cli = RAGQueryCLI(args.url)
    
    # Check health first
    if not cli.check_health():
        print(f"\n💡 Make sure the RAG server is running:")
        print(f"   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000")
        sys.exit(1)
    
    if args.batch_test:
        cli.batch_test()
    elif args.interactive:
        cli.interactive_mode()
    elif args.query:
        cli.query(args.query, args.top_k, args.citations, args.enhanced)
    else:
        # Default to interactive mode
        cli.interactive_mode()

if __name__ == "__main__":
    main()
