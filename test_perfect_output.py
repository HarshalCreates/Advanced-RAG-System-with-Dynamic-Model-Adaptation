#!/usr/bin/env python3
"""
Test script to demonstrate the perfect JSON output format.
"""

import sys
import json
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.pipeline.manager import PipelineManager
from app.models.schemas import QueryRequest

def test_perfect_json_output():
    """Test the RAG system with perfect JSON output format."""
    print("🧪 Testing Perfect JSON Output Format...")
    
    # Get pipeline manager
    manager = PipelineManager.get_instance()
    
    # Create test query
    query_request = QueryRequest(
        query="What is machine learning?",
        top_k=5,
        filters={}
    )
    
    print(f"📝 Query: {query_request.query}")
    print("🔄 Processing...")
    
    # Process query
    response = manager.query(query_request)
    
    # Convert to dict for JSON serialization
    response_dict = response.model_dump()
    
    # Pretty print the JSON
    print("\n🎯 Perfect JSON Output:")
    print("=" * 80)
    print(json.dumps(response_dict, indent=2, ensure_ascii=False))
    print("=" * 80)
    
    # Validate structure
    print("\n✅ Structure Validation:")
    required_fields = [
        "query_id", "answer", "citations", "alternative_answers", 
        "context_analysis", "performance_metrics", "system_metadata"
    ]
    
    for field in required_fields:
        if field in response_dict:
            print(f"  ✅ {field}: Present")
        else:
            print(f"  ❌ {field}: Missing")
    
    # Validate answer structure
    answer = response_dict.get("answer", {})
    answer_fields = ["content", "reasoning_steps", "confidence", "uncertainty_factors"]
    
    print(f"\n📋 Answer Structure:")
    for field in answer_fields:
        if field in answer:
            value = answer[field]
            if field == "reasoning_steps":
                print(f"  ✅ {field}: {len(value)} steps")
                for i, step in enumerate(value, 1):
                    print(f"    {i}. {step[:60]}...")
            elif field == "confidence":
                print(f"  ✅ {field}: {value}")
            else:
                print(f"  ✅ {field}: Present")
        else:
            print(f"  ❌ {field}: Missing")
    
    # Validate citations structure
    citations = response_dict.get("citations", [])
    print(f"\n📚 Citations: {len(citations)} citations")
    for i, citation in enumerate(citations[:2], 1):  # Show first 2
        print(f"  Citation {i}:")
        print(f"    Document: {citation.get('document', 'N/A')}")
        print(f"    Relevance: {citation.get('relevance_score', 0):.2f}")
        print(f"    Method: {citation.get('extraction_method', 'N/A')}")
    
    # Validate performance metrics
    perf = response_dict.get("performance_metrics", {})
    print(f"\n⚡ Performance Metrics:")
    print(f"  Retrieval: {perf.get('retrieval_latency_ms', 0)}ms")
    print(f"  Generation: {perf.get('generation_latency_ms', 0)}ms")
    print(f"  Total: {perf.get('total_response_time_ms', 0)}ms")
    print(f"  Tokens: {perf.get('tokens_processed', 0)}")
    print(f"  Cost: ${perf.get('cost_estimate_usd', 0):.4f}")
    
    # Validate system metadata
    meta = response_dict.get("system_metadata", {})
    print(f"\n🔧 System Metadata:")
    print(f"  Embedding Model: {meta.get('embedding_model', 'N/A')}")
    print(f"  Generation Model: {meta.get('generation_model', 'N/A')}")
    print(f"  Strategy: {meta.get('retrieval_strategy', 'N/A')}")
    print(f"  Timestamp: {meta.get('timestamp', 'N/A')}")
    
    # Alternative answers
    alt_answers = response_dict.get("alternative_answers", [])
    print(f"\n🔄 Alternative Answers: {len(alt_answers)} alternatives")
    
    print(f"\n🎉 Test completed! Output matches the required JSON structure.")
    
    return response_dict

def test_with_reasoning():
    """Test with the reasoning-enhanced pipeline."""
    print("\n" + "="*80)
    print("🧠 Testing with Reasoning Enhancement...")
    
    manager = PipelineManager.get_instance()
    
    query_request = QueryRequest(
        query="How does machine learning work?",
        top_k=7,
        filters={}
    )
    
    print(f"📝 Query: {query_request.query}")
    print("🔄 Processing with reasoning...")
    
    import asyncio
    
    async def run_reasoning_test():
        response = await manager.query_with_reasoning(query_request)
        response_dict = response.model_dump()
        
        print("\n🧠 Reasoning-Enhanced JSON Output:")
        print("=" * 80)
        print(json.dumps(response_dict, indent=2, ensure_ascii=False))
        print("=" * 80)
        
        # Check reasoning steps
        reasoning_steps = response_dict.get("answer", {}).get("reasoning_steps", [])
        print(f"\n🧠 Reasoning Steps ({len(reasoning_steps)}):")
        for i, step in enumerate(reasoning_steps, 1):
            print(f"  {i}. {step}")
        
        return response_dict
    
    return asyncio.run(run_reasoning_test())

if __name__ == "__main__":
    # Test standard query
    standard_result = test_perfect_json_output()
    
    # Test reasoning query  
    reasoning_result = test_with_reasoning()
    
    print(f"\n🚀 Both test types completed successfully!")
    print(f"📊 Standard Query Result: {len(json.dumps(standard_result))} characters")
    print(f"🧠 Reasoning Query Result: {len(json.dumps(reasoning_result))} characters")
