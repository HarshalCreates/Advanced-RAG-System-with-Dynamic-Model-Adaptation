#!/usr/bin/env python3
"""Quick start script to test the Advanced RAG system."""

import os
import sys
import time
import requests
import json
from pathlib import Path

def print_header():
    print("🚀 Advanced RAG System - Quick Start Guide")
    print("=" * 50)
    print()

def check_server_running():
    """Check if the FastAPI server is running."""
    try:
        response = requests.get("http://localhost:8002/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def create_sample_documents():
    """Create sample documents for testing."""
    print("📁 Creating sample documents...")
    
    # Create uploads directory
    uploads_dir = Path("uploads/sample_docs")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample documents with different content types
    
    # 1. Technical document with code
    tech_doc = """# Python Programming Guide

## Introduction to Python

Python is a high-level programming language known for its simplicity and readability.

## Basic Python Function

Here's a simple Python function:

```python
def calculate_area(radius):
    pi = 3.14159
    area = pi * radius ** 2
    return area

# Usage example
circle_area = calculate_area(5)
print(f"Area of circle: {circle_area}")
```

## Object-Oriented Programming

```python
class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a, b):
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
```

Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.
"""
    
    # 2. Mathematical document
    math_doc = """# Mathematical Formulas and Concepts

## Quadratic Formula

The quadratic formula is used to solve quadratic equations of the form $ax^2 + bx + c = 0$:

$$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$

## Calculus - Integration

The fundamental theorem of calculus states that:

$$\\int_a^b f'(x) dx = f(b) - f(a)$$

## Basic Integration Examples

1. $\\int x^n dx = \\frac{x^{n+1}}{n+1} + C$ (for $n \\neq -1$)

2. $\\int \\sin(x) dx = -\\cos(x) + C$

3. $\\int e^x dx = e^x + C$

## Statistics

The normal distribution probability density function is:

$$f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-\\frac{1}{2}(\\frac{x-\\mu}{\\sigma})^2}$$

Where $\\mu$ is the mean and $\\sigma$ is the standard deviation.
"""
    
    # 3. Recent AI developments document
    ai_doc = """# Artificial Intelligence Developments in 2024

## Recent Breakthroughs

### January 2024
- New transformer architecture improvements showed 15% better performance
- GPT-4 Turbo released with enhanced reasoning capabilities

### March 2024  
- Multimodal AI systems achieved human-level performance on visual reasoning
- Open-source alternatives to proprietary models gained significant traction

### June 2024
- Breakthrough in AI code generation with 95% accuracy on programming tasks
- Real-time language translation reached near-perfect accuracy

## Machine Learning Algorithms

Popular algorithms in 2024:
1. Transformer-based models for NLP
2. Diffusion models for image generation  
3. Reinforcement learning for robotics
4. Graph neural networks for recommendation systems

## Future Predictions

Experts predict that by 2025:
- AI will be integrated into 80% of software applications
- Autonomous systems will become mainstream
- AI-generated content will be indistinguishable from human-created content

## Programming in AI

```python
import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        
    def forward(self, src, tgt):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        output = self.transformer(src_emb, tgt_emb)
        return output
```
"""
    
    # 4. JavaScript tutorial
    js_doc = """# JavaScript Modern Features Guide

## Introduction

JavaScript has evolved significantly with ES6+ features.

## Arrow Functions

```javascript
// Traditional function
function add(a, b) {
    return a + b;
}

// Arrow function
const add = (a, b) => a + b;

// Multiple lines
const processData = (data) => {
    const filtered = data.filter(item => item.active);
    const mapped = filtered.map(item => item.value);
    return mapped.reduce((sum, val) => sum + val, 0);
};
```

## Async/Await

```javascript
async function fetchUserData(userId) {
    try {
        const response = await fetch(`/api/users/${userId}`);
        const userData = await response.json();
        return userData;
    } catch (error) {
        console.error('Error fetching user data:', error);
        throw error;
    }
}

// Usage
fetchUserData(123)
    .then(user => console.log(user))
    .catch(err => console.error(err));
```

## Classes and Modules

```javascript
class DataProcessor {
    constructor(apiKey) {
        this.apiKey = apiKey;
        this.cache = new Map();
    }
    
    async process(data) {
        if (this.cache.has(data.id)) {
            return this.cache.get(data.id);
        }
        
        const processed = await this.transform(data);
        this.cache.set(data.id, processed);
        return processed;
    }
    
    transform(data) {
        // Processing logic
        return { ...data, processed: true, timestamp: Date.now() };
    }
}

export default DataProcessor;
```
"""
    
    # Write documents to files
    documents = [
        ("python_guide.md", tech_doc),
        ("mathematics_formulas.md", math_doc),
        ("ai_developments_2024.md", ai_doc),
        ("javascript_tutorial.md", js_doc)
    ]
    
    for filename, content in documents:
        file_path = uploads_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ Created: {filename}")
    
    print(f"📁 Sample documents created in: {uploads_dir}")
    return str(uploads_dir)

def ingest_documents(directory_path):
    """Ingest documents into the RAG system."""
    print("\n📤 Ingesting documents into RAG system...")
    
    try:
        response = requests.post(
            "http://localhost:8002/api/ingest/directory",
            data={"path": directory_path},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("  ✅ Documents ingested successfully!")
            print(f"  📊 Processed: {result.get('processed_files', 'N/A')} files")
            return True
        else:
            print(f"  ❌ Ingestion failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error during ingestion: {e}")
        return False

def test_queries():
    """Test different types of queries to showcase advanced features."""
    print("\n🔍 Testing Advanced RAG Features...")
    
    test_cases = [
        {
            "name": "Mathematical Query",
            "query": "What is the quadratic formula?",
            "description": "Tests mathematical formula extraction"
        },
        {
            "name": "Code-Related Query", 
            "query": "Show me Python code for calculating area of a circle",
            "description": "Tests code snippet detection"
        },
        {
            "name": "Recent Developments Query",
            "query": "What happened in AI development in 2024?",
            "description": "Tests temporal filtering"
        },
        {
            "name": "Programming Comparison",
            "query": "Compare Python and JavaScript functions",
            "description": "Tests intelligent query routing"
        },
        {
            "name": "Technical Definition",
            "query": "What is a transformer in machine learning?",
            "description": "Tests hybrid retrieval with reranking"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Query: '{test_case['query']}'")
        print(f"   Testing: {test_case['description']}")
        
        try:
            response = requests.post(
                "http://localhost:8002/api/query",
                json={
                    "query": test_case['query'],
                    "top_k": 3
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print("   ✅ Query successful!")
                
                # Extract key information
                answer = result.get('answer', {})
                citations = result.get('citations', [])
                enhanced_features = result.get('enhanced_features', {})
                
                print(f"   📝 Answer (preview): {answer.get('content', '')[:100]}...")
                print(f"   🎯 Confidence: {answer.get('confidence', 0):.2f}")
                print(f"   📚 Citations: {len(citations)} documents")
                
                # Show advanced features
                if enhanced_features:
                    query_class = enhanced_features.get('query_classification', {})
                    content_analysis = enhanced_features.get('content_analysis', {})
                    
                    print(f"   🧠 Query Type: {query_class.get('type', 'N/A')}")
                    print(f"   🧮 Math Formulas: {content_analysis.get('math_formulas_found', 0)}")
                    print(f"   💻 Code Snippets: {content_analysis.get('code_snippets_found', 0)}")
                    print(f"   🔄 Reranking: {enhanced_features.get('cross_encoder_reranking', False)}")
                
                results.append({
                    'test': test_case['name'],
                    'success': True,
                    'confidence': answer.get('confidence', 0),
                    'citations': len(citations)
                })
                
            else:
                print(f"   ❌ Query failed: {response.status_code}")
                results.append({'test': test_case['name'], 'success': False})
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({'test': test_case['name'], 'success': False})
    
    return results

def show_results_summary(results):
    """Show summary of test results."""
    print("\n📊 TEST RESULTS SUMMARY")
    print("=" * 30)
    
    successful = sum(1 for r in results if r.get('success', False))
    total = len(results)
    success_rate = (successful / total) * 100 if total > 0 else 0
    
    print(f"✅ Successful queries: {successful}/{total}")
    print(f"📈 Success rate: {success_rate:.1f}%")
    
    if successful > 0:
        avg_confidence = sum(r.get('confidence', 0) for r in results if r.get('success', False)) / successful
        avg_citations = sum(r.get('citations', 0) for r in results if r.get('success', False)) / successful
        
        print(f"🎯 Average confidence: {avg_confidence:.2f}")
        print(f"📚 Average citations: {avg_citations:.1f}")
    
    print("\n🎉 Advanced RAG System Testing Complete!")
    
    if success_rate >= 80:
        print("🟢 System Status: EXCELLENT - All features working!")
    elif success_rate >= 60:
        print("🟡 System Status: GOOD - Minor issues detected")
    else:
        print("🔴 System Status: NEEDS ATTENTION")

def main():
    """Main function to run the quick start guide."""
    print_header()
    
    # Check if server is running
    print("🔍 Checking if RAG server is running...")
    if not check_server_running():
        print("❌ Server not running! Please start it first:")
        print("   python start_server.py")
        print("   OR: python -m uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload")
        print("\nThen run this script again.")
        sys.exit(1)
    
    print("✅ Server is running!")
    
    # Create sample documents
    docs_path = create_sample_documents()
    
    # Wait a moment
    print("\n⏳ Waiting 2 seconds before ingestion...")
    time.sleep(2)
    
    # Ingest documents
    if not ingest_documents(docs_path):
        print("❌ Document ingestion failed! Please check the server logs.")
        sys.exit(1)
    
    # Wait for indexing to complete
    print("\n⏳ Waiting 3 seconds for indexing to complete...")
    time.sleep(3)
    
    # Test queries
    results = test_queries()
    
    # Show summary
    show_results_summary(results)
    
    print("\n" + "="*50)
    print("🎯 NEXT STEPS:")
    print("1. Visit the admin dashboard: http://localhost:8002/api/admin/dashboard")
    print("2. Start the Chainlit UI: cd ui/chainlit && chainlit run langchain_app.py --port 8001")
    print("3. Add your own documents to uploads/sample_docs/")
    print("4. Try more advanced queries!")
    print("\n📖 For more details, see: USER_GUIDE.md")

if __name__ == "__main__":
    main()
