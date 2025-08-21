# 🚀 Advanced RAG System - Complete User Guide

## 📁 **Step 1: Document Ingestion (Adding Your Data)**

### **Method 1: Using the Admin Dashboard (Recommended)**

1. **Start the system:**
```bash
# Make sure you're in the advanced_rag directory
cd /Users/vrajshrimali/Desktop/RAG\ Advance\ Project/advanced_rag

# Install dependencies (if not done)
pip install -r requirements.txt

# Start the FastAPI server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

2. **Access the Admin Dashboard:**
   - Open your browser and go to: `http://localhost:8000`
   - This will redirect to: `http://localhost:8000/api/admin/dashboard`

3. **Upload Documents:**
   - Click on "Upload Documents" section
   - Select your files (PDF, DOCX, TXT, XLSX, CSV)
   - Click "Upload" to process them

### **Method 2: Using the Upload Directory**

1. **Create uploads directory:**
```bash
mkdir -p uploads/documents
```

2. **Copy your documents:**
```bash
# Copy your PDFs, Word docs, etc. to the uploads folder
cp /path/to/your/documents/* uploads/documents/
```

3. **Ingest from directory using API:**
```bash
curl -X POST "http://localhost:8000/api/ingest-directory" \
  -H "Content-Type: application/json" \
  -d '{"directory_path": "uploads/documents"}'
```

### **Method 3: Using the API Directly**

```bash
# Upload a single file
curl -X POST "http://localhost:8000/api/ingest" \
  -F "file=@/path/to/your/document.pdf"
```

---

## 🔍 **Step 2: Getting Outputs (Querying the System)**

### **Method 1: Using the Chainlit Chat UI (Most User-Friendly)**

1. **Start the Chainlit UI:**
```bash
# In a new terminal, navigate to the UI directory
cd ui/chainlit

# Start Chainlit
chainlit run langchain_app.py --host 0.0.0.0 --port 8001
```

2. **Access the Chat Interface:**
   - Open: `http://localhost:8001`
   - Type your questions in natural language
   - Get AI-powered responses with citations

**Example Queries:**
```
"What is machine learning?"
"How do I install Python?"
"Show me Python code for sorting arrays"
"What mathematical formulas are mentioned in the documents?"
"Recent developments in AI from 2023"
```

### **Method 2: Using the REST API**

```bash
# Basic query
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?",
    "top_k": 5
  }'
```

### **Method 3: Using WebSocket for Streaming Responses**

```javascript
// Connect to WebSocket for real-time streaming
const ws = new WebSocket('ws://localhost:8000/ws/query');

ws.onopen = function() {
    ws.send(JSON.stringify({
        query: "Explain machine learning algorithms",
        top_k: 3
    }));
};

ws.onmessage = function(event) {
    const response = JSON.parse(event.data);
    console.log('Streaming response:', response);
};
```

---

## 📊 **Step 3: Advanced Features Usage**

### **🧮 Mathematical Content Queries**

The system now understands mathematical formulas!

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find equations related to quadratic formula",
    "top_k": 3
  }'
```

**What it can find:**
- LaTeX formulas: `$x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}$`
- Mathematical expressions and theorems
- Variables and constants in equations

### **💻 Code-Related Queries**

Perfect for technical documentation!

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Python function for file handling",
    "top_k": 5
  }'
```

**What it can find:**
- Code snippets in any programming language
- Function definitions and class structures
- Programming tutorials and examples

### **⏰ Time-Aware Queries**

The system understands temporal context!

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What happened in AI development in 2023?",
    "top_k": 3
  }'
```

**What it understands:**
- "Recent developments"
- "What happened in 2020?"
- "Latest updates"
- "Historical timeline"

### **🌍 Multi-Language Support**

Automatic translation and cross-language search!

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "¿Qué es la inteligencia artificial?",
    "top_k": 3
  }'
```

---

## 📁 **Step 4: Recommended Document Organization**

### **Best Practices for Document Ingestion:**

1. **Create organized folders:**
```
uploads/
├── technical_docs/          # Programming, API docs
├── research_papers/         # Academic papers with formulas
├── business_docs/          # Reports, presentations
├── manuals/               # User guides, tutorials
└── multilingual/         # Documents in different languages
```

2. **Supported file types:**
   - **PDFs**: Research papers, reports, books
   - **Word Documents**: Articles, documentation
   - **Excel/CSV**: Data tables, spreadsheets
   - **Text Files**: Code files, plain text docs
   - **Images**: Documents with OCR processing

3. **Document naming:**
   - Use descriptive names: `machine_learning_guide_2024.pdf`
   - Include dates when relevant: `quarterly_report_Q1_2024.docx`
   - Avoid special characters: Use underscores instead of spaces

---

## 🎯 **Step 5: Example Complete Workflow**

### **Complete Example: Technical Documentation System**

1. **Prepare your documents:**
```bash
# Create structure
mkdir -p uploads/tech_docs/{python,javascript,ai_research,math_papers}

# Copy your files
cp ~/Documents/python_tutorial.pdf uploads/tech_docs/python/
cp ~/Documents/js_reference.pdf uploads/tech_docs/javascript/
cp ~/Documents/ai_paper_2024.pdf uploads/tech_docs/ai_research/
cp ~/Documents/calculus_formulas.pdf uploads/tech_docs/math_papers/
```

2. **Start the system:**
```bash
# Terminal 1: Start the API server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start the UI (optional)
cd ui/chainlit && chainlit run langchain_app.py --port 8001
```

3. **Ingest all documents:**
```bash
# Ingest the entire directory
curl -X POST "http://localhost:8000/api/ingest-directory" \
  -H "Content-Type: application/json" \
  -d '{"directory_path": "uploads/tech_docs"}'
```

4. **Query the system:**
```bash
# Technical query
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to create a Python class for file handling?",
    "top_k": 3
  }'

# Mathematical query
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me the derivative formula for x squared",
    "top_k": 2
  }'

# AI research query
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Latest transformer architecture improvements in 2024",
    "top_k": 5
  }'
```

---

## 📋 **Step 6: Understanding the Enhanced Output**

### **New Output Format with Advanced Features:**

```json
{
  "query_id": "uuid-string",
  "answer": {
    "content": "Enhanced AI-generated answer",
    "reasoning_steps": ["Analysis step", "Synthesis step"],
    "confidence": 0.87,
    "uncertainty_factors": ["Limited sources", "Complex topic"]
  },
  "citations": [
    {
      "document": "ai_paper_2024.pdf",
      "pages": [12, 13],
      "chunk_id": "chunk_001",
      "excerpt": "Relevant text with mathematical formulas",
      "relevance_score": 0.94,
      "credibility_score": 0.89,
      "extraction_method": "cross_encoder_reranking",
      "content_type": "mathematical",  // NEW: Detected content type
      "code_snippets": [...],          // NEW: Extracted code
      "math_formulas": [...],          // NEW: Mathematical content
      "temporal_relevance": 0.95       // NEW: Time-based relevance
    }
  ],
  "enhanced_features": {              // NEW: Advanced features info
    "query_classification": {
      "type": "code_related",
      "confidence": 0.92,
      "routing_strategy": "hybrid_with_reranking"
    },
    "content_analysis": {
      "math_formulas_found": 3,
      "code_snippets_found": 2,
      "languages_detected": ["python", "javascript"],
      "temporal_context": "recent"
    },
    "translation_applied": false,
    "cross_encoder_reranking": true
  }
}
```

---

## 🛠 **Step 7: Monitoring and Analytics**

### **Admin Dashboard Features:**

1. **Access analytics:**
   - Go to: `http://localhost:8000/api/admin/dashboard`
   - View document statistics
   - Monitor query performance
   - Check system health

2. **Query routing insights:**
```bash
curl "http://localhost:8000/api/evaluation/analytics/dashboard"
```

3. **Performance metrics:**
```bash
curl "http://localhost:8000/api/evaluation/performance/benchmarks"
```

---

## 🚨 **Troubleshooting Common Issues**

### **1. Documents not being ingested:**
```bash
# Check if the uploads directory exists
ls -la uploads/

# Verify file permissions
chmod 755 uploads/
chmod 644 uploads/documents/*
```

### **2. No results for queries:**
```bash
# Check if documents are indexed
curl "http://localhost:8000/api/health"

# Re-index if needed
curl -X POST "http://localhost:8000/api/ingest-directory" \
  -H "Content-Type: application/json" \
  -d '{"directory_path": "uploads/documents"}'
```

### **3. Mathematical formulas not detected:**
- Ensure your PDFs contain text (not just images)
- Use documents with proper LaTeX formatting
- Check if SymPy is installed: `pip install sympy`

### **4. Code snippets not recognized:**
- Verify Pygments is installed: `pip install pygments`
- Use proper code formatting (markdown code blocks)
- Check supported languages in the system

---

## 🎉 **Quick Start Summary**

**For immediate testing:**

1. **Start the system:**
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

2. **Add a test document:**
```bash
# Create a simple test file
echo "Machine learning is a subset of artificial intelligence. The quadratic formula is \$x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}\$." > test_doc.txt

# Upload it
curl -X POST "http://localhost:8000/api/ingest" -F "file=@test_doc.txt"
```

3. **Query it:**
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "top_k": 3}'
```

4. **See the magic happen!** 🎯

Your Advanced RAG system is now ready to handle:
- 📊 Technical documentation with math and code
- 🌍 Multi-language content
- ⏰ Time-sensitive queries  
- 🔍 Intelligent query routing
- 🎯 Enhanced relevance with cross-encoder reranking

**Happy querying! 🚀**
