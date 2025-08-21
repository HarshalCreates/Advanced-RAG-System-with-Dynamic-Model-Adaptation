#!/bin/bash

echo "🚀 Optimized RAG System Deployment (No Model Containerization)"
echo "=============================================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "📋 Prerequisites:"
echo "1. Docker and Docker Compose installed"
echo "2. Ollama installed on host machine (optional, for local models)"
echo "3. API keys configured (optional, for cloud models)"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating default configuration..."
    cat > .env << EOF
# RAG System Configuration
EMBEDDING_BACKEND=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2
GENERATION_BACKEND=echo
GENERATION_MODEL=gpt-4o
RETRIEVER_BACKEND=faiss
LOG_LEVEL=INFO
ENVIRONMENT=production

# Optional: Add your API keys here
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
# COHERE_API_KEY=your_key_here
EOF
    echo "✅ Created .env file with default settings"
fi

echo "🔧 Building optimized containers (should be much faster now)..."
echo "   - Removed heavy ML libraries (transformers, torch, etc.)"
echo "   - Models will be accessed externally"
echo ""

# Build and start services
docker-compose up --build -d

echo ""
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are healthy
echo "🔍 Checking service health..."
if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
    echo "✅ API service is healthy"
else
    echo "❌ API service is not responding"
fi

if curl -f http://localhost:8501/healthz > /dev/null 2>&1; then
    echo "✅ UI service is healthy"
else
    echo "❌ UI service is not responding"
fi

echo ""
echo "🎉 Deployment Complete!"
echo ""
echo "📱 Access Points:"
echo "   - API: http://localhost:8000"
echo "   - UI: http://localhost:8501"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo "🔧 Model Configuration Options:"
echo ""
echo "Option 1: Use Cloud Models (Recommended)"
echo "   - Edit .env file and add your API keys:"
echo "     OPENAI_API_KEY=your_key_here"
echo "     ANTHROPIC_API_KEY=your_key_here"
echo "   - Restart: docker-compose restart"
echo ""
echo "Option 2: Use Local Ollama Models"
echo "   - Install Ollama on your host machine:"
echo "     curl -fsSL https://ollama.ai/install.sh | sh"
echo "   - Start Ollama: ollama serve"
echo "   - Pull models: ollama pull llama3.2:3b"
echo "   - The containers will automatically connect to host Ollama"
echo ""
echo "Option 3: Use Echo Mode (Default)"
echo "   - No setup required, uses simple text generation"
echo "   - Good for testing and development"
echo ""
echo "🛠️  Management Commands:"
echo "   - View logs: docker-compose logs -f"
echo "   - Stop services: docker-compose down"
echo "   - Restart: docker-compose restart"
echo "   - Rebuild: docker-compose up --build -d"
echo ""
echo "📚 Next Steps:"
echo "1. Upload documents at http://localhost:8501"
echo "2. Ask questions and test the system"
echo "3. Configure your preferred model backend"
