@echo off
echo 🚀 Optimized RAG System Deployment (No Model Containerization)
echo ==============================================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker first.
    pause
    exit /b 1
)

echo 📋 Prerequisites:
echo 1. Docker and Docker Compose installed
echo 2. Ollama installed on host machine (optional, for local models)
echo 3. API keys configured (optional, for cloud models)
echo.

REM Check if .env exists
if not exist .env (
    echo ⚠️  No .env file found. Creating default configuration...
    (
        echo # RAG System Configuration
        echo EMBEDDING_BACKEND=sentence-transformers
        echo EMBEDDING_MODEL=all-MiniLM-L6-v2
        echo GENERATION_BACKEND=echo
        echo GENERATION_MODEL=gpt-4o
        echo RETRIEVER_BACKEND=faiss
        echo LOG_LEVEL=INFO
        echo ENVIRONMENT=production
        echo.
        echo # Optional: Add your API keys here
        echo # OPENAI_API_KEY=your_key_here
        echo # ANTHROPIC_API_KEY=your_key_here
        echo # COHERE_API_KEY=your_key_here
    ) > .env
    echo ✅ Created .env file with default settings
)

echo 🔧 Building optimized containers (should be much faster now)...
echo    - Removed heavy ML libraries (transformers, torch, etc.)
echo    - Models will be accessed externally
echo.

REM Build and start services
docker-compose up --build -d

echo.
echo ⏳ Waiting for services to start...
timeout /t 10 /nobreak >nul

REM Check if services are healthy
echo 🔍 Checking service health...
curl -f http://localhost:8000/api/health >nul 2>&1
if errorlevel 1 (
    echo ❌ API service is not responding
) else (
    echo ✅ API service is healthy
)

curl -f http://localhost:8501/healthz >nul 2>&1
if errorlevel 1 (
    echo ❌ UI service is not responding
) else (
    echo ✅ UI service is healthy
)

echo.
echo 🎉 Deployment Complete!
echo.
echo 📱 Access Points:
echo    - API: http://localhost:8000
echo    - UI: http://localhost:8501
echo    - API Docs: http://localhost:8000/docs
echo.
echo 🔧 Model Configuration Options:
echo.
echo Option 1: Use Cloud Models (Recommended)
echo    - Edit .env file and add your API keys:
echo      OPENAI_API_KEY=your_key_here
echo      ANTHROPIC_API_KEY=your_key_here
echo    - Restart: docker-compose restart
echo.
echo Option 2: Use Local Ollama Models
echo    - Install Ollama on your host machine:
echo      Download from https://ollama.ai/download
echo    - Start Ollama: ollama serve
echo    - Pull models: ollama pull llama3.2:3b
echo    - The containers will automatically connect to host Ollama
echo.
echo Option 3: Use Echo Mode (Default)
echo    - No setup required, uses simple text generation
echo    - Good for testing and development
echo.
echo 🛠️  Management Commands:
echo    - View logs: docker-compose logs -f
echo    - Stop services: docker-compose down
echo    - Restart: docker-compose restart
echo    - Rebuild: docker-compose up --build -d
echo.
echo 📚 Next Steps:
echo 1. Upload documents at http://localhost:8501
echo 2. Ask questions and test the system
echo 3. Configure your preferred model backend
echo.
pause

