# Advanced Multi-Modal RAG System

A production-grade Retrieval-Augmented Generation (RAG) stack with multi-modal ingestion (PDF, DOCX, XLSX, CSV, TXT, images with OCR), hybrid retrieval (dense + TF‑IDF + BM25 with fusion), real-time model adaptation, FastAPI backend (REST + WebSockets), admin dashboard, and Chainlit chat UI.

## 🚀 Features

- **Multi-modal Ingestion**: Handle PDF, DOCX, XLSX, CSV, TXT, and images with OCR fallback
- **Structure-aware Chunking**: Page-aware chunking for PDFs preserving document hierarchy
- **Hybrid Retrieval**: Dense vectors + TF‑IDF + BM25 with intelligent fusion ranking
- **Dynamic Model Adaptation**: Hot-swap embedding and generation models in real-time
- **FastAPI Backend**: REST API + WebSocket streaming + admin dashboard
- **Chainlit UI**: Modern chat interface with real-time model switching
- **Docker Support**: Complete containerization with Docker Compose
- **Observability**: Prometheus metrics, structured logging, security headers

## 🏗️ Architecture

```
advanced_rag/
├── app/                    # FastAPI application
│   ├── main.py            # App factory & routing
│   ├── api/               # REST & WebSocket endpoints
│   ├── models/            # Pydantic schemas & config
│   ├── pipeline/          # Ingestion & query pipeline
│   ├── retrieval/         # Hybrid retrieval service
│   ├── embeddings/        # Embedding model factory
│   ├── generation/        # Generation model factory
│   ├── admin/             # Admin dashboard & routes
│   └── observability/     # Metrics & logging
├── ui/chainlit/           # Chainlit chat interface
├── docs/                  # Documentation & reports
├── tests/                 # Unit & integration tests
└── docker-compose.yml     # Multi-service orchestration
```

## 📋 Requirements

- Python 3.11+
- System dependencies for OCR:
  - **macOS**: `brew install tesseract poppler`
  - **Ubuntu/Debian**: `sudo apt install tesseract-ocr libtesseract-dev poppler-utils`

## ⚙️ Quick Start

### Local Development

1. **Clone and Setup**
```bash
git clone <your-repo-url>
cd advanced_rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Run API Server**
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

3. **Run Chainlit UI** (Optional)
```bash
export RAG_API_BASE="http://127.0.0.1:8000"
chainlit run ui/chainlit/langchain_app.py -w --host 127.0.0.1 --port 8501
```

### Docker Deployment

```bash
docker compose up --build
```

- **API**: http://localhost:8000
- **Dashboard**: http://localhost:8000/api/admin/dashboard
- **UI**: http://localhost:8501

## 🔧 Configuration

Create `.env` file for API keys (optional):
```bash
OPENAI_API_KEY=your_key
COHERE_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
ADMIN_API_KEY=optional_admin_key
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

## 📥 Ingestion

### Web Dashboard
- Navigate to Admin Dashboard → Ingest Files
- Upload documents directly through the web interface

### API Endpoints
```bash
# Upload files
curl -X POST "http://127.0.0.1:8000/api/ingest/upload" \
  -F "files=@/path/to/document.pdf"

# Ingest directory
curl -X POST "http://127.0.0.1:8000/api/ingest/directory" \
  -F "path=./data/uploads"
```

## 🔍 Querying

### REST API
```bash
curl -X POST "http://127.0.0.1:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"Your question here","top_k":5,"filters":{}}'
```

### WebSocket Streaming
Connect to `/ws/stream` for real-time token streaming.

## 🎛️ Admin Dashboard

Access at `/api/admin/dashboard`:
- System health monitoring
- Quick query testing
- File ingestion management
- Performance metrics

## 🧪 Testing

```bash
pytest -q
```

## 🐛 Troubleshooting

- **Port conflicts**: Kill existing processes or use different ports
- **OCR issues**: Ensure Tesseract and Poppler are installed
- **API errors**: Check environment variables and API keys
- **404 errors**: Root path redirects to admin dashboard

## 📚 Documentation

- **API Docs**: http://127.0.0.1:8000/docs (Swagger UI)
- **Metrics**: http://127.0.0.1:8000/metrics (Prometheus)
- **Health Check**: http://127.0.0.1:8000/api/health

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

Built with FastAPI, Chainlit, FAISS, and modern AI/ML libraries.

---

**Advanced RAG System** - Production-ready multi-modal retrieval and generation.
