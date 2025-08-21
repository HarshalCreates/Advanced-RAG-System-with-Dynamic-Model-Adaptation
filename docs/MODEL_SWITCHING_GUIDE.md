# 🦙 Production RAG with Real-Time Model Switching

This guide covers the complete production-ready RAG system with real-time model switching between GPT, Claude, and Llama models through an intuitive Chainlit interface.

## 🚀 Quick Start

### 1. Start the Complete System
```bash
# Start everything (Ollama + RAG API + Chainlit UI)
python start_production_rag.py
```

This will:
- ✅ Start Ollama service
- ✅ Pull recommended Llama models
- ✅ Start RAG API server (port 8000)
- ✅ Start Chainlit UI (port 8001)
- ✅ Configure all services automatically

### 2. Access the Interface
Open your browser to: **http://127.0.0.1:8001**

### 3. Switch Models in Real-Time
- **GUI Method**: Click the gear icon ⚙️ → Select backend and model
- **Quick Select**: Use the "⚡ Quick Model Select" dropdown
- **Command Method**: Type `/model ollama llama3.2` in chat
- **API Method**: POST to `/api/admin/hot-swap/generation`

## 🎯 Available Models

### 🦙 Ollama (Local)
- **llama3.2** - Latest Llama model (3B/7B versions)
- **llama3.1** - Previous stable version
- **mixtral** - Mixture of Experts model
- **codellama** - Code-specialized model
- **gemma2** - Google's Gemma model
- **phi3** - Microsoft's Phi-3 model

### 🤖 OpenAI (Cloud)
- **gpt-4o** - Latest GPT-4 Optimized
- **gpt-4o-mini** - Faster, more affordable
- **gpt-4-turbo** - Previous generation
- **gpt-3.5-turbo** - Cost-effective option

### 🧠 Anthropic (Cloud)
- **claude-3-5-sonnet-20241022** - Latest Claude
- **claude-3-5-haiku-20241022** - Faster Claude
- **claude-3-opus-20240229** - Most capable

## 🔧 Configuration

### Environment Variables
```bash
# API Keys (optional for cloud models)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Service URLs
RAG_API_BASE=http://127.0.0.1:8000
OLLAMA_BASE_URL=http://127.0.0.1:11434

# Admin access
RAG_ADMIN_API_KEY=your_admin_key
```

### Settings Panel Options
- **🔧 Generation Backend** - Choose between ollama, openai, anthropic
- **🤖 Model** - Select specific model for chosen backend
- **⚡ Quick Model Select** - Fast switching with backend:model format
- **📊 Retrieval Top K** - Number of documents to retrieve (1-20)
- **🧠 Show Reasoning Steps** - Display AI reasoning process
- **📚 Show All Citations** - Show all vs. top citations only

## 🎮 Usage Examples

### Switch to Llama Model
```bash
# In Chainlit chat
/model ollama llama3.2

# Via API
curl -X POST "http://127.0.0.1:8000/api/admin/hot-swap/generation?backend=ollama&model=llama3.2"
```

### Switch to GPT-4
```bash
# In Chainlit chat
/model openai gpt-4o

# Via settings panel
Backend: openai → Model: gpt-4o
```

### Check System Status
```bash
# In Chainlit chat
/status

# View available models
/refresh
```

## 🛠️ Advanced Features

### Model Registry
The system includes a comprehensive model registry with:
- Model availability validation
- Configuration parameters (context window, max tokens)
- Automatic model discovery for Ollama
- Fallback mechanisms

### Production Features
- **Health Monitoring** - Service health checks and automatic recovery
- **Resource Management** - Memory and CPU optimization
- **Cost Tracking** - Usage and cost monitoring for cloud models
- **Security** - Authentication and access control
- **Logging** - Comprehensive logging and metrics

### API Endpoints

#### Model Management
```bash
# Get available models
GET /api/admin/models/available

# Get current model status
GET /api/admin/models/status

# Hot-swap generation model
POST /api/admin/hot-swap/generation?backend=ollama&model=llama3.2

# Hot-swap embedding model  
POST /api/admin/hot-swap/embeddings?backend=openai&model=text-embedding-ada-002
```

#### Query with Specific Models
```bash
# Standard query (uses current model)
POST /api/query
{
  "query": "What is machine learning?",
  "top_k": 5
}

# Secure query with full features
POST /api/secure/query
{
  "query": "Explain neural networks",
  "top_k": 10,
  "anonymize_pii": true,
  "priority": 1
}
```

## 📊 Monitoring

### Real-time Metrics
- Response times per model
- Token usage and costs
- Model availability status
- Resource utilization

### Health Checks
```bash
# System health
GET /health

# Comprehensive status
GET /api/status/comprehensive

# Ollama status
GET http://127.0.0.1:11434/api/tags
```

## 🚨 Troubleshooting

### Ollama Issues
```bash
# Check if Ollama is running
curl http://127.0.0.1:11434/api/tags

# Start Ollama manually
ollama serve

# Pull a model
ollama pull llama3.2
```

### Model Switching Failures
1. **Check model availability**: Use `/refresh` command
2. **Verify API keys**: For cloud models (OpenAI, Anthropic)
3. **Check logs**: Look at server console output
4. **Test connection**: Use `/status` command

### Performance Issues
1. **Local models**: Ensure sufficient RAM (8GB+ for 7B models)
2. **Cloud models**: Check API rate limits
3. **Network**: Verify internet connectivity for cloud models

## 🔐 Security

### Authentication
- Admin API key protection for model switching
- User authentication for secure queries
- Role-based access control

### Privacy
- PII detection and anonymization
- GDPR compliance features
- Data processing consent management

## 📈 Optimization

### Cost Optimization
- Automatic model selection based on query complexity
- Usage monitoring and budget alerts
- Recommendations for cost-effective models

### Performance Optimization
- Model caching and warm-up
- Intelligent load balancing
- Resource usage optimization

## 🧪 Testing

### Test Model Switching
```bash
# Run comprehensive tests
python test_model_switching.py

# Test specific model
python -c "
from app.generation.factory import GenerationFactory
factory = GenerationFactory('ollama', 'llama3.2')
client = factory.build()
print(client.complete('Hello', 'Say hi!'))
"
```

### Validate Installation
1. Start the production system: `python start_production_rag.py`
2. Check all services are running
3. Try switching between different models
4. Upload a document and test queries

## 📚 Additional Resources

- **Ollama Models**: https://ollama.ai/models
- **OpenAI API**: https://platform.openai.com/docs
- **Anthropic API**: https://docs.anthropic.com/
- **Chainlit Documentation**: https://docs.chainlit.io/

## 🎉 Success Indicators

When everything is working correctly:
- ✅ All services start without errors
- ✅ Ollama models are pulled and available
- ✅ Model switching works in UI
- ✅ Queries return responses from selected models
- ✅ Performance metrics are displayed
- ✅ Status commands work properly

---

**🚀 You now have a production-ready RAG system with real-time model switching capabilities!**
