# 🐳 Docker Deployment Guide

## 📋 Overview

This guide covers deploying the Advanced RAG System using Docker. The containerized setup provides:

- **🚀 Easy deployment** across different environments
- **🔧 Consistent configuration** with environment variables
- **📈 Scalable architecture** with optional services
- **🛡️ Production-ready** security and monitoring
- **🦙 Local LLM support** via Ollama integration

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Nginx       │    │   Chainlit UI   │    │   FastAPI       │
│  (Port 80/443)  │◄──►│   (Port 8001)   │◄──►│   Backend       │
│   Load Balancer │    │   Frontend      │    │   (Port 8000)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐             │
         └─────────────►│     Redis       │◄────────────┘
                        │   (Port 6379)   │
                        │     Cache       │
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │     Ollama      │
                        │  (Port 11434)   │
                        │  Local Models   │
                        └─────────────────┘
```

### Services Overview

| Service | Purpose | Port | Profile |
|---------|---------|------|---------|
| **api** | FastAPI backend with RAG pipeline | 8000 | default |
| **ui** | Chainlit frontend interface | 8001 | default |
| **ollama** | Local LLM service (Llama models) | 11434 | ollama |
| **redis** | Caching and session storage | 6379 | cache |
| **nginx** | Reverse proxy and load balancer | 80/443 | nginx |

---

## 🚀 Quick Start

### Prerequisites

- **Docker** 20.10+ and **Docker Compose** 2.0+
- **8GB RAM** minimum (16GB recommended for Ollama)
- **10GB disk space** for images and models

### 1️⃣ Clone and Setup

```bash
git clone https://github.com/HarshalCreates/Advanced-RAG-System-with-Dynamic-Model-Adaptation.git
cd Advanced-RAG-System-with-Dynamic-Model-Adaptation/advanced_rag
```

### 2️⃣ Configure Environment

```bash
# Copy environment template
cp docker/environment.template .env

# Edit with your API keys
nano .env  # or your preferred editor
```

**Required Environment Variables:**
```bash
# API Keys (at least one required)
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# Security
ADMIN_API_KEY=your-secure-admin-key
```

### 3️⃣ Start the System

Choose your deployment profile:

**Basic (API + UI only):**
```bash
# Linux/macOS
./docker/start.sh basic

# Windows
docker\start.bat basic
```

**With Local Models (recommended):**
```bash
# Linux/macOS
./docker/start.sh ollama

# Windows
docker\start.bat ollama
```

**Full Production Stack:**
```bash
# Linux/macOS
./docker/start.sh full

# Windows
docker\start.bat full
```

### 4️⃣ Access the System

- **Web UI**: http://localhost:8001
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## 📊 Deployment Profiles

### 🏃‍♂️ Basic Profile (Fastest Start)

**Services**: API + UI  
**Memory**: ~3GB  
**Use case**: Development, testing, cloud-only models

```bash
docker/start.sh basic
```

**Features:**
- ✅ FastAPI backend
- ✅ Chainlit UI
- ✅ OpenAI/Anthropic models
- ❌ Local models
- ❌ Caching
- ❌ Load balancing

### 🦙 Ollama Profile (Local Models)

**Services**: API + UI + Ollama  
**Memory**: ~8GB  
**Use case**: Local inference, privacy, cost optimization

```bash
docker/start.sh ollama
```

**Features:**
- ✅ Everything from Basic
- ✅ Local Llama models
- ✅ Automatic model pulling
- ✅ Zero API costs
- ❌ Caching
- ❌ Load balancing

### 🏢 Full Profile (Production)

**Services**: All services  
**Memory**: ~12GB  
**Use case**: Production deployment, high availability

```bash
docker/start.sh full
```

**Features:**
- ✅ Everything from Ollama
- ✅ Redis caching
- ✅ Nginx load balancer
- ✅ SSL support
- ✅ Rate limiting
- ✅ Production monitoring

---

## 🔧 Configuration

### Environment Variables

```bash
# Core API Keys
OPENAI_API_KEY=sk-...                    # OpenAI API key
ANTHROPIC_API_KEY=...                    # Anthropic API key
COHERE_API_KEY=...                       # Cohere API key (optional)

# Security
ADMIN_API_KEY=your-secure-key            # Admin endpoints protection

# Service Configuration
OLLAMA_BASE_URL=http://ollama:11434      # Ollama service URL
LOG_LEVEL=INFO                           # Logging level
API_WORKERS=1                            # FastAPI workers

# Resource Limits
API_MEMORY_LIMIT=4G                      # API container memory
UI_MEMORY_LIMIT=2G                       # UI container memory
OLLAMA_MEMORY_LIMIT=8G                   # Ollama memory

# Profiles
COMPOSE_PROFILES=ollama                  # Active profiles
```

### Advanced Configuration

**Custom Models:**
```bash
# Add to .env
EMBEDDING_MODEL=text-embedding-3-large
DEFAULT_TOP_K=10
MAX_TOKENS=4096
TEMPERATURE=0.1
```

**Performance Tuning:**
```bash
# Multi-worker setup
API_WORKERS=4

# Memory optimization
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1
```

**Security Hardening:**
```bash
# Generate secure admin key
openssl rand -hex 32

# Custom ports (if needed)
API_PORT=8000
UI_PORT=8001
OLLAMA_PORT=11434
```

---

## 📈 Management Commands

### Using the Startup Scripts

**Linux/macOS:**
```bash
./docker/start.sh [command]
```

**Windows:**
```batch
docker\start.bat [command]
```

### Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `basic` | Start API + UI only | `./docker/start.sh basic` |
| `ollama` | Start with local models | `./docker/start.sh ollama` |
| `full` | Start full stack | `./docker/start.sh full` |
| `status` | Show service status | `./docker/start.sh status` |
| `logs` | View logs (all or specific) | `./docker/start.sh logs api` |
| `stop` | Stop all services | `./docker/start.sh stop` |
| `restart` | Restart services | `./docker/start.sh restart` |
| `cleanup` | Stop and remove all | `./docker/start.sh cleanup` |
| `build` | Build images | `./docker/start.sh build` |
| `pull` | Pull latest images | `./docker/start.sh pull` |

### Direct Docker Compose Commands

```bash
# Start specific profile
docker compose --profile ollama up -d

# View logs
docker compose logs -f api

# Scale services
docker compose up -d --scale api=3

# Execute commands in containers
docker compose exec api python cli_query.py "What is AI?"

# Check service health
docker compose ps
```

---

## 🛠️ Development

### Building Custom Images

```bash
# Build with custom tag
docker build -t my-rag-system:latest .

# Build specific stage
docker build --target production -t my-rag-system:prod .

# Build with build args
docker build --build-arg PYTHON_VERSION=3.11 .
```

### Hot Reloading for Development

```bash
# Mount source code for live editing
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

Create `docker-compose.dev.yml`:
```yaml
version: '3.8'
services:
  api:
    volumes:
      - ./app:/app/app:ro
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Debugging

```bash
# Enter container shell
docker compose exec api bash

# Debug specific service
docker compose exec api python -c "import app.main; print('API loaded')"

# Check service logs
docker compose logs --tail=50 -f api
```

---

## 🔍 Monitoring & Health Checks

### Health Endpoints

| Service | Health Check URL | Status Codes |
|---------|------------------|--------------|
| API | `http://localhost:8000/health` | 200 = healthy |
| UI | `http://localhost:8001/` | 200 = healthy |
| Ollama | `http://localhost:11434/` | 200 = healthy |

### Container Health Monitoring

```bash
# Check container health
docker compose ps

# View health check logs
docker inspect advanced_rag_api | grep -A 10 Health

# Manual health checks
curl -f http://localhost:8000/health
curl -f http://localhost:8001/
```

### Resource Monitoring

```bash
# Container resource usage
docker stats

# Disk usage
docker system df

# Volume inspection
docker volume ls
docker volume inspect advanced_rag_rag_data
```

---

## 🔒 Security & Production

### SSL/TLS Configuration

1. **Generate SSL certificates:**
```bash
mkdir -p docker/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout docker/ssl/key.pem \
    -out docker/ssl/cert.pem
```

2. **Update Nginx configuration:**
```bash
# Edit docker/nginx.conf
# Uncomment HTTPS server block
# Update server_name with your domain
```

3. **Start with SSL:**
```bash
docker/start.sh full
```

### Security Best Practices

**Environment Security:**
```bash
# Use strong admin keys
ADMIN_API_KEY=$(openssl rand -hex 32)

# Restrict file permissions
chmod 600 .env
```

**Network Security:**
```bash
# Internal network only
OLLAMA_BASE_URL=http://ollama:11434  # Not localhost

# Use nginx for external access
# Block direct access to internal ports
```

**Container Security:**
```bash
# Non-root user in containers
USER app

# Read-only filesystems where possible
# Resource limits enforced
# Security scanning enabled
```

---

## 🚨 Troubleshooting

### Common Issues

#### 1. **Container won't start**
```bash
# Check logs
docker compose logs api

# Common fixes
docker compose down && docker compose up -d
docker system prune -f
```

#### 2. **Port conflicts**
```bash
# Check what's using ports
netstat -tulpn | grep :8000

# Use different ports
echo "API_PORT=8010" >> .env
echo "UI_PORT=8011" >> .env
```

#### 3. **Memory issues**
```bash
# Check memory usage
docker stats

# Reduce memory limits
echo "API_MEMORY_LIMIT=2G" >> .env
echo "OLLAMA_MEMORY_LIMIT=4G" >> .env
```

#### 4. **Ollama model download fails**
```bash
# Manual model pull
docker compose exec ollama ollama pull llama3.2

# Check available models
docker compose exec ollama ollama list

# Clear model cache
docker volume rm advanced_rag_ollama_data
```

#### 5. **Permission denied on volumes**
```bash
# Fix volume permissions
sudo chown -R 1000:1000 data uploads logs

# Or use different volume mounting
# Edit docker-compose.yml volumes section
```

#### 6. **API key not working**
```bash
# Verify environment file
cat .env | grep API_KEY

# Check container environment
docker compose exec api env | grep API_KEY

# Restart after env changes
docker compose restart
```

### Performance Optimization

```bash
# Enable Docker BuildKit
export DOCKER_BUILDKIT=1

# Use layer caching
docker compose build --cache-from my-rag-system:latest

# Optimize memory
echo 'vm.max_map_count=262144' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### Log Analysis

```bash
# Structured log viewing
docker compose logs --tail=100 api | grep ERROR

# Export logs
docker compose logs --no-color > rag_system.log

# Real-time monitoring
docker compose logs -f | grep -E "(ERROR|WARNING)"
```

---

## 📦 Backup & Migration

### Data Backup

```bash
# Backup volumes
docker run --rm -v advanced_rag_rag_data:/data -v $(pwd):/backup \
    alpine tar czf /backup/rag_data.tar.gz /data

# Backup Ollama models
docker run --rm -v advanced_rag_ollama_data:/data -v $(pwd):/backup \
    alpine tar czf /backup/ollama_models.tar.gz /data
```

### Migration

```bash
# Export configuration
docker compose config > docker-compose.backup.yml

# Migrate to new host
scp -r advanced_rag/ user@newhost:/opt/
ssh user@newhost "cd /opt/advanced_rag && ./docker/start.sh ollama"
```

---

## 🎯 Production Deployment

### Cloud Deployment (AWS/GCP/Azure)

```bash
# Use Docker Machine or cloud-specific tools
docker-machine create --driver amazonec2 rag-production

# Or use managed container services
# - AWS ECS/Fargate
# - Google Cloud Run
# - Azure Container Instances
```

### Kubernetes Deployment

```yaml
# Example k8s deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: api
        image: your-registry/rag-system:latest
        ports:
        - containerPort: 8000
```

### CI/CD Integration

```yaml
# Example GitHub Actions workflow
name: Build and Deploy
on:
  push:
    branches: [main]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Build Docker image
      run: docker build -t rag-system:${{ github.sha }} .
    - name: Deploy
      run: docker compose up -d
```

---

## 📋 Maintenance

### Regular Tasks

```bash
# Weekly maintenance script
#!/bin/bash
docker compose down
docker system prune -f
docker volume prune -f
docker/start.sh ollama
```

### Updates

```bash
# Update system
git pull origin main
docker compose build --no-cache
docker compose up -d
```

### Monitoring Setup

```bash
# Add monitoring stack
docker compose --profile monitoring up -d

# View metrics
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000
```

---

## 🆘 Support

### Getting Help

1. **Check logs first:**
   ```bash
   docker/start.sh logs
   ```

2. **Verify configuration:**
   ```bash
   docker/start.sh status
   ```

3. **Try cleanup and restart:**
   ```bash
   docker/start.sh cleanup
   docker/start.sh ollama
   ```

4. **Report issues:**
   - Include logs and configuration
   - Specify deployment profile used
   - Mention system specifications

### Useful Resources

- **Docker Documentation**: https://docs.docker.com/
- **Compose File Reference**: https://docs.docker.com/compose/compose-file/
- **Ollama Documentation**: https://ollama.ai/docs
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Chainlit Documentation**: https://docs.chainlit.io/

---

<div align="center">

**🎉 Your Advanced RAG System is now containerized and ready for production!**

[⬆ Back to Top](#-docker-deployment-guide)

</div>
