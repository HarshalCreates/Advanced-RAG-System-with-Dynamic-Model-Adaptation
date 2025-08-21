# Docker Deployment Guide - Advanced RAG System

## Overview

This guide provides comprehensive instructions for deploying the Advanced RAG System using Docker with full image processing capabilities including OCR and fallback mechanisms.

## Prerequisites

### System Requirements
- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
- **Docker Compose** (included with Docker Desktop)
- **Git** (for cloning the repository)
- **4GB+ RAM** (recommended for optimal performance)
- **10GB+ free disk space**

### Docker Installation
1. **Windows/Mac**: Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. **Linux**: Follow [Docker Engine installation guide](https://docs.docker.com/engine/install/)
3. **Verify installation**: Run `docker --version` and `docker-compose --version`

## Quick Start

### 1. Clone and Navigate
```bash
git clone <your-repo-url>
cd advanced_rag
```

### 2. Set Environment Variables
Create a `.env` file in the project root:
```bash
# API Keys (required for LLM functionality)
OPENAI_API_KEY=your_openai_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Admin API Key (for secure access)
ADMIN_API_KEY=your_admin_api_key_here

# Optional: Ollama for local LLM
OLLAMA_BASE_URL=http://ollama:11434
```

### 3. Deploy with Scripts

#### Windows (PowerShell)
```powershell
# Full deployment
.\docker-deploy.ps1 deploy

# Or step by step
.\docker-deploy.ps1 build
.\docker-deploy.ps1 start
```

#### Linux/Mac (Bash)
```bash
# Make script executable
chmod +x docker-deploy.sh

# Full deployment
./docker-deploy.sh deploy

# Or step by step
./docker-deploy.sh build
./docker-deploy.sh start
```

### 4. Access the System
- **Web UI**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## Manual Deployment

### 1. Build Images
```bash
docker-compose build --no-cache
```

### 2. Create Directories
```bash
mkdir -p data/uploads data/vectorstore logs
```

### 3. Start Services
```bash
docker-compose up -d
```

### 4. Check Status
```bash
docker-compose ps
```

## Image Processing Features

### What's Included in Docker

#### ✅ **Full OCR Capabilities**
- **Tesseract OCR** with multiple language support:
  - English, French, German, Spanish
  - Italian, Portuguese, Russian
  - Chinese (Simplified), Japanese
- **Automatic text extraction** from images
- **Multi-language detection** and processing

#### ✅ **Fallback Mechanisms**
- **Metadata extraction** when OCR fails
- **Image analysis** (format, size, dimensions, colors)
- **Graceful degradation** for missing dependencies

#### ✅ **Advanced Processing**
- **PDF processing** with mixed content support
- **Table extraction** from documents
- **Figure and diagram analysis**
- **Code snippet detection**
- **Mathematical formula extraction**

### Testing Image Processing

#### 1. Test OCR Functionality
```bash
# Test Tesseract installation
docker exec advanced_rag_api tesseract --version

# Test language support
docker exec advanced_rag_api tesseract --list-langs
```

#### 2. Test Image Upload
1. Open http://localhost:8501
2. Navigate to the upload section
3. Upload an image with text
4. Verify that text is extracted and indexed

#### 3. Test Fallback Processing
```bash
# Test fallback processor
docker exec advanced_rag_api python -c "
from app.intelligence.fallback_image_processor import fallback_processor
from PIL import Image
import io

img = Image.new('RGB', (100, 100), color='red')
img_buffer = io.BytesIO()
img.save(img_buffer, format='PNG')
result = fallback_processor.extract_text_from_image(img_buffer.getvalue())
print('Fallback result:', result)
"
```

## Service Management

### Available Commands

#### Using Scripts
```bash
# Windows
.\docker-deploy.ps1 [command]

# Linux/Mac
./docker-deploy.sh [command]
```

#### Using Docker Compose
```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Restart services
docker-compose restart

# Check status
docker-compose ps
```

### Service Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web UI        │    │   API Server    │    │   Ollama        │
│   (Port 8501)   │◄──►│   (Port 8000)   │◄──►│   (Port 11434)  │
│   Chainlit      │    │   FastAPI       │    │   (Optional)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         │                       ▼
         │              ┌─────────────────┐
         │              │   Vector Store  │
         │              │   (Local)       │
         │              └─────────────────┘
         │
         ▼
┌─────────────────┐
│   File Storage  │
│   (Uploads)     │
└─────────────────┘
```

## Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | Yes* | - |
| `COHERE_API_KEY` | Cohere API key for embeddings | Yes* | - |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude | No | - |
| `ADMIN_API_KEY` | Admin API key for secure access | No | - |
| `OLLAMA_BASE_URL` | Ollama service URL | No | http://ollama:11434 |
| `TESSERACT_CMD` | Tesseract executable path | No | /usr/bin/tesseract |
| `ENABLE_OCR` | Enable OCR processing | No | true |
| `FALLBACK_IMAGE_PROCESSING` | Enable fallback processing | No | true |

*At least one LLM API key is required

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./data` | `/app/data` | Persistent data storage |
| `./uploads` | `/app/data/uploads` | Uploaded files |
| `./logs` | `/app/logs` | Application logs |

## Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Check what's using the ports
netstat -tulpn | grep :8000
netstat -tulpn | grep :8501

# Stop conflicting services or change ports in docker-compose.yml
```

#### 2. Docker Build Fails
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache
```

#### 3. Image Processing Not Working
```bash
# Check Tesseract installation
docker exec advanced_rag_api tesseract --version

# Check logs for errors
docker-compose logs api

# Test fallback processor
docker exec advanced_rag_api python -c "
from app.intelligence.fallback_image_processor import fallback_processor
print('Fallback processor available')
"
```

#### 4. Services Not Starting
```bash
# Check service status
docker-compose ps

# View detailed logs
docker-compose logs api
docker-compose logs ui

# Check resource usage
docker stats
```

### Performance Optimization

#### 1. Resource Allocation
```yaml
# Add to docker-compose.yml services
services:
  api:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

#### 2. Volume Optimization
```bash
# Use named volumes for better performance
docker volume create rag_data
docker volume create rag_uploads
```

#### 3. Image Optimization
```bash
# Use multi-stage builds for smaller images
# (Already implemented in Dockerfile)
```

## Monitoring and Logs

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f ui

# Last 100 lines
docker-compose logs --tail=100 api
```

### Health Checks
```bash
# Check API health
curl http://localhost:8000/api/health

# Check UI health
curl http://localhost:8501/healthz
```

### Resource Monitoring
```bash
# Container stats
docker stats

# Disk usage
docker system df
```

## Security Considerations

### 1. API Key Management
- Store API keys in `.env` file (not in version control)
- Use environment variables for sensitive data
- Rotate API keys regularly

### 2. Network Security
- Services communicate over internal Docker network
- Only necessary ports exposed to host
- Admin API key required for sensitive operations

### 3. File Upload Security
- File type validation
- Size limits enforced
- Malware scanning (consider adding)

## Backup and Recovery

### Data Backup
```bash
# Backup data directory
tar -czf rag_backup_$(date +%Y%m%d).tar.gz data/

# Backup specific volumes
docker run --rm -v rag_data:/data -v $(pwd):/backup alpine tar czf /backup/data_backup.tar.gz -C /data .
```

### Recovery
```bash
# Restore from backup
tar -xzf rag_backup_20231201.tar.gz

# Restart services
docker-compose restart
```

## Scaling and Production

### Production Considerations

#### 1. Reverse Proxy
```nginx
# Nginx configuration example
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### 2. SSL/TLS
```bash
# Use Let's Encrypt or other SSL provider
# Configure in reverse proxy
```

#### 3. Database
```yaml
# Add external database service
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: rag_system
      POSTGRES_USER: rag_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
```

## Support and Maintenance

### Regular Maintenance
```bash
# Weekly cleanup
docker system prune -f
docker image prune -f

# Monthly updates
docker-compose pull
docker-compose build --no-cache
```

### Monitoring Setup
- Set up log aggregation (ELK stack, Graylog)
- Configure alerting for service failures
- Monitor resource usage and performance

## Conclusion

The Docker deployment provides a complete, production-ready RAG system with:

- ✅ **Full image processing capabilities** with OCR and fallback mechanisms
- ✅ **Multi-language support** for text extraction
- ✅ **Scalable architecture** ready for production
- ✅ **Comprehensive monitoring** and logging
- ✅ **Security best practices** implemented
- ✅ **Easy deployment** and management scripts

Your RAG system is now ready to process images and documents with enterprise-grade capabilities! 🚀
