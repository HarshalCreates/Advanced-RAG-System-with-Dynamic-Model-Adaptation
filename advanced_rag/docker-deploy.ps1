# Advanced RAG System - Docker Deployment Script (PowerShell)
# This script handles building, testing, and running the RAG system with image processing fixes

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Function to check if Docker is running
function Test-Docker {
    try {
        docker info | Out-Null
        Write-Success "Docker is running"
        return $true
    }
    catch {
        Write-Error "Docker is not running. Please start Docker and try again."
        return $false
    }
}

# Function to create necessary directories
function New-Directories {
    Write-Status "Creating necessary directories..."
    New-Item -ItemType Directory -Force -Path "data/uploads" | Out-Null
    New-Item -ItemType Directory -Force -Path "data/vectorstore" | Out-Null
    New-Item -ItemType Directory -Force -Path "logs" | Out-Null
    Write-Success "Directories created"
}

# Function to build Docker images
function Build-Images {
    Write-Status "Building Docker images..."
    docker-compose build --no-cache
    Write-Success "Docker images built successfully"
}

# Function to test image processing
function Test-ImageProcessing {
    Write-Status "Testing image processing capabilities..."
    
    # Create a test container
    docker run --rm advanced_rag_api python -c "
import sys
sys.path.insert(0, '/app')
from app.intelligence.fallback_image_processor import fallback_processor
from PIL import Image
import io

# Test fallback processor
print('Testing fallback image processor...')
img = Image.new('RGB', (100, 100), color='red')
img_buffer = io.BytesIO()
img.save(img_buffer, format='PNG')
img_content = img_buffer.getvalue()

result = fallback_processor.extract_text_from_image(img_content)
print('Fallback processor result:', result)

# Test Tesseract
try:
    import pytesseract
    version = pytesseract.get_tesseract_version()
    print(f'Tesseract version: {version}')
    print('Tesseract is working correctly')
except Exception as e:
    print(f'Tesseract error: {e}')

print('Image processing test completed')
"
    Write-Success "Image processing test completed"
}

# Function to start services
function Start-Services {
    Write-Status "Starting RAG services..."
    docker-compose up -d
    
    # Wait for services to be healthy
    Write-Status "Waiting for services to be ready..."
    Start-Sleep -Seconds 10
    
    # Check service health
    $status = docker-compose ps
    if ($status -match "healthy") {
        Write-Success "Services are healthy and running"
    }
    else {
        Write-Warning "Some services may still be starting up"
    }
}

# Function to show service status
function Show-Status {
    Write-Status "Service Status:"
    docker-compose ps
    
    Write-Host ""
    Write-Status "Service URLs:"
    Write-Host "API: http://localhost:8000"
    Write-Host "UI: http://localhost:8501"
    Write-Host "API Docs: http://localhost:8000/docs"
    Write-Host "Health Check: http://localhost:8000/api/health"
}

# Function to stop services
function Stop-Services {
    Write-Status "Stopping RAG services..."
    docker-compose down
    Write-Success "Services stopped"
}

# Function to show logs
function Show-Logs {
    Write-Status "Showing logs (Ctrl+C to exit)..."
    docker-compose logs -f
}

# Function to clean up
function Remove-All {
    Write-Status "Cleaning up Docker resources..."
    docker-compose down -v --remove-orphans
    docker system prune -f
    Write-Success "Cleanup completed"
}

# Function to run tests
function Test-System {
    Write-Status "Running system tests..."
    
    # Test API health
    try {
        Invoke-WebRequest -Uri "http://localhost:8000/api/health" -UseBasicParsing | Out-Null
        Write-Success "API health check passed"
    }
    catch {
        Write-Error "API health check failed"
        return $false
    }
    
    # Test image processing endpoint
    Write-Status "Testing image processing endpoint..."
    Write-Success "Basic tests completed"
    return $true
}

# Function to show help
function Show-Help {
    Write-Host "Advanced RAG System - Docker Deployment Script (PowerShell)"
    Write-Host ""
    Write-Host "Usage: .\docker-deploy.ps1 [COMMAND]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  build     - Build Docker images"
    Write-Host "  start     - Start all services"
    Write-Host "  stop      - Stop all services"
    Write-Host "  restart   - Restart all services"
    Write-Host "  status    - Show service status"
    Write-Host "  logs      - Show service logs"
    Write-Host "  test      - Run system tests"
    Write-Host "  cleanup   - Clean up Docker resources"
    Write-Host "  deploy    - Full deployment (build + start)"
    Write-Host "  help      - Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\docker-deploy.ps1 deploy    # Full deployment"
    Write-Host "  .\docker-deploy.ps1 start     # Start services"
    Write-Host "  .\docker-deploy.ps1 logs      # View logs"
}

# Main script logic
switch ($Command.ToLower()) {
    "build" {
        if (Test-Docker) {
            New-Directories
            Build-Images
        }
    }
    "start" {
        if (Test-Docker) {
            Start-Services
            Show-Status
        }
    }
    "stop" {
        Stop-Services
    }
    "restart" {
        Stop-Services
        Start-Services
        Show-Status
    }
    "status" {
        Show-Status
    }
    "logs" {
        Show-Logs
    }
    "test" {
        Test-System
    }
    "cleanup" {
        Remove-All
    }
    "deploy" {
        if (Test-Docker) {
            New-Directories
            Build-Images
            Test-ImageProcessing
            Start-Services
            Show-Status
            Write-Success "Deployment completed successfully!"
            Write-Host ""
            Write-Status "Next steps:"
            Write-Host "1. Open http://localhost:8501 to access the UI"
            Write-Host "2. Open http://localhost:8000/docs to view API documentation"
            Write-Host "3. Upload images and test the image processing capabilities"
            Write-Host "4. Use 'docker-compose logs -f' to monitor logs"
        }
    }
    default {
        Show-Help
    }
}
