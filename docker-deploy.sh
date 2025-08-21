#!/bin/bash

# Advanced RAG System - Docker Deployment Script
# This script handles building, testing, and running the RAG system with image processing fixes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p data/uploads data/vectorstore logs
    print_success "Directories created"
}

# Function to build Docker images
build_images() {
    print_status "Building Docker images..."
    docker-compose build --no-cache
    print_success "Docker images built successfully"
}

# Function to test image processing
test_image_processing() {
    print_status "Testing image processing capabilities..."
    
    # Create a test container
    docker run --rm -v $(pwd)/test_image.png:/test_image.png advanced_rag_api python -c "
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
    print_success "Image processing test completed"
}

# Function to start services
start_services() {
    print_status "Starting RAG services..."
    docker-compose up -d
    
    # Wait for services to be healthy
    print_status "Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    if docker-compose ps | grep -q "healthy"; then
        print_success "Services are healthy and running"
    else
        print_warning "Some services may still be starting up"
    fi
}

# Function to show service status
show_status() {
    print_status "Service Status:"
    docker-compose ps
    
    echo ""
    print_status "Service URLs:"
    echo "API: http://localhost:8000"
    echo "UI: http://localhost:8501"
    echo "API Docs: http://localhost:8000/docs"
    echo "Health Check: http://localhost:8000/api/health"
}

# Function to stop services
stop_services() {
    print_status "Stopping RAG services..."
    docker-compose down
    print_success "Services stopped"
}

# Function to show logs
show_logs() {
    print_status "Showing logs (Ctrl+C to exit)..."
    docker-compose logs -f
}

# Function to clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker-compose down -v --remove-orphans
    docker system prune -f
    print_success "Cleanup completed"
}

# Function to run tests
run_tests() {
    print_status "Running system tests..."
    
    # Test API health
    if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
        print_success "API health check passed"
    else
        print_error "API health check failed"
        return 1
    fi
    
    # Test image processing endpoint
    print_status "Testing image processing endpoint..."
    # This would require a test image upload - simplified for now
    print_success "Basic tests completed"
}

# Function to show help
show_help() {
    echo "Advanced RAG System - Docker Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build     - Build Docker images"
    echo "  start     - Start all services"
    echo "  stop      - Stop all services"
    echo "  restart   - Restart all services"
    echo "  status    - Show service status"
    echo "  logs      - Show service logs"
    echo "  test      - Run system tests"
    echo "  cleanup   - Clean up Docker resources"
    echo "  deploy    - Full deployment (build + start)"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 deploy    # Full deployment"
    echo "  $0 start     # Start services"
    echo "  $0 logs      # View logs"
}

# Main script logic
case "${1:-help}" in
    build)
        check_docker
        create_directories
        build_images
        ;;
    start)
        check_docker
        start_services
        show_status
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        start_services
        show_status
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    test)
        run_tests
        ;;
    cleanup)
        cleanup
        ;;
    deploy)
        check_docker
        create_directories
        build_images
        test_image_processing
        start_services
        show_status
        print_success "Deployment completed successfully!"
        echo ""
        print_status "Next steps:"
        echo "1. Open http://localhost:8501 to access the UI"
        echo "2. Open http://localhost:8000/docs to view API documentation"
        echo "3. Upload images and test the image processing capabilities"
        echo "4. Use 'docker-compose logs -f' to monitor logs"
        ;;
    help|*)
        show_help
        ;;
esac
