#!/bin/bash

# Advanced RAG System - Docker Startup Script
# This script provides easy commands to start different configurations of the system

set -e

# Color codes for output
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
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to check if docker-compose is available
check_docker_compose() {
    if command -v docker-compose >/dev/null 2>&1; then
        COMPOSE_CMD="docker-compose"
    elif docker compose version >/dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        print_error "Neither docker-compose nor 'docker compose' is available"
        exit 1
    fi
    print_success "Using: $COMPOSE_CMD"
}

# Function to create required directories
setup_directories() {
    print_status "Setting up required directories..."
    mkdir -p data uploads logs
    chmod 755 data uploads logs
    print_success "Directories created"
}

# Function to check for environment file
check_environment() {
    if [ ! -f .env ]; then
        print_warning "No .env file found"
        if [ -f docker/environment.template ]; then
            print_status "Copying environment template..."
            cp docker/environment.template .env
            print_warning "Please edit .env file with your API keys before starting the system"
            return 1
        else
            print_error "No environment template found"
            return 1
        fi
    fi
    print_success "Environment file found"
    return 0
}

# Function to pull latest images
pull_images() {
    print_status "Pulling latest Docker images..."
    $COMPOSE_CMD pull
    print_success "Images updated"
}

# Function to build images
build_images() {
    print_status "Building Docker images..."
    $COMPOSE_CMD build --no-cache
    print_success "Images built"
}

# Function to start basic services (API + UI)
start_basic() {
    print_status "Starting basic services (API + UI)..."
    $COMPOSE_CMD up -d api ui
    print_success "Basic services started"
    print_status "API available at: http://localhost:8000"
    print_status "UI available at: http://localhost:8001"
}

# Function to start with Ollama
start_with_ollama() {
    print_status "Starting services with Ollama..."
    $COMPOSE_CMD --profile ollama up -d
    print_success "Services with Ollama started"
    print_status "API available at: http://localhost:8000"
    print_status "UI available at: http://localhost:8001"
    print_status "Ollama available at: http://localhost:11434"
    
    print_status "Pulling recommended Ollama models..."
    sleep 10  # Wait for Ollama to start
    docker exec advanced_rag_ollama ollama pull llama3.2 || print_warning "Failed to pull llama3.2"
    docker exec advanced_rag_ollama ollama pull codellama || print_warning "Failed to pull codellama"
}

# Function to start full stack (with Redis and Nginx)
start_full() {
    print_status "Starting full stack..."
    $COMPOSE_CMD --profile full up -d
    print_success "Full stack started"
    print_status "Access via Nginx: http://localhost:80"
    print_status "API direct: http://localhost:8000"
    print_status "UI direct: http://localhost:8001"
    print_status "Ollama: http://localhost:11434"
    print_status "Redis: localhost:6379"
}

# Function to show status
show_status() {
    print_status "System Status:"
    $COMPOSE_CMD ps
    echo ""
    print_status "Service Health:"
    $COMPOSE_CMD exec api curl -s http://localhost:8000/health || print_warning "API not responding"
    $COMPOSE_CMD exec ui curl -s http://localhost:8001/ || print_warning "UI not responding"
}

# Function to stop services
stop_services() {
    print_status "Stopping services..."
    $COMPOSE_CMD down
    print_success "Services stopped"
}

# Function to stop and clean up
cleanup() {
    print_status "Stopping and cleaning up..."
    $COMPOSE_CMD down -v --remove-orphans
    docker system prune -f
    print_success "Cleanup completed"
}

# Function to show logs
show_logs() {
    local service=${1:-}
    if [ -n "$service" ]; then
        print_status "Showing logs for $service..."
        $COMPOSE_CMD logs -f "$service"
    else
        print_status "Showing all logs..."
        $COMPOSE_CMD logs -f
    fi
}

# Function to restart services
restart_services() {
    print_status "Restarting services..."
    $COMPOSE_CMD restart
    print_success "Services restarted"
}

# Function to show help
show_help() {
    echo "Advanced RAG System - Docker Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  basic      Start basic services (API + UI)"
    echo "  ollama     Start with Ollama support"
    echo "  full       Start full stack (with Redis and Nginx)"
    echo "  status     Show system status"
    echo "  logs       Show logs (optionally for specific service)"
    echo "  stop       Stop all services"
    echo "  restart    Restart all services"
    echo "  cleanup    Stop and clean up all resources"
    echo "  build      Build Docker images"
    echo "  pull       Pull latest images"
    echo "  setup      Setup directories and environment"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 basic                 # Start API and UI only"
    echo "  $0 ollama                # Start with local Llama models"
    echo "  $0 full                  # Start everything including Nginx"
    echo "  $0 logs api              # Show API logs"
    echo "  $0 logs                  # Show all logs"
    echo ""
}

# Main script logic
main() {
    case "${1:-help}" in
        "basic")
            check_docker
            check_docker_compose
            setup_directories
            if check_environment; then
                start_basic
                show_status
            fi
            ;;
        "ollama")
            check_docker
            check_docker_compose
            setup_directories
            if check_environment; then
                start_with_ollama
                show_status
            fi
            ;;
        "full")
            check_docker
            check_docker_compose
            setup_directories
            if check_environment; then
                start_full
                show_status
            fi
            ;;
        "status")
            check_docker_compose
            show_status
            ;;
        "logs")
            check_docker_compose
            show_logs "$2"
            ;;
        "stop")
            check_docker_compose
            stop_services
            ;;
        "restart")
            check_docker_compose
            restart_services
            ;;
        "cleanup")
            check_docker_compose
            cleanup
            ;;
        "build")
            check_docker
            check_docker_compose
            build_images
            ;;
        "pull")
            check_docker
            check_docker_compose
            pull_images
            ;;
        "setup")
            setup_directories
            check_environment
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function with all arguments
main "$@"
