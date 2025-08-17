@echo off
REM Advanced RAG System - Docker Startup Script for Windows
REM This script provides easy commands to start different configurations of the system

setlocal enabledelayedexpansion

REM Function to print status messages
:print_status
echo [INFO] %~1
goto :eof

:print_success
echo [SUCCESS] %~1
goto :eof

:print_warning
echo [WARNING] %~1
goto :eof

:print_error
echo [ERROR] %~1
goto :eof

REM Check if Docker is running
:check_docker
docker info >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker is not running. Please start Docker and try again."
    exit /b 1
)
call :print_success "Docker is running"
goto :eof

REM Check for docker-compose
:check_docker_compose
where docker-compose >nul 2>&1
if not errorlevel 1 (
    set COMPOSE_CMD=docker-compose
    goto :compose_found
)

docker compose version >nul 2>&1
if not errorlevel 1 (
    set COMPOSE_CMD=docker compose
    goto :compose_found
)

call :print_error "Neither docker-compose nor 'docker compose' is available"
exit /b 1

:compose_found
call :print_success "Using: %COMPOSE_CMD%"
goto :eof

REM Setup directories
:setup_directories
call :print_status "Setting up required directories..."
if not exist data mkdir data
if not exist uploads mkdir uploads
if not exist logs mkdir logs
call :print_success "Directories created"
goto :eof

REM Check environment file
:check_environment
if not exist .env (
    call :print_warning "No .env file found"
    if exist docker\environment.template (
        call :print_status "Copying environment template..."
        copy docker\environment.template .env >nul
        call :print_warning "Please edit .env file with your API keys before starting the system"
        exit /b 1
    ) else (
        call :print_error "No environment template found"
        exit /b 1
    )
)
call :print_success "Environment file found"
goto :eof

REM Start basic services
:start_basic
call :print_status "Starting basic services (API + UI)..."
%COMPOSE_CMD% up -d api ui
call :print_success "Basic services started"
call :print_status "API available at: http://localhost:8000"
call :print_status "UI available at: http://localhost:8001"
goto :eof

REM Start with Ollama
:start_with_ollama
call :print_status "Starting services with Ollama..."
%COMPOSE_CMD% --profile ollama up -d
call :print_success "Services with Ollama started"
call :print_status "API available at: http://localhost:8000"
call :print_status "UI available at: http://localhost:8001"
call :print_status "Ollama available at: http://localhost:11434"

call :print_status "Pulling recommended Ollama models..."
timeout /t 10 >nul
docker exec advanced_rag_ollama ollama pull llama3.2 || call :print_warning "Failed to pull llama3.2"
docker exec advanced_rag_ollama ollama pull codellama || call :print_warning "Failed to pull codellama"
goto :eof

REM Start full stack
:start_full
call :print_status "Starting full stack..."
%COMPOSE_CMD% --profile full up -d
call :print_success "Full stack started"
call :print_status "Access via Nginx: http://localhost:80"
call :print_status "API direct: http://localhost:8000"
call :print_status "UI direct: http://localhost:8001"
call :print_status "Ollama: http://localhost:11434"
call :print_status "Redis: localhost:6379"
goto :eof

REM Show status
:show_status
call :print_status "System Status:"
%COMPOSE_CMD% ps
echo.
call :print_status "Service Health:"
%COMPOSE_CMD% exec api curl -s http://localhost:8000/health || call :print_warning "API not responding"
%COMPOSE_CMD% exec ui curl -s http://localhost:8001/ || call :print_warning "UI not responding"
goto :eof

REM Stop services
:stop_services
call :print_status "Stopping services..."
%COMPOSE_CMD% down
call :print_success "Services stopped"
goto :eof

REM Cleanup
:cleanup
call :print_status "Stopping and cleaning up..."
%COMPOSE_CMD% down -v --remove-orphans
docker system prune -f
call :print_success "Cleanup completed"
goto :eof

REM Show logs
:show_logs
if "%~2"=="" (
    call :print_status "Showing all logs..."
    %COMPOSE_CMD% logs -f
) else (
    call :print_status "Showing logs for %~2..."
    %COMPOSE_CMD% logs -f %~2
)
goto :eof

REM Restart services
:restart_services
call :print_status "Restarting services..."
%COMPOSE_CMD% restart
call :print_success "Services restarted"
goto :eof

REM Show help
:show_help
echo Advanced RAG System - Docker Management Script for Windows
echo.
echo Usage: %~nx0 [COMMAND]
echo.
echo Commands:
echo   basic      Start basic services (API + UI)
echo   ollama     Start with Ollama support
echo   full       Start full stack (with Redis and Nginx)
echo   status     Show system status
echo   logs       Show logs (optionally for specific service)
echo   stop       Stop all services
echo   restart    Restart all services
echo   cleanup    Stop and clean up all resources
echo   build      Build Docker images
echo   pull       Pull latest images
echo   setup      Setup directories and environment
echo   help       Show this help message
echo.
echo Examples:
echo   %~nx0 basic                 # Start API and UI only
echo   %~nx0 ollama                # Start with local Llama models
echo   %~nx0 full                  # Start everything including Nginx
echo   %~nx0 logs api              # Show API logs
echo   %~nx0 logs                  # Show all logs
echo.
goto :eof

REM Main logic
set command=%1
if "%command%"=="" set command=help

if "%command%"=="basic" (
    call :check_docker
    call :check_docker_compose
    call :setup_directories
    call :check_environment
    if not errorlevel 1 (
        call :start_basic
        call :show_status
    )
) else if "%command%"=="ollama" (
    call :check_docker
    call :check_docker_compose
    call :setup_directories
    call :check_environment
    if not errorlevel 1 (
        call :start_with_ollama
        call :show_status
    )
) else if "%command%"=="full" (
    call :check_docker
    call :check_docker_compose
    call :setup_directories
    call :check_environment
    if not errorlevel 1 (
        call :start_full
        call :show_status
    )
) else if "%command%"=="status" (
    call :check_docker_compose
    call :show_status
) else if "%command%"=="logs" (
    call :check_docker_compose
    call :show_logs %*
) else if "%command%"=="stop" (
    call :check_docker_compose
    call :stop_services
) else if "%command%"=="restart" (
    call :check_docker_compose
    call :restart_services
) else if "%command%"=="cleanup" (
    call :check_docker_compose
    call :cleanup
) else if "%command%"=="build" (
    call :check_docker
    call :check_docker_compose
    call :print_status "Building Docker images..."
    %COMPOSE_CMD% build --no-cache
    call :print_success "Images built"
) else if "%command%"=="pull" (
    call :check_docker
    call :check_docker_compose
    call :print_status "Pulling latest Docker images..."
    %COMPOSE_CMD% pull
    call :print_success "Images updated"
) else if "%command%"=="setup" (
    call :setup_directories
    call :check_environment
) else (
    call :show_help
)
