@echo off
REM Advanced RAG System - Docker Deployment Script (Windows Batch)
REM This script handles building, testing, and running the RAG system with image processing fixes

setlocal enabledelayedexpansion

if "%1"=="" (
    echo Advanced RAG System - Docker Deployment Script (Windows)
    echo.
    echo Usage: docker-deploy.bat [COMMAND]
    echo.
    echo Commands:
    echo   build     - Build Docker images
    echo   start     - Start all services
    echo   stop      - Stop all services
    echo   restart   - Restart all services
    echo   status    - Show service status
    echo   logs      - Show service logs
    echo   test      - Run system tests
    echo   cleanup   - Clean up Docker resources
    echo   deploy    - Full deployment (build + start)
    echo   help      - Show this help message
    echo.
    echo Examples:
    echo   docker-deploy.bat deploy    # Full deployment
    echo   docker-deploy.bat start     # Start services
    echo   docker-deploy.bat logs      # View logs
    goto :eof
)

echo [INFO] Advanced RAG System - Docker Deployment
echo =============================================

if "%1"=="build" goto :build
if "%1"=="start" goto :start
if "%1"=="stop" goto :stop
if "%1"=="restart" goto :restart
if "%1"=="status" goto :status
if "%1"=="logs" goto :logs
if "%1"=="test" goto :test
if "%1"=="cleanup" goto :cleanup
if "%1"=="deploy" goto :deploy
if "%1"=="help" goto :help
goto :help

:check_docker
echo [INFO] Checking Docker installation...
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running. Please start Docker and try again.
    exit /b 1
)
echo [SUCCESS] Docker is running
goto :eof

:create_directories
echo [INFO] Creating necessary directories...
if not exist "data\uploads" mkdir "data\uploads"
if not exist "data\vectorstore" mkdir "data\vectorstore"
if not exist "logs" mkdir "logs"
echo [SUCCESS] Directories created
goto :eof

:build
call :check_docker
call :create_directories
echo [INFO] Building Docker images...
docker-compose build --no-cache
if errorlevel 1 (
    echo [ERROR] Docker build failed
    exit /b 1
)
echo [SUCCESS] Docker images built successfully
goto :eof

:start
call :check_docker
echo [INFO] Starting RAG services...
docker-compose up -d
if errorlevel 1 (
    echo [ERROR] Failed to start services
    exit /b 1
)
echo [INFO] Waiting for services to be ready...
timeout /t 10 /nobreak >nul
echo [SUCCESS] Services started
call :status
goto :eof

:stop
echo [INFO] Stopping RAG services...
docker-compose down
echo [SUCCESS] Services stopped
goto :eof

:restart
call :stop
call :start
goto :eof

:status
echo [INFO] Service Status:
docker-compose ps
echo.
echo [INFO] Service URLs:
echo API: http://localhost:8000
echo UI: http://localhost:8501
echo API Docs: http://localhost:8000/docs
echo Health Check: http://localhost:8000/api/health
goto :eof

:logs
echo [INFO] Showing logs (Ctrl+C to exit)...
docker-compose logs -f
goto :eof

:test
echo [INFO] Running system tests...
echo [INFO] Testing API health...
curl -f http://localhost:8000/api/health >nul 2>&1
if errorlevel 1 (
    echo [ERROR] API health check failed
    exit /b 1
)
echo [SUCCESS] API health check passed
echo [SUCCESS] Basic tests completed
goto :eof

:cleanup
echo [INFO] Cleaning up Docker resources...
docker-compose down -v --remove-orphans
docker system prune -f
echo [SUCCESS] Cleanup completed
goto :eof

:deploy
echo [INFO] Starting full deployment...
call :build
if errorlevel 1 goto :eof
call :start
if errorlevel 1 goto :eof
echo [SUCCESS] Deployment completed successfully!
echo.
echo [INFO] Next steps:
echo 1. Open http://localhost:8501 to access the UI
echo 2. Open http://localhost:8000/docs to view API documentation
echo 3. Upload images and test the image processing capabilities
echo 4. Use 'docker-compose logs -f' to monitor logs
goto :eof

:help
echo Advanced RAG System - Docker Deployment Script (Windows)
echo.
echo Usage: docker-deploy.bat [COMMAND]
echo.
echo Commands:
echo   build     - Build Docker images
echo   start     - Start all services
echo   stop      - Stop all services
echo   restart   - Restart all services
echo   status    - Show service status
echo   logs      - Show service logs
echo   test      - Run system tests
echo   cleanup   - Clean up Docker resources
echo   deploy    - Full deployment (build + start)
echo   help      - Show this help message
echo.
echo Examples:
echo   docker-deploy.bat deploy    # Full deployment
echo   docker-deploy.bat start     # Start services
echo   docker-deploy.bat logs      # View logs
goto :eof
