from __future__ import annotations

from typing import Dict, Optional, Any

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from pathlib import Path

from app.embeddings.factory import EmbeddingFactory
from app.generation.factory import GenerationFactory
from app.models.config import Settings, get_settings
from app.retrieval.service import HybridRetrievalService
from app.deployment.rollback import RollbackManager
from app.deployment.ab_testing import ABTestingManager
from app.deployment.canary import CanaryManager
from app.deployment.load_balancer import LoadBalancerManager
from app.deployment.validation import DeploymentValidator, ValidationLevel
from app.monitoring.alerting import AlertingManager
from app.monitoring.dashboard import MonitoringOrchestrator


admin_router = APIRouter(prefix="/admin")


def _auth(api_key: str | None) -> None:
    s = get_settings()
    if s.admin_api_key and api_key != s.admin_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


@admin_router.post("/hot-swap/embeddings")
def hot_swap_embeddings(backend: str, model: str, x_api_key: str | None = Header(default=None)) -> Dict[str, str]:
    _auth(x_api_key)
    s = get_settings()
    s.embedding_backend = backend
    s.embedding_model = model
    return {"status": "ok", "embedding_backend": backend, "embedding_model": model}


@admin_router.post("/hot-swap/generation")
def hot_swap_generation(backend: str, model: str, x_api_key: str | None = Header(default=None)) -> Dict[str, str]:
    _auth(x_api_key)
    
    # Validate backend and model
    from app.generation.factory import ModelRegistry
    if not ModelRegistry.is_model_available(backend, model):
        # Still allow it but with a warning
        pass
    
    try:
        # Test the model by creating a factory instance
        factory = GenerationFactory(backend=backend, model=model)
        client = factory.build()
        
        # Update settings
        s = get_settings()
        s.generation_backend = backend
        s.generation_model = model
        
        return {
            "status": "ok", 
            "generation_backend": backend, 
            "generation_model": model,
            "message": f"Successfully switched to {backend}:{model}"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to initialize {backend}:{model} - {str(e)}")


@admin_router.post("/hot-swap/retriever")
def hot_swap_retriever(backend: str, x_api_key: str | None = Header(default=None)) -> Dict[str, str]:
    _auth(x_api_key)
    s = get_settings()
    s.retriever_backend = backend
    return {"status": "ok", "retriever_backend": backend}


@admin_router.get("/models/available")
def get_available_models(x_api_key: str | None = Header(default=None)) -> Dict[str, Dict]:
    """Get all available models organized by backend."""
    _auth(x_api_key)
    
    from app.generation.factory import ModelRegistry
    
    available_models = ModelRegistry.get_available_models()
    
    # Check Ollama status and get real available models
    try:
        import httpx
        with httpx.Client(base_url="http://127.0.0.1:11434", timeout=5.0) as client:
            response = client.get("/api/tags")
            if response.status_code == 200:
                data = response.json()
                ollama_models = [model["name"] for model in data.get("models", [])]
                available_models["ollama"] = {
                    model: ModelRegistry.get_model_config("ollama", model) or {"max_tokens": 2048, "context_window": 4096}
                    for model in ollama_models
                }
                # Add a status indicator
                available_models["_ollama_status"] = "running"
            else:
                available_models["_ollama_status"] = "not_running"
    except Exception:
        available_models["_ollama_status"] = "not_running"
    
    return available_models


@admin_router.get("/models/status")
def get_model_status(x_api_key: str | None = Header(default=None)) -> Dict[str, str]:
    """Get current model configuration status."""
    _auth(x_api_key)
    
    s = get_settings()
    
    return {
        "generation_backend": s.generation_backend,
        "generation_model": s.generation_model,
        "embedding_backend": s.embedding_backend,
        "embedding_model": s.embedding_model,
        "retriever_backend": s.retriever_backend
    }


@admin_router.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    html_path = Path(__file__).parent / "templates" / "dashboard.html"
    return HTMLResponse(html_path.read_text())


# ===== DEPLOYMENT MANAGEMENT =====

# Initialize deployment managers (lazy initialization)
_rollback_manager = None
_ab_testing_manager = None
_canary_manager = None
_load_balancer_manager = None
_deployment_validator = None
_alerting_manager = None
_monitoring_orchestrator = None

def get_rollback_manager() -> RollbackManager:
    global _rollback_manager
    if _rollback_manager is None:
        _rollback_manager = RollbackManager()
    return _rollback_manager

def get_monitoring_orchestrator() -> MonitoringOrchestrator:
    global _monitoring_orchestrator
    if _monitoring_orchestrator is None:
        _monitoring_orchestrator = MonitoringOrchestrator()
    return _monitoring_orchestrator


# ===== MONITORING & DASHBOARDS =====

@admin_router.get("/monitoring/summary")
def get_monitoring_summary(x_api_key: str | None = Header(default=None)) -> Dict[str, Any]:
    """Get monitoring system summary."""
    _auth(x_api_key)
    
    monitoring = get_monitoring_orchestrator()
    return monitoring.get_monitoring_summary()

@admin_router.get("/monitoring/health")
def get_system_health(x_api_key: str | None = Header(default=None)) -> Dict[str, Any]:
    """Get system health summary."""
    _auth(x_api_key)
    
    monitoring = get_monitoring_orchestrator()
    return monitoring.get_system_health_summary()

@admin_router.get("/monitoring/dashboards")
def get_available_dashboards(x_api_key: str | None = Header(default=None)) -> Dict[str, Any]:
    """Get available monitoring dashboards."""
    _auth(x_api_key)
    
    monitoring = get_monitoring_orchestrator()
    dashboards = monitoring.dashboard_manager.get_all_dashboards_summary()
    return {"dashboards": dashboards}

@admin_router.get("/monitoring/dashboard/{dashboard_id}")
def get_dashboard_data(dashboard_id: str, x_api_key: str | None = Header(default=None)) -> Dict[str, Any]:
    """Get dashboard data."""
    _auth(x_api_key)
    
    monitoring = get_monitoring_orchestrator()
    dashboard_data = monitoring.dashboard_manager.get_dashboard_data(dashboard_id)
    
    if not dashboard_data:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    
    return dashboard_data

@admin_router.get("/monitoring/dashboard/{dashboard_id}/html")
def get_dashboard_html(dashboard_id: str, x_api_key: str | None = Header(default=None)) -> HTMLResponse:
    """Get dashboard as HTML page."""
    _auth(x_api_key)
    
    monitoring = get_monitoring_orchestrator()
    html_content = monitoring.dashboard_manager.generate_dashboard_html(dashboard_id)
    
    if not html_content:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    
    return HTMLResponse(content=html_content)

@admin_router.post("/monitoring/start")
async def start_monitoring(x_api_key: str | None = Header(default=None)) -> Dict[str, str]:
    """Start monitoring system."""
    _auth(x_api_key)
    
    monitoring = get_monitoring_orchestrator()
    
    # Start monitoring in background
    import asyncio
    asyncio.create_task(monitoring.start_monitoring())
    
    return {"status": "started", "message": "Monitoring system started"}


# ===== ROLLBACK MANAGEMENT =====

@admin_router.post("/rollback/emergency")
async def emergency_rollback(x_api_key: str | None = Header(default=None)) -> Dict[str, str]:
    """Trigger emergency rollback to last known good configuration."""
    _auth(x_api_key)
    
    rollback_manager = get_rollback_manager()
    success = await rollback_manager.emergency_rollback()
    
    return {
        "status": "success" if success else "failed",
        "message": "Emergency rollback completed" if success else "Emergency rollback failed"
    }

@admin_router.get("/rollback/stats")
def get_rollback_stats(x_api_key: str | None = Header(default=None)) -> Dict[str, Any]:
    """Get rollback statistics."""
    _auth(x_api_key)
    
    rollback_manager = get_rollback_manager()
    return rollback_manager.get_rollback_stats()


