"""Enhanced API routes with new security and infrastructure features."""
from __future__ import annotations

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Request, status
from pydantic import BaseModel

from app.security.auth import (
    get_current_user, require_permission, require_role, 
    User, Role, Permission, get_user_manager, UserManager
)
from app.security.privacy import PrivacyManager
from app.security.advanced_security import SecurityManager
from app.api.query_manager import QueryManager
from app.monitoring.anomaly_detection import AnomalyDetectionManager
from app.monitoring.cost_optimization import CostTrackingManager
from app.models.schemas import QueryRequest, RAGResponse


enhanced_router = APIRouter()

# Global managers (lazy initialization)
_privacy_manager: PrivacyManager | None = None
_security_manager: SecurityManager | None = None
_query_manager: QueryManager | None = None
_anomaly_detector: AnomalyDetectionManager | None = None
_cost_tracker: CostTrackingManager | None = None


def get_privacy_manager() -> PrivacyManager:
    global _privacy_manager
    if _privacy_manager is None:
        _privacy_manager = PrivacyManager()
    return _privacy_manager


def get_security_manager() -> SecurityManager:
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


def get_query_manager() -> QueryManager:
    global _query_manager
    if _query_manager is None:
        _query_manager = QueryManager()
    return _query_manager


def get_anomaly_detector() -> AnomalyDetectionManager:
    global _anomaly_detector
    if _anomaly_detector is None:
        _anomaly_detector = AnomalyDetectionManager()
    return _anomaly_detector


def get_cost_tracker() -> CostTrackingManager:
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTrackingManager()
    return _cost_tracker


# ===== AUTHENTICATION ROUTES =====

class LoginRequest(BaseModel):
    username: str
    password: str


class UserCreateRequest(BaseModel):
    username: str
    email: str
    password: str
    role: str = "user"


@enhanced_router.post("/auth/login")
async def login(
    request: LoginRequest,
    user_manager: UserManager = Depends(get_user_manager)
):
    """Authenticate user and return access token."""
    user = user_manager.authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    access_token = user_manager.create_access_token(user)
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions]
        }
    }


@enhanced_router.post("/auth/register")
async def register(
    request: UserCreateRequest,
    user_manager: UserManager = Depends(get_user_manager)
):
    """Register a new user."""
    try:
        role = Role(request.role)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid role")
    
    try:
        user = user_manager.create_user(
            username=request.username,
            email=request.email,
            password=request.password,
            role=role
        )
        
        return {
            "message": "User created successfully",
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@enhanced_router.get("/auth/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return {
        "user_id": current_user.user_id,
        "username": current_user.username,
        "email": current_user.email,
        "role": current_user.role.value,
        "permissions": [p.value for p in current_user.permissions],
        "document_access": current_user.document_access,
        "last_login": current_user.last_login,
        "is_active": current_user.is_active
    }


# ===== SECURE QUERY ROUTES =====

class SecureQueryRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Dict[str, Any] = {}
    anonymize_pii: bool = True
    priority: int = 1


@enhanced_router.post("/secure/query", response_model=RAGResponse)
async def secure_query(
    request: SecureQueryRequest,
    http_request: Request,
    current_user: User = Depends(require_permission(Permission.QUERY_SYSTEM)),
    privacy_manager: PrivacyManager = Depends(get_privacy_manager),
    security_manager: SecurityManager = Depends(get_security_manager),
    query_manager: QueryManager = Depends(get_query_manager),
    anomaly_detector: AnomalyDetectionManager = Depends(get_anomaly_detector),
    cost_tracker: CostTrackingManager = Depends(get_cost_tracker)
):
    """Secure query endpoint with full security, privacy, and cost tracking."""
    
    # Get client IP
    client_ip = getattr(http_request.client, 'host', 'unknown') if http_request.client else 'unknown'
    
    # 1. Security Analysis
    is_secure, security_reason, security_threat = security_manager.analyze_query_security(
        request.query, current_user.user_id, client_ip
    )
    
    if not is_secure:
        raise HTTPException(status_code=400, detail=f"Security check failed: {security_reason}")
    
    # 2. PII Detection and Anonymization
    original_query = request.query
    if request.anonymize_pii:
        anonymized_query, pii_detected = privacy_manager.anonymize_query(request.query)
        if pii_detected:
            request.query = anonymized_query
            # Record PII detection
            privacy_manager.record_data_processing(
                user_id=current_user.user_id,
                activity_type="pii_anonymization",
                data_categories=[privacy_manager.consent_manager.DataCategory.CONTENT],
                purpose="query_processing"
            )
    
    # 3. Check data processing consent
    has_consent = privacy_manager.check_processing_consent(current_user.user_id, "query_processing")
    if not has_consent:
        raise HTTPException(
            status_code=403, 
            detail="Data processing consent required. Please provide consent for query processing."
        )
    
    # 4. Cost Estimation
    estimated_cost_data = {
        "query": request.query,
        "top_k": request.top_k,
        "embedding_backend": "openai",  # Would be dynamic
        "embedding_model": "text-embedding-ada-002",
        "generation_backend": "openai",
        "generation_model": "gpt-4o",
        "context_tokens": 500,
        "response_tokens": 150
    }
    estimated_cost = cost_tracker.estimate_query_cost(estimated_cost_data)
    
    # 5. Submit to Query Manager for resource management
    from app.pipeline.manager import PipelineManager
    
    async def execute_secure_query():
        manager = PipelineManager.get_instance()
        query_request = QueryRequest(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )
        return manager.query(query_request)
    
    query_id = await query_manager.submit_query(
        query_text=request.query,
        query_function=execute_secure_query,
        user_id=current_user.user_id,
        priority=request.priority,
        estimated_resources={"memory_mb": 150.0, "cpu_percent": 15.0}
    )
    
    # Wait for query completion (simplified - in production use async polling)
    import asyncio
    max_wait = 30
    wait_time = 0
    
    while wait_time < max_wait:
        query_status = await query_manager.get_query_status(query_id)
        if query_status and query_status.status.value in ["completed", "failed", "cancelled"]:
            break
        await asyncio.sleep(1)
        wait_time += 1
    
    if wait_time >= max_wait:
        raise HTTPException(status_code=408, detail="Query timeout")
    
    # Execute the query (for now, direct execution)
    query_request = QueryRequest(
        query=request.query,
        top_k=request.top_k,
        filters=request.filters
    )
    
    manager = PipelineManager.get_instance()
    result = manager.query(query_request)
    
    # 6. Update behavioral profile for anomaly detection
    anomaly_detector.update_user_behavior(current_user.user_id, {
        "query_length": len(request.query),
        "query_topic": "general",  # Would be extracted from query
        "session_duration": 300  # Estimated
    })
    
    # 7. Record actual costs
    actual_cost_data = {
        **estimated_cost_data,
        "response_time_ms": result.performance_metrics.total_response_time_ms,
        "cpu_cores": 0.1,
        "memory_gb": 0.15
    }
    
    cost_tracker.record_query_cost(actual_cost_data, current_user.user_id)
    
    # 8. Detect anomalies in response metrics
    anomaly_detector.analyze_metric(
        "response_time_ms", 
        result.performance_metrics.total_response_time_ms,
        user_id=current_user.user_id,
        metadata={"query_length": len(request.query)}
    )
    
    # 9. Record data processing activity
    privacy_manager.record_data_processing(
        user_id=current_user.user_id,
        activity_type="query_processing",
        data_categories=[privacy_manager.consent_manager.DataCategory.CONTENT],
        purpose="information_retrieval"
    )
    
    return result


# ===== PRIVACY & COMPLIANCE ROUTES =====

class ConsentRequest(BaseModel):
    data_categories: List[str]
    purpose: str
    expires_days: Optional[int] = None


@enhanced_router.post("/privacy/consent")
async def give_consent(
    request: ConsentRequest,
    current_user: User = Depends(get_current_user),
    privacy_manager: PrivacyManager = Depends(get_privacy_manager)
):
    """Give consent for data processing."""
    from app.security.privacy import DataCategory
    
    try:
        data_categories = [DataCategory(cat) for cat in request.data_categories]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid data category")
    
    consent_id = privacy_manager.consent_manager.give_consent(
        user_id=current_user.user_id,
        data_categories=data_categories,
        purpose=request.purpose,
        expires_days=request.expires_days
    )
    
    return {"consent_id": consent_id, "message": "Consent recorded successfully"}


@enhanced_router.get("/privacy/my-data")
async def get_my_data(
    current_user: User = Depends(get_current_user),
    privacy_manager: PrivacyManager = Depends(get_privacy_manager)
):
    """Get user's personal data (GDPR Article 15)."""
    return privacy_manager.handle_data_subject_request(current_user.user_id, "access")


@enhanced_router.delete("/privacy/my-data")
async def delete_my_data(
    current_user: User = Depends(get_current_user),
    privacy_manager: PrivacyManager = Depends(get_privacy_manager)
):
    """Request data deletion (GDPR Article 17)."""
    return privacy_manager.handle_data_subject_request(current_user.user_id, "erasure")


@enhanced_router.get("/privacy/export")
async def export_my_data(
    current_user: User = Depends(get_current_user),
    privacy_manager: PrivacyManager = Depends(get_privacy_manager)
):
    """Export user data (GDPR Article 20)."""
    return privacy_manager.handle_data_subject_request(current_user.user_id, "portability")


# ===== MONITORING ROUTES =====

@enhanced_router.get("/monitoring/security")
async def get_security_stats(
    hours: int = 24,
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYTICS)),
    security_manager: SecurityManager = Depends(get_security_manager)
):
    """Get security statistics."""
    return security_manager.get_security_stats(hours)


@enhanced_router.get("/monitoring/anomalies")
async def get_anomaly_stats(
    hours: int = 24,
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYTICS)),
    anomaly_detector: AnomalyDetectionManager = Depends(get_anomaly_detector)
):
    """Get anomaly detection statistics."""
    return anomaly_detector.get_anomaly_summary(hours)


@enhanced_router.get("/monitoring/costs")
async def get_cost_stats(
    hours: int = 24,
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYTICS)),
    cost_tracker: CostTrackingManager = Depends(get_cost_tracker)
):
    """Get cost tracking statistics."""
    return cost_tracker.get_cost_summary(hours)


@enhanced_router.get("/monitoring/cost-optimization")
async def get_cost_optimization(
    days: int = 7,
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYTICS)),
    cost_tracker: CostTrackingManager = Depends(get_cost_tracker)
):
    """Get cost optimization recommendations."""
    recommendations = cost_tracker.get_optimization_recommendations(days)
    return {
        "recommendations": [
            {
                "strategy": rec.strategy.value,
                "description": rec.description,
                "estimated_savings_usd": rec.estimated_savings_usd,
                "estimated_savings_percent": rec.estimated_savings_percent,
                "implementation_effort": rec.implementation_effort,
                "priority": rec.priority
            }
            for rec in recommendations
        ]
    }


@enhanced_router.get("/monitoring/budgets")
async def get_budget_status(
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYTICS)),
    cost_tracker: CostTrackingManager = Depends(get_cost_tracker)
):
    """Get budget status."""
    return {"budgets": cost_tracker.get_budget_status()}


# ===== ADMIN ROUTES =====

@enhanced_router.get("/admin/users")
async def list_users(
    current_user: User = Depends(require_role(Role.ADMIN)),
    user_manager: UserManager = Depends(get_user_manager)
):
    """List all users (admin only)."""
    users = user_manager.get_all_users()
    return {
        "users": [
            {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "is_active": user.is_active,
                "created_at": user.created_at,
                "last_login": user.last_login
            }
            for user in users
        ]
    }


class UserUpdateRequest(BaseModel):
    role: Optional[str] = None
    is_active: Optional[bool] = None


@enhanced_router.patch("/admin/users/{user_id}")
async def update_user(
    user_id: str,
    request: UserUpdateRequest,
    current_user: User = Depends(require_role(Role.ADMIN)),
    user_manager: UserManager = Depends(get_user_manager)
):
    """Update user (admin only)."""
    target_user = user_manager.get_user_by_id(user_id)
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if request.role:
        try:
            new_role = Role(request.role)
            user_manager.update_user_role(user_id, new_role)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid role")
    
    if request.is_active is not None and not request.is_active:
        user_manager.deactivate_user(user_id)
    
    return {"message": "User updated successfully"}


@enhanced_router.get("/admin/privacy-summary")
async def get_privacy_summary(
    current_user: User = Depends(require_role(Role.ADMIN)),
    privacy_manager: PrivacyManager = Depends(get_privacy_manager)
):
    """Get privacy compliance summary (admin only)."""
    return privacy_manager.get_privacy_summary()


# ===== SYSTEM STATUS ROUTE =====

@enhanced_router.get("/status/comprehensive")
async def get_comprehensive_status(
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYTICS)),
    security_manager: SecurityManager = Depends(get_security_manager),
    anomaly_detector: AnomalyDetectionManager = Depends(get_anomaly_detector),
    cost_tracker: CostTrackingManager = Depends(get_cost_tracker),
    query_manager: QueryManager = Depends(get_query_manager)
):
    """Get comprehensive system status."""
    
    # Get all stats
    security_stats = security_manager.get_security_stats(24)
    anomaly_stats = anomaly_detector.get_anomaly_summary(24)
    cost_stats = cost_tracker.get_cost_summary(24)
    system_stats = query_manager.get_system_stats()
    budget_status = cost_tracker.get_budget_status()
    
    # Calculate overall health score
    health_factors = []
    
    # Security health (inversely related to threats)
    security_health = max(0, 100 - security_stats["total_threats"] * 2)
    health_factors.append(security_health)
    
    # Anomaly health (inversely related to anomalies)
    anomaly_health = max(0, 100 - anomaly_stats["total_anomalies"] * 5)
    health_factors.append(anomaly_health)
    
    # Resource health (based on capacity utilization)
    resource_health = max(0, 100 - system_stats["capacity_utilization"] * 100)
    health_factors.append(resource_health)
    
    # Budget health (inversely related to overspend)
    budget_health = 100
    for budget in budget_status:
        if budget["spend_ratio"] > 1.0:  # Over budget
            budget_health = min(budget_health, 50)
        elif budget["spend_ratio"] > 0.9:  # Near budget
            budget_health = min(budget_health, 80)
    health_factors.append(budget_health)
    
    overall_health = sum(health_factors) / len(health_factors)
    
    return {
        "overall_health_score": round(overall_health, 1),
        "security": security_stats,
        "anomalies": anomaly_stats,
        "costs": cost_stats,
        "system_resources": system_stats,
        "budgets": budget_status,
        "recommendations": [
            "Monitor security threats closely" if security_stats["total_threats"] > 10 else None,
            "Investigate anomalies" if anomaly_stats["total_anomalies"] > 5 else None,
            "Review cost optimization" if cost_stats["total_cost_usd"] > 5.0 else None,
            "Scale resources" if system_stats["capacity_utilization"] > 0.8 else None
        ]
    }
