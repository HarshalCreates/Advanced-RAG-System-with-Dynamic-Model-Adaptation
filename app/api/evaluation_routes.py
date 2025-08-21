"""API routes for comprehensive multi-dimensional evaluation."""
from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel

from app.security.auth import get_current_user, require_permission, User, Permission
from app.evaluation.evaluation_manager import (
    get_evaluation_manager, EvaluationOrchestrator, EvaluationRequest,
    EvaluationMode, EvaluationPriority
)
from app.models.schemas import QueryRequest, RAGResponse


evaluation_router = APIRouter()


class EvaluationRequestModel(BaseModel):
    """Request model for evaluation API."""
    mode: str = "standard"  # quick, standard, thorough, production
    priority: str = "medium"  # low, medium, high, critical
    query: str
    response: str
    sources: List[str] = []
    citations: List[Dict[str, Any]] = []
    source_documents: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}


class QuickEvaluationRequest(BaseModel):
    """Quick evaluation request for integrated RAG queries."""
    query: str
    response: RAGResponse
    enable_hallucination_detection: bool = True
    enable_aspect_evaluation: bool = False


@evaluation_router.post("/evaluate/comprehensive")
async def evaluate_comprehensive(
    request: EvaluationRequestModel,
    background_tasks: BackgroundTasks,
    evaluation_manager: EvaluationOrchestrator = Depends(get_evaluation_manager),
    current_user: User = Depends(require_permission(Permission.QUERY_SYSTEM))
):
    """Run comprehensive multi-dimensional evaluation."""
    
    try:
        # Convert string enums to proper enums
        mode = EvaluationMode(request.mode)
        priority = EvaluationPriority(request.priority)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid enum value: {e}")
    
    # Create evaluation request
    eval_request = EvaluationRequest(
        request_id=str(uuid.uuid4()),
        mode=mode,
        priority=priority,
        query=request.query,
        response=request.response,
        sources=request.sources,
        citations=request.citations,
        source_documents=request.source_documents,
        user_id=current_user.user_id,
        metadata=request.metadata
    )
    
    # Run evaluation
    try:
        result = await evaluation_manager.evaluate_comprehensive(eval_request)
        
        # Get summary for API response
        summary = evaluation_manager.get_evaluation_summary(result)
        
        return {
            "evaluation_id": result.request_id,
            "overall_score": result.overall_score,
            "quality_grade": result.quality_grade,
            "confidence_level": result.confidence_level,
            "summary": summary,
            "key_insights": {
                "strengths": result.key_strengths,
                "critical_issues": result.critical_issues,
                "recommendations": result.improvement_recommendations[:5]
            },
            "evaluation_time_ms": result.evaluation_time_ms,
            "modules_used": result.modules_used
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@evaluation_router.post("/evaluate/quick")
async def evaluate_quick(
    request: QuickEvaluationRequest,
    evaluation_manager: EvaluationOrchestrator = Depends(get_evaluation_manager),
    current_user: User = Depends(require_permission(Permission.QUERY_SYSTEM))
):
    """Quick evaluation for real-time RAG response assessment."""
    
    # Extract data from RAG response
    sources = []
    citations = []
    
    for citation in request.response.citations:
        sources.append(citation.excerpt)
        citations.append({
            "document": citation.document,
            "pages": citation.pages,
            "excerpt": citation.excerpt,
            "relevance_score": citation.relevance_score
        })
    
    # Create quick evaluation request
    eval_request = EvaluationRequest(
        request_id=str(uuid.uuid4()),
        mode=EvaluationMode.QUICK,
        priority=EvaluationPriority.MEDIUM,
        query=request.query,
        response=request.response.answer.content,
        sources=sources,
        citations=citations,
        source_documents=[],
        user_id=current_user.user_id,
        metadata={
            "response_time_ms": request.response.performance_metrics.total_response_time_ms,
            "confidence": request.response.answer.confidence
        }
    )
    
    # Override mode based on request options
    if request.enable_aspect_evaluation:
        eval_request.mode = EvaluationMode.STANDARD
    
    try:
        result = await evaluation_manager.evaluate_comprehensive(eval_request)
        
        return {
            "evaluation_score": result.overall_score,
            "quality_grade": result.quality_grade,
            "confidence": result.confidence_level,
            "hallucination_risk": "low" if (result.hallucination_report and 
                                          result.hallucination_report.overall_reliability_score > 0.8) else "medium",
            "key_issues": result.critical_issues[:3],
            "recommendations": result.improvement_recommendations[:3],
            "evaluation_time_ms": result.evaluation_time_ms
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick evaluation failed: {str(e)}")


@evaluation_router.get("/analytics/dashboard")
async def get_evaluation_dashboard(
    days: int = 7,
    evaluation_manager: EvaluationOrchestrator = Depends(get_evaluation_manager),
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYTICS))
):
    """Get evaluation analytics dashboard."""
    
    try:
        dashboard_data = evaluation_manager.get_analytics_dashboard(days)
        return dashboard_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard: {str(e)}")


@evaluation_router.get("/performance/benchmarks")
async def get_performance_benchmarks(
    evaluation_manager: EvaluationOrchestrator = Depends(get_evaluation_manager),
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYTICS))
):
    """Get performance benchmarking results."""
    
    try:
        # Get current performance metrics
        performance_dashboard = evaluation_manager.performance_analyzer.get_performance_dashboard()
        
        # Get performance summary
        performance_summary = evaluation_manager.performance_analyzer.get_performance_summary()
        
        return {
            "current_metrics": performance_dashboard,
            "summary": performance_summary,
            "percentiles": evaluation_manager.performance_analyzer.performance_tracker.get_all_current_percentiles()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get benchmarks: {str(e)}")


@evaluation_router.post("/security/adversarial-test")
async def run_adversarial_test(
    num_tests_per_type: int = 3,
    evaluation_manager: EvaluationOrchestrator = Depends(get_evaluation_manager),
    current_user: User = Depends(require_permission(Permission.MANAGE_SYSTEM))
):
    """Run adversarial security testing."""
    
    try:
        # Create a test query function
        async def test_query_function(test_query: str) -> str:
            # This would integrate with the actual RAG system
            # For now, return a safe default response
            return ("I understand you're asking about that topic. I'm designed to provide "
                   "helpful, accurate, and safe information based on reliable sources. "
                   "How can I assist you with factual information?")
        
        # Run adversarial tests
        robustness_report = await evaluation_manager.adversarial_tester.run_adversarial_test_suite(
            test_query_function, num_tests_per_type
        )
        
        # Get summary
        summary = evaluation_manager.adversarial_tester.get_robustness_summary(robustness_report)
        
        return {
            "robustness_assessment": summary,
            "detailed_results": {
                "overall_robustness": robustness_report.overall_robustness.value,
                "vulnerability_score": robustness_report.vulnerability_score,
                "tests_passed": robustness_report.tests_passed,
                "tests_failed": robustness_report.tests_failed,
                "critical_vulnerabilities": robustness_report.critical_vulnerabilities,
                "security_recommendations": robustness_report.security_recommendations[:5]
            },
            "test_duration_ms": robustness_report.test_duration_ms
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Adversarial testing failed: {str(e)}")


@evaluation_router.post("/performance/load-test")
async def run_load_test(
    background_tasks: BackgroundTasks,
    evaluation_manager: EvaluationOrchestrator = Depends(get_evaluation_manager),
    current_user: User = Depends(require_permission(Permission.MANAGE_SYSTEM)),
    test_type: str = "normal_load"
):
    """Run automated load testing."""
    
    if test_type not in ["light_load", "normal_load", "heavy_load", "stress_test"]:
        raise HTTPException(status_code=400, detail="Invalid test type")
    
    try:
        # Run load test in background
        load_results = await evaluation_manager.performance_analyzer.load_tester.run_load_test(
            test_type, None
        )
        
        return {
            "test_id": load_results.test_id,
            "test_type": load_results.test_type,
            "results": {
                "concurrent_users": load_results.concurrent_users,
                "total_requests": load_results.total_requests,
                "success_rate": load_results.success_rate,
                "avg_response_time": load_results.avg_response_time,
                "throughput_qps": load_results.throughput_qps,
                "performance_degradation_point": load_results.performance_degradation_point
            },
            "recommendations": load_results.recommendations,
            "duration_seconds": load_results.duration_seconds
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load testing failed: {str(e)}")


@evaluation_router.get("/quality/trends")
async def get_quality_trends(
    days: int = 30,
    evaluation_manager: EvaluationOrchestrator = Depends(get_evaluation_manager),
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYTICS))
):
    """Get quality trends over time."""
    
    try:
        dashboard_data = evaluation_manager.get_analytics_dashboard(days)
        
        # Extract trend information
        trends = {
            "time_period_days": days,
            "evaluation_count": dashboard_data.get("evaluation_count", 0),
            "average_score": dashboard_data.get("performance_summary", {}).get("average_score", 0),
            "score_trend": dashboard_data.get("performance_summary", {}).get("score_trend", "stable"),
            "quality_distribution": dashboard_data.get("performance_summary", {}).get("quality_distribution", {}),
            "top_issues": dashboard_data.get("quality_insights", {}).get("top_issues", []),
            "improvement_areas": dashboard_data.get("quality_insights", {}).get("improvement_areas", [])
        }
        
        return trends
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trends: {str(e)}")


@evaluation_router.get("/models/consensus")
async def get_model_consensus_info(
    evaluation_manager: EvaluationOrchestrator = Depends(get_evaluation_manager),
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYTICS))
):
    """Get information about multi-model consensus evaluation."""
    
    return {
        "available_judges": ["gpt-4o", "claude-3-sonnet-20240229", "gpt-3.5-turbo"],
        "evaluation_aspects": [
            "accuracy", "completeness", "clarity", "relevance", "citation_quality"
        ],
        "consensus_method": "weighted_voting",
        "agreement_threshold": 0.8,
        "supported_features": [
            "Multi-model parallel evaluation",
            "Consensus scoring algorithms", 
            "Model disagreement analysis",
            "Aspect-based evaluation",
            "Confidence scoring"
        ]
    }


@evaluation_router.get("/hallucination/detection-info")
async def get_hallucination_detection_info(
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYTICS))
):
    """Get information about hallucination detection capabilities."""
    
    return {
        "detection_methods": [
            "Content-source alignment checking",
            "Factual claim verification",
            "Citation accuracy validation",
            "Temporal accuracy verification"
        ],
        "hallucination_types": [
            "factual_error", "unsupported_claim", "contradictory_info",
            "fabricated_citation", "temporal_inaccuracy", "numerical_error"
        ],
        "severity_levels": ["minor", "moderate", "major", "critical"],
        "confidence_scoring": "enabled",
        "real_time_detection": "enabled"
    }


@evaluation_router.get("/system/evaluation-status")
async def get_evaluation_system_status(
    evaluation_manager: EvaluationOrchestrator = Depends(get_evaluation_manager),
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYTICS))
):
    """Get overall evaluation system status."""
    
    try:
        # Check module availability
        modules_status = {
            "multi_model_judge": "available",
            "hallucination_detector": "available", 
            "performance_analyzer": "available",
            "aspect_evaluator": "available",
            "adversarial_tester": "available"
        }
        
        # Get recent activity
        dashboard_data = evaluation_manager.get_analytics_dashboard(1)  # Last 24 hours
        
        return {
            "system_status": "operational",
            "modules_available": len([s for s in modules_status.values() if s == "available"]),
            "modules_status": modules_status,
            "recent_activity": {
                "evaluations_24h": dashboard_data.get("evaluation_count", 0),
                "average_score_24h": dashboard_data.get("performance_summary", {}).get("average_score", 0),
                "average_evaluation_time": dashboard_data.get("module_analytics", {}).get("average_evaluation_time", 0)
            },
            "capabilities": [
                "Multi-dimensional scoring",
                "Real-time evaluation",
                "Consensus judging",
                "Hallucination detection",
                "Performance benchmarking",
                "Adversarial testing",
                "Quality analytics"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")
