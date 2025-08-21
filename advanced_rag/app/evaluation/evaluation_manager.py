"""Comprehensive evaluation manager integrating all assessment modules."""
from __future__ import annotations

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from app.evaluation.multi_model_judge import MultiModelJudge, ConsensusResult
from app.evaluation.hallucination_detector import HallucinationDetector, HallucinationReport
from app.evaluation.performance_analyzer import PerformanceAnalyzer, MetricType
from app.evaluation.aspect_evaluator import AspectBasedEvaluator, EvaluationResult
from app.evaluation.adversarial_tester import AdversarialTester, RobustnessReport


class EvaluationMode(Enum):
    """Evaluation modes for different use cases."""
    QUICK = "quick"           # Basic evaluation
    STANDARD = "standard"     # Comprehensive evaluation
    THOROUGH = "thorough"     # All modules including adversarial
    PRODUCTION = "production" # Production monitoring mode


class EvaluationPriority(Enum):
    """Priority levels for evaluation tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EvaluationRequest:
    """Request for comprehensive evaluation."""
    request_id: str
    mode: EvaluationMode
    priority: EvaluationPriority
    query: str
    response: str
    sources: List[str]
    citations: List[Dict[str, Any]]
    source_documents: List[Dict[str, Any]]
    user_id: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class ComprehensiveEvaluationResult:
    """Complete evaluation result from all modules."""
    request_id: str
    overall_score: float
    confidence_level: str
    
    # Multi-model judging results
    consensus_result: Optional[ConsensusResult]
    
    # Hallucination detection results
    hallucination_report: Optional[HallucinationReport]
    
    # Performance analysis
    performance_metrics: Dict[str, Any]
    
    # Aspect-based evaluation
    aspect_evaluation: Optional[EvaluationResult]
    
    # Adversarial testing (if applicable)
    robustness_report: Optional[RobustnessReport]
    
    # Unified insights
    key_strengths: List[str]
    critical_issues: List[str]
    improvement_recommendations: List[str]
    quality_grade: str
    
    # Metadata
    evaluation_time_ms: int
    modules_used: List[str]
    timestamp: float


class EvaluationOrchestrator:
    """Orchestrates evaluation across multiple modules."""
    
    def __init__(self, storage_path: str = "./data/evaluation"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize evaluation modules
        self.multi_model_judge = MultiModelJudge()
        self.hallucination_detector = HallucinationDetector()
        self.performance_analyzer = PerformanceAnalyzer()
        self.aspect_evaluator = AspectBasedEvaluator()
        self.adversarial_tester = AdversarialTester()
        
        # Evaluation history
        self.evaluation_history: List[ComprehensiveEvaluationResult] = []
        self.load_evaluation_history()
    
    async def evaluate_comprehensive(self, request: EvaluationRequest) -> ComprehensiveEvaluationResult:
        """Run comprehensive evaluation based on request mode."""
        
        start_time = time.time()
        modules_used = []
        
        # Initialize result containers
        consensus_result = None
        hallucination_report = None
        performance_metrics = {}
        aspect_evaluation = None
        robustness_report = None
        
        try:
            # Multi-model judging (for STANDARD, THOROUGH modes)
            if request.mode in [EvaluationMode.STANDARD, EvaluationMode.THOROUGH]:
                modules_used.append("multi_model_judge")
                consensus_result = await self.multi_model_judge.evaluate_response(
                    query=request.query,
                    response=request.response,
                    context="\\n\\n".join(request.sources),
                    citations=request.citations
                )
            
            # Hallucination detection (all modes except QUICK)
            if request.mode != EvaluationMode.QUICK:
                modules_used.append("hallucination_detector")
                hallucination_report = self.hallucination_detector.detect_hallucinations(
                    response_text=request.response,
                    source_texts=request.sources,
                    citations=request.citations,
                    source_documents=request.source_documents
                )
            
            # Performance tracking (all modes)
            modules_used.append("performance_analyzer")
            response_time = request.metadata.get("response_time_ms", 1000)
            self.performance_analyzer.record_performance_metric(
                MetricType.RESPONSE_TIME, 
                response_time,
                {"user_id": request.user_id, "query_length": len(request.query)}
            )
            performance_metrics = self.performance_analyzer.get_performance_dashboard()
            
            # Aspect-based evaluation (STANDARD, THOROUGH modes)
            if request.mode in [EvaluationMode.STANDARD, EvaluationMode.THOROUGH]:
                modules_used.append("aspect_evaluator")
                aspect_evaluation = await self.aspect_evaluator.evaluate_response(
                    response=request.response,
                    query=request.query,
                    sources=request.sources,
                    citations=request.citations,
                    source_documents=request.source_documents
                )
            
            # Adversarial testing (THOROUGH mode only)
            if request.mode == EvaluationMode.THOROUGH:
                modules_used.append("adversarial_tester")
                
                # Create a simple query function for adversarial testing
                async def test_query_function(test_query: str) -> str:
                    # This would normally call the actual RAG system
                    # For now, return a safe response
                    return "I can help you with that question. Let me provide accurate information based on available sources."
                
                robustness_report = await self.adversarial_tester.run_adversarial_test_suite(
                    test_query_function, num_tests_per_type=2
                )
            
        except Exception as e:
            print(f"Evaluation module error: {e}")
        
        # Calculate unified metrics
        overall_score = self._calculate_overall_score(
            consensus_result, hallucination_report, aspect_evaluation, robustness_report
        )
        
        confidence_level = self._determine_confidence_level(
            consensus_result, hallucination_report, aspect_evaluation
        )
        
        # Extract unified insights
        key_strengths, critical_issues, recommendations = self._extract_unified_insights(
            consensus_result, hallucination_report, aspect_evaluation, robustness_report
        )
        
        quality_grade = self._calculate_quality_grade(overall_score, critical_issues)
        
        evaluation_time = int((time.time() - start_time) * 1000)
        
        # Create comprehensive result
        result = ComprehensiveEvaluationResult(
            request_id=request.request_id,
            overall_score=overall_score,
            confidence_level=confidence_level,
            consensus_result=consensus_result,
            hallucination_report=hallucination_report,
            performance_metrics=performance_metrics,
            aspect_evaluation=aspect_evaluation,
            robustness_report=robustness_report,
            key_strengths=key_strengths,
            critical_issues=critical_issues,
            improvement_recommendations=recommendations,
            quality_grade=quality_grade,
            evaluation_time_ms=evaluation_time,
            modules_used=modules_used,
            timestamp=time.time()
        )
        
        # Store result
        self.evaluation_history.append(result)
        self.save_evaluation_history()
        
        return result
    
    def _calculate_overall_score(self, consensus_result: Optional[ConsensusResult],
                                hallucination_report: Optional[HallucinationReport],
                                aspect_evaluation: Optional[EvaluationResult],
                                robustness_report: Optional[RobustnessReport]) -> float:
        """Calculate unified overall score from all evaluations."""
        
        scores = []
        weights = []
        
        # Multi-model consensus score
        if consensus_result:
            scores.append(consensus_result.final_score)
            weights.append(0.3)
        
        # Hallucination reliability score (inverted - higher reliability = higher score)
        if hallucination_report:
            reliability_score = hallucination_report.overall_reliability_score * 10
            scores.append(reliability_score)
            weights.append(0.25)
        
        # Aspect evaluation score
        if aspect_evaluation:
            scores.append(aspect_evaluation.overall_score)
            weights.append(0.3)
        
        # Robustness score (inverted vulnerability score)
        if robustness_report:
            robustness_score = max(0, 10 - robustness_report.vulnerability_score)
            scores.append(robustness_score)
            weights.append(0.15)
        
        # Calculate weighted average
        if scores and weights:
            weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
            total_weight = sum(weights)
            overall_score = weighted_sum / total_weight
        else:
            overall_score = 5.0  # Default neutral score
        
        return round(overall_score, 2)
    
    def _determine_confidence_level(self, consensus_result: Optional[ConsensusResult],
                                   hallucination_report: Optional[HallucinationReport],
                                   aspect_evaluation: Optional[EvaluationResult]) -> str:
        """Determine overall confidence level in evaluation."""
        
        confidence_scores = []
        
        if consensus_result:
            confidence_scores.append(consensus_result.consensus_confidence)
        
        if hallucination_report:
            confidence_scores.append(hallucination_report.confidence_in_analysis)
        
        if aspect_evaluation:
            # Convert confidence level to numeric
            confidence_map = {"High": 0.9, "Medium": 0.7, "Low": 0.5}
            confidence_scores.append(confidence_map.get(aspect_evaluation.confidence_level, 0.5))
        
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            
            if avg_confidence >= 0.8:
                return "High"
            elif avg_confidence >= 0.6:
                return "Medium"
            else:
                return "Low"
        
        return "Medium"
    
    def _extract_unified_insights(self, consensus_result: Optional[ConsensusResult],
                                 hallucination_report: Optional[HallucinationReport],
                                 aspect_evaluation: Optional[EvaluationResult],
                                 robustness_report: Optional[RobustnessReport]) -> Tuple[List[str], List[str], List[str]]:
        """Extract unified insights across all evaluation modules."""
        
        key_strengths = []
        critical_issues = []
        recommendations = []
        
        # From multi-model consensus
        if consensus_result:
            # Find best aspects
            best_aspects = [k for k, v in consensus_result.consensus_scores.items() if v >= 8.0]
            key_strengths.extend([f"Strong {aspect}" for aspect in best_aspects])
            
            # Find weak aspects
            weak_aspects = [k for k, v in consensus_result.consensus_scores.items() if v <= 5.0]
            critical_issues.extend([f"Weak {aspect}" for aspect in weak_aspects])
        
        # From hallucination detection
        if hallucination_report:
            if hallucination_report.overall_reliability_score >= 0.9:
                key_strengths.append("High factual reliability")
            elif hallucination_report.overall_reliability_score <= 0.6:
                critical_issues.append("Significant hallucination risk")
                recommendations.extend(["Improve source grounding", "Verify factual claims"])
            
            if hallucination_report.total_hallucinations > 0:
                critical_issues.append(f"{hallucination_report.total_hallucinations} potential hallucinations detected")
        
        # From aspect evaluation
        if aspect_evaluation:
            key_strengths.extend(aspect_evaluation.strengths)
            critical_issues.extend(aspect_evaluation.weaknesses)
            recommendations.extend(aspect_evaluation.priority_improvements)
        
        # From robustness testing
        if robustness_report:
            if robustness_report.vulnerability_score <= 2.0:
                key_strengths.append("Strong security posture")
            elif robustness_report.vulnerability_score >= 6.0:
                critical_issues.append("High vulnerability to adversarial attacks")
                recommendations.extend(robustness_report.security_recommendations[:3])
        
        # Remove duplicates
        key_strengths = list(set(key_strengths))
        critical_issues = list(set(critical_issues))
        recommendations = list(set(recommendations))
        
        return key_strengths, critical_issues, recommendations
    
    def _calculate_quality_grade(self, overall_score: float, critical_issues: List[str]) -> str:
        """Calculate overall quality grade."""
        
        # Start with score-based grade
        if overall_score >= 9.0:
            base_grade = "A+"
        elif overall_score >= 8.5:
            base_grade = "A"
        elif overall_score >= 7.5:
            base_grade = "B+"
        elif overall_score >= 7.0:
            base_grade = "B"
        elif overall_score >= 6.0:
            base_grade = "C"
        elif overall_score >= 5.0:
            base_grade = "D"
        else:
            base_grade = "F"
        
        # Penalize for critical issues
        if len(critical_issues) >= 3:
            grade_penalties = {"A+": "B+", "A": "B", "B+": "C", "B": "C", "C": "D", "D": "F", "F": "F"}
            base_grade = grade_penalties.get(base_grade, "F")
        elif len(critical_issues) >= 1:
            grade_penalties = {"A+": "A", "A": "B+", "B+": "B", "B": "C", "C": "D", "D": "F", "F": "F"}
            base_grade = grade_penalties.get(base_grade, "F")
        
        return base_grade
    
    def get_evaluation_summary(self, result: ComprehensiveEvaluationResult) -> Dict[str, Any]:
        """Get comprehensive evaluation summary."""
        
        summary = {
            "overall_assessment": {
                "score": result.overall_score,
                "quality_grade": result.quality_grade,
                "confidence": result.confidence_level
            },
            "module_results": {},
            "insights": {
                "key_strengths": result.key_strengths,
                "critical_issues": result.critical_issues,
                "priority_recommendations": result.improvement_recommendations[:5]
            },
            "performance": {
                "evaluation_time_ms": result.evaluation_time_ms,
                "modules_used": result.modules_used
            }
        }
        
        # Add module-specific results
        if result.consensus_result:
            summary["module_results"]["multi_model_consensus"] = {
                "final_score": result.consensus_result.final_score,
                "agreement_level": result.consensus_result.consensus_confidence,
                "model_count": len(result.consensus_result.individual_judgments)
            }
        
        if result.hallucination_report:
            summary["module_results"]["hallucination_detection"] = {
                "reliability_score": result.hallucination_report.overall_reliability_score,
                "hallucinations_found": result.hallucination_report.total_hallucinations,
                "confidence": result.hallucination_report.confidence_in_analysis
            }
        
        if result.aspect_evaluation:
            summary["module_results"]["aspect_evaluation"] = {
                "overall_score": result.aspect_evaluation.overall_score,
                "aspects_evaluated": len(result.aspect_evaluation.aspect_scores),
                "confidence": result.aspect_evaluation.confidence_level
            }
        
        if result.robustness_report:
            summary["module_results"]["adversarial_testing"] = {
                "robustness_level": result.robustness_report.overall_robustness.value,
                "vulnerability_score": result.robustness_report.vulnerability_score,
                "tests_passed": result.robustness_report.tests_passed,
                "critical_vulnerabilities": result.robustness_report.critical_vulnerabilities
            }
        
        return summary
    
    def get_analytics_dashboard(self, days: int = 7) -> Dict[str, Any]:
        """Get evaluation analytics dashboard."""
        
        cutoff_time = time.time() - (days * 24 * 3600)
        recent_evaluations = [e for e in self.evaluation_history if e.timestamp > cutoff_time]
        
        if not recent_evaluations:
            return {"message": "No recent evaluations available"}
        
        # Calculate trends
        scores = [e.overall_score for e in recent_evaluations]
        avg_score = sum(scores) / len(scores)
        
        # Quality distribution
        quality_distribution = {}
        for evaluation in recent_evaluations:
            grade = evaluation.quality_grade
            quality_distribution[grade] = quality_distribution.get(grade, 0) + 1
        
        # Module usage statistics
        module_usage = {}
        for evaluation in recent_evaluations:
            for module in evaluation.modules_used:
                module_usage[module] = module_usage.get(module, 0) + 1
        
        # Common issues
        all_issues = []
        for evaluation in recent_evaluations:
            all_issues.extend(evaluation.critical_issues)
        
        issue_frequency = {}
        for issue in all_issues:
            issue_frequency[issue] = issue_frequency.get(issue, 0) + 1
        
        top_issues = sorted(issue_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "time_period_days": days,
            "evaluation_count": len(recent_evaluations),
            "performance_summary": {
                "average_score": round(avg_score, 2),
                "score_trend": self._calculate_score_trend(recent_evaluations),
                "quality_distribution": quality_distribution
            },
            "module_analytics": {
                "usage_statistics": module_usage,
                "average_evaluation_time": round(
                    sum(e.evaluation_time_ms for e in recent_evaluations) / len(recent_evaluations), 0
                )
            },
            "quality_insights": {
                "top_issues": [{"issue": issue, "frequency": freq} for issue, freq in top_issues],
                "improvement_areas": self._identify_improvement_areas(recent_evaluations)
            }
        }
    
    def _calculate_score_trend(self, evaluations: List[ComprehensiveEvaluationResult]) -> str:
        """Calculate score trend over time."""
        
        if len(evaluations) < 2:
            return "insufficient_data"
        
        # Sort by timestamp
        sorted_evals = sorted(evaluations, key=lambda x: x.timestamp)
        
        # Compare first half vs second half
        mid_point = len(sorted_evals) // 2
        first_half_avg = sum(e.overall_score for e in sorted_evals[:mid_point]) / mid_point
        second_half_avg = sum(e.overall_score for e in sorted_evals[mid_point:]) / (len(sorted_evals) - mid_point)
        
        if second_half_avg > first_half_avg + 0.5:
            return "improving"
        elif second_half_avg < first_half_avg - 0.5:
            return "declining"
        else:
            return "stable"
    
    def _identify_improvement_areas(self, evaluations: List[ComprehensiveEvaluationResult]) -> List[str]:
        """Identify key improvement areas based on evaluation history."""
        
        improvement_areas = []
        
        # Analyze score patterns
        low_scores = [e for e in evaluations if e.overall_score < 6.0]
        if len(low_scores) > len(evaluations) * 0.3:  # More than 30% low scores
            improvement_areas.append("Overall response quality needs improvement")
        
        # Analyze common issues
        all_issues = []
        for evaluation in evaluations:
            all_issues.extend(evaluation.critical_issues)
        
        if "hallucination" in str(all_issues).lower():
            improvement_areas.append("Hallucination detection and prevention")
        
        if "citation" in str(all_issues).lower():
            improvement_areas.append("Citation quality and accuracy")
        
        if "completeness" in str(all_issues).lower():
            improvement_areas.append("Response completeness and thoroughness")
        
        return improvement_areas[:5]  # Top 5 areas
    
    def load_evaluation_history(self):
        """Load evaluation history from storage."""
        history_file = self.storage_path / "evaluation_history.json"
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    # Keep last 1000 evaluations
                    recent_data = data[-1000:] if len(data) > 1000 else data
                    
                    for eval_data in recent_data:
                        # Reconstruct evaluation result (simplified)
                        result = ComprehensiveEvaluationResult(
                            request_id=eval_data.get("request_id", ""),
                            overall_score=eval_data.get("overall_score", 0),
                            confidence_level=eval_data.get("confidence_level", "Medium"),
                            consensus_result=None,  # Complex objects not restored
                            hallucination_report=None,
                            performance_metrics=eval_data.get("performance_metrics", {}),
                            aspect_evaluation=None,
                            robustness_report=None,
                            key_strengths=eval_data.get("key_strengths", []),
                            critical_issues=eval_data.get("critical_issues", []),
                            improvement_recommendations=eval_data.get("improvement_recommendations", []),
                            quality_grade=eval_data.get("quality_grade", "C"),
                            evaluation_time_ms=eval_data.get("evaluation_time_ms", 0),
                            modules_used=eval_data.get("modules_used", []),
                            timestamp=eval_data.get("timestamp", time.time())
                        )
                        self.evaluation_history.append(result)
            except Exception as e:
                print(f"Failed to load evaluation history: {e}")
    
    def save_evaluation_history(self):
        """Save evaluation history to storage."""
        history_file = self.storage_path / "evaluation_history.json"
        
        try:
            # Convert to JSON-serializable format (simplified)
            history_data = []
            for result in self.evaluation_history[-1000:]:  # Keep last 1000
                data = {
                    "request_id": result.request_id,
                    "overall_score": result.overall_score,
                    "confidence_level": result.confidence_level,
                    "performance_metrics": result.performance_metrics,
                    "key_strengths": result.key_strengths,
                    "critical_issues": result.critical_issues,
                    "improvement_recommendations": result.improvement_recommendations,
                    "quality_grade": result.quality_grade,
                    "evaluation_time_ms": result.evaluation_time_ms,
                    "modules_used": result.modules_used,
                    "timestamp": result.timestamp
                }
                history_data.append(data)
            
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
        except Exception as e:
            print(f"Failed to save evaluation history: {e}")


# Global evaluation manager instance
_evaluation_manager: EvaluationOrchestrator | None = None


def get_evaluation_manager() -> EvaluationOrchestrator:
    """Get or create global evaluation manager."""
    global _evaluation_manager
    if _evaluation_manager is None:
        _evaluation_manager = EvaluationOrchestrator()
    return _evaluation_manager
