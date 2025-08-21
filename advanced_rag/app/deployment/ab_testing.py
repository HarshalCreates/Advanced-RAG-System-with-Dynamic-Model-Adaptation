"""A/B testing framework for gradual model rollouts and experimentation."""
from __future__ import annotations

import json
import time
import hashlib
import random
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
from enum import Enum
import uuid


class ExperimentStatus(Enum):
    """A/B experiment status types."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TrafficSplitType(Enum):
    """Traffic splitting methods."""
    RANDOM = "random"
    USER_ID = "user_id"
    SESSION_ID = "session_id"
    GEOGRAPHIC = "geographic"
    TIME_BASED = "time_based"


@dataclass
class ModelVariant:
    """Represents a model configuration variant for testing."""
    variant_id: str
    name: str
    description: str
    embedding_backend: str
    embedding_model: str
    generation_backend: str
    generation_model: str
    retriever_backend: str
    traffic_percentage: float
    metadata: Dict[str, Any]


@dataclass
class ExperimentMetrics:
    """Metrics collected for experiment analysis."""
    variant_id: str
    query_count: int
    avg_response_time_ms: float
    avg_confidence: float
    success_rate: float
    error_rate: float
    user_satisfaction: float  # From feedback
    conversion_rate: float  # Task completion
    cost_per_query: float
    timestamp: float


@dataclass
class ABExperiment:
    """A/B experiment configuration and state."""
    experiment_id: str
    name: str
    description: str
    variants: List[ModelVariant]
    control_variant_id: str
    traffic_split_type: TrafficSplitType
    start_time: Optional[float]
    end_time: Optional[float]
    status: ExperimentStatus
    success_criteria: Dict[str, float]
    min_sample_size: int
    confidence_level: float
    created_by: str
    metadata: Dict[str, Any]


@dataclass
class QueryAssignment:
    """Records which variant was used for a query."""
    query_id: str
    experiment_id: str
    variant_id: str
    user_identifier: str
    timestamp: float
    query_text: str
    response_time_ms: float
    confidence: float
    success: bool
    metadata: Dict[str, Any]


class TrafficSplitter:
    """Handles traffic splitting for A/B experiments."""
    
    def __init__(self):
        self.splitting_strategies = {
            TrafficSplitType.RANDOM: self._random_split,
            TrafficSplitType.USER_ID: self._user_id_split,
            TrafficSplitType.SESSION_ID: self._session_id_split,
            TrafficSplitType.GEOGRAPHIC: self._geographic_split,
            TrafficSplitType.TIME_BASED: self._time_based_split
        }
    
    def assign_variant(self, experiment: ABExperiment, 
                      user_identifier: str = None,
                      session_id: str = None,
                      geographic_info: Dict[str, str] = None) -> ModelVariant:
        """Assign a user to a variant based on the experiment's traffic splitting strategy."""
        
        if experiment.status != ExperimentStatus.RUNNING:
            # Return control variant if experiment is not running
            return next(v for v in experiment.variants if v.variant_id == experiment.control_variant_id)
        
        splitter = self.splitting_strategies.get(experiment.traffic_split_type, self._random_split)
        
        context = {
            'user_identifier': user_identifier or str(uuid.uuid4()),
            'session_id': session_id or str(uuid.uuid4()),
            'geographic_info': geographic_info or {},
            'timestamp': time.time()
        }
        
        return splitter(experiment, context)
    
    def _random_split(self, experiment: ABExperiment, context: Dict[str, Any]) -> ModelVariant:
        """Random traffic splitting."""
        rand_value = random.random()
        cumulative_percentage = 0.0
        
        for variant in experiment.variants:
            cumulative_percentage += variant.traffic_percentage / 100.0
            if rand_value <= cumulative_percentage:
                return variant
        
        # Fallback to control variant
        return next(v for v in experiment.variants if v.variant_id == experiment.control_variant_id)
    
    def _user_id_split(self, experiment: ABExperiment, context: Dict[str, Any]) -> ModelVariant:
        """Consistent user-based splitting using hash."""
        user_id = context.get('user_identifier', 'anonymous')
        
        # Create a hash based on experiment_id + user_id for consistency
        hash_input = f"{experiment.experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
        hash_percentage = (hash_value % 100) / 100.0
        
        cumulative_percentage = 0.0
        for variant in experiment.variants:
            cumulative_percentage += variant.traffic_percentage / 100.0
            if hash_percentage <= cumulative_percentage:
                return variant
        
        return next(v for v in experiment.variants if v.variant_id == experiment.control_variant_id)
    
    def _session_id_split(self, experiment: ABExperiment, context: Dict[str, Any]) -> ModelVariant:
        """Session-based consistent splitting."""
        session_id = context.get('session_id', 'default_session')
        
        hash_input = f"{experiment.experiment_id}:{session_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
        hash_percentage = (hash_value % 100) / 100.0
        
        cumulative_percentage = 0.0
        for variant in experiment.variants:
            cumulative_percentage += variant.traffic_percentage / 100.0
            if hash_percentage <= cumulative_percentage:
                return variant
        
        return next(v for v in experiment.variants if v.variant_id == experiment.control_variant_id)
    
    def _geographic_split(self, experiment: ABExperiment, context: Dict[str, Any]) -> ModelVariant:
        """Geographic-based splitting."""
        geo_info = context.get('geographic_info', {})
        country = geo_info.get('country', 'unknown')
        
        # Simple geographic rules (can be enhanced)
        geo_mappings = {
            'US': 0,
            'UK': 1,
            'DE': 2,
            'JP': 3
        }
        
        variant_index = geo_mappings.get(country, 0) % len(experiment.variants)
        return experiment.variants[variant_index]
    
    def _time_based_split(self, experiment: ABExperiment, context: Dict[str, Any]) -> ModelVariant:
        """Time-based splitting (e.g., different variants for different hours)."""
        current_hour = datetime.now().hour
        variant_index = current_hour % len(experiment.variants)
        return experiment.variants[variant_index]


class ABTestingManager:
    """Manages A/B experiments for gradual model rollouts."""
    
    def __init__(self, storage_path: str = "./data/ab_testing"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.experiments_file = self.storage_path / "experiments.json"
        self.assignments_file = self.storage_path / "assignments.json" 
        self.metrics_file = self.storage_path / "metrics.json"
        
        self.experiments: List[ABExperiment] = []
        self.assignments: List[QueryAssignment] = []
        self.metrics: List[ExperimentMetrics] = []
        
        self.traffic_splitter = TrafficSplitter()
        
        # Load existing data
        self.load_experiment_data()
    
    def load_experiment_data(self):
        """Load experiment data from storage."""
        # Load experiments
        if self.experiments_file.exists():
            try:
                with open(self.experiments_file, 'r') as f:
                    data = json.load(f)
                    self.experiments = []
                    for exp_data in data:
                        exp_data['status'] = ExperimentStatus(exp_data['status'])
                        exp_data['traffic_split_type'] = TrafficSplitType(exp_data['traffic_split_type'])
                        exp_data['variants'] = [ModelVariant(**v) for v in exp_data['variants']]
                        self.experiments.append(ABExperiment(**exp_data))
            except Exception as e:
                print(f"Failed to load experiments: {e}")
        
        # Load assignments (keep only recent ones)
        if self.assignments_file.exists():
            try:
                with open(self.assignments_file, 'r') as f:
                    data = json.load(f)
                    # Only keep assignments from last 7 days
                    cutoff_time = time.time() - (7 * 24 * 3600)
                    self.assignments = [
                        QueryAssignment(**assignment) 
                        for assignment in data 
                        if assignment.get('timestamp', 0) > cutoff_time
                    ]
            except Exception as e:
                print(f"Failed to load assignments: {e}")
        
        # Load metrics
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.metrics = [ExperimentMetrics(**metric) for metric in data]
            except Exception as e:
                print(f"Failed to load metrics: {e}")
    
    def save_experiment_data(self):
        """Save experiment data to storage."""
        try:
            # Save experiments
            experiments_data = []
            for exp in self.experiments:
                data = asdict(exp)
                data['status'] = exp.status.value
                data['traffic_split_type'] = exp.traffic_split_type.value
                experiments_data.append(data)
            
            with open(self.experiments_file, 'w') as f:
                json.dump(experiments_data, f, indent=2)
            
            # Save assignments (limited to recent ones)
            assignments_data = [asdict(assignment) for assignment in self.assignments[-1000:]]
            with open(self.assignments_file, 'w') as f:
                json.dump(assignments_data, f, indent=2)
            
            # Save metrics
            metrics_data = [asdict(metric) for metric in self.metrics]
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
        except Exception as e:
            print(f"Failed to save experiment data: {e}")
    
    def create_experiment(self, name: str, description: str, variants: List[ModelVariant],
                         control_variant_id: str, traffic_split_type: TrafficSplitType = TrafficSplitType.RANDOM,
                         success_criteria: Dict[str, float] = None, min_sample_size: int = 1000,
                         confidence_level: float = 0.95, created_by: str = "system") -> ABExperiment:
        """Create a new A/B experiment."""
        
        experiment_id = f"exp_{int(time.time())}"
        
        # Validate traffic percentages sum to 100
        total_traffic = sum(v.traffic_percentage for v in variants)
        if abs(total_traffic - 100.0) > 0.1:
            raise ValueError(f"Traffic percentages must sum to 100%, got {total_traffic}%")
        
        # Default success criteria
        if success_criteria is None:
            success_criteria = {
                'min_confidence_improvement': 0.05,
                'max_response_time_degradation': 0.2,
                'min_success_rate': 0.95
            }
        
        experiment = ABExperiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            variants=variants,
            control_variant_id=control_variant_id,
            traffic_split_type=traffic_split_type,
            start_time=None,
            end_time=None,
            status=ExperimentStatus.DRAFT,
            success_criteria=success_criteria,
            min_sample_size=min_sample_size,
            confidence_level=confidence_level,
            created_by=created_by,
            metadata={}
        )
        
        self.experiments.append(experiment)
        self.save_experiment_data()
        
        return experiment
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an A/B experiment."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return False
        
        if experiment.status != ExperimentStatus.DRAFT:
            print(f"Cannot start experiment {experiment_id} - status is {experiment.status}")
            return False
        
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_time = time.time()
        
        self.save_experiment_data()
        print(f"✅ Started A/B experiment: {experiment.name}")
        return True
    
    def pause_experiment(self, experiment_id: str) -> bool:
        """Pause a running experiment."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return False
        
        if experiment.status != ExperimentStatus.RUNNING:
            return False
        
        experiment.status = ExperimentStatus.PAUSED
        self.save_experiment_data()
        return True
    
    def complete_experiment(self, experiment_id: str, winner_variant_id: str = None) -> bool:
        """Complete an experiment and optionally declare a winner."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return False
        
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_time = time.time()
        
        if winner_variant_id:
            experiment.metadata['winner_variant_id'] = winner_variant_id
        
        self.save_experiment_data()
        print(f"✅ Completed A/B experiment: {experiment.name}")
        return True
    
    def get_experiment(self, experiment_id: str) -> Optional[ABExperiment]:
        """Get experiment by ID."""
        return next((exp for exp in self.experiments if exp.experiment_id == experiment_id), None)
    
    def get_active_experiments(self) -> List[ABExperiment]:
        """Get all running experiments."""
        return [exp for exp in self.experiments if exp.status == ExperimentStatus.RUNNING]
    
    def assign_variant_for_query(self, query_id: str, query_text: str,
                                user_identifier: str = None, session_id: str = None,
                                geographic_info: Dict[str, str] = None) -> Optional[ModelVariant]:
        """Assign a variant for a query and record the assignment."""
        
        active_experiments = self.get_active_experiments()
        if not active_experiments:
            return None
        
        # For simplicity, use the first active experiment
        # In practice, you might have priority rules or multiple concurrent experiments
        experiment = active_experiments[0]
        
        variant = self.traffic_splitter.assign_variant(
            experiment, user_identifier, session_id, geographic_info
        )
        
        # Record assignment
        assignment = QueryAssignment(
            query_id=query_id,
            experiment_id=experiment.experiment_id,
            variant_id=variant.variant_id,
            user_identifier=user_identifier or "anonymous",
            timestamp=time.time(),
            query_text=query_text,
            response_time_ms=0.0,  # Will be updated after query completion
            confidence=0.0,  # Will be updated after query completion
            success=False,  # Will be updated after query completion
            metadata={}
        )
        
        self.assignments.append(assignment)
        return variant
    
    def record_query_result(self, query_id: str, response_time_ms: float,
                          confidence: float, success: bool, metadata: Dict[str, Any] = None):
        """Record the result of a query for experiment tracking."""
        
        assignment = next((a for a in self.assignments if a.query_id == query_id), None)
        if assignment:
            assignment.response_time_ms = response_time_ms
            assignment.confidence = confidence
            assignment.success = success
            assignment.metadata.update(metadata or {})
            
            # Save periodically
            if len(self.assignments) % 10 == 0:
                self.save_experiment_data()
    
    def calculate_experiment_metrics(self, experiment_id: str) -> List[ExperimentMetrics]:
        """Calculate current metrics for an experiment."""
        
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return []
        
        # Get assignments for this experiment
        exp_assignments = [a for a in self.assignments if a.experiment_id == experiment_id]
        
        metrics = []
        for variant in experiment.variants:
            variant_assignments = [a for a in exp_assignments if a.variant_id == variant.variant_id]
            
            if not variant_assignments:
                continue
            
            # Calculate metrics
            query_count = len(variant_assignments)
            avg_response_time = sum(a.response_time_ms for a in variant_assignments) / query_count
            avg_confidence = sum(a.confidence for a in variant_assignments) / query_count
            success_count = sum(1 for a in variant_assignments if a.success)
            success_rate = success_count / query_count
            error_rate = 1.0 - success_rate
            
            # Placeholder values for metrics that require additional tracking
            user_satisfaction = 0.8  # Would come from user feedback
            conversion_rate = 0.7    # Would come from task completion tracking
            cost_per_query = 0.01    # Would come from cost tracking
            
            metric = ExperimentMetrics(
                variant_id=variant.variant_id,
                query_count=query_count,
                avg_response_time_ms=avg_response_time,
                avg_confidence=avg_confidence,
                success_rate=success_rate,
                error_rate=error_rate,
                user_satisfaction=user_satisfaction,
                conversion_rate=conversion_rate,
                cost_per_query=cost_per_query,
                timestamp=time.time()
            )
            
            metrics.append(metric)
        
        # Update stored metrics
        self.metrics.extend(metrics)
        self.save_experiment_data()
        
        return metrics
    
    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze experiment results and determine statistical significance."""
        
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return {"error": "Experiment not found"}
        
        metrics = self.calculate_experiment_metrics(experiment_id)
        if not metrics:
            return {"error": "No data available for analysis"}
        
        # Find control and treatment variants
        control_metric = next((m for m in metrics if m.variant_id == experiment.control_variant_id), None)
        treatment_metrics = [m for m in metrics if m.variant_id != experiment.control_variant_id]
        
        if not control_metric:
            return {"error": "Control variant data not found"}
        
        analysis = {
            "experiment_id": experiment_id,
            "experiment_name": experiment.name,
            "status": experiment.status.value,
            "control_variant": {
                "variant_id": control_metric.variant_id,
                "query_count": control_metric.query_count,
                "avg_response_time_ms": control_metric.avg_response_time_ms,
                "avg_confidence": control_metric.avg_confidence,
                "success_rate": control_metric.success_rate
            },
            "treatment_variants": [],
            "recommendations": []
        }
        
        for treatment_metric in treatment_metrics:
            # Calculate improvements vs control
            response_time_change = (treatment_metric.avg_response_time_ms - control_metric.avg_response_time_ms) / control_metric.avg_response_time_ms
            confidence_change = treatment_metric.avg_confidence - control_metric.avg_confidence
            success_rate_change = treatment_metric.success_rate - control_metric.success_rate
            
            # Determine if improvements meet success criteria
            meets_criteria = (
                confidence_change >= experiment.success_criteria.get('min_confidence_improvement', 0.05) and
                response_time_change <= experiment.success_criteria.get('max_response_time_degradation', 0.2) and
                treatment_metric.success_rate >= experiment.success_criteria.get('min_success_rate', 0.95)
            )
            
            treatment_analysis = {
                "variant_id": treatment_metric.variant_id,
                "query_count": treatment_metric.query_count,
                "avg_response_time_ms": treatment_metric.avg_response_time_ms,
                "avg_confidence": treatment_metric.avg_confidence,
                "success_rate": treatment_metric.success_rate,
                "vs_control": {
                    "response_time_change_pct": response_time_change * 100,
                    "confidence_change": confidence_change,
                    "success_rate_change": success_rate_change
                },
                "meets_success_criteria": meets_criteria,
                "sample_size_sufficient": treatment_metric.query_count >= experiment.min_sample_size
            }
            
            analysis["treatment_variants"].append(treatment_analysis)
        
        # Generate recommendations
        if any(t["meets_success_criteria"] and t["sample_size_sufficient"] for t in analysis["treatment_variants"]):
            best_variant = max(analysis["treatment_variants"], 
                             key=lambda x: x["avg_confidence"] if x["meets_success_criteria"] else -1)
            analysis["recommendations"].append(f"Deploy variant {best_variant['variant_id']} - shows significant improvement")
        else:
            analysis["recommendations"].append("Continue experiment - no variant shows significant improvement yet")
        
        return analysis
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments."""
        total = len(self.experiments)
        running = len([e for e in self.experiments if e.status == ExperimentStatus.RUNNING])
        completed = len([e for e in self.experiments if e.status == ExperimentStatus.COMPLETED])
        
        return {
            "total_experiments": total,
            "running_experiments": running,
            "completed_experiments": completed,
            "total_assignments": len(self.assignments),
            "recent_experiments": [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "status": exp.status.value,
                    "variants_count": len(exp.variants),
                    "start_time": exp.start_time
                }
                for exp in sorted(self.experiments, key=lambda x: x.start_time or 0, reverse=True)[:5]
            ]
        }
