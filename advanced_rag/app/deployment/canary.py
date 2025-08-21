"""Canary deployment support for gradual rollouts."""
from __future__ import annotations

import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
from enum import Enum
import uuid


class CanaryStatus(Enum):
    """Canary deployment status types."""
    PREPARING = "preparing"
    RUNNING = "running"
    PROMOTING = "promoting"
    COMPLETED = "completed"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"


class CanaryStrategy(Enum):
    """Canary deployment strategies."""
    TRAFFIC_BASED = "traffic_based"  # Gradually increase traffic percentage
    USER_BASED = "user_based"       # Deploy to specific user groups first
    GEOGRAPHIC = "geographic"        # Deploy to specific regions first
    TIME_BASED = "time_based"        # Deploy during specific time windows


@dataclass
class CanaryConfig:
    """Configuration for canary deployment."""
    deployment_id: str
    name: str
    description: str
    strategy: CanaryStrategy
    
    # Model configurations
    stable_config: Dict[str, str]  # Current stable model config
    canary_config: Dict[str, str]  # New canary model config
    
    # Traffic management
    initial_traffic_percentage: float = 5.0
    final_traffic_percentage: float = 100.0
    traffic_increment: float = 5.0
    increment_interval_minutes: int = 10
    
    # Success criteria
    success_criteria: Dict[str, float] = None
    max_duration_minutes: int = 120
    
    # Rollback conditions
    rollback_thresholds: Dict[str, float] = None
    
    # Target groups (for user/geographic strategies)
    target_groups: List[str] = None
    
    metadata: Dict[str, Any] = None


@dataclass
class CanaryMetrics:
    """Metrics for canary deployment analysis."""
    deployment_id: str
    timestamp: float
    traffic_percentage: float
    
    # Performance metrics
    stable_response_time_ms: float
    canary_response_time_ms: float
    stable_success_rate: float
    canary_success_rate: float
    stable_confidence_avg: float
    canary_confidence_avg: float
    
    # Business metrics
    stable_user_satisfaction: float
    canary_user_satisfaction: float
    stable_conversion_rate: float
    canary_conversion_rate: float
    
    # Resource metrics
    stable_cost_per_query: float
    canary_cost_per_query: float
    
    # Comparison metrics
    response_time_improvement: float
    success_rate_improvement: float
    confidence_improvement: float


@dataclass
class CanaryDeployment:
    """Represents an active canary deployment."""
    deployment_id: str
    config: CanaryConfig
    status: CanaryStatus
    current_traffic_percentage: float
    
    start_time: float
    last_increment_time: float
    completion_time: Optional[float]
    
    metrics_history: List[CanaryMetrics]
    decision_log: List[Dict[str, Any]]
    
    rollback_reason: Optional[str]
    metadata: Dict[str, Any]


class TrafficManager:
    """Manages traffic routing for canary deployments."""
    
    def __init__(self):
        self.routing_rules: Dict[str, Dict[str, Any]] = {}
        self.user_assignments: Dict[str, str] = {}  # user_id -> variant
    
    def set_canary_traffic(self, deployment_id: str, traffic_percentage: float, 
                          strategy: CanaryStrategy = CanaryStrategy.TRAFFIC_BASED):
        """Set traffic percentage for canary deployment."""
        self.routing_rules[deployment_id] = {
            'traffic_percentage': traffic_percentage,
            'strategy': strategy,
            'updated_at': time.time()
        }
    
    def should_use_canary(self, deployment_id: str, user_id: str = None,
                         user_groups: List[str] = None, geographic_region: str = None) -> bool:
        """Determine if a request should use the canary version."""
        
        if deployment_id not in self.routing_rules:
            return False
        
        rule = self.routing_rules[deployment_id]
        strategy = CanaryStrategy(rule.get('strategy', CanaryStrategy.TRAFFIC_BASED))
        traffic_percentage = rule['traffic_percentage']
        
        if strategy == CanaryStrategy.TRAFFIC_BASED:
            # Simple random routing based on percentage
            import random
            return random.random() * 100 < traffic_percentage
        
        elif strategy == CanaryStrategy.USER_BASED:
            # Consistent user-based routing
            if user_id:
                # Check if user is already assigned
                if user_id in self.user_assignments:
                    return self.user_assignments[user_id] == 'canary'
                
                # Assign user based on hash and percentage
                import hashlib
                user_hash = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16)
                use_canary = (user_hash % 100) < traffic_percentage
                
                self.user_assignments[user_id] = 'canary' if use_canary else 'stable'
                return use_canary
            
            return False
        
        elif strategy == CanaryStrategy.GEOGRAPHIC:
            # Geographic-based routing
            target_regions = rule.get('target_regions', [])
            if geographic_region and geographic_region in target_regions:
                return True
            return False
        
        elif strategy == CanaryStrategy.TIME_BASED:
            # Time-based routing (e.g., only during business hours)
            current_hour = datetime.now().hour
            allowed_hours = rule.get('allowed_hours', list(range(24)))
            if current_hour in allowed_hours:
                import random
                return random.random() * 100 < traffic_percentage
            return False
        
        return False
    
    def get_assignment_stats(self, deployment_id: str) -> Dict[str, Any]:
        """Get statistics about traffic assignment."""
        if deployment_id not in self.routing_rules:
            return {}
        
        rule = self.routing_rules[deployment_id]
        
        # Count user assignments
        canary_users = sum(1 for assignment in self.user_assignments.values() if assignment == 'canary')
        stable_users = sum(1 for assignment in self.user_assignments.values() if assignment == 'stable')
        total_users = canary_users + stable_users
        
        actual_percentage = (canary_users / total_users * 100) if total_users > 0 else 0
        
        return {
            'target_percentage': rule['traffic_percentage'],
            'actual_percentage': actual_percentage,
            'canary_users': canary_users,
            'stable_users': stable_users,
            'total_users': total_users,
            'strategy': rule.get('strategy', CanaryStrategy.TRAFFIC_BASED.value)
        }


class CanaryAnalyzer:
    """Analyzes canary deployment performance and makes decisions."""
    
    def __init__(self):
        self.analysis_history: List[Dict[str, Any]] = []
    
    async def collect_canary_metrics(self, deployment: CanaryDeployment) -> CanaryMetrics:
        """Collect current metrics for canary analysis."""
        
        # In a real implementation, this would collect metrics from both stable and canary versions
        # For now, we'll simulate metrics collection
        current_time = time.time()
        
        # Simulate metrics - in practice, these would come from monitoring systems
        stable_metrics = {
            'response_time_ms': 150.0,
            'success_rate': 0.98,
            'confidence_avg': 0.75,
            'user_satisfaction': 0.85,
            'conversion_rate': 0.65,
            'cost_per_query': 0.02
        }
        
        # Canary metrics with some variation
        canary_metrics = {
            'response_time_ms': stable_metrics['response_time_ms'] * 0.95,  # 5% improvement
            'success_rate': stable_metrics['success_rate'] + 0.01,          # Slight improvement
            'confidence_avg': stable_metrics['confidence_avg'] + 0.05,      # Confidence boost
            'user_satisfaction': stable_metrics['user_satisfaction'] + 0.02,
            'conversion_rate': stable_metrics['conversion_rate'] + 0.03,
            'cost_per_query': stable_metrics['cost_per_query'] * 1.1        # 10% cost increase
        }
        
        # Calculate improvements
        response_time_improvement = (stable_metrics['response_time_ms'] - canary_metrics['response_time_ms']) / stable_metrics['response_time_ms']
        success_rate_improvement = canary_metrics['success_rate'] - stable_metrics['success_rate']
        confidence_improvement = canary_metrics['confidence_avg'] - stable_metrics['confidence_avg']
        
        metrics = CanaryMetrics(
            deployment_id=deployment.deployment_id,
            timestamp=current_time,
            traffic_percentage=deployment.current_traffic_percentage,
            
            stable_response_time_ms=stable_metrics['response_time_ms'],
            canary_response_time_ms=canary_metrics['response_time_ms'],
            stable_success_rate=stable_metrics['success_rate'],
            canary_success_rate=canary_metrics['success_rate'],
            stable_confidence_avg=stable_metrics['confidence_avg'],
            canary_confidence_avg=canary_metrics['confidence_avg'],
            
            stable_user_satisfaction=stable_metrics['user_satisfaction'],
            canary_user_satisfaction=canary_metrics['user_satisfaction'],
            stable_conversion_rate=stable_metrics['conversion_rate'],
            canary_conversion_rate=canary_metrics['conversion_rate'],
            
            stable_cost_per_query=stable_metrics['cost_per_query'],
            canary_cost_per_query=canary_metrics['cost_per_query'],
            
            response_time_improvement=response_time_improvement,
            success_rate_improvement=success_rate_improvement,
            confidence_improvement=confidence_improvement
        )
        
        return metrics
    
    def analyze_deployment_health(self, deployment: CanaryDeployment, 
                                current_metrics: CanaryMetrics) -> Dict[str, Any]:
        """Analyze the health of a canary deployment."""
        
        config = deployment.config
        criteria = config.success_criteria or {
            'min_success_rate': 0.95,
            'max_response_time_degradation': 0.2,
            'min_confidence_improvement': 0.0
        }
        
        rollback_thresholds = config.rollback_thresholds or {
            'max_error_rate_increase': 0.05,
            'max_response_time_increase': 0.5,
            'min_success_rate': 0.90
        }
        
        analysis = {
            'healthy': True,
            'should_rollback': False,
            'should_promote': False,
            'can_increase_traffic': True,
            'issues': [],
            'recommendations': [],
            'metrics_summary': {
                'response_time_improvement': current_metrics.response_time_improvement,
                'success_rate_improvement': current_metrics.success_rate_improvement,
                'confidence_improvement': current_metrics.confidence_improvement
            }
        }
        
        # Check rollback conditions
        if current_metrics.canary_success_rate < rollback_thresholds['min_success_rate']:
            analysis['should_rollback'] = True
            analysis['healthy'] = False
            analysis['issues'].append(f"Canary success rate too low: {current_metrics.canary_success_rate:.3f}")
        
        error_rate_increase = (1.0 - current_metrics.canary_success_rate) - (1.0 - current_metrics.stable_success_rate)
        if error_rate_increase > rollback_thresholds['max_error_rate_increase']:
            analysis['should_rollback'] = True
            analysis['healthy'] = False
            analysis['issues'].append(f"Error rate increased by {error_rate_increase:.3f}")
        
        response_time_increase = (current_metrics.canary_response_time_ms - current_metrics.stable_response_time_ms) / current_metrics.stable_response_time_ms
        if response_time_increase > rollback_thresholds['max_response_time_increase']:
            analysis['should_rollback'] = True
            analysis['healthy'] = False
            analysis['issues'].append(f"Response time increased by {response_time_increase:.1%}")
        
        # Check promotion conditions (if at high traffic and meeting criteria)
        if (deployment.current_traffic_percentage >= 90.0 and 
            current_metrics.canary_success_rate >= criteria['min_success_rate'] and
            current_metrics.response_time_improvement >= -criteria['max_response_time_degradation'] and
            current_metrics.confidence_improvement >= criteria['min_confidence_improvement']):
            
            analysis['should_promote'] = True
            analysis['recommendations'].append("Canary is performing well - ready for full promotion")
        
        # Check if we can safely increase traffic
        if (current_metrics.canary_success_rate >= criteria['min_success_rate'] and
            response_time_increase <= criteria['max_response_time_degradation']):
            analysis['can_increase_traffic'] = True
        else:
            analysis['can_increase_traffic'] = False
            analysis['recommendations'].append("Hold current traffic level - metrics need stabilization")
        
        # Performance insights
        if current_metrics.response_time_improvement > 0.1:
            analysis['recommendations'].append(f"Excellent response time improvement: {current_metrics.response_time_improvement:.1%}")
        
        if current_metrics.confidence_improvement > 0.05:
            analysis['recommendations'].append(f"Significant confidence improvement: {current_metrics.confidence_improvement:.2f}")
        
        # Record analysis
        self.analysis_history.append({
            'deployment_id': deployment.deployment_id,
            'timestamp': time.time(),
            'analysis': analysis,
            'metrics': asdict(current_metrics)
        })
        
        return analysis


class CanaryManager:
    """Manages canary deployments."""
    
    def __init__(self, storage_path: str = "./data/canary"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.deployments_file = self.storage_path / "canary_deployments.json"
        
        self.active_deployments: Dict[str, CanaryDeployment] = {}
        self.completed_deployments: List[CanaryDeployment] = []
        
        self.traffic_manager = TrafficManager()
        self.analyzer = CanaryAnalyzer()
        
        # Load existing deployments
        self.load_deployments()
    
    def load_deployments(self):
        """Load canary deployments from storage."""
        if self.deployments_file.exists():
            try:
                with open(self.deployments_file, 'r') as f:
                    data = json.load(f)
                    
                    for deployment_data in data.get('active', []):
                        deployment_data['status'] = CanaryStatus(deployment_data['status'])
                        deployment_data['config']['strategy'] = CanaryStrategy(deployment_data['config']['strategy'])
                        
                        config_data = deployment_data['config']
                        config_data['strategy'] = CanaryStrategy(config_data['strategy'])
                        deployment_data['config'] = CanaryConfig(**config_data)
                        
                        deployment_data['metrics_history'] = [
                            CanaryMetrics(**m) for m in deployment_data['metrics_history']
                        ]
                        
                        deployment = CanaryDeployment(**deployment_data)
                        self.active_deployments[deployment.deployment_id] = deployment
                    
                    for deployment_data in data.get('completed', []):
                        deployment_data['status'] = CanaryStatus(deployment_data['status'])
                        deployment_data['config']['strategy'] = CanaryStrategy(deployment_data['config']['strategy'])
                        
                        config_data = deployment_data['config']
                        config_data['strategy'] = CanaryStrategy(config_data['strategy'])
                        deployment_data['config'] = CanaryConfig(**config_data)
                        
                        deployment_data['metrics_history'] = [
                            CanaryMetrics(**m) for m in deployment_data['metrics_history']
                        ]
                        
                        deployment = CanaryDeployment(**deployment_data)
                        self.completed_deployments.append(deployment)
                        
            except Exception as e:
                print(f"Failed to load canary deployments: {e}")
    
    def save_deployments(self):
        """Save canary deployments to storage."""
        try:
            data = {
                'active': [],
                'completed': []
            }
            
            for deployment in self.active_deployments.values():
                deployment_data = asdict(deployment)
                deployment_data['status'] = deployment.status.value
                deployment_data['config']['strategy'] = deployment.config.strategy.value
                data['active'].append(deployment_data)
            
            for deployment in self.completed_deployments:
                deployment_data = asdict(deployment)
                deployment_data['status'] = deployment.status.value
                deployment_data['config']['strategy'] = deployment.config.strategy.value
                data['completed'].append(deployment_data)
            
            with open(self.deployments_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Failed to save canary deployments: {e}")
    
    def create_canary_deployment(self, config: CanaryConfig) -> CanaryDeployment:
        """Create a new canary deployment."""
        
        if config.deployment_id in self.active_deployments:
            raise ValueError(f"Canary deployment {config.deployment_id} already exists")
        
        deployment = CanaryDeployment(
            deployment_id=config.deployment_id,
            config=config,
            status=CanaryStatus.PREPARING,
            current_traffic_percentage=0.0,
            start_time=time.time(),
            last_increment_time=0.0,
            completion_time=None,
            metrics_history=[],
            decision_log=[],
            rollback_reason=None,
            metadata={}
        )
        
        self.active_deployments[config.deployment_id] = deployment
        self.save_deployments()
        
        return deployment
    
    async def start_canary_deployment(self, deployment_id: str) -> bool:
        """Start a canary deployment."""
        
        deployment = self.active_deployments.get(deployment_id)
        if not deployment or deployment.status != CanaryStatus.PREPARING:
            return False
        
        # Apply canary configuration to the system
        try:
            from app.pipeline.manager import PipelineManager
            pipeline_manager = PipelineManager.get_instance()
            
            # For now, we'll just set the configuration
            # In a real implementation, you'd set up traffic routing
            canary_config = deployment.config.canary_config
            
            # The actual model switching would be handled by traffic routing
            # This is just to demonstrate the integration
            print(f"🐣 Starting canary deployment: {deployment.config.name}")
            print(f"   Canary config: {canary_config}")
            
            deployment.status = CanaryStatus.RUNNING
            deployment.current_traffic_percentage = deployment.config.initial_traffic_percentage
            deployment.last_increment_time = time.time()
            
            # Set initial traffic routing
            self.traffic_manager.set_canary_traffic(
                deployment_id,
                deployment.config.initial_traffic_percentage,
                deployment.config.strategy
            )
            
            # Log decision
            deployment.decision_log.append({
                'timestamp': time.time(),
                'action': 'start_deployment',
                'traffic_percentage': deployment.current_traffic_percentage,
                'reason': 'Initial canary start'
            })
            
            self.save_deployments()
            return True
            
        except Exception as e:
            print(f"Failed to start canary deployment: {e}")
            deployment.status = CanaryStatus.FAILED
            self.save_deployments()
            return False
    
    async def manage_canary_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Manage an active canary deployment - check metrics and make decisions."""
        
        deployment = self.active_deployments.get(deployment_id)
        if not deployment or deployment.status != CanaryStatus.RUNNING:
            return {'error': 'Deployment not found or not running'}
        
        current_time = time.time()
        config = deployment.config
        
        # Collect current metrics
        metrics = await self.analyzer.collect_canary_metrics(deployment)
        deployment.metrics_history.append(metrics)
        
        # Analyze deployment health
        analysis = self.analyzer.analyze_deployment_health(deployment, metrics)
        
        decision = {
            'timestamp': current_time,
            'deployment_id': deployment_id,
            'current_traffic': deployment.current_traffic_percentage,
            'analysis': analysis,
            'action_taken': 'none'
        }
        
        # Make deployment decisions
        if analysis['should_rollback']:
            await self._rollback_canary(deployment, "Health check failure")
            decision['action_taken'] = 'rollback'
            decision['reason'] = f"Rollback triggered: {', '.join(analysis['issues'])}"
        
        elif analysis['should_promote']:
            await self._promote_canary(deployment)
            decision['action_taken'] = 'promote'
            decision['reason'] = "Canary ready for full promotion"
        
        elif (analysis['can_increase_traffic'] and 
              deployment.current_traffic_percentage < config.final_traffic_percentage and
              current_time - deployment.last_increment_time >= config.increment_interval_minutes * 60):
            
            # Increase traffic
            new_percentage = min(
                deployment.current_traffic_percentage + config.traffic_increment,
                config.final_traffic_percentage
            )
            
            deployment.current_traffic_percentage = new_percentage
            deployment.last_increment_time = current_time
            
            self.traffic_manager.set_canary_traffic(deployment_id, new_percentage, config.strategy)
            
            decision['action_taken'] = 'increase_traffic'
            decision['new_traffic_percentage'] = new_percentage
            decision['reason'] = f"Increased traffic to {new_percentage}% - metrics healthy"
        
        # Check timeout
        max_duration_seconds = config.max_duration_minutes * 60
        if current_time - deployment.start_time > max_duration_seconds:
            if deployment.current_traffic_percentage >= 90.0:
                await self._promote_canary(deployment)
                decision['action_taken'] = 'promote'
                decision['reason'] = "Max duration reached - promoting to stable"
            else:
                await self._rollback_canary(deployment, "Max duration exceeded without reaching promotion criteria")
                decision['action_taken'] = 'rollback'
                decision['reason'] = "Timeout - rolling back"
        
        # Log decision
        deployment.decision_log.append(decision)
        self.save_deployments()
        
        return decision
    
    async def _rollback_canary(self, deployment: CanaryDeployment, reason: str):
        """Rollback a canary deployment."""
        deployment.status = CanaryStatus.ROLLING_BACK
        deployment.rollback_reason = reason
        
        # Remove traffic routing
        if deployment.deployment_id in self.traffic_manager.routing_rules:
            del self.traffic_manager.routing_rules[deployment.deployment_id]
        
        # Restore stable configuration
        try:
            from app.pipeline.manager import PipelineManager
            pipeline_manager = PipelineManager.get_instance()
            
            stable_config = deployment.config.stable_config
            
            # Apply stable configuration
            pipeline_manager.swap_embeddings(
                stable_config.get('embedding_backend', 'hash'),
                stable_config.get('embedding_model', 'fallback')
            )
            pipeline_manager.swap_generation(
                stable_config.get('generation_backend', 'echo'),
                stable_config.get('generation_model', 'fallback')
            )
            
            print(f"🔄 Rolled back canary deployment: {deployment.config.name}")
            print(f"   Reason: {reason}")
            
            deployment.status = CanaryStatus.FAILED
            deployment.completion_time = time.time()
            
            # Move to completed deployments
            self.completed_deployments.append(deployment)
            del self.active_deployments[deployment.deployment_id]
            
        except Exception as e:
            print(f"Failed to rollback canary deployment: {e}")
    
    async def _promote_canary(self, deployment: CanaryDeployment):
        """Promote canary to stable (100% traffic)."""
        deployment.status = CanaryStatus.PROMOTING
        
        try:
            from app.pipeline.manager import PipelineManager
            pipeline_manager = PipelineManager.get_instance()
            
            canary_config = deployment.config.canary_config
            
            # Apply canary configuration as the new stable
            pipeline_manager.swap_embeddings(
                canary_config.get('embedding_backend', 'hash'),
                canary_config.get('embedding_model', 'fallback')
            )
            pipeline_manager.swap_generation(
                canary_config.get('generation_backend', 'echo'),
                canary_config.get('generation_model', 'fallback')
            )
            
            # Remove traffic routing (100% stable now uses canary config)
            if deployment.deployment_id in self.traffic_manager.routing_rules:
                del self.traffic_manager.routing_rules[deployment.deployment_id]
            
            print(f"🎉 Promoted canary deployment to stable: {deployment.config.name}")
            
            deployment.status = CanaryStatus.COMPLETED
            deployment.completion_time = time.time()
            deployment.current_traffic_percentage = 100.0
            
            # Move to completed deployments
            self.completed_deployments.append(deployment)
            del self.active_deployments[deployment.deployment_id]
            
        except Exception as e:
            print(f"Failed to promote canary deployment: {e}")
            deployment.status = CanaryStatus.FAILED
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a canary deployment."""
        deployment = self.active_deployments.get(deployment_id)
        if not deployment:
            # Check completed deployments
            deployment = next((d for d in self.completed_deployments if d.deployment_id == deployment_id), None)
        
        if not deployment:
            return None
        
        latest_metrics = deployment.metrics_history[-1] if deployment.metrics_history else None
        
        return {
            'deployment_id': deployment.deployment_id,
            'name': deployment.config.name,
            'status': deployment.status.value,
            'current_traffic_percentage': deployment.current_traffic_percentage,
            'start_time': deployment.start_time,
            'completion_time': deployment.completion_time,
            'duration_minutes': (time.time() - deployment.start_time) / 60,
            'rollback_reason': deployment.rollback_reason,
            'latest_metrics': asdict(latest_metrics) if latest_metrics else None,
            'decision_count': len(deployment.decision_log),
            'traffic_stats': self.traffic_manager.get_assignment_stats(deployment_id)
        }
    
    def get_active_deployments(self) -> List[Dict[str, Any]]:
        """Get all active canary deployments."""
        return [
            {
                'deployment_id': deployment.deployment_id,
                'name': deployment.config.name,
                'status': deployment.status.value,
                'current_traffic_percentage': deployment.current_traffic_percentage,
                'start_time': deployment.start_time,
                'duration_minutes': (time.time() - deployment.start_time) / 60
            }
            for deployment in self.active_deployments.values()
        ]
    
    def should_use_canary_for_request(self, user_id: str = None, 
                                    user_groups: List[str] = None,
                                    geographic_region: str = None) -> Optional[Dict[str, str]]:
        """Determine if a request should use canary configuration."""
        
        for deployment_id, deployment in self.active_deployments.items():
            if deployment.status == CanaryStatus.RUNNING:
                if self.traffic_manager.should_use_canary(deployment_id, user_id, user_groups, geographic_region):
                    return {
                        'deployment_id': deployment_id,
                        'config': deployment.config.canary_config
                    }
        
        return None
    
    async def start_monitoring(self, check_interval: int = 300):  # 5 minutes
        """Start continuous monitoring of canary deployments."""
        print(f"🐣 Starting canary deployment monitoring (check interval: {check_interval}s)")
        
        while True:
            try:
                for deployment_id in list(self.active_deployments.keys()):
                    decision = await self.manage_canary_deployment(deployment_id)
                    if decision.get('action_taken') != 'none':
                        print(f"🐣 Canary {deployment_id}: {decision['action_taken']} - {decision.get('reason', '')}")
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                print(f"Error in canary monitoring loop: {e}")
                await asyncio.sleep(check_interval)
