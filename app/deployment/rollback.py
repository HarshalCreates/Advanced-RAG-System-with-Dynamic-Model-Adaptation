"""Automated rollback mechanisms for failed deployments."""
from __future__ import annotations

import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
from enum import Enum

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


class DeploymentStatus(Enum):
    """Deployment status types."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ModelConfig:
    """Model configuration snapshot."""
    embedding_backend: str
    embedding_model: str
    generation_backend: str
    generation_model: str
    retriever_backend: str
    timestamp: float
    config_id: str


@dataclass
class HealthMetrics:
    """System health metrics."""
    response_time_ms: float
    error_rate: float
    success_rate: float
    confidence_avg: float
    throughput_qps: float
    timestamp: float


@dataclass
class DeploymentRecord:
    """Record of a deployment attempt."""
    deployment_id: str
    config_before: ModelConfig
    config_after: ModelConfig
    status: DeploymentStatus
    start_time: float
    end_time: Optional[float]
    health_before: Optional[HealthMetrics]
    health_after: Optional[HealthMetrics]
    rollback_reason: Optional[str]
    metadata: Dict[str, Any]


class HealthChecker:
    """Monitors system health and performance."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.baseline_metrics: Optional[HealthMetrics] = None
        
    async def get_health_metrics(self) -> HealthMetrics:
        """Get current system health metrics."""
        if not AIOHTTP_AVAILABLE:
            # Fallback metrics for testing
            return HealthMetrics(
                response_time_ms=100.0,
                error_rate=0.01,
                success_rate=0.99,
                confidence_avg=0.75,
                throughput_qps=10.0,
                timestamp=time.time()
            )
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test multiple queries to get average metrics
                start_time = time.time()
                
                test_queries = [
                    "What is machine learning?",
                    "How does AI work?",
                    "Explain neural networks"
                ]
                
                success_count = 0
                total_confidence = 0.0
                
                for query in test_queries:
                    try:
                        async with session.post(
                            f"{self.base_url}/api/query",
                            json={"query": query, "top_k": 3, "filters": {}},
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                success_count += 1
                                total_confidence += result.get("answer", {}).get("confidence", 0.0)
                    except Exception:
                        pass
                
                end_time = time.time()
                total_time = (end_time - start_time) * 1000  # Convert to ms
                
                success_rate = success_count / len(test_queries)
                error_rate = 1.0 - success_rate
                avg_response_time = total_time / len(test_queries)
                avg_confidence = total_confidence / max(success_count, 1)
                throughput = len(test_queries) / (total_time / 1000)  # QPS
                
                return HealthMetrics(
                    response_time_ms=avg_response_time,
                    error_rate=error_rate,
                    success_rate=success_rate,
                    confidence_avg=avg_confidence,
                    throughput_qps=throughput,
                    timestamp=time.time()
                )
                
        except Exception as e:
            print(f"Health check failed: {e}")
            # Return degraded metrics
            return HealthMetrics(
                response_time_ms=999.0,
                error_rate=1.0,
                success_rate=0.0,
                confidence_avg=0.0,
                throughput_qps=0.0,
                timestamp=time.time()
            )
    
    def set_baseline(self, metrics: HealthMetrics):
        """Set baseline metrics for comparison."""
        self.baseline_metrics = metrics
    
    def is_healthy(self, current: HealthMetrics, thresholds: Dict[str, float] = None) -> Tuple[bool, List[str]]:
        """Check if current metrics indicate healthy system."""
        if thresholds is None:
            thresholds = {
                'max_response_time_ms': 1000.0,
                'max_error_rate': 0.05,
                'min_success_rate': 0.95,
                'min_confidence': 0.3,
                'min_throughput_qps': 1.0
            }
        
        issues = []
        
        if current.response_time_ms > thresholds['max_response_time_ms']:
            issues.append(f"High response time: {current.response_time_ms:.1f}ms")
        
        if current.error_rate > thresholds['max_error_rate']:
            issues.append(f"High error rate: {current.error_rate:.1%}")
        
        if current.success_rate < thresholds['min_success_rate']:
            issues.append(f"Low success rate: {current.success_rate:.1%}")
        
        if current.confidence_avg < thresholds['min_confidence']:
            issues.append(f"Low confidence: {current.confidence_avg:.2f}")
        
        if current.throughput_qps < thresholds['min_throughput_qps']:
            issues.append(f"Low throughput: {current.throughput_qps:.1f} QPS")
        
        # Compare with baseline if available
        if self.baseline_metrics:
            baseline = self.baseline_metrics
            
            # Response time degradation
            if current.response_time_ms > baseline.response_time_ms * 1.5:
                issues.append(f"Response time degraded: {current.response_time_ms:.1f}ms vs baseline {baseline.response_time_ms:.1f}ms")
            
            # Error rate increase
            if current.error_rate > baseline.error_rate * 2.0:
                issues.append(f"Error rate increased: {current.error_rate:.1%} vs baseline {baseline.error_rate:.1%}")
            
            # Confidence drop
            if current.confidence_avg < baseline.confidence_avg * 0.8:
                issues.append(f"Confidence dropped: {current.confidence_avg:.2f} vs baseline {baseline.confidence_avg:.2f}")
        
        return len(issues) == 0, issues


class RollbackManager:
    """Manages automated rollback for failed deployments."""
    
    def __init__(self, storage_path: str = "./data/deployments"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.deployments_file = self.storage_path / "deployments.json"
        self.configs_file = self.storage_path / "config_snapshots.json"
        
        self.deployments: List[DeploymentRecord] = []
        self.config_snapshots: List[ModelConfig] = []
        
        self.health_checker = HealthChecker()
        
        # Load existing data
        self.load_deployment_history()
        
        # Rollback policies
        self.rollback_policies = {
            'health_check_interval': 30,  # seconds
            'max_health_check_failures': 3,
            'rollback_timeout': 300,  # 5 minutes
            'auto_rollback_enabled': True
        }
    
    def load_deployment_history(self):
        """Load deployment history from storage."""
        if self.deployments_file.exists():
            try:
                with open(self.deployments_file, 'r') as f:
                    data = json.load(f)
                    self.deployments = [
                        DeploymentRecord(
                            **{**record, 'status': DeploymentStatus(record['status'])}
                        ) for record in data
                    ]
            except Exception as e:
                print(f"Failed to load deployments: {e}")
        
        if self.configs_file.exists():
            try:
                with open(self.configs_file, 'r') as f:
                    data = json.load(f)
                    self.config_snapshots = [ModelConfig(**config) for config in data]
            except Exception as e:
                print(f"Failed to load config snapshots: {e}")
    
    def save_deployment_history(self):
        """Save deployment history to storage."""
        try:
            # Save deployments
            deployments_data = []
            for deployment in self.deployments:
                data = asdict(deployment)
                data['status'] = deployment.status.value  # Convert enum to string
                deployments_data.append(data)
            
            with open(self.deployments_file, 'w') as f:
                json.dump(deployments_data, f, indent=2)
            
            # Save config snapshots
            configs_data = [asdict(config) for config in self.config_snapshots]
            with open(self.configs_file, 'w') as f:
                json.dump(configs_data, f, indent=2)
                
        except Exception as e:
            print(f"Failed to save deployment history: {e}")
    
    def create_config_snapshot(self, pipeline_manager) -> ModelConfig:
        """Create a snapshot of current model configuration."""
        settings = pipeline_manager.settings
        
        config = ModelConfig(
            embedding_backend=settings.embedding_backend,
            embedding_model=settings.embedding_model,
            generation_backend=settings.generation_backend,
            generation_model=settings.generation_model,
            retriever_backend=settings.retriever_backend,
            timestamp=time.time(),
            config_id=f"config_{int(time.time())}"
        )
        
        self.config_snapshots.append(config)
        self.save_deployment_history()
        
        return config
    
    async def safe_deployment(self, deployment_function: Callable, 
                            pipeline_manager, deployment_params: Dict[str, Any],
                            validation_timeout: int = 300) -> DeploymentRecord:
        """Perform a safe deployment with automatic rollback capability."""
        
        deployment_id = f"deployment_{int(time.time())}"
        
        # 1. Create config snapshot before deployment
        config_before = self.create_config_snapshot(pipeline_manager)
        
        # 2. Get baseline health metrics
        health_before = await self.health_checker.get_health_metrics()
        self.health_checker.set_baseline(health_before)
        
        # 3. Create deployment record
        deployment = DeploymentRecord(
            deployment_id=deployment_id,
            config_before=config_before,
            config_after=config_before,  # Will be updated after deployment
            status=DeploymentStatus.PENDING,
            start_time=time.time(),
            end_time=None,
            health_before=health_before,
            health_after=None,
            rollback_reason=None,
            metadata=deployment_params
        )
        
        self.deployments.append(deployment)
        
        try:
            # 4. Execute deployment
            deployment.status = DeploymentStatus.IN_PROGRESS
            print(f"🚀 Starting deployment {deployment_id}")
            
            # Execute the deployment function
            await deployment_function(**deployment_params)
            
            # 5. Create post-deployment config snapshot
            config_after = self.create_config_snapshot(pipeline_manager)
            deployment.config_after = config_after
            
            # 6. Monitor health for validation period
            print(f"🔍 Monitoring deployment health for {validation_timeout}s...")
            
            validation_start = time.time()
            health_check_failures = 0
            
            while time.time() - validation_start < validation_timeout:
                await asyncio.sleep(self.rollback_policies['health_check_interval'])
                
                current_health = await self.health_checker.get_health_metrics()
                is_healthy, issues = self.health_checker.is_healthy(current_health)
                
                if not is_healthy:
                    health_check_failures += 1
                    print(f"⚠️  Health check failed ({health_check_failures}/{self.rollback_policies['max_health_check_failures']}): {', '.join(issues)}")
                    
                    if health_check_failures >= self.rollback_policies['max_health_check_failures']:
                        # Trigger automatic rollback
                        print(f"🔄 Triggering automatic rollback due to health check failures")
                        rollback_success = await self.rollback_deployment(deployment_id, "Health check failures")
                        
                        deployment.status = DeploymentStatus.ROLLED_BACK if rollback_success else DeploymentStatus.FAILED
                        deployment.rollback_reason = f"Health check failures: {', '.join(issues)}"
                        deployment.health_after = current_health
                        deployment.end_time = time.time()
                        
                        self.save_deployment_history()
                        return deployment
                else:
                    health_check_failures = 0  # Reset on successful check
            
            # 7. Deployment successful
            final_health = await self.health_checker.get_health_metrics()
            deployment.health_after = final_health
            deployment.status = DeploymentStatus.SUCCESS
            deployment.end_time = time.time()
            
            print(f"✅ Deployment {deployment_id} completed successfully")
            
        except Exception as e:
            print(f"❌ Deployment {deployment_id} failed: {e}")
            
            # Automatic rollback on deployment failure
            rollback_success = await self.rollback_deployment(deployment_id, f"Deployment error: {str(e)}")
            deployment.status = DeploymentStatus.ROLLED_BACK if rollback_success else DeploymentStatus.FAILED
            deployment.rollback_reason = f"Deployment error: {str(e)}"
            deployment.end_time = time.time()
        
        self.save_deployment_history()
        return deployment
    
    async def rollback_deployment(self, deployment_id: str, reason: str = "") -> bool:
        """Rollback a specific deployment to its previous configuration."""
        
        # Find the deployment
        deployment = next((d for d in self.deployments if d.deployment_id == deployment_id), None)
        if not deployment:
            print(f"❌ Deployment {deployment_id} not found")
            return False
        
        print(f"🔄 Rolling back deployment {deployment_id}: {reason}")
        
        try:
            # Get the pipeline manager instance
            from app.pipeline.manager import PipelineManager
            pipeline_manager = PipelineManager.get_instance()
            
            # Restore previous configuration
            config = deployment.config_before
            
            # Apply rollback configuration
            pipeline_manager.swap_embeddings(config.embedding_backend, config.embedding_model)
            pipeline_manager.swap_generation(config.generation_backend, config.generation_model)
            pipeline_manager.swap_retriever(config.retriever_backend)
            
            print(f"✅ Rollback completed for deployment {deployment_id}")
            return True
            
        except Exception as e:
            print(f"❌ Rollback failed for deployment {deployment_id}: {e}")
            return False
    
    def get_deployment_history(self, limit: int = 10) -> List[DeploymentRecord]:
        """Get recent deployment history."""
        return sorted(self.deployments, key=lambda d: d.start_time, reverse=True)[:limit]
    
    def get_rollback_candidates(self) -> List[DeploymentRecord]:
        """Get deployments that can be rolled back."""
        return [
            d for d in self.deployments 
            if d.status == DeploymentStatus.SUCCESS and d.end_time and d.end_time > time.time() - 86400  # Last 24 hours
        ]
    
    async def emergency_rollback(self) -> bool:
        """Emergency rollback to the last known good configuration."""
        successful_deployments = [
            d for d in self.deployments 
            if d.status == DeploymentStatus.SUCCESS
        ]
        
        if not successful_deployments:
            print("❌ No successful deployments found for emergency rollback")
            return False
        
        # Get the most recent successful deployment
        last_good = sorted(successful_deployments, key=lambda d: d.end_time or 0, reverse=True)[0]
        
        print(f"🚨 Emergency rollback to deployment {last_good.deployment_id}")
        return await self.rollback_deployment(last_good.deployment_id, "Emergency rollback")
    
    def get_rollback_stats(self) -> Dict[str, Any]:
        """Get rollback statistics."""
        total = len(self.deployments)
        if total == 0:
            return {"total_deployments": 0}
        
        successful = len([d for d in self.deployments if d.status == DeploymentStatus.SUCCESS])
        failed = len([d for d in self.deployments if d.status == DeploymentStatus.FAILED])
        rolled_back = len([d for d in self.deployments if d.status == DeploymentStatus.ROLLED_BACK])
        
        return {
            "total_deployments": total,
            "successful": successful,
            "failed": failed,
            "rolled_back": rolled_back,
            "success_rate": successful / total,
            "rollback_rate": rolled_back / total,
            "health_policies": self.rollback_policies
        }
