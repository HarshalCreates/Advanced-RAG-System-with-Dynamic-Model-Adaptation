"""Pre-deployment validation and compatibility checking."""
from __future__ import annotations

import json
import time
import asyncio
import tempfile
import subprocess
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
from enum import Enum
import sys
import importlib


class ValidationLevel(Enum):
    """Validation levels."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    PRODUCTION = "production"


class ValidationStatus(Enum):
    """Validation status types."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


class ValidationType(Enum):
    """Types of validation checks."""
    DEPENDENCY = "dependency"
    CONFIGURATION = "configuration"
    COMPATIBILITY = "compatibility"
    PERFORMANCE = "performance"
    SECURITY = "security"
    INTEGRATION = "integration"
    LOAD_TEST = "load_test"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_id: str
    check_name: str
    validation_type: ValidationType
    status: ValidationStatus
    message: str
    details: Dict[str, Any]
    duration_ms: float
    timestamp: float
    recommendations: List[str] = None


@dataclass
class DeploymentValidationConfig:
    """Configuration for deployment validation."""
    config_id: str
    name: str
    validation_level: ValidationLevel
    
    # Target configuration to validate
    target_config: Dict[str, str]
    
    # Validation settings
    timeout_seconds: int = 300
    parallel_execution: bool = True
    fail_fast: bool = False
    
    # Test data and scenarios
    test_queries: List[str] = None
    load_test_duration_seconds: int = 60
    load_test_concurrent_users: int = 10
    
    # Environment settings
    test_environment_url: str = "http://localhost:8000"
    
    # Validation rules
    min_response_time_ms: float = 2000.0
    min_success_rate: float = 0.95
    min_confidence_score: float = 0.3
    max_memory_usage_mb: float = 2048.0
    max_cpu_usage_percent: float = 80.0
    
    metadata: Dict[str, Any] = None


class DependencyValidator:
    """Validates dependencies and environment."""
    
    def __init__(self):
        self.python_version = sys.version_info
    
    async def validate_python_version(self) -> ValidationResult:
        """Validate Python version compatibility."""
        start_time = time.time()
        
        min_version = (3, 11, 0)
        current_version = self.python_version[:3]
        
        if current_version >= min_version:
            status = ValidationStatus.PASSED
            message = f"Python version {'.'.join(map(str, current_version))} is compatible"
            details = {"required": min_version, "actual": current_version}
            recommendations = []
        else:
            status = ValidationStatus.FAILED
            message = f"Python version {'.'.join(map(str, current_version))} is below minimum {'.'.join(map(str, min_version))}"
            details = {"required": min_version, "actual": current_version}
            recommendations = [f"Upgrade Python to version {'.'.join(map(str, min_version))} or higher"]
        
        duration = (time.time() - start_time) * 1000
        
        return ValidationResult(
            check_id="python_version",
            check_name="Python Version Compatibility",
            validation_type=ValidationType.DEPENDENCY,
            status=status,
            message=message,
            details=details,
            duration_ms=duration,
            timestamp=time.time(),
            recommendations=recommendations
        )
    
    async def validate_required_packages(self) -> ValidationResult:
        """Validate that all required packages are installed."""
        start_time = time.time()
        
        required_packages = [
            "fastapi", "uvicorn", "pydantic", "pydantic-settings",
            "sentence-transformers", "faiss-cpu", "scikit-learn",
            "openai", "anthropic", "cohere", "requests", "aiohttp",
            "prometheus-fastapi-instrumentator", "chainlit"
        ]
        
        missing_packages = []
        installed_packages = {}
        
        for package in required_packages:
            try:
                module = importlib.import_module(package.replace("-", "_"))
                version = getattr(module, "__version__", "unknown")
                installed_packages[package] = version
            except ImportError:
                missing_packages.append(package)
        
        if not missing_packages:
            status = ValidationStatus.PASSED
            message = f"All {len(required_packages)} required packages are installed"
            recommendations = []
        else:
            status = ValidationStatus.FAILED
            message = f"{len(missing_packages)} required packages are missing"
            recommendations = [f"Install missing packages: pip install {' '.join(missing_packages)}"]
        
        details = {
            "required_packages": required_packages,
            "installed_packages": installed_packages,
            "missing_packages": missing_packages
        }
        
        duration = (time.time() - start_time) * 1000
        
        return ValidationResult(
            check_id="required_packages",
            check_name="Required Packages Availability",
            validation_type=ValidationType.DEPENDENCY,
            status=status,
            message=message,
            details=details,
            duration_ms=duration,
            timestamp=time.time(),
            recommendations=recommendations
        )
    
    async def validate_system_resources(self) -> ValidationResult:
        """Validate system resources (memory, disk space)."""
        start_time = time.time()
        
        try:
            import psutil
            
            # Memory check
            memory = psutil.virtual_memory()
            available_memory_gb = memory.available / (1024**3)
            
            # Disk space check
            disk = psutil.disk_usage('/')
            available_disk_gb = disk.free / (1024**3)
            
            # CPU count
            cpu_count = psutil.cpu_count()
            
            min_memory_gb = 2.0
            min_disk_gb = 5.0
            min_cpu_cores = 2
            
            issues = []
            if available_memory_gb < min_memory_gb:
                issues.append(f"Insufficient memory: {available_memory_gb:.1f}GB available, {min_memory_gb}GB required")
            
            if available_disk_gb < min_disk_gb:
                issues.append(f"Insufficient disk space: {available_disk_gb:.1f}GB available, {min_disk_gb}GB required")
            
            if cpu_count < min_cpu_cores:
                issues.append(f"Insufficient CPU cores: {cpu_count} available, {min_cpu_cores} required")
            
            if not issues:
                status = ValidationStatus.PASSED
                message = "System resources are adequate"
                recommendations = []
            else:
                status = ValidationStatus.WARNING if len(issues) == 1 else ValidationStatus.FAILED
                message = f"Resource issues detected: {'; '.join(issues)}"
                recommendations = ["Consider upgrading system resources for optimal performance"]
            
            details = {
                "memory_available_gb": available_memory_gb,
                "disk_available_gb": available_disk_gb,
                "cpu_cores": cpu_count,
                "requirements": {
                    "min_memory_gb": min_memory_gb,
                    "min_disk_gb": min_disk_gb,
                    "min_cpu_cores": min_cpu_cores
                }
            }
            
        except ImportError:
            status = ValidationStatus.WARNING
            message = "psutil not available - cannot check system resources"
            details = {}
            recommendations = ["Install psutil for system resource monitoring: pip install psutil"]
        
        duration = (time.time() - start_time) * 1000
        
        return ValidationResult(
            check_id="system_resources",
            check_name="System Resources Check",
            validation_type=ValidationType.DEPENDENCY,
            status=status,
            message=message,
            details=details,
            duration_ms=duration,
            timestamp=time.time(),
            recommendations=recommendations
        )


class ConfigurationValidator:
    """Validates configuration compatibility and correctness."""
    
    async def validate_model_configurations(self, target_config: Dict[str, str]) -> ValidationResult:
        """Validate model configuration compatibility."""
        start_time = time.time()
        
        embedding_backend = target_config.get("embedding_backend", "hash")
        embedding_model = target_config.get("embedding_model", "fallback")
        generation_backend = target_config.get("generation_backend", "echo")
        generation_model = target_config.get("generation_model", "fallback")
        retriever_backend = target_config.get("retriever_backend", "inmemory")
        
        issues = []
        warnings = []
        
        # Validate embedding configuration
        valid_embedding_backends = ["openai", "cohere", "sentence-transformers", "hash"]
        if embedding_backend not in valid_embedding_backends:
            issues.append(f"Invalid embedding backend: {embedding_backend}")
        
        # Model-specific validations
        if embedding_backend == "openai" and not embedding_model.startswith("text-embedding-"):
            warnings.append(f"Unusual OpenAI embedding model: {embedding_model}")
        
        if embedding_backend == "sentence-transformers" and "/" not in embedding_model:
            warnings.append(f"SentenceTransformers model should include organization: {embedding_model}")
        
        # Validate generation configuration
        valid_generation_backends = ["openai", "anthropic", "ollama", "echo"]
        if generation_backend not in valid_generation_backends:
            issues.append(f"Invalid generation backend: {generation_backend}")
        
        # Validate retriever configuration
        valid_retriever_backends = ["faiss", "inmemory", "chroma", "pinecone", "weaviate"]
        if retriever_backend not in valid_retriever_backends:
            issues.append(f"Invalid retriever backend: {retriever_backend}")
        
        # Check for compatible combinations
        if embedding_backend == "hash" and retriever_backend == "faiss":
            warnings.append("Hash embeddings with FAISS may not provide meaningful semantic search")
        
        if generation_backend == "echo" and embedding_backend != "hash":
            warnings.append("Using real embeddings with echo generation may not be ideal for testing")
        
        # Determine status
        if issues:
            status = ValidationStatus.FAILED
            message = f"Configuration validation failed: {'; '.join(issues)}"
        elif warnings:
            status = ValidationStatus.WARNING
            message = f"Configuration validation passed with warnings: {'; '.join(warnings)}"
        else:
            status = ValidationStatus.PASSED
            message = "All model configurations are valid"
        
        recommendations = []
        if embedding_backend == "hash":
            recommendations.append("Consider using real embedding models for production")
        if generation_backend == "echo":
            recommendations.append("Configure real LLM for production use")
        
        details = {
            "embedding_backend": embedding_backend,
            "embedding_model": embedding_model,
            "generation_backend": generation_backend,
            "generation_model": generation_model,
            "retriever_backend": retriever_backend,
            "issues": issues,
            "warnings": warnings
        }
        
        duration = (time.time() - start_time) * 1000
        
        return ValidationResult(
            check_id="model_configurations",
            check_name="Model Configuration Validation",
            validation_type=ValidationType.CONFIGURATION,
            status=status,
            message=message,
            details=details,
            duration_ms=duration,
            timestamp=time.time(),
            recommendations=recommendations
        )
    
    async def validate_api_keys(self, target_config: Dict[str, str]) -> ValidationResult:
        """Validate API key availability and validity."""
        start_time = time.time()
        
        from app.models.config import get_settings
        settings = get_settings()
        
        embedding_backend = target_config.get("embedding_backend", "hash")
        generation_backend = target_config.get("generation_backend", "echo")
        
        missing_keys = []
        invalid_keys = []
        valid_keys = []
        
        # Check required API keys based on backends
        key_requirements = {
            "openai": {"key": settings.openai_api_key, "service": "OpenAI"},
            "cohere": {"key": settings.cohere_api_key, "service": "Cohere"},
            "anthropic": {"key": settings.anthropic_api_key, "service": "Anthropic"}
        }
        
        required_services = set()
        if embedding_backend in ["openai", "cohere"]:
            required_services.add(embedding_backend)
        if generation_backend in ["openai", "anthropic"]:
            required_services.add(generation_backend)
        
        for service in required_services:
            if service in key_requirements:
                key_info = key_requirements[service]
                if not key_info["key"]:
                    missing_keys.append(key_info["service"])
                elif len(key_info["key"]) < 10:  # Basic sanity check
                    invalid_keys.append(key_info["service"])
                else:
                    valid_keys.append(key_info["service"])
        
        # Determine status
        if missing_keys:
            status = ValidationStatus.FAILED
            message = f"Missing API keys for: {', '.join(missing_keys)}"
        elif invalid_keys:
            status = ValidationStatus.WARNING
            message = f"Potentially invalid API keys for: {', '.join(invalid_keys)}"
        else:
            status = ValidationStatus.PASSED
            message = "All required API keys are available"
        
        recommendations = []
        for service in missing_keys:
            recommendations.append(f"Set {service.upper()}_API_KEY environment variable")
        
        details = {
            "required_services": list(required_services),
            "missing_keys": missing_keys,
            "invalid_keys": invalid_keys,
            "valid_keys": valid_keys
        }
        
        duration = (time.time() - start_time) * 1000
        
        return ValidationResult(
            check_id="api_keys",
            check_name="API Keys Validation",
            validation_type=ValidationType.CONFIGURATION,
            status=status,
            message=message,
            details=details,
            duration_ms=duration,
            timestamp=time.time(),
            recommendations=recommendations
        )


class PerformanceValidator:
    """Validates performance characteristics."""
    
    async def validate_response_time(self, config: DeploymentValidationConfig) -> ValidationResult:
        """Validate API response time."""
        start_time = time.time()
        
        try:
            import aiohttp
            
            test_queries = config.test_queries or [
                "What is machine learning?",
                "How does artificial intelligence work?",
                "Explain deep learning"
            ]
            
            response_times = []
            success_count = 0
            
            async with aiohttp.ClientSession() as session:
                for query in test_queries:
                    query_start = time.time()
                    try:
                        async with session.post(
                            f"{config.test_environment_url}/api/query",
                            json={"query": query, "top_k": 3, "filters": {}},
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as response:
                            query_time = (time.time() - query_start) * 1000
                            response_times.append(query_time)
                            
                            if response.status == 200:
                                success_count += 1
                    except Exception:
                        response_times.append(30000)  # 30s timeout
            
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
            success_rate = success_count / len(test_queries)
            
            issues = []
            if avg_response_time > config.min_response_time_ms:
                issues.append(f"Average response time too high: {avg_response_time:.1f}ms")
            
            if success_rate < config.min_success_rate:
                issues.append(f"Success rate too low: {success_rate:.1%}")
            
            if not issues:
                status = ValidationStatus.PASSED
                message = f"Performance validation passed (avg: {avg_response_time:.1f}ms, success: {success_rate:.1%})"
            else:
                status = ValidationStatus.FAILED
                message = f"Performance issues: {'; '.join(issues)}"
            
            recommendations = []
            if avg_response_time > config.min_response_time_ms:
                recommendations.append("Consider optimizing model configuration or scaling resources")
            
            details = {
                "avg_response_time_ms": avg_response_time,
                "max_response_time_ms": max_response_time,
                "min_response_time_ms": min_response_time,
                "success_rate": success_rate,
                "test_queries_count": len(test_queries),
                "individual_response_times": response_times
            }
            
        except ImportError:
            status = ValidationStatus.SKIPPED
            message = "aiohttp not available - skipping response time validation"
            details = {}
            recommendations = ["Install aiohttp for performance testing"]
        
        duration = (time.time() - start_time) * 1000
        
        return ValidationResult(
            check_id="response_time",
            check_name="Response Time Validation",
            validation_type=ValidationType.PERFORMANCE,
            status=status,
            message=message,
            details=details,
            duration_ms=duration,
            timestamp=time.time(),
            recommendations=recommendations
        )
    
    async def validate_load_capacity(self, config: DeploymentValidationConfig) -> ValidationResult:
        """Validate system under load."""
        start_time = time.time()
        
        if config.validation_level in [ValidationLevel.BASIC, ValidationLevel.STANDARD]:
            status = ValidationStatus.SKIPPED
            message = "Load testing skipped for basic/standard validation levels"
            details = {}
            recommendations = ["Run comprehensive validation for load testing"]
            duration = (time.time() - start_time) * 1000
            
            return ValidationResult(
                check_id="load_capacity",
                check_name="Load Capacity Validation",
                validation_type=ValidationType.LOAD_TEST,
                status=status,
                message=message,
                details=details,
                duration_ms=duration,
                timestamp=time.time(),
                recommendations=recommendations
            )
        
        try:
            import aiohttp
            import asyncio
            
            concurrent_users = config.load_test_concurrent_users
            duration_seconds = config.load_test_duration_seconds
            test_query = "What is the system status?"
            
            async def send_request(session):
                try:
                    async with session.post(
                        f"{config.test_environment_url}/api/query",
                        json={"query": test_query, "top_k": 1, "filters": {}},
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        return response.status == 200, time.time()
                except Exception:
                    return False, time.time()
            
            async def load_test():
                results = []
                end_time = time.time() + duration_seconds
                
                async with aiohttp.ClientSession() as session:
                    while time.time() < end_time:
                        # Send concurrent requests
                        tasks = [send_request(session) for _ in range(concurrent_users)]
                        batch_results = await asyncio.gather(*tasks)
                        results.extend(batch_results)
                        
                        # Small delay between batches
                        await asyncio.sleep(1)
                
                return results
            
            print(f"🧪 Running load test: {concurrent_users} concurrent users for {duration_seconds}s")
            load_results = await load_test()
            
            if load_results:
                success_count = sum(1 for success, _ in load_results if success)
                total_requests = len(load_results)
                success_rate = success_count / total_requests
                
                requests_per_second = total_requests / duration_seconds
                
                if success_rate >= config.min_success_rate:
                    status = ValidationStatus.PASSED
                    message = f"Load test passed: {success_rate:.1%} success rate, {requests_per_second:.1f} RPS"
                else:
                    status = ValidationStatus.FAILED
                    message = f"Load test failed: {success_rate:.1%} success rate below threshold"
                
                details = {
                    "concurrent_users": concurrent_users,
                    "duration_seconds": duration_seconds,
                    "total_requests": total_requests,
                    "successful_requests": success_count,
                    "success_rate": success_rate,
                    "requests_per_second": requests_per_second
                }
                
                recommendations = []
                if success_rate < config.min_success_rate:
                    recommendations.append("System may need scaling or optimization for production load")
            else:
                status = ValidationStatus.FAILED
                message = "Load test failed - no successful requests"
                details = {"error": "No successful requests during load test"}
                recommendations = ["Check system availability and configuration"]
            
        except ImportError:
            status = ValidationStatus.SKIPPED
            message = "aiohttp not available - skipping load testing"
            details = {}
            recommendations = ["Install aiohttp for load testing"]
        
        duration = (time.time() - start_time) * 1000
        
        return ValidationResult(
            check_id="load_capacity",
            check_name="Load Capacity Validation",
            validation_type=ValidationType.LOAD_TEST,
            status=status,
            message=message,
            details=details,
            duration_ms=duration,
            timestamp=time.time(),
            recommendations=recommendations
        )


class IntegrationValidator:
    """Validates integration with external services."""
    
    async def validate_service_connectivity(self, config: DeploymentValidationConfig) -> ValidationResult:
        """Validate connectivity to external services."""
        start_time = time.time()
        
        try:
            import aiohttp
            
            # Test endpoints
            endpoints_to_test = [
                {"name": "Health Check", "url": f"{config.test_environment_url}/api/health", "expected_status": 200},
                {"name": "API Documentation", "url": f"{config.test_environment_url}/docs", "expected_status": 200},
                {"name": "Metrics Endpoint", "url": f"{config.test_environment_url}/metrics", "expected_status": 200}
            ]
            
            connectivity_results = []
            
            async with aiohttp.ClientSession() as session:
                for endpoint in endpoints_to_test:
                    try:
                        async with session.get(
                            endpoint["url"],
                            timeout=aiohttp.ClientTimeout(total=10)
                        ) as response:
                            success = response.status == endpoint["expected_status"]
                            connectivity_results.append({
                                "name": endpoint["name"],
                                "url": endpoint["url"],
                                "status_code": response.status,
                                "expected_status": endpoint["expected_status"],
                                "success": success
                            })
                    except Exception as e:
                        connectivity_results.append({
                            "name": endpoint["name"],
                            "url": endpoint["url"],
                            "error": str(e),
                            "success": False
                        })
            
            successful_connections = sum(1 for result in connectivity_results if result["success"])
            total_connections = len(connectivity_results)
            
            if successful_connections == total_connections:
                status = ValidationStatus.PASSED
                message = f"All {total_connections} service endpoints are accessible"
            elif successful_connections > 0:
                status = ValidationStatus.WARNING
                message = f"{successful_connections}/{total_connections} service endpoints are accessible"
            else:
                status = ValidationStatus.FAILED
                message = "No service endpoints are accessible"
            
            recommendations = []
            for result in connectivity_results:
                if not result["success"]:
                    recommendations.append(f"Fix connectivity to {result['name']}: {result['url']}")
            
            details = {
                "connectivity_results": connectivity_results,
                "successful_connections": successful_connections,
                "total_connections": total_connections
            }
            
        except ImportError:
            status = ValidationStatus.SKIPPED
            message = "aiohttp not available - skipping connectivity validation"
            details = {}
            recommendations = ["Install aiohttp for connectivity testing"]
        
        duration = (time.time() - start_time) * 1000
        
        return ValidationResult(
            check_id="service_connectivity",
            check_name="Service Connectivity Validation",
            validation_type=ValidationType.INTEGRATION,
            status=status,
            message=message,
            details=details,
            duration_ms=duration,
            timestamp=time.time(),
            recommendations=recommendations
        )


class DeploymentValidator:
    """Main deployment validation orchestrator."""
    
    def __init__(self, storage_path: str = "./data/validation"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.results_file = self.storage_path / "validation_results.json"
        
        # Initialize validators
        self.dependency_validator = DependencyValidator()
        self.config_validator = ConfigurationValidator()
        self.performance_validator = PerformanceValidator()
        self.integration_validator = IntegrationValidator()
        
        # Validation history
        self.validation_history: List[Dict[str, Any]] = []
        self.load_validation_history()
    
    def load_validation_history(self):
        """Load validation history from storage."""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    self.validation_history = json.load(f)
            except Exception as e:
                print(f"Failed to load validation history: {e}")
    
    def save_validation_history(self):
        """Save validation history to storage."""
        try:
            with open(self.results_file, 'w') as f:
                json.dump(self.validation_history[-100:], f, indent=2)  # Keep last 100 results
        except Exception as e:
            print(f"Failed to save validation history: {e}")
    
    async def run_validation(self, config: DeploymentValidationConfig) -> Dict[str, Any]:
        """Run comprehensive deployment validation."""
        
        validation_start_time = time.time()
        print(f"🔍 Starting {config.validation_level.value} deployment validation: {config.name}")
        
        # Define validation checks based on level
        validation_checks = self._get_validation_checks(config)
        
        results = []
        failed_checks = 0
        warning_checks = 0
        
        if config.parallel_execution:
            # Run checks in parallel
            tasks = []
            for check_func, check_args in validation_checks:
                if check_args:
                    tasks.append(check_func(*check_args))
                else:
                    tasks.append(check_func())
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    check_name = validation_checks[i][0].__name__
                    error_result = ValidationResult(
                        check_id=f"error_{i}",
                        check_name=check_name,
                        validation_type=ValidationType.CONFIGURATION,
                        status=ValidationStatus.FAILED,
                        message=f"Validation check failed: {str(result)}",
                        details={"error": str(result)},
                        duration_ms=0,
                        timestamp=time.time(),
                        recommendations=["Check system configuration and try again"]
                    )
                    processed_results.append(error_result)
                else:
                    processed_results.append(result)
            
            results = processed_results
        else:
            # Run checks sequentially
            for check_func, check_args in validation_checks:
                try:
                    if check_args:
                        result = await check_func(*check_args)
                    else:
                        result = await check_func()
                    
                    results.append(result)
                    
                    # Fail fast if enabled
                    if config.fail_fast and result.status == ValidationStatus.FAILED:
                        print(f"❌ Fail-fast triggered by check: {result.check_name}")
                        break
                        
                except Exception as e:
                    error_result = ValidationResult(
                        check_id=f"error_{len(results)}",
                        check_name=check_func.__name__,
                        validation_type=ValidationType.CONFIGURATION,
                        status=ValidationStatus.FAILED,
                        message=f"Validation check failed: {str(e)}",
                        details={"error": str(e)},
                        duration_ms=0,
                        timestamp=time.time(),
                        recommendations=["Check system configuration and try again"]
                    )
                    results.append(error_result)
                    
                    if config.fail_fast:
                        break
        
        # Analyze results
        for result in results:
            if result.status == ValidationStatus.FAILED:
                failed_checks += 1
            elif result.status == ValidationStatus.WARNING:
                warning_checks += 1
        
        total_duration = (time.time() - validation_start_time) * 1000
        
        # Determine overall status
        if failed_checks > 0:
            overall_status = "FAILED"
            overall_message = f"Validation failed: {failed_checks} failed checks"
        elif warning_checks > 0:
            overall_status = "WARNING"
            overall_message = f"Validation passed with warnings: {warning_checks} warnings"
        else:
            overall_status = "PASSED"
            overall_message = "All validation checks passed"
        
        # Create validation report
        validation_report = {
            "validation_id": f"validation_{int(validation_start_time)}",
            "config_name": config.name,
            "validation_level": config.validation_level.value,
            "overall_status": overall_status,
            "overall_message": overall_message,
            "start_time": validation_start_time,
            "duration_ms": total_duration,
            "target_config": config.target_config,
            "summary": {
                "total_checks": len(results),
                "passed_checks": len([r for r in results if r.status == ValidationStatus.PASSED]),
                "warning_checks": warning_checks,
                "failed_checks": failed_checks,
                "skipped_checks": len([r for r in results if r.status == ValidationStatus.SKIPPED])
            },
            "results": [asdict(result) for result in results],
            "recommendations": self._generate_overall_recommendations(results),
            "next_steps": self._generate_next_steps(overall_status, results)
        }
        
        # Save to history
        self.validation_history.append(validation_report)
        self.save_validation_history()
        
        # Print summary
        self._print_validation_summary(validation_report)
        
        return validation_report
    
    def _get_validation_checks(self, config: DeploymentValidationConfig) -> List[tuple]:
        """Get validation checks based on validation level."""
        
        checks = []
        
        # Basic checks (all levels)
        checks.extend([
            (self.dependency_validator.validate_python_version, ()),
            (self.dependency_validator.validate_required_packages, ()),
            (self.config_validator.validate_model_configurations, (config.target_config,)),
            (self.integration_validator.validate_service_connectivity, (config,))
        ])
        
        # Standard level and above
        if config.validation_level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE, ValidationLevel.PRODUCTION]:
            checks.extend([
                (self.dependency_validator.validate_system_resources, ()),
                (self.config_validator.validate_api_keys, (config.target_config,)),
                (self.performance_validator.validate_response_time, (config,))
            ])
        
        # Comprehensive level and above
        if config.validation_level in [ValidationLevel.COMPREHENSIVE, ValidationLevel.PRODUCTION]:
            checks.extend([
                (self.performance_validator.validate_load_capacity, (config,))
            ])
        
        return checks
    
    def _generate_overall_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate overall recommendations based on validation results."""
        
        recommendations = []
        failed_results = [r for r in results if r.status == ValidationStatus.FAILED]
        warning_results = [r for r in results if r.status == ValidationStatus.WARNING]
        
        if failed_results:
            recommendations.append("❌ Critical issues must be resolved before deployment")
            for result in failed_results:
                if result.recommendations:
                    recommendations.extend(result.recommendations)
        
        if warning_results:
            recommendations.append("⚠️ Address warnings for optimal performance")
            for result in warning_results:
                if result.recommendations:
                    recommendations.extend(result.recommendations)
        
        if not failed_results and not warning_results:
            recommendations.append("✅ System is ready for deployment")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _generate_next_steps(self, overall_status: str, results: List[ValidationResult]) -> List[str]:
        """Generate next steps based on validation results."""
        
        if overall_status == "PASSED":
            return [
                "✅ Proceed with deployment",
                "🔄 Consider setting up monitoring and alerting",
                "📊 Plan for load testing in production environment"
            ]
        elif overall_status == "WARNING":
            return [
                "⚠️ Review and address warnings",
                "🔄 Re-run validation after addressing issues",
                "📋 Document known limitations for production team"
            ]
        else:  # FAILED
            return [
                "❌ Do not proceed with deployment",
                "🔧 Fix critical issues identified in validation",
                "🔍 Re-run validation after fixes",
                "📞 Consult with development team if issues persist"
            ]
    
    def _print_validation_summary(self, report: Dict[str, Any]):
        """Print a formatted validation summary."""
        
        print(f"\\n📋 VALIDATION SUMMARY")
        print("=" * 50)
        print(f"Configuration: {report['config_name']}")
        print(f"Validation Level: {report['validation_level']}")
        print(f"Overall Status: {report['overall_status']}")
        print(f"Duration: {report['duration_ms']:.0f}ms")
        print(f"\\nResults Summary:")
        summary = report['summary']
        print(f"  ✅ Passed: {summary['passed_checks']}")
        print(f"  ⚠️  Warnings: {summary['warning_checks']}")
        print(f"  ❌ Failed: {summary['failed_checks']}")
        print(f"  ⏭️  Skipped: {summary['skipped_checks']}")
        
        if report['recommendations']:
            print(f"\\n📝 Recommendations:")
            for rec in report['recommendations'][:5]:  # Show top 5
                print(f"  • {rec}")
        
        if report['next_steps']:
            print(f"\\n🚀 Next Steps:")
            for step in report['next_steps']:
                print(f"  • {step}")
        
        print("=" * 50)
    
    def create_validation_config(self, name: str, target_config: Dict[str, str],
                               validation_level: ValidationLevel = ValidationLevel.STANDARD,
                               **kwargs) -> DeploymentValidationConfig:
        """Create a validation configuration."""
        
        config_id = f"validation_{int(time.time())}"
        
        config = DeploymentValidationConfig(
            config_id=config_id,
            name=name,
            validation_level=validation_level,
            target_config=target_config,
            **kwargs
        )
        
        return config
    
    def get_validation_history_summary(self) -> Dict[str, Any]:
        """Get summary of validation history."""
        
        if not self.validation_history:
            return {"total_validations": 0}
        
        total = len(self.validation_history)
        passed = len([v for v in self.validation_history if v["overall_status"] == "PASSED"])
        warnings = len([v for v in self.validation_history if v["overall_status"] == "WARNING"])
        failed = len([v for v in self.validation_history if v["overall_status"] == "FAILED"])
        
        return {
            "total_validations": total,
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
            "success_rate": passed / total if total > 0 else 0,
            "recent_validations": [
                {
                    "validation_id": v["validation_id"],
                    "config_name": v["config_name"],
                    "overall_status": v["overall_status"],
                    "start_time": v["start_time"],
                    "duration_ms": v["duration_ms"]
                }
                for v in sorted(self.validation_history, key=lambda x: x["start_time"], reverse=True)[:5]
            ]
        }
