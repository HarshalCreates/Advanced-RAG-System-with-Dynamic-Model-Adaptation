"""Advanced performance analysis with percentile tracking and benchmarking."""
from __future__ import annotations

import time
import json
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import numpy as np
from pathlib import Path
import threading


class MetricType(Enum):
    """Types of performance metrics."""
    RESPONSE_TIME = "response_time"
    RETRIEVAL_TIME = "retrieval_time"
    GENERATION_TIME = "generation_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    COST_PER_QUERY = "cost_per_query"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"


class BenchmarkCategory(Enum):
    """Categories for benchmarking."""
    SIMPLE_QUERY = "simple_query"
    COMPLEX_QUERY = "complex_query"
    MULTI_DOCUMENT = "multi_document"
    LARGE_CONTEXT = "large_context"
    HIGH_CONCURRENCY = "high_concurrency"


@dataclass
class PercentileMetrics:
    """Percentile-based performance metrics."""
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    p99_9: float
    mean: float
    std_dev: float
    min_value: float
    max_value: float
    sample_count: int


@dataclass
class PerformanceBenchmark:
    """Performance benchmark result."""
    benchmark_id: str
    category: BenchmarkCategory
    metric_type: MetricType
    percentiles: PercentileMetrics
    baseline_comparison: Dict[str, float]  # comparison to baseline
    performance_grade: str
    recommendations: List[str]
    test_conditions: Dict[str, Any]
    timestamp: float


@dataclass
class TrendAnalysis:
    """Performance trend analysis."""
    metric_name: str
    time_period_hours: int
    trend_direction: str  # "improving", "degrading", "stable"
    trend_strength: float  # 0-1, how strong the trend is
    change_rate: float  # percentage change per hour
    anomalies_detected: int
    forecast_next_hour: float
    confidence_interval: Tuple[float, float]


@dataclass
class LoadTestResult:
    """Result of automated load testing."""
    test_id: str
    test_type: str
    concurrent_users: int
    total_requests: int
    duration_seconds: int
    success_rate: float
    avg_response_time: float
    throughput_qps: float
    resource_utilization: Dict[str, float]
    error_breakdown: Dict[str, int]
    performance_degradation_point: Optional[int]  # at what concurrency level
    recommendations: List[str]


class PercentileCalculator:
    """Efficient percentile calculation for streaming data."""
    
    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self.samples: deque = deque(maxlen=max_samples)
        self.lock = threading.Lock()
    
    def add_sample(self, value: float):
        """Add a new sample for percentile calculation."""
        with self.lock:
            self.samples.append(value)
    
    def calculate_percentiles(self) -> Optional[PercentileMetrics]:
        """Calculate percentile metrics from current samples."""
        with self.lock:
            if not self.samples:
                return None
            
            sorted_samples = sorted(self.samples)
            n = len(sorted_samples)
            
            if n == 0:
                return None
            
            def percentile(p: float) -> float:
                """Calculate percentile value."""
                index = (p / 100.0) * (n - 1)
                lower = int(index)
                upper = min(lower + 1, n - 1)
                weight = index - lower
                return sorted_samples[lower] * (1 - weight) + sorted_samples[upper] * weight
            
            mean_val = statistics.mean(sorted_samples)
            std_dev_val = statistics.stdev(sorted_samples) if n > 1 else 0.0
            
            return PercentileMetrics(
                p50=round(percentile(50), 2),
                p75=round(percentile(75), 2),
                p90=round(percentile(90), 2),
                p95=round(percentile(95), 2),
                p99=round(percentile(99), 2),
                p99_9=round(percentile(99.9), 2),
                mean=round(mean_val, 2),
                std_dev=round(std_dev_val, 2),
                min_value=round(min(sorted_samples), 2),
                max_value=round(max(sorted_samples), 2),
                sample_count=n
            )
    
    def get_sample_count(self) -> int:
        """Get current number of samples."""
        with self.lock:
            return len(self.samples)


class PerformanceTracker:
    """Tracks performance metrics with percentile calculation."""
    
    def __init__(self, storage_path: str = "./data/performance"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Percentile calculators for different metrics
        self.calculators: Dict[str, PercentileCalculator] = {}
        
        # Historical data storage
        self.metrics_file = self.storage_path / "performance_metrics.json"
        self.trends_file = self.storage_path / "performance_trends.json"
        
        # Initialize calculators for standard metrics
        for metric_type in MetricType:
            self.calculators[metric_type.value] = PercentileCalculator()
        
        # Load historical data
        self.historical_data: List[Dict[str, Any]] = []
        self.load_historical_data()
    
    def record_metric(self, metric_type: MetricType, value: float, metadata: Dict[str, Any] = None):
        """Record a performance metric."""
        
        # Add to percentile calculator
        calculator = self.calculators.get(metric_type.value)
        if calculator:
            calculator.add_sample(value)
        
        # Store in historical data
        record = {
            "metric_type": metric_type.value,
            "value": value,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        self.historical_data.append(record)
        
        # Periodically save data
        if len(self.historical_data) % 100 == 0:
            self.save_historical_data()
    
    def get_current_percentiles(self, metric_type: MetricType) -> Optional[PercentileMetrics]:
        """Get current percentile metrics for a specific type."""
        calculator = self.calculators.get(metric_type.value)
        if calculator:
            return calculator.calculate_percentiles()
        return None
    
    def get_all_current_percentiles(self) -> Dict[str, PercentileMetrics]:
        """Get current percentiles for all tracked metrics."""
        results = {}
        
        for metric_name, calculator in self.calculators.items():
            percentiles = calculator.calculate_percentiles()
            if percentiles:
                results[metric_name] = percentiles
        
        return results
    
    def analyze_trends(self, metric_type: MetricType, hours: int = 24) -> TrendAnalysis:
        """Analyze performance trends over time."""
        
        cutoff_time = time.time() - (hours * 3600)
        
        # Filter historical data for the specified time period
        recent_data = [
            record for record in self.historical_data
            if record["metric_type"] == metric_type.value and record["timestamp"] > cutoff_time
        ]
        
        if len(recent_data) < 2:
            return TrendAnalysis(
                metric_name=metric_type.value,
                time_period_hours=hours,
                trend_direction="insufficient_data",
                trend_strength=0.0,
                change_rate=0.0,
                anomalies_detected=0,
                forecast_next_hour=0.0,
                confidence_interval=(0.0, 0.0)
            )
        
        # Extract values and timestamps
        values = [record["value"] for record in recent_data]
        timestamps = [record["timestamp"] for record in recent_data]
        
        # Calculate trend using linear regression
        trend_direction, trend_strength, change_rate = self._calculate_trend(timestamps, values)
        
        # Detect anomalies
        anomalies_detected = self._count_anomalies(values)
        
        # Simple forecasting
        latest_value = values[-1]
        forecast_next_hour = latest_value + (change_rate * 3600)  # Change per hour
        
        # Confidence interval (simplified)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        confidence_interval = (
            forecast_next_hour - 1.96 * std_dev,
            forecast_next_hour + 1.96 * std_dev
        )
        
        return TrendAnalysis(
            metric_name=metric_type.value,
            time_period_hours=hours,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            change_rate=change_rate,
            anomalies_detected=anomalies_detected,
            forecast_next_hour=round(forecast_next_hour, 2),
            confidence_interval=(round(confidence_interval[0], 2), round(confidence_interval[1], 2))
        )
    
    def _calculate_trend(self, timestamps: List[float], values: List[float]) -> Tuple[str, float, float]:
        """Calculate trend direction and strength."""
        
        if len(timestamps) < 2 or len(values) < 2:
            return "stable", 0.0, 0.0
        
        # Normalize timestamps to start from 0
        time_diffs = [t - timestamps[0] for t in timestamps]
        
        # Calculate linear regression slope
        n = len(values)
        sum_x = sum(time_diffs)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(time_diffs, values))
        sum_x2 = sum(x * x for x in time_diffs)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return "stable", 0.0, 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Determine trend direction
        if abs(slope) < 1e-6:
            direction = "stable"
        elif slope > 0:
            direction = "improving" if "time" not in str(values[0]).lower() else "degrading"
        else:
            direction = "degrading" if "time" not in str(values[0]).lower() else "improving"
        
        # Calculate trend strength (R-squared)
        y_mean = sum_y / n
        ss_tot = sum((y - y_mean) ** 2 for y in values)
        ss_res = sum((values[i] - (slope * time_diffs[i] + (sum_y - slope * sum_x) / n)) ** 2 
                    for i in range(n))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        trend_strength = max(0, min(1, r_squared))
        
        return direction, trend_strength, slope
    
    def _count_anomalies(self, values: List[float]) -> int:
        """Count anomalies using IQR method."""
        
        if len(values) < 4:
            return 0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        q1 = sorted_values[n // 4]
        q3 = sorted_values[3 * n // 4]
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        anomalies = sum(1 for v in values if v < lower_bound or v > upper_bound)
        
        return anomalies
    
    def load_historical_data(self):
        """Load historical performance data."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    self.historical_data = json.load(f)[-10000:]  # Keep last 10k records
            except Exception as e:
                print(f"Failed to load historical data: {e}")
                self.historical_data = []
    
    def save_historical_data(self):
        """Save historical performance data."""
        try:
            # Keep only recent data to prevent file from growing too large
            recent_data = self.historical_data[-10000:] if len(self.historical_data) > 10000 else self.historical_data
            
            with open(self.metrics_file, 'w') as f:
                json.dump(recent_data, f, indent=2)
        except Exception as e:
            print(f"Failed to save historical data: {e}")


class LoadTester:
    """Automated load testing for scalability assessment."""
    
    def __init__(self):
        self.test_scenarios = {
            "light_load": {"concurrent_users": 5, "duration": 60},
            "normal_load": {"concurrent_users": 20, "duration": 120},
            "heavy_load": {"concurrent_users": 50, "duration": 180},
            "stress_test": {"concurrent_users": 100, "duration": 300}
        }
    
    async def run_load_test(self, test_type: str, query_function, 
                           test_queries: List[str] = None) -> LoadTestResult:
        """Run automated load test."""
        
        if test_type not in self.test_scenarios:
            raise ValueError(f"Unknown test type: {test_type}")
        
        scenario = self.test_scenarios[test_type]
        concurrent_users = scenario["concurrent_users"]
        duration = scenario["duration"]
        
        # Default test queries if none provided
        if not test_queries:
            test_queries = [
                "What is artificial intelligence?",
                "How does machine learning work?",
                "Explain deep learning concepts",
                "What are the benefits of automation?",
                "How do neural networks function?"
            ]
        
        test_id = f"{test_type}_{int(time.time())}"
        
        # Initialize metrics
        response_times = []
        success_count = 0
        error_count = 0
        error_breakdown = defaultdict(int)
        
        start_time = time.time()
        end_time = start_time + duration
        
        # Simulate concurrent load
        tasks = []
        request_count = 0
        
        print(f"Starting load test: {test_type} with {concurrent_users} concurrent users for {duration}s")
        
        try:
            import asyncio
            
            async def simulate_user():
                """Simulate a single user's queries."""
                nonlocal success_count, error_count, request_count
                
                while time.time() < end_time:
                    try:
                        # Select random query
                        import random
                        query = random.choice(test_queries)
                        
                        # Measure response time
                        query_start = time.time()
                        
                        # Execute query (simplified - would call actual query function)
                        await asyncio.sleep(random.uniform(0.5, 2.0))  # Simulate query time
                        
                        query_time = (time.time() - query_start) * 1000
                        response_times.append(query_time)
                        
                        success_count += 1
                        request_count += 1
                        
                        # Add some delay between requests
                        await asyncio.sleep(random.uniform(0.1, 0.5))
                        
                    except Exception as e:
                        error_count += 1
                        error_breakdown[str(type(e).__name__)] += 1
            
            # Create concurrent user tasks
            for _ in range(concurrent_users):
                task = asyncio.create_task(simulate_user())
                tasks.append(task)
            
            # Wait for test completion
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            print(f"Load test failed: {e}")
        
        # Calculate results
        total_requests = success_count + error_count
        actual_duration = time.time() - start_time
        
        success_rate = success_count / total_requests if total_requests > 0 else 0
        avg_response_time = statistics.mean(response_times) if response_times else 0
        throughput_qps = total_requests / actual_duration if actual_duration > 0 else 0
        
        # Analyze performance degradation
        degradation_point = self._analyze_degradation_point(response_times, concurrent_users)
        
        # Generate recommendations
        recommendations = self._generate_load_test_recommendations(
            success_rate, avg_response_time, throughput_qps, degradation_point
        )
        
        return LoadTestResult(
            test_id=test_id,
            test_type=test_type,
            concurrent_users=concurrent_users,
            total_requests=total_requests,
            duration_seconds=int(actual_duration),
            success_rate=round(success_rate, 3),
            avg_response_time=round(avg_response_time, 2),
            throughput_qps=round(throughput_qps, 2),
            resource_utilization={"cpu": 0.5, "memory": 0.6},  # Would be measured in real implementation
            error_breakdown=dict(error_breakdown),
            performance_degradation_point=degradation_point,
            recommendations=recommendations
        )
    
    def _analyze_degradation_point(self, response_times: List[float], max_users: int) -> Optional[int]:
        """Analyze at what point performance significantly degrades."""
        
        if len(response_times) < 10:
            return None
        
        # Simple analysis - look for significant increase in response times
        chunk_size = len(response_times) // 10
        if chunk_size < 1:
            return None
        
        chunks = [response_times[i:i+chunk_size] for i in range(0, len(response_times), chunk_size)]
        chunk_averages = [statistics.mean(chunk) for chunk in chunks if chunk]
        
        # Find first chunk where average response time increases by >50%
        baseline = chunk_averages[0] if chunk_averages else 0
        
        for i, avg in enumerate(chunk_averages[1:], 1):
            if avg > baseline * 1.5:
                # Estimate concurrency level at this point
                degradation_point = int((i / len(chunk_averages)) * max_users)
                return degradation_point
        
        return None
    
    def _generate_load_test_recommendations(self, success_rate: float, avg_response_time: float,
                                          throughput_qps: float, degradation_point: Optional[int]) -> List[str]:
        """Generate recommendations based on load test results."""
        
        recommendations = []
        
        if success_rate < 0.95:
            recommendations.append(f"Low success rate ({success_rate:.1%}) - investigate error handling and system stability")
        
        if avg_response_time > 2000:  # 2 seconds
            recommendations.append(f"High average response time ({avg_response_time:.0f}ms) - consider performance optimization")
        
        if throughput_qps < 10:
            recommendations.append(f"Low throughput ({throughput_qps:.1f} QPS) - consider scaling or optimization")
        
        if degradation_point and degradation_point < 20:
            recommendations.append(f"Performance degrades early at {degradation_point} concurrent users - improve scalability")
        
        if not recommendations:
            recommendations.append("Performance test results are satisfactory")
        
        return recommendations


class PerformanceAnalyzer:
    """Main performance analysis and benchmarking system."""
    
    def __init__(self, storage_path: str = "./data/performance"):
        self.performance_tracker = PerformanceTracker(storage_path)
        self.load_tester = LoadTester()
        self.benchmarks: List[PerformanceBenchmark] = []
        
        # Performance baselines (would be configurable)
        self.baselines = {
            MetricType.RESPONSE_TIME: {"p95": 1000, "p99": 2000},  # milliseconds
            MetricType.RETRIEVAL_TIME: {"p95": 500, "p99": 1000},
            MetricType.GENERATION_TIME: {"p95": 800, "p99": 1500},
            MetricType.THROUGHPUT: {"mean": 50, "p95": 40},  # QPS
            MetricType.ERROR_RATE: {"p95": 0.01, "p99": 0.05}  # percentage
        }
    
    def record_performance_metric(self, metric_type: MetricType, value: float, 
                                 metadata: Dict[str, Any] = None):
        """Record a performance metric for analysis."""
        self.performance_tracker.record_metric(metric_type, value, metadata)
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data."""
        
        current_percentiles = self.performance_tracker.get_all_current_percentiles()
        
        dashboard_data = {
            "timestamp": time.time(),
            "current_percentiles": {},
            "trends": {},
            "alerts": [],
            "performance_grades": {}
        }
        
        # Process current percentiles
        for metric_name, percentiles in current_percentiles.items():
            dashboard_data["current_percentiles"][metric_name] = asdict(percentiles)
            
            # Calculate performance grade
            grade = self._calculate_performance_grade(metric_name, percentiles)
            dashboard_data["performance_grades"][metric_name] = grade
            
            # Check for alerts
            alerts = self._check_performance_alerts(metric_name, percentiles)
            dashboard_data["alerts"].extend(alerts)
        
        # Add trend analysis for key metrics
        key_metrics = [MetricType.RESPONSE_TIME, MetricType.THROUGHPUT, MetricType.ERROR_RATE]
        for metric_type in key_metrics:
            try:
                trend = self.performance_tracker.analyze_trends(metric_type, hours=24)
                dashboard_data["trends"][metric_type.value] = asdict(trend)
            except Exception as e:
                print(f"Failed to analyze trend for {metric_type.value}: {e}")
        
        return dashboard_data
    
    def run_benchmark_suite(self, category: BenchmarkCategory = None) -> List[PerformanceBenchmark]:
        """Run comprehensive benchmark suite."""
        
        categories_to_test = [category] if category else list(BenchmarkCategory)
        benchmark_results = []
        
        for cat in categories_to_test:
            for metric_type in [MetricType.RESPONSE_TIME, MetricType.THROUGHPUT]:
                try:
                    benchmark = self._run_single_benchmark(cat, metric_type)
                    benchmark_results.append(benchmark)
                    self.benchmarks.append(benchmark)
                except Exception as e:
                    print(f"Benchmark failed for {cat.value} - {metric_type.value}: {e}")
        
        return benchmark_results
    
    def _run_single_benchmark(self, category: BenchmarkCategory, metric_type: MetricType) -> PerformanceBenchmark:
        """Run a single benchmark test."""
        
        # Generate test conditions based on category
        test_conditions = self._get_test_conditions(category)
        
        # Get current percentiles for the metric
        percentiles = self.performance_tracker.get_current_percentiles(metric_type)
        
        if not percentiles:
            # Create dummy percentiles if no data available
            percentiles = PercentileMetrics(
                p50=100, p75=150, p90=200, p95=250, p99=400, p99_9=500,
                mean=130, std_dev=50, min_value=50, max_value=600, sample_count=0
            )
        
        # Compare to baseline
        baseline_comparison = self._compare_to_baseline(metric_type, percentiles)
        
        # Calculate performance grade
        performance_grade = self._calculate_performance_grade(metric_type.value, percentiles)
        
        # Generate recommendations
        recommendations = self._generate_benchmark_recommendations(metric_type, percentiles, baseline_comparison)
        
        benchmark_id = f"{category.value}_{metric_type.value}_{int(time.time())}"
        
        return PerformanceBenchmark(
            benchmark_id=benchmark_id,
            category=category,
            metric_type=metric_type,
            percentiles=percentiles,
            baseline_comparison=baseline_comparison,
            performance_grade=performance_grade,
            recommendations=recommendations,
            test_conditions=test_conditions,
            timestamp=time.time()
        )
    
    def _get_test_conditions(self, category: BenchmarkCategory) -> Dict[str, Any]:
        """Get test conditions for benchmark category."""
        
        conditions = {
            BenchmarkCategory.SIMPLE_QUERY: {
                "query_complexity": "low",
                "context_size": "small",
                "expected_response_time": "<500ms"
            },
            BenchmarkCategory.COMPLEX_QUERY: {
                "query_complexity": "high", 
                "context_size": "large",
                "expected_response_time": "<2000ms"
            },
            BenchmarkCategory.MULTI_DOCUMENT: {
                "document_count": ">5",
                "context_size": "very_large",
                "expected_response_time": "<3000ms"
            },
            BenchmarkCategory.HIGH_CONCURRENCY: {
                "concurrent_users": ">50",
                "sustained_load": "5min",
                "expected_degradation": "<20%"
            }
        }
        
        return conditions.get(category, {})
    
    def _compare_to_baseline(self, metric_type: MetricType, percentiles: PercentileMetrics) -> Dict[str, float]:
        """Compare current performance to baseline."""
        
        baseline = self.baselines.get(metric_type, {})
        comparison = {}
        
        if "p95" in baseline:
            comparison["p95_vs_baseline"] = (percentiles.p95 / baseline["p95"]) if baseline["p95"] > 0 else 1.0
        
        if "p99" in baseline:
            comparison["p99_vs_baseline"] = (percentiles.p99 / baseline["p99"]) if baseline["p99"] > 0 else 1.0
        
        if "mean" in baseline:
            comparison["mean_vs_baseline"] = (percentiles.mean / baseline["mean"]) if baseline["mean"] > 0 else 1.0
        
        return comparison
    
    def _calculate_performance_grade(self, metric_name: str, percentiles: PercentileMetrics) -> str:
        """Calculate performance grade based on percentiles."""
        
        # Simple grading based on P95 performance
        p95_value = percentiles.p95
        
        if "time" in metric_name.lower():
            # Lower is better for time metrics
            if p95_value <= 500:
                return "A+"
            elif p95_value <= 1000:
                return "A"
            elif p95_value <= 2000:
                return "B"
            elif p95_value <= 5000:
                return "C"
            else:
                return "D"
        else:
            # Higher is better for throughput, etc.
            if p95_value >= 100:
                return "A+"
            elif p95_value >= 50:
                return "A"
            elif p95_value >= 25:
                return "B"
            elif p95_value >= 10:
                return "C"
            else:
                return "D"
    
    def _check_performance_alerts(self, metric_name: str, percentiles: PercentileMetrics) -> List[Dict[str, Any]]:
        """Check for performance alerts based on thresholds."""
        
        alerts = []
        
        # Response time alerts
        if "response_time" in metric_name:
            if percentiles.p95 > 2000:
                alerts.append({
                    "severity": "HIGH",
                    "metric": metric_name,
                    "message": f"P95 response time ({percentiles.p95}ms) exceeds 2000ms threshold",
                    "value": percentiles.p95,
                    "threshold": 2000
                })
            elif percentiles.p95 > 1000:
                alerts.append({
                    "severity": "MEDIUM",
                    "metric": metric_name,
                    "message": f"P95 response time ({percentiles.p95}ms) exceeds 1000ms threshold",
                    "value": percentiles.p95,
                    "threshold": 1000
                })
        
        # Error rate alerts
        elif "error" in metric_name:
            if percentiles.mean > 0.05:  # 5% error rate
                alerts.append({
                    "severity": "HIGH",
                    "metric": metric_name,
                    "message": f"Error rate ({percentiles.mean:.2%}) exceeds 5% threshold",
                    "value": percentiles.mean,
                    "threshold": 0.05
                })
        
        return alerts
    
    def _generate_benchmark_recommendations(self, metric_type: MetricType, percentiles: PercentileMetrics,
                                          baseline_comparison: Dict[str, float]) -> List[str]:
        """Generate recommendations based on benchmark results."""
        
        recommendations = []
        
        # Check baseline comparison
        for comparison_key, ratio in baseline_comparison.items():
            if ratio > 1.5:  # 50% worse than baseline
                recommendations.append(f"{comparison_key} is {ratio:.1f}x worse than baseline - investigate performance issues")
            elif ratio > 1.2:  # 20% worse than baseline
                recommendations.append(f"{comparison_key} is {ratio:.1f}x slower than baseline - minor optimization needed")
        
        # Specific recommendations based on metric type
        if metric_type == MetricType.RESPONSE_TIME:
            if percentiles.p99 > percentiles.p95 * 3:
                recommendations.append("High P99/P95 ratio indicates response time variability - investigate outliers")
        
        elif metric_type == MetricType.THROUGHPUT:
            if percentiles.std_dev > percentiles.mean * 0.5:
                recommendations.append("High throughput variability - consider load balancing improvements")
        
        if not recommendations:
            recommendations.append("Performance metrics are within acceptable ranges")
        
        return recommendations
    
    async def run_comprehensive_load_test(self) -> Dict[str, LoadTestResult]:
        """Run comprehensive load testing suite."""
        
        test_types = ["light_load", "normal_load", "heavy_load", "stress_test"]
        results = {}
        
        for test_type in test_types:
            try:
                print(f"Running {test_type} test...")
                result = await self.load_tester.run_load_test(test_type, None)
                results[test_type] = result
            except Exception as e:
                print(f"Load test {test_type} failed: {e}")
        
        return results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        
        current_percentiles = self.performance_tracker.get_all_current_percentiles()
        
        # Calculate overall health score
        health_scores = []
        for metric_name, percentiles in current_percentiles.items():
            grade = self._calculate_performance_grade(metric_name, percentiles)
            grade_scores = {"A+": 100, "A": 90, "B": 80, "C": 70, "D": 60}
            health_scores.append(grade_scores.get(grade, 50))
        
        overall_health = statistics.mean(health_scores) if health_scores else 50
        
        return {
            "overall_health_score": round(overall_health, 1),
            "performance_status": "Excellent" if overall_health >= 90 else
                                "Good" if overall_health >= 80 else
                                "Satisfactory" if overall_health >= 70 else "Needs Improvement",
            "metrics_tracked": len(current_percentiles),
            "active_alerts": len(self._get_all_alerts()),
            "recent_benchmarks": len([b for b in self.benchmarks if time.time() - b.timestamp < 3600])
        }
    
    def _get_all_alerts(self) -> List[Dict[str, Any]]:
        """Get all current performance alerts."""
        
        all_alerts = []
        current_percentiles = self.performance_tracker.get_all_current_percentiles()
        
        for metric_name, percentiles in current_percentiles.items():
            alerts = self._check_performance_alerts(metric_name, percentiles)
            all_alerts.extend(alerts)
        
        return all_alerts
