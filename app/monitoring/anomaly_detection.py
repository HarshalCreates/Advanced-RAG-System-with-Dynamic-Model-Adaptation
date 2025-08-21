"""ML-based anomaly detection for system monitoring."""
from __future__ import annotations

import json
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone
from pathlib import Path
from collections import deque
import statistics


class AnomalyType(Enum):
    """Types of anomalies."""
    STATISTICAL_OUTLIER = "statistical_outlier"
    TREND_ANOMALY = "trend_anomaly"
    SEASONAL_ANOMALY = "seasonal_anomaly"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    USAGE_PATTERN_ANOMALY = "usage_pattern_anomaly"


class AnomalySeverity(Enum):
    """Anomaly severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection."""
    anomaly_id: str
    metric_name: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    score: float  # 0-1, higher = more anomalous
    detected_value: float
    expected_range: Tuple[float, float]
    timestamp: float
    description: str
    recommendations: List[str]
    metadata: Dict[str, Any]


@dataclass
class MetricBaseline:
    """Baseline statistics for a metric."""
    metric_name: str
    mean: float
    std: float
    min_value: float
    max_value: float
    median: float
    percentile_95: float
    percentile_99: float
    trend_slope: float
    seasonal_patterns: Dict[str, float]
    last_updated: float


class StatisticalDetector:
    """Statistical anomaly detection using z-score and IQR methods."""
    
    def __init__(self, window_size: int = 100, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.metric_data: Dict[str, deque] = {}
        self.baselines: Dict[str, MetricBaseline] = {}
    
    def add_data_point(self, metric_name: str, value: float, timestamp: float):
        """Add a new data point for a metric."""
        if metric_name not in self.metric_data:
            self.metric_data[metric_name] = deque(maxlen=self.window_size)
        
        self.metric_data[metric_name].append((timestamp, value))
        
        # Update baseline if we have enough data
        if len(self.metric_data[metric_name]) >= 10:
            self._update_baseline(metric_name)
    
    def _update_baseline(self, metric_name: str):
        """Update baseline statistics for a metric."""
        data_points = list(self.metric_data[metric_name])
        values = [point[1] for point in data_points]
        
        if len(values) < 2:
            return
        
        # Calculate basic statistics
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        min_val = min(values)
        max_val = max(values)
        median = statistics.median(values)
        
        # Calculate percentiles
        sorted_values = sorted(values)
        p95_idx = int(0.95 * len(sorted_values))
        p99_idx = int(0.99 * len(sorted_values))
        percentile_95 = sorted_values[p95_idx] if p95_idx < len(sorted_values) else max_val
        percentile_99 = sorted_values[p99_idx] if p99_idx < len(sorted_values) else max_val
        
        # Calculate trend (simple linear regression slope)
        if len(data_points) >= 3:
            timestamps = [point[0] for point in data_points]
            trend_slope = self._calculate_trend(timestamps, values)
        else:
            trend_slope = 0.0
        
        # Detect seasonal patterns (simplified - hourly patterns)
        seasonal_patterns = self._detect_seasonal_patterns(data_points)
        
        self.baselines[metric_name] = MetricBaseline(
            metric_name=metric_name,
            mean=mean,
            std=std,
            min_value=min_val,
            max_value=max_val,
            median=median,
            percentile_95=percentile_95,
            percentile_99=percentile_99,
            trend_slope=trend_slope,
            seasonal_patterns=seasonal_patterns,
            last_updated=time.time()
        )
    
    def _calculate_trend(self, timestamps: List[float], values: List[float]) -> float:
        """Calculate linear trend slope."""
        n = len(timestamps)
        if n < 2:
            return 0.0
        
        # Normalize timestamps to avoid numerical issues
        t_norm = [(t - timestamps[0]) for t in timestamps]
        
        # Calculate slope using least squares
        sum_x = sum(t_norm)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(t_norm, values))
        sum_x2 = sum(x * x for x in t_norm)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _detect_seasonal_patterns(self, data_points: List[Tuple[float, float]]) -> Dict[str, float]:
        """Detect simple seasonal patterns (hour of day)."""
        if len(data_points) < 24:  # Need at least 24 hours of data
            return {}
        
        hourly_values: Dict[int, List[float]] = {}
        
        for timestamp, value in data_points:
            hour = datetime.fromtimestamp(timestamp).hour
            if hour not in hourly_values:
                hourly_values[hour] = []
            hourly_values[hour].append(value)
        
        # Calculate average for each hour
        hourly_averages = {}
        for hour, values in hourly_values.items():
            if len(values) >= 2:  # Need multiple samples
                hourly_averages[f"hour_{hour}"] = statistics.mean(values)
        
        return hourly_averages
    
    def detect_anomalies(self, metric_name: str, value: float, timestamp: float) -> List[AnomalyDetectionResult]:
        """Detect anomalies in a new data point."""
        anomalies = []
        
        baseline = self.baselines.get(metric_name)
        if not baseline:
            return anomalies  # No baseline yet
        
        # Z-score based detection
        if baseline.std > 0:
            z_score = abs((value - baseline.mean) / baseline.std)
            if z_score > self.z_threshold:
                severity = self._calculate_severity(z_score, self.z_threshold)
                
                anomaly = AnomalyDetectionResult(
                    anomaly_id=f"stat_{metric_name}_{int(timestamp)}",
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    severity=severity,
                    score=min(z_score / (self.z_threshold * 2), 1.0),
                    detected_value=value,
                    expected_range=(baseline.mean - 2*baseline.std, baseline.mean + 2*baseline.std),
                    timestamp=timestamp,
                    description=f"Statistical outlier: z-score = {z_score:.2f}",
                    recommendations=self._get_outlier_recommendations(metric_name, value, baseline),
                    metadata={"z_score": z_score, "baseline_mean": baseline.mean, "baseline_std": baseline.std}
                )
                anomalies.append(anomaly)
        
        # IQR-based detection
        q1 = baseline.median - (baseline.percentile_95 - baseline.median) / 2
        q3 = baseline.percentile_95
        iqr = q3 - q1
        
        if iqr > 0:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            if value < lower_bound or value > upper_bound:
                severity = AnomalySeverity.MEDIUM if value > baseline.percentile_99 else AnomalySeverity.LOW
                
                anomaly = AnomalyDetectionResult(
                    anomaly_id=f"iqr_{metric_name}_{int(timestamp)}",
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    severity=severity,
                    score=min(abs(value - baseline.median) / (3 * iqr), 1.0),
                    detected_value=value,
                    expected_range=(lower_bound, upper_bound),
                    timestamp=timestamp,
                    description=f"IQR outlier: value outside expected range",
                    recommendations=self._get_outlier_recommendations(metric_name, value, baseline),
                    metadata={"iqr": iqr, "q1": q1, "q3": q3}
                )
                anomalies.append(anomaly)
        
        # Trend anomaly detection
        if len(self.metric_data[metric_name]) >= 5:
            recent_trend = self._calculate_recent_trend(metric_name)
            if abs(recent_trend - baseline.trend_slope) > 2 * abs(baseline.trend_slope) and abs(recent_trend) > 0.1:
                severity = AnomalySeverity.HIGH if abs(recent_trend) > abs(baseline.trend_slope) * 3 else AnomalySeverity.MEDIUM
                
                anomaly = AnomalyDetectionResult(
                    anomaly_id=f"trend_{metric_name}_{int(timestamp)}",
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.TREND_ANOMALY,
                    severity=severity,
                    score=min(abs(recent_trend - baseline.trend_slope) / max(abs(baseline.trend_slope), 0.1), 1.0),
                    detected_value=value,
                    expected_range=(baseline.mean - baseline.std, baseline.mean + baseline.std),
                    timestamp=timestamp,
                    description=f"Trend anomaly: recent trend {recent_trend:.4f} vs baseline {baseline.trend_slope:.4f}",
                    recommendations=self._get_trend_recommendations(metric_name, recent_trend),
                    metadata={"recent_trend": recent_trend, "baseline_trend": baseline.trend_slope}
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _calculate_recent_trend(self, metric_name: str, window: int = 10) -> float:
        """Calculate trend for recent data points."""
        data_points = list(self.metric_data[metric_name])
        recent_points = data_points[-window:] if len(data_points) >= window else data_points
        
        if len(recent_points) < 2:
            return 0.0
        
        timestamps = [point[0] for point in recent_points]
        values = [point[1] for point in recent_points]
        
        return self._calculate_trend(timestamps, values)
    
    def _calculate_severity(self, z_score: float, threshold: float) -> AnomalySeverity:
        """Calculate anomaly severity based on z-score."""
        if z_score > threshold * 3:
            return AnomalySeverity.CRITICAL
        elif z_score > threshold * 2:
            return AnomalySeverity.HIGH
        elif z_score > threshold * 1.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    def _get_outlier_recommendations(self, metric_name: str, value: float, baseline: MetricBaseline) -> List[str]:
        """Get recommendations for statistical outliers."""
        recommendations = []
        
        if "response_time" in metric_name.lower():
            if value > baseline.mean:
                recommendations.extend([
                    "Check system resources (CPU, memory)",
                    "Review recent deployment changes",
                    "Analyze query complexity"
                ])
        elif "error" in metric_name.lower():
            if value > baseline.mean:
                recommendations.extend([
                    "Check application logs for errors",
                    "Verify external service dependencies",
                    "Review recent configuration changes"
                ])
        elif "memory" in metric_name.lower():
            if value > baseline.mean:
                recommendations.extend([
                    "Check for memory leaks",
                    "Review caching strategies",
                    "Consider scaling resources"
                ])
        
        return recommendations
    
    def _get_trend_recommendations(self, metric_name: str, trend: float) -> List[str]:
        """Get recommendations for trend anomalies."""
        recommendations = []
        
        if trend > 0:  # Increasing trend
            if "response_time" in metric_name.lower():
                recommendations.extend([
                    "Performance degradation detected",
                    "Consider scaling resources",
                    "Review system optimization"
                ])
            elif "error" in metric_name.lower():
                recommendations.extend([
                    "Error rate increasing",
                    "Investigate root cause",
                    "Check system stability"
                ])
        else:  # Decreasing trend
            if "response_time" in metric_name.lower():
                recommendations.append("Performance improvement detected")
            elif "error" in metric_name.lower():
                recommendations.append("Error rate decreasing - system stabilizing")
        
        return recommendations


class BehavioralDetector:
    """Detects behavioral anomalies in user patterns."""
    
    def __init__(self):
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.global_patterns: Dict[str, Any] = {}
    
    def update_user_profile(self, user_id: str, activity_data: Dict[str, Any]):
        """Update user behavioral profile."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'query_frequency': deque(maxlen=100),
                'query_lengths': deque(maxlen=100),
                'session_durations': deque(maxlen=50),
                'preferred_topics': {},
                'activity_times': deque(maxlen=200),
                'last_updated': time.time()
            }
        
        profile = self.user_profiles[user_id]
        
        # Update activity data
        if 'query_length' in activity_data:
            profile['query_lengths'].append(activity_data['query_length'])
        
        if 'query_topic' in activity_data:
            topic = activity_data['query_topic']
            profile['preferred_topics'][topic] = profile['preferred_topics'].get(topic, 0) + 1
        
        if 'session_duration' in activity_data:
            profile['session_durations'].append(activity_data['session_duration'])
        
        profile['activity_times'].append(time.time())
        profile['last_updated'] = time.time()
    
    def detect_behavioral_anomalies(self, user_id: str, current_activity: Dict[str, Any]) -> List[AnomalyDetectionResult]:
        """Detect behavioral anomalies for a user."""
        anomalies = []
        
        if user_id not in self.user_profiles:
            return anomalies  # No profile yet
        
        profile = self.user_profiles[user_id]
        current_time = time.time()
        
        # Check query frequency anomaly
        recent_activities = [t for t in profile['activity_times'] if current_time - t < 3600]  # Last hour
        if len(recent_activities) > 100:  # More than 100 queries per hour
            anomaly = AnomalyDetectionResult(
                anomaly_id=f"behavior_freq_{user_id}_{int(current_time)}",
                metric_name="query_frequency",
                anomaly_type=AnomalyType.BEHAVIORAL_ANOMALY,
                severity=AnomalySeverity.HIGH,
                score=min(len(recent_activities) / 100, 1.0),
                detected_value=len(recent_activities),
                expected_range=(0, 50),
                timestamp=current_time,
                description=f"Unusual query frequency: {len(recent_activities)} queries in last hour",
                recommendations=["Monitor for automated/bot behavior", "Check rate limiting"],
                metadata={"user_id": user_id, "hourly_queries": len(recent_activities)}
            )
            anomalies.append(anomaly)
        
        # Check query length anomaly
        if 'query_length' in current_activity and profile['query_lengths']:
            avg_length = statistics.mean(profile['query_lengths'])
            current_length = current_activity['query_length']
            
            if current_length > avg_length * 5:  # 5x longer than usual
                anomaly = AnomalyDetectionResult(
                    anomaly_id=f"behavior_length_{user_id}_{int(current_time)}",
                    metric_name="query_length",
                    anomaly_type=AnomalyType.BEHAVIORAL_ANOMALY,
                    severity=AnomalySeverity.MEDIUM,
                    score=min(current_length / (avg_length * 5), 1.0),
                    detected_value=current_length,
                    expected_range=(0, avg_length * 2),
                    timestamp=current_time,
                    description=f"Unusually long query: {current_length} chars vs avg {avg_length:.0f}",
                    recommendations=["Check for injection attempts", "Validate query complexity"],
                    metadata={"user_id": user_id, "avg_length": avg_length}
                )
                anomalies.append(anomaly)
        
        return anomalies


class AnomalyDetectionManager:
    """Main anomaly detection system."""
    
    def __init__(self, storage_path: str = "./data/anomaly_detection"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.anomalies_file = self.storage_path / "detected_anomalies.json"
        self.baselines_file = self.storage_path / "metric_baselines.json"
        
        # Detectors
        self.statistical_detector = StatisticalDetector()
        self.behavioral_detector = BehavioralDetector()
        
        # Storage
        self.detected_anomalies: List[AnomalyDetectionResult] = []
        
        self.load_data()
    
    def load_data(self):
        """Load anomaly detection data."""
        # Load detected anomalies
        if self.anomalies_file.exists():
            try:
                with open(self.anomalies_file, 'r') as f:
                    data = json.load(f)
                    for anomaly_data in data[-500:]:  # Keep last 500 anomalies
                        anomaly_data['anomaly_type'] = AnomalyType(anomaly_data['anomaly_type'])
                        anomaly_data['severity'] = AnomalySeverity(anomaly_data['severity'])
                        anomaly = AnomalyDetectionResult(**anomaly_data)
                        self.detected_anomalies.append(anomaly)
            except Exception as e:
                print(f"Failed to load anomalies: {e}")
        
        # Load baselines
        if self.baselines_file.exists():
            try:
                with open(self.baselines_file, 'r') as f:
                    data = json.load(f)
                    for baseline_data in data:
                        baseline = MetricBaseline(**baseline_data)
                        self.statistical_detector.baselines[baseline.metric_name] = baseline
            except Exception as e:
                print(f"Failed to load baselines: {e}")
    
    def save_data(self):
        """Save anomaly detection data."""
        try:
            # Save anomalies
            anomalies_data = []
            for anomaly in self.detected_anomalies[-500:]:  # Keep last 500
                data = asdict(anomaly)
                data['anomaly_type'] = anomaly.anomaly_type.value
                data['severity'] = anomaly.severity.value
                anomalies_data.append(data)
            
            with open(self.anomalies_file, 'w') as f:
                json.dump(anomalies_data, f, indent=2)
            
            # Save baselines
            baselines_data = []
            for baseline in self.statistical_detector.baselines.values():
                baselines_data.append(asdict(baseline))
            
            with open(self.baselines_file, 'w') as f:
                json.dump(baselines_data, f, indent=2)
                
        except Exception as e:
            print(f"Failed to save anomaly detection data: {e}")
    
    def analyze_metric(self, metric_name: str, value: float, timestamp: float = None, 
                      user_id: str = None, metadata: Dict[str, Any] = None) -> List[AnomalyDetectionResult]:
        """Analyze a metric value for anomalies."""
        if timestamp is None:
            timestamp = time.time()
        
        anomalies = []
        
        # Add data point to statistical detector
        self.statistical_detector.add_data_point(metric_name, value, timestamp)
        
        # Run statistical anomaly detection
        stat_anomalies = self.statistical_detector.detect_anomalies(metric_name, value, timestamp)
        anomalies.extend(stat_anomalies)
        
        # Run behavioral anomaly detection if user_id provided
        if user_id and metadata:
            behavioral_anomalies = self.behavioral_detector.detect_behavioral_anomalies(user_id, metadata)
            anomalies.extend(behavioral_anomalies)
        
        # Store detected anomalies
        for anomaly in anomalies:
            self.detected_anomalies.append(anomaly)
        
        # Save data periodically
        if len(self.detected_anomalies) % 10 == 0:
            self.save_data()
        
        return anomalies
    
    def update_user_behavior(self, user_id: str, activity_data: Dict[str, Any]):
        """Update user behavioral profile."""
        self.behavioral_detector.update_user_profile(user_id, activity_data)
    
    def get_anomaly_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get anomaly detection summary."""
        cutoff_time = time.time() - (hours * 3600)
        recent_anomalies = [a for a in self.detected_anomalies if a.timestamp > cutoff_time]
        
        # Group by type
        type_counts = {}
        for anomaly_type in AnomalyType:
            type_counts[anomaly_type.value] = len([a for a in recent_anomalies if a.anomaly_type == anomaly_type])
        
        # Group by severity
        severity_counts = {}
        for severity in AnomalySeverity:
            severity_counts[severity.value] = len([a for a in recent_anomalies if a.severity == severity])
        
        # Top metrics with anomalies
        metric_counts = {}
        for anomaly in recent_anomalies:
            metric_counts[anomaly.metric_name] = metric_counts.get(anomaly.metric_name, 0) + 1
        
        top_metrics = sorted(metric_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "time_period_hours": hours,
            "total_anomalies": len(recent_anomalies),
            "anomaly_types": type_counts,
            "severity_levels": severity_counts,
            "top_anomalous_metrics": top_metrics,
            "baseline_metrics": len(self.statistical_detector.baselines),
            "user_profiles": len(self.behavioral_detector.user_profiles),
            "recent_critical_anomalies": [
                {
                    "anomaly_id": a.anomaly_id,
                    "metric": a.metric_name,
                    "type": a.anomaly_type.value,
                    "severity": a.severity.value,
                    "score": a.score,
                    "description": a.description,
                    "timestamp": a.timestamp
                }
                for a in sorted(recent_anomalies, key=lambda x: x.score, reverse=True)[:5]
                if a.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]
            ]
        }
    
    def get_anomaly_by_id(self, anomaly_id: str) -> Optional[AnomalyDetectionResult]:
        """Get anomaly by ID."""
        return next((a for a in self.detected_anomalies if a.anomaly_id == anomaly_id), None)
    
    def get_metric_baseline(self, metric_name: str) -> Optional[MetricBaseline]:
        """Get baseline for a metric."""
        return self.statistical_detector.baselines.get(metric_name)
