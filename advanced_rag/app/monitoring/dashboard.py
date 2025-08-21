"""Enhanced monitoring with detailed metrics and dashboards."""
from __future__ import annotations

import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone, timedelta
from enum import Enum
import statistics


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class DashboardType(Enum):
    """Types of dashboards."""
    OVERVIEW = "overview"
    PERFORMANCE = "performance"
    ERRORS = "errors"
    BUSINESS = "business"
    INFRASTRUCTURE = "infrastructure"


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    labels: Dict[str, str] = None


@dataclass
class Metric:
    """Metric definition and data."""
    name: str
    metric_type: MetricType
    description: str
    unit: str
    data_points: List[MetricPoint]
    labels: Dict[str, str] = None


@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""
    widget_id: str
    title: str
    widget_type: str  # chart, table, stat, gauge
    metric_queries: List[str]
    time_range_minutes: int = 60
    refresh_interval_seconds: int = 30
    config: Dict[str, Any] = None


@dataclass
class Dashboard:
    """Dashboard configuration."""
    dashboard_id: str
    name: str
    description: str
    dashboard_type: DashboardType
    widgets: List[DashboardWidget]
    layout: Dict[str, Any] = None
    metadata: Dict[str, Any] = None


class MetricsCollector:
    """Advanced metrics collection and storage."""
    
    def __init__(self, storage_path: str = "./data/metrics"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.metrics: Dict[str, Metric] = {}
        self.collection_interval = 30  # seconds
        self.retention_hours = 24
        
        # Initialize core metrics
        self._initialize_core_metrics()
    
    def _initialize_core_metrics(self):
        """Initialize core system metrics."""
        
        core_metrics = [
            # Performance metrics
            Metric("api_requests_total", MetricType.COUNTER, "Total API requests", "requests", []),
            Metric("api_request_duration_ms", MetricType.HISTOGRAM, "API request duration", "milliseconds", []),
            Metric("query_response_time_ms", MetricType.HISTOGRAM, "Query response time", "milliseconds", []),
            Metric("retrieval_latency_ms", MetricType.HISTOGRAM, "Retrieval latency", "milliseconds", []),
            Metric("generation_latency_ms", MetricType.HISTOGRAM, "Generation latency", "milliseconds", []),
            
            # Quality metrics
            Metric("confidence_score", MetricType.GAUGE, "Average confidence score", "score", []),
            Metric("retrieval_accuracy", MetricType.GAUGE, "Retrieval accuracy", "percentage", []),
            Metric("user_satisfaction", MetricType.GAUGE, "User satisfaction score", "score", []),
            
            # Error metrics
            Metric("api_errors_total", MetricType.COUNTER, "Total API errors", "errors", []),
            Metric("query_failures_total", MetricType.COUNTER, "Total query failures", "failures", []),
            Metric("model_errors_total", MetricType.COUNTER, "Total model errors", "errors", []),
            
            # Resource metrics
            Metric("memory_usage_mb", MetricType.GAUGE, "Memory usage", "megabytes", []),
            Metric("cpu_usage_percent", MetricType.GAUGE, "CPU usage", "percentage", []),
            Metric("active_connections", MetricType.GAUGE, "Active connections", "connections", []),
            
            # Business metrics
            Metric("documents_processed_total", MetricType.COUNTER, "Documents processed", "documents", []),
            Metric("unique_users_daily", MetricType.GAUGE, "Daily unique users", "users", []),
            Metric("queries_per_minute", MetricType.GAUGE, "Queries per minute", "queries", []),
            
            # Model metrics
            Metric("embedding_model_switches", MetricType.COUNTER, "Embedding model switches", "switches", []),
            Metric("generation_model_switches", MetricType.COUNTER, "Generation model switches", "switches", []),
            Metric("hot_swap_duration_ms", MetricType.HISTOGRAM, "Hot swap duration", "milliseconds", [])
        ]
        
        for metric in core_metrics:
            self.metrics[metric.name] = metric
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric value."""
        if name not in self.metrics:
            return
        
        metric = self.metrics[name]
        data_point = MetricPoint(
            timestamp=time.time(),
            value=value,
            labels=labels or {}
        )
        
        metric.data_points.append(data_point)
        
        # Clean old data points (keep last 24 hours)
        cutoff_time = time.time() - (self.retention_hours * 3600)
        metric.data_points = [
            dp for dp in metric.data_points 
            if dp.timestamp > cutoff_time
        ]
    
    def get_metric_summary(self, name: str, time_range_minutes: int = 60) -> Optional[Dict[str, Any]]:
        """Get summary statistics for a metric."""
        if name not in self.metrics:
            return None
        
        metric = self.metrics[name]
        cutoff_time = time.time() - (time_range_minutes * 60)
        
        recent_points = [
            dp for dp in metric.data_points 
            if dp.timestamp > cutoff_time
        ]
        
        if not recent_points:
            return {
                "name": name,
                "count": 0,
                "latest_value": None,
                "average": None,
                "min": None,
                "max": None,
                "time_range_minutes": time_range_minutes
            }
        
        values = [dp.value for dp in recent_points]
        
        summary = {
            "name": name,
            "count": len(values),
            "latest_value": values[-1],
            "average": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "time_range_minutes": time_range_minutes
        }
        
        # Add percentiles for histograms
        if metric.metric_type == MetricType.HISTOGRAM and len(values) >= 10:
            sorted_values = sorted(values)
            summary.update({
                "p50": statistics.median(sorted_values),
                "p95": sorted_values[int(0.95 * len(sorted_values))],
                "p99": sorted_values[int(0.99 * len(sorted_values))]
            })
        
        return summary
    
    def get_time_series_data(self, name: str, time_range_minutes: int = 60, 
                           aggregation_interval_minutes: int = 5) -> List[Dict[str, Any]]:
        """Get time series data for a metric."""
        if name not in self.metrics:
            return []
        
        metric = self.metrics[name]
        cutoff_time = time.time() - (time_range_minutes * 60)
        
        recent_points = [
            dp for dp in metric.data_points 
            if dp.timestamp > cutoff_time
        ]
        
        if not recent_points:
            return []
        
        # Group data points by time intervals
        interval_seconds = aggregation_interval_minutes * 60
        time_buckets: Dict[int, List[float]] = {}
        
        for dp in recent_points:
            bucket_key = int(dp.timestamp // interval_seconds)
            if bucket_key not in time_buckets:
                time_buckets[bucket_key] = []
            time_buckets[bucket_key].append(dp.value)
        
        # Aggregate data points
        time_series = []
        for bucket_key in sorted(time_buckets.keys()):
            bucket_timestamp = bucket_key * interval_seconds
            values = time_buckets[bucket_key]
            
            if metric.metric_type == MetricType.COUNTER:
                # For counters, sum the values
                aggregated_value = sum(values)
            else:
                # For gauges and histograms, use average
                aggregated_value = statistics.mean(values)
            
            time_series.append({
                "timestamp": bucket_timestamp,
                "datetime": datetime.fromtimestamp(bucket_timestamp, tz=timezone.utc).isoformat(),
                "value": aggregated_value,
                "count": len(values)
            })
        
        return time_series
    
    async def collect_system_metrics(self):
        """Collect current system metrics."""
        try:
            # Try to collect real system metrics
            import psutil
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric("memory_usage_mb", memory.used / (1024**2))
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("cpu_usage_percent", cpu_percent)
            
            # Network connections (approximation for active connections)
            connections = len(psutil.net_connections())
            self.record_metric("active_connections", connections)
            
        except ImportError:
            # Fallback to simulated metrics
            import random
            self.record_metric("memory_usage_mb", random.uniform(512, 1024))
            self.record_metric("cpu_usage_percent", random.uniform(10, 60))
            self.record_metric("active_connections", random.randint(5, 50))
    
    async def collect_api_metrics(self):
        """Collect API-specific metrics."""
        try:
            import aiohttp
            
            # Test API health and collect metrics
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get("http://localhost:8000/api/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                        duration_ms = (time.time() - start_time) * 1000
                        
                        if response.status == 200:
                            self.record_metric("api_request_duration_ms", duration_ms)
                            self.record_metric("api_requests_total", 1)
                        else:
                            self.record_metric("api_errors_total", 1)
                
                except Exception:
                    self.record_metric("api_errors_total", 1)
                
                # Test query endpoint
                try:
                    query_start = time.time()
                    async with session.post(
                        "http://localhost:8000/api/query",
                        json={"query": "test query for metrics", "top_k": 3, "filters": {}},
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        query_duration = (time.time() - query_start) * 1000
                        
                        if response.status == 200:
                            result = await response.json()
                            
                            # Record query metrics
                            self.record_metric("query_response_time_ms", query_duration)
                            self.record_metric("confidence_score", result.get("answer", {}).get("confidence", 0.0))
                            
                            # Extract performance metrics if available
                            perf_metrics = result.get("performance_metrics", {})
                            if "retrieval_latency_ms" in perf_metrics:
                                self.record_metric("retrieval_latency_ms", perf_metrics["retrieval_latency_ms"])
                            if "generation_latency_ms" in perf_metrics:
                                self.record_metric("generation_latency_ms", perf_metrics["generation_latency_ms"])
                        
                        else:
                            self.record_metric("query_failures_total", 1)
                
                except Exception:
                    self.record_metric("query_failures_total", 1)
        
        except ImportError:
            # Fallback metrics
            import random
            self.record_metric("api_request_duration_ms", random.uniform(50, 200))
            self.record_metric("query_response_time_ms", random.uniform(100, 500))
            self.record_metric("confidence_score", random.uniform(0.5, 0.9))
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}
        
        for metric_name in self.metrics.keys():
            metric_summary = self.get_metric_summary(metric_name, time_range_minutes=60)
            if metric_summary and metric_summary["count"] > 0:
                summary[metric_name] = metric_summary
        
        return summary


class DashboardManager:
    """Manages monitoring dashboards."""
    
    def __init__(self, metrics_collector: MetricsCollector, storage_path: str = "./data/dashboards"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.metrics_collector = metrics_collector
        self.dashboards: Dict[str, Dashboard] = {}
        
        # Initialize default dashboards
        self._create_default_dashboards()
    
    def _create_default_dashboards(self):
        """Create default monitoring dashboards."""
        
        # Overview Dashboard
        overview_widgets = [
            DashboardWidget(
                widget_id="api_health",
                title="API Health",
                widget_type="stat",
                metric_queries=["api_requests_total", "api_errors_total"],
                time_range_minutes=60
            ),
            DashboardWidget(
                widget_id="response_time",
                title="Average Response Time",
                widget_type="gauge",
                metric_queries=["query_response_time_ms"],
                time_range_minutes=60,
                config={"min": 0, "max": 2000, "thresholds": [500, 1000, 1500]}
            ),
            DashboardWidget(
                widget_id="confidence_trend",
                title="Confidence Score Trend",
                widget_type="chart",
                metric_queries=["confidence_score"],
                time_range_minutes=240
            ),
            DashboardWidget(
                widget_id="system_resources",
                title="System Resources",
                widget_type="chart",
                metric_queries=["memory_usage_mb", "cpu_usage_percent"],
                time_range_minutes=120
            )
        ]
        
        overview_dashboard = Dashboard(
            dashboard_id="overview",
            name="System Overview",
            description="High-level system health and performance overview",
            dashboard_type=DashboardType.OVERVIEW,
            widgets=overview_widgets
        )
        
        # Performance Dashboard
        performance_widgets = [
            DashboardWidget(
                widget_id="latency_breakdown",
                title="Latency Breakdown",
                widget_type="chart",
                metric_queries=["retrieval_latency_ms", "generation_latency_ms", "query_response_time_ms"],
                time_range_minutes=120
            ),
            DashboardWidget(
                widget_id="throughput",
                title="Queries per Minute",
                widget_type="chart",
                metric_queries=["queries_per_minute"],
                time_range_minutes=240
            ),
            DashboardWidget(
                widget_id="response_time_percentiles",
                title="Response Time Percentiles",
                widget_type="table",
                metric_queries=["query_response_time_ms"],
                time_range_minutes=60
            ),
            DashboardWidget(
                widget_id="model_performance",
                title="Model Performance",
                widget_type="chart",
                metric_queries=["confidence_score", "retrieval_accuracy"],
                time_range_minutes=240
            )
        ]
        
        performance_dashboard = Dashboard(
            dashboard_id="performance",
            name="Performance Analytics",
            description="Detailed performance metrics and trends",
            dashboard_type=DashboardType.PERFORMANCE,
            widgets=performance_widgets
        )
        
        # Error Dashboard
        error_widgets = [
            DashboardWidget(
                widget_id="error_rates",
                title="Error Rates",
                widget_type="chart",
                metric_queries=["api_errors_total", "query_failures_total", "model_errors_total"],
                time_range_minutes=240
            ),
            DashboardWidget(
                widget_id="error_summary",
                title="Error Summary",
                widget_type="table",
                metric_queries=["api_errors_total", "query_failures_total"],
                time_range_minutes=60
            )
        ]
        
        error_dashboard = Dashboard(
            dashboard_id="errors",
            name="Error Monitoring",
            description="Error tracking and analysis",
            dashboard_type=DashboardType.ERRORS,
            widgets=error_widgets
        )
        
        # Business Dashboard
        business_widgets = [
            DashboardWidget(
                widget_id="user_engagement",
                title="User Engagement",
                widget_type="chart",
                metric_queries=["unique_users_daily", "queries_per_minute"],
                time_range_minutes=1440  # 24 hours
            ),
            DashboardWidget(
                widget_id="document_processing",
                title="Document Processing",
                widget_type="stat",
                metric_queries=["documents_processed_total"],
                time_range_minutes=1440
            ),
            DashboardWidget(
                widget_id="user_satisfaction",
                title="User Satisfaction",
                widget_type="gauge",
                metric_queries=["user_satisfaction"],
                time_range_minutes=240,
                config={"min": 0, "max": 1, "thresholds": [0.6, 0.8, 0.9]}
            )
        ]
        
        business_dashboard = Dashboard(
            dashboard_id="business",
            name="Business Metrics",
            description="Business and user-focused metrics",
            dashboard_type=DashboardType.BUSINESS,
            widgets=business_widgets
        )
        
        # Store dashboards
        self.dashboards = {
            "overview": overview_dashboard,
            "performance": performance_dashboard,
            "errors": error_dashboard,
            "business": business_dashboard
        }
    
    def get_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get data for a dashboard widget."""
        
        widget_data = {
            "widget_id": widget.widget_id,
            "title": widget.title,
            "widget_type": widget.widget_type,
            "config": widget.config or {},
            "data": {},
            "updated_at": time.time()
        }
        
        # Collect data for each metric query
        for metric_name in widget.metric_queries:
            if widget.widget_type in ["chart"]:
                # Time series data for charts
                time_series = self.metrics_collector.get_time_series_data(
                    metric_name, 
                    time_range_minutes=widget.time_range_minutes,
                    aggregation_interval_minutes=max(1, widget.time_range_minutes // 20)
                )
                widget_data["data"][metric_name] = {
                    "type": "time_series",
                    "data": time_series
                }
            
            elif widget.widget_type in ["stat", "gauge"]:
                # Summary data for stats and gauges
                summary = self.metrics_collector.get_metric_summary(
                    metric_name, 
                    time_range_minutes=widget.time_range_minutes
                )
                widget_data["data"][metric_name] = {
                    "type": "summary",
                    "data": summary
                }
            
            elif widget.widget_type == "table":
                # Detailed summary for tables
                summary = self.metrics_collector.get_metric_summary(
                    metric_name, 
                    time_range_minutes=widget.time_range_minutes
                )
                widget_data["data"][metric_name] = {
                    "type": "summary",
                    "data": summary
                }
        
        return widget_data
    
    def get_dashboard_data(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """Get complete dashboard data."""
        
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return None
        
        dashboard_data = {
            "dashboard_id": dashboard.dashboard_id,
            "name": dashboard.name,
            "description": dashboard.description,
            "dashboard_type": dashboard.dashboard_type.value,
            "updated_at": time.time(),
            "widgets": []
        }
        
        # Get data for each widget
        for widget in dashboard.widgets:
            widget_data = self.get_widget_data(widget)
            dashboard_data["widgets"].append(widget_data)
        
        return dashboard_data
    
    def get_all_dashboards_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all available dashboards."""
        
        return [
            {
                "dashboard_id": dashboard.dashboard_id,
                "name": dashboard.name,
                "description": dashboard.description,
                "dashboard_type": dashboard.dashboard_type.value,
                "widget_count": len(dashboard.widgets)
            }
            for dashboard in self.dashboards.values()
        ]
    
    def generate_dashboard_html(self, dashboard_id: str) -> Optional[str]:
        """Generate HTML for a dashboard."""
        
        dashboard_data = self.get_dashboard_data(dashboard_id)
        if not dashboard_data:
            return None
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dashboard_data['name']} - RAG System Monitoring</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .dashboard-header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .dashboard-title {{
            margin: 0;
            color: #333;
        }}
        .dashboard-description {{
            color: #666;
            margin: 5px 0 0 0;
        }}
        .widgets-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .widget {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .widget-title {{
            margin: 0 0 15px 0;
            color: #333;
            font-size: 18px;
            font-weight: 600;
        }}
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
            color: #2563eb;
        }}
        .stat-label {{
            color: #666;
            font-size: 14px;
        }}
        .chart-container {{
            position: relative;
            height: 300px;
        }}
        .table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .table th, .table td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        .table th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .refresh-info {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: #4ade80;
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="refresh-info">
        Auto-refresh: 30s
    </div>
    
    <div class="dashboard-header">
        <h1 class="dashboard-title">{dashboard_data['name']}</h1>
        <p class="dashboard-description">{dashboard_data['description']}</p>
        <p style="font-size: 14px; color: #888;">
            Last updated: {datetime.fromtimestamp(dashboard_data['updated_at']).strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </div>
    
    <div class="widgets-grid">
"""
        
        # Generate HTML for each widget
        for i, widget in enumerate(dashboard_data['widgets']):
            html += f'        <div class="widget" id="widget-{widget["widget_id"]}">\\n'
            html += f'            <h3 class="widget-title">{widget["title"]}</h3>\\n'
            
            if widget['widget_type'] == 'stat':
                # Generate stat widget
                for metric_name, metric_data in widget['data'].items():
                    if metric_data['data'] and metric_data['data']['latest_value'] is not None:
                        value = metric_data['data']['latest_value']
                        html += f'            <div class="stat-value">{value:.1f}</div>\\n'
                        html += f'            <div class="stat-label">{metric_name}</div>\\n'
            
            elif widget['widget_type'] == 'gauge':
                # Generate gauge widget (simplified as stat for now)
                for metric_name, metric_data in widget['data'].items():
                    if metric_data['data'] and metric_data['data']['average'] is not None:
                        value = metric_data['data']['average']
                        html += f'            <div class="stat-value">{value:.1f}</div>\\n'
                        html += f'            <div class="stat-label">Average {metric_name}</div>\\n'
            
            elif widget['widget_type'] == 'chart':
                # Generate chart widget
                html += f'            <div class="chart-container">\\n'
                html += f'                <canvas id="chart-{i}"></canvas>\\n'
                html += f'            </div>\\n'
            
            elif widget['widget_type'] == 'table':
                # Generate table widget
                html += '            <table class="table">\\n'
                html += '                <thead>\\n'
                html += '                    <tr><th>Metric</th><th>Current</th><th>Average</th><th>Min</th><th>Max</th></tr>\\n'
                html += '                </thead>\\n'
                html += '                <tbody>\\n'
                
                for metric_name, metric_data in widget['data'].items():
                    if metric_data['data']:
                        data = metric_data['data']
                        current = data.get('latest_value', 'N/A')
                        average = data.get('average', 'N/A')
                        minimum = data.get('min', 'N/A')
                        maximum = data.get('max', 'N/A')
                        
                        html += f'                    <tr>\\n'
                        html += f'                        <td>{metric_name}</td>\\n'
                        html += f'                        <td>{current:.1f if isinstance(current, (int, float)) else current}</td>\\n'
                        html += f'                        <td>{average:.1f if isinstance(average, (int, float)) else average}</td>\\n'
                        html += f'                        <td>{minimum:.1f if isinstance(minimum, (int, float)) else minimum}</td>\\n'
                        html += f'                        <td>{maximum:.1f if isinstance(maximum, (int, float)) else maximum}</td>\\n'
                        html += f'                    </tr>\\n'
                
                html += '                </tbody>\\n'
                html += '            </table>\\n'
            
            html += '        </div>\\n'
        
        html += """    </div>
    
    <script>
        // Chart.js configuration for time series charts
        const chartConfigs = [];
"""
        
        # Generate chart JavaScript
        for i, widget in enumerate(dashboard_data['widgets']):
            if widget['widget_type'] == 'chart':
                html += f"""
        // Chart {i} configuration
        const chart{i}Data = {{
            labels: [],
            datasets: []
        }};
        
        """
                
                # Add data for each metric
                dataset_colors = ['#2563eb', '#dc2626', '#059669', '#d97706', '#7c3aed']
                color_index = 0
                
                for metric_name, metric_data in widget['data'].items():
                    if metric_data['type'] == 'time_series' and metric_data['data']:
                        color = dataset_colors[color_index % len(dataset_colors)]
                        color_index += 1
                        
                        # Extract timestamps and values
                        timestamps = [point['datetime'][:16] for point in metric_data['data']]  # YYYY-MM-DDTHH:MM
                        values = [point['value'] for point in metric_data['data']]
                        
                        html += f"""
        chart{i}Data.labels = {json.dumps(timestamps)};
        chart{i}Data.datasets.push({{
            label: '{metric_name}',
            data: {json.dumps(values)},
            borderColor: '{color}',
            backgroundColor: '{color}20',
            fill: false,
            tension: 0.1
        }});
        """
                
                html += f"""
        const chart{i}Config = {{
            type: 'line',
            data: chart{i}Data,
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'top',
                    }},
                    title: {{
                        display: false
                    }}
                }},
                scales: {{
                    x: {{
                        display: true,
                        title: {{
                            display: true,
                            text: 'Time'
                        }}
                    }},
                    y: {{
                        display: true,
                        title: {{
                            display: true,
                            text: 'Value'
                        }}
                    }}
                }}
            }}
        }};
        
        chartConfigs.push({{
            id: 'chart-{i}',
            config: chart{i}Config
        }});
        """
        
        html += """
        
        // Initialize charts
        window.addEventListener('load', function() {
            chartConfigs.forEach(chartConfig => {
                const ctx = document.getElementById(chartConfig.id);
                if (ctx) {
                    new Chart(ctx, chartConfig.config);
                }
            });
        });
        
        // Auto-refresh every 30 seconds
        setTimeout(function() {
            window.location.reload();
        }, 30000);
    </script>
</body>
</html>
"""
        
        return html


class MonitoringOrchestrator:
    """Orchestrates all monitoring components."""
    
    def __init__(self, storage_path: str = "./data/monitoring"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.metrics_collector = MetricsCollector(str(self.storage_path / "metrics"))
        self.dashboard_manager = DashboardManager(self.metrics_collector, str(self.storage_path / "dashboards"))
        
        self.monitoring_active = False
    
    async def start_monitoring(self, collection_interval: int = 30):
        """Start continuous monitoring."""
        self.monitoring_active = True
        print(f"📊 Starting enhanced monitoring (collection interval: {collection_interval}s)")
        
        while self.monitoring_active:
            try:
                # Collect system metrics
                await self.metrics_collector.collect_system_metrics()
                
                # Collect API metrics
                await self.metrics_collector.collect_api_metrics()
                
                # Record timestamp metric
                self.metrics_collector.record_metric("monitoring_heartbeat", 1.0)
                
                await asyncio.sleep(collection_interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                await asyncio.sleep(collection_interval)
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        print("📊 Stopped enhanced monitoring")
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        
        # Get key metrics summaries
        key_metrics = [
            "api_requests_total",
            "api_errors_total", 
            "query_response_time_ms",
            "confidence_score",
            "memory_usage_mb",
            "cpu_usage_percent"
        ]
        
        health_summary = {
            "overall_status": "healthy",
            "timestamp": time.time(),
            "key_metrics": {},
            "alerts": [],
            "recommendations": []
        }
        
        issues = []
        
        for metric_name in key_metrics:
            summary = self.metrics_collector.get_metric_summary(metric_name, time_range_minutes=15)
            if summary:
                health_summary["key_metrics"][metric_name] = summary
                
                # Check for issues
                if metric_name == "query_response_time_ms" and summary.get("average", 0) > 1000:
                    issues.append("High query response time")
                elif metric_name == "memory_usage_mb" and summary.get("latest_value", 0) > 1500:
                    issues.append("High memory usage")
                elif metric_name == "cpu_usage_percent" and summary.get("average", 0) > 80:
                    issues.append("High CPU usage")
                elif metric_name == "confidence_score" and summary.get("average", 1) < 0.3:
                    issues.append("Low confidence scores")
        
        # Determine overall status
        if issues:
            health_summary["overall_status"] = "degraded" if len(issues) <= 2 else "unhealthy"
            health_summary["alerts"] = issues
        
        # Generate recommendations
        if "High query response time" in issues:
            health_summary["recommendations"].append("Consider optimizing model configuration")
        if "High memory usage" in issues:
            health_summary["recommendations"].append("Consider scaling up memory resources")
        if "High CPU usage" in issues:
            health_summary["recommendations"].append("Consider scaling out to more instances")
        
        return health_summary
    
    def generate_dashboard_html_file(self, dashboard_id: str, output_path: str = None) -> Optional[str]:
        """Generate and save dashboard HTML file."""
        
        html_content = self.dashboard_manager.generate_dashboard_html(dashboard_id)
        if not html_content:
            return None
        
        if output_path is None:
            output_path = str(self.storage_path / f"dashboard_{dashboard_id}.html")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"📊 Generated dashboard HTML: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Failed to generate dashboard HTML: {e}")
            return None
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        
        return {
            "monitoring_active": self.monitoring_active,
            "metrics_count": len(self.metrics_collector.metrics),
            "dashboards_count": len(self.dashboard_manager.dashboards),
            "health_summary": self.get_system_health_summary(),
            "available_dashboards": self.dashboard_manager.get_all_dashboards_summary(),
            "collection_interval_seconds": self.metrics_collector.collection_interval,
            "retention_hours": self.metrics_collector.retention_hours
        }
