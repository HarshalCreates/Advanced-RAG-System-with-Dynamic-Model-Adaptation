"""Comprehensive alerting system with notifications and escalation."""
from __future__ import annotations

import json
import time
import asyncio
import smtplib
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
from enum import Enum
from email.mime.text import MIMEText as MimeText
from email.mime.multipart import MIMEMultipart as MimeMultipart

try:
    import aiohttp
    import requests
    NOTIFICATION_AVAILABLE = True
except ImportError:
    NOTIFICATION_AVAILABLE = False


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status types."""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class NotificationType(Enum):
    """Notification channel types."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    CONSOLE = "console"


@dataclass
class AlertCondition:
    """Defines when an alert should trigger."""
    metric_name: str
    operator: str  # >, <, >=, <=, ==, !=
    threshold_value: float
    duration_seconds: int  # How long condition must persist
    description: str


@dataclass
class NotificationChannel:
    """Configuration for notification delivery."""
    channel_id: str
    channel_type: NotificationType
    config: Dict[str, Any]  # Channel-specific configuration
    enabled: bool = True


@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    description: str
    conditions: List[AlertCondition]
    severity: AlertSeverity
    notification_channels: List[str]  # Channel IDs
    cooldown_seconds: int = 300  # 5 minutes
    auto_resolve: bool = True
    metadata: Dict[str, Any] = None


@dataclass
class Alert:
    """Active alert instance."""
    alert_id: str
    rule_id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    triggered_at: float
    resolved_at: Optional[float]
    acknowledged_at: Optional[float]
    acknowledged_by: Optional[str]
    escalated_at: Optional[float]
    trigger_values: Dict[str, float]
    metadata: Dict[str, Any]


class MetricsCollector:
    """Collects system metrics for alerting."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.metrics_cache: Dict[str, float] = {}
        self.last_collection_time = 0.0
    
    async def collect_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        current_time = time.time()
        
        # Collect only every 30 seconds to avoid overload
        if current_time - self.last_collection_time < 30:
            return self.metrics_cache
        
        metrics = {}
        
        try:
            if NOTIFICATION_AVAILABLE:
                async with aiohttp.ClientSession() as session:
                    # Health endpoint
                    async with session.get(f"{self.api_base_url}/api/health", timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            metrics['api_health'] = 1.0
                        else:
                            metrics['api_health'] = 0.0
                    
                    # Test query for performance metrics
                    start_time = time.time()
                    try:
                        async with session.post(
                            f"{self.api_base_url}/api/query",
                            json={"query": "test metric collection", "top_k": 1, "filters": {}},
                            timeout=aiohttp.ClientTimeout(total=15)
                        ) as response:
                            query_time = (time.time() - start_time) * 1000
                            metrics['response_time_ms'] = query_time
                            
                            if response.status == 200:
                                result = await response.json()
                                metrics['query_success_rate'] = 1.0
                                metrics['confidence_score'] = result.get("answer", {}).get("confidence", 0.0)
                                metrics['retrieval_latency_ms'] = result.get("performance_metrics", {}).get("retrieval_latency_ms", 0.0)
                                metrics['generation_latency_ms'] = result.get("performance_metrics", {}).get("generation_latency_ms", 0.0)
                            else:
                                metrics['query_success_rate'] = 0.0
                                metrics['confidence_score'] = 0.0
                    except Exception:
                        metrics['query_success_rate'] = 0.0
                        metrics['response_time_ms'] = 9999.0
                        metrics['confidence_score'] = 0.0
                    
                    # Prometheus metrics if available
                    try:
                        async with session.get(f"{self.api_base_url}/metrics", timeout=aiohttp.ClientTimeout(total=5)) as response:
                            if response.status == 200:
                                metrics_text = await response.text()
                                # Parse simple metrics from Prometheus format
                                for line in metrics_text.split('\\n'):
                                    if line.startswith('python_gc_objects_collected_total') and 'generation="0"' in line:
                                        try:
                                            value = float(line.split()[-1])
                                            metrics['memory_pressure'] = min(value / 1000, 1.0)  # Normalize
                                        except:
                                            pass
                    except Exception:
                        pass
            
            else:
                # Fallback metrics when aiohttp not available
                metrics = {
                    'api_health': 1.0,
                    'response_time_ms': 100.0,
                    'query_success_rate': 0.99,
                    'confidence_score': 0.75,
                    'retrieval_latency_ms': 50.0,
                    'generation_latency_ms': 100.0,
                    'memory_pressure': 0.2
                }
        
        except Exception as e:
            print(f"Metrics collection failed: {e}")
            # Return degraded metrics
            metrics = {
                'api_health': 0.0,
                'response_time_ms': 9999.0,
                'query_success_rate': 0.0,
                'confidence_score': 0.0,
                'retrieval_latency_ms': 0.0,
                'generation_latency_ms': 0.0,
                'memory_pressure': 1.0
            }
        
        # Add system-level metrics
        metrics.update({
            'current_timestamp': current_time,
            'uptime_hours': (current_time - self.last_collection_time) / 3600 if self.last_collection_time > 0 else 0,
            'error_rate': 1.0 - metrics.get('query_success_rate', 0.0)
        })
        
        self.metrics_cache = metrics
        self.last_collection_time = current_time
        
        return metrics


class NotificationDelivery:
    """Handles notification delivery through various channels."""
    
    def __init__(self):
        self.channels: Dict[str, NotificationChannel] = {}
        self.delivery_history: List[Dict[str, Any]] = []
    
    def add_channel(self, channel: NotificationChannel):
        """Add a notification channel."""
        self.channels[channel.channel_id] = channel
    
    async def send_notification(self, alert: Alert, channel_ids: List[str]) -> Dict[str, bool]:
        """Send notification to specified channels."""
        delivery_results = {}
        
        for channel_id in channel_ids:
            channel = self.channels.get(channel_id)
            if not channel or not channel.enabled:
                delivery_results[channel_id] = False
                continue
            
            try:
                if channel.channel_type == NotificationType.EMAIL:
                    success = await self._send_email(alert, channel)
                elif channel.channel_type == NotificationType.SLACK:
                    success = await self._send_slack(alert, channel)
                elif channel.channel_type == NotificationType.WEBHOOK:
                    success = await self._send_webhook(alert, channel)
                elif channel.channel_type == NotificationType.CONSOLE:
                    success = self._send_console(alert, channel)
                else:
                    success = False
                
                delivery_results[channel_id] = success
                
                # Record delivery attempt
                self.delivery_history.append({
                    'alert_id': alert.alert_id,
                    'channel_id': channel_id,
                    'channel_type': channel.channel_type.value,
                    'success': success,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                print(f"Failed to send notification to {channel_id}: {e}")
                delivery_results[channel_id] = False
        
        return delivery_results
    
    async def _send_email(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send email notification."""
        try:
            config = channel.config
            
            # Create message
            msg = MimeMultipart()
            msg['From'] = config['from_email']
            msg['To'] = config['to_email']
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Email body
            body = f"""
            Alert: {alert.title}
            Severity: {alert.severity.value.upper()}
            Description: {alert.description}
            Triggered At: {datetime.fromtimestamp(alert.triggered_at, tz=timezone.utc).isoformat()}
            
            Trigger Values:
            {json.dumps(alert.trigger_values, indent=2)}
            
            Metadata:
            {json.dumps(alert.metadata, indent=2)}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(config['smtp_server'], config.get('smtp_port', 587))
            if config.get('use_tls', True):
                server.starttls()
            if config.get('username') and config.get('password'):
                server.login(config['username'], config['password'])
            
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            print(f"Email delivery failed: {e}")
            return False
    
    async def _send_slack(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send Slack notification."""
        if not NOTIFICATION_AVAILABLE:
            return False
        
        try:
            config = channel.config
            webhook_url = config['webhook_url']
            
            # Create Slack message
            color_map = {
                AlertSeverity.LOW: '#36a64f',
                AlertSeverity.MEDIUM: '#ff9500', 
                AlertSeverity.HIGH: '#ff0000',
                AlertSeverity.CRITICAL: '#8B0000'
            }
            
            payload = {
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, '#ff0000'),
                        "title": f"{alert.severity.value.upper()}: {alert.title}",
                        "text": alert.description,
                        "fields": [
                            {
                                "title": "Alert ID",
                                "value": alert.alert_id,
                                "short": True
                            },
                            {
                                "title": "Triggered At",
                                "value": datetime.fromtimestamp(alert.triggered_at, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
                                "short": True
                            }
                        ],
                        "footer": "Advanced RAG Monitoring",
                        "ts": int(alert.triggered_at)
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    return response.status == 200
            
        except Exception as e:
            print(f"Slack delivery failed: {e}")
            return False
    
    async def _send_webhook(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send webhook notification."""
        if not NOTIFICATION_AVAILABLE:
            return False
        
        try:
            config = channel.config
            webhook_url = config['webhook_url']
            
            # Create webhook payload
            payload = {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "triggered_at": alert.triggered_at,
                "trigger_values": alert.trigger_values,
                "metadata": alert.metadata
            }
            
            headers = config.get('headers', {})
            if config.get('auth_token'):
                headers['Authorization'] = f"Bearer {config['auth_token']}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, headers=headers) as response:
                    return response.status in [200, 201, 202]
            
        except Exception as e:
            print(f"Webhook delivery failed: {e}")
            return False
    
    def _send_console(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send console notification."""
        try:
            severity_symbols = {
                AlertSeverity.LOW: "ℹ️",
                AlertSeverity.MEDIUM: "⚠️",
                AlertSeverity.HIGH: "🚨",
                AlertSeverity.CRITICAL: "🔥"
            }
            
            symbol = severity_symbols.get(alert.severity, "🚨")
            timestamp = datetime.fromtimestamp(alert.triggered_at, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            
            print(f"\\n{symbol} ALERT [{alert.severity.value.upper()}] {symbol}")
            print(f"Title: {alert.title}")
            print(f"Description: {alert.description}")
            print(f"Triggered At: {timestamp}")
            print(f"Alert ID: {alert.alert_id}")
            
            if alert.trigger_values:
                print("Trigger Values:")
                for key, value in alert.trigger_values.items():
                    print(f"  {key}: {value}")
            
            print("-" * 50)
            
            return True
            
        except Exception as e:
            print(f"Console delivery failed: {e}")
            return False


class AlertingManager:
    """Main alerting system manager."""
    
    def __init__(self, storage_path: str = "./data/alerting"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.rules_file = self.storage_path / "alert_rules.json"
        self.alerts_file = self.storage_path / "active_alerts.json"
        self.channels_file = self.storage_path / "notification_channels.json"
        
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: List[Alert] = []
        self.metrics_collector = MetricsCollector()
        self.notification_delivery = NotificationDelivery()
        
        # Condition tracking for persistence requirements
        self.condition_violations: Dict[str, List[float]] = {}  # rule_id -> timestamps
        
        # Load configuration
        self.load_configuration()
        
        # Setup default notification channels
        self._setup_default_channels()
        
        # Setup default alert rules
        self._setup_default_rules()
    
    def load_configuration(self):
        """Load alerting configuration from storage."""
        # Load alert rules
        if self.rules_file.exists():
            try:
                with open(self.rules_file, 'r') as f:
                    data = json.load(f)
                    self.alert_rules = []
                    for rule_data in data:
                        rule_data['severity'] = AlertSeverity(rule_data['severity'])
                        rule_data['conditions'] = [AlertCondition(**cond) for cond in rule_data['conditions']]
                        self.alert_rules.append(AlertRule(**rule_data))
            except Exception as e:
                print(f"Failed to load alert rules: {e}")
        
        # Load active alerts
        if self.alerts_file.exists():
            try:
                with open(self.alerts_file, 'r') as f:
                    data = json.load(f)
                    self.active_alerts = []
                    for alert_data in data:
                        alert_data['severity'] = AlertSeverity(alert_data['severity'])
                        alert_data['status'] = AlertStatus(alert_data['status'])
                        self.active_alerts.append(Alert(**alert_data))
            except Exception as e:
                print(f"Failed to load active alerts: {e}")
        
        # Load notification channels
        if self.channels_file.exists():
            try:
                with open(self.channels_file, 'r') as f:
                    data = json.load(f)
                    for channel_data in data:
                        channel_data['channel_type'] = NotificationType(channel_data['channel_type'])
                        channel = NotificationChannel(**channel_data)
                        self.notification_delivery.add_channel(channel)
            except Exception as e:
                print(f"Failed to load notification channels: {e}")
    
    def save_configuration(self):
        """Save alerting configuration to storage."""
        try:
            # Save alert rules
            rules_data = []
            for rule in self.alert_rules:
                data = asdict(rule)
                data['severity'] = rule.severity.value
                rules_data.append(data)
            
            with open(self.rules_file, 'w') as f:
                json.dump(rules_data, f, indent=2)
            
            # Save active alerts
            alerts_data = []
            for alert in self.active_alerts:
                data = asdict(alert)
                data['severity'] = alert.severity.value
                data['status'] = alert.status.value
                alerts_data.append(data)
            
            with open(self.alerts_file, 'w') as f:
                json.dump(alerts_data, f, indent=2)
            
            # Save notification channels
            channels_data = []
            for channel in self.notification_delivery.channels.values():
                data = asdict(channel)
                data['channel_type'] = channel.channel_type.value
                channels_data.append(data)
            
            with open(self.channels_file, 'w') as f:
                json.dump(channels_data, f, indent=2)
                
        except Exception as e:
            print(f"Failed to save alerting configuration: {e}")
    
    def _setup_default_channels(self):
        """Setup default notification channels."""
        # Console channel (always available)
        console_channel = NotificationChannel(
            channel_id="console",
            channel_type=NotificationType.CONSOLE,
            config={},
            enabled=True
        )
        self.notification_delivery.add_channel(console_channel)
    
    def _setup_default_rules(self):
        """Setup default alert rules."""
        if not self.alert_rules:  # Only setup if no rules exist
            default_rules = [
                AlertRule(
                    rule_id="high_response_time",
                    name="High Response Time",
                    description="Alert when API response time is too high",
                    conditions=[
                        AlertCondition(
                            metric_name="response_time_ms",
                            operator=">",
                            threshold_value=2000.0,
                            duration_seconds=60,
                            description="Response time > 2000ms for 1 minute"
                        )
                    ],
                    severity=AlertSeverity.HIGH,
                    notification_channels=["console"],
                    cooldown_seconds=300,
                    auto_resolve=True
                ),
                AlertRule(
                    rule_id="api_down",
                    name="API Service Down",
                    description="Alert when API health check fails",
                    conditions=[
                        AlertCondition(
                            metric_name="api_health",
                            operator="<",
                            threshold_value=1.0,
                            duration_seconds=30,
                            description="API health check failing for 30 seconds"
                        )
                    ],
                    severity=AlertSeverity.CRITICAL,
                    notification_channels=["console"],
                    cooldown_seconds=180,
                    auto_resolve=True
                ),
                AlertRule(
                    rule_id="low_confidence",
                    name="Low Confidence Scores",
                    description="Alert when confidence scores are consistently low",
                    conditions=[
                        AlertCondition(
                            metric_name="confidence_score",
                            operator="<",
                            threshold_value=0.3,
                            duration_seconds=300,
                            description="Confidence score < 0.3 for 5 minutes"
                        )
                    ],
                    severity=AlertSeverity.MEDIUM,
                    notification_channels=["console"],
                    cooldown_seconds=600,
                    auto_resolve=True
                ),
                AlertRule(
                    rule_id="high_error_rate",
                    name="High Error Rate",
                    description="Alert when error rate is too high",
                    conditions=[
                        AlertCondition(
                            metric_name="error_rate",
                            operator=">",
                            threshold_value=0.1,
                            duration_seconds=120,
                            description="Error rate > 10% for 2 minutes"
                        )
                    ],
                    severity=AlertSeverity.HIGH,
                    notification_channels=["console"],
                    cooldown_seconds=300,
                    auto_resolve=True
                )
            ]
            
            self.alert_rules.extend(default_rules)
            self.save_configuration()
    
    async def check_alerts(self) -> List[Alert]:
        """Check all alert conditions and trigger alerts if needed."""
        current_time = time.time()
        
        # Collect current metrics
        try:
            metrics = await self.metrics_collector.collect_metrics()
        except Exception as e:
            print(f"Failed to collect metrics for alerting: {e}")
            return []
        
        new_alerts = []
        
        for rule in self.alert_rules:
            # Check if rule is in cooldown
            last_alert = next((a for a in self.active_alerts 
                             if a.rule_id == rule.rule_id and a.status == AlertStatus.OPEN), None)
            
            if last_alert and (current_time - last_alert.triggered_at) < rule.cooldown_seconds:
                continue  # Still in cooldown
            
            # Evaluate all conditions for this rule
            all_conditions_met = True
            trigger_values = {}
            
            for condition in rule.conditions:
                metric_value = metrics.get(condition.metric_name)
                if metric_value is None:
                    all_conditions_met = False
                    break
                
                # Evaluate condition
                condition_met = self._evaluate_condition(condition, metric_value)
                trigger_values[condition.metric_name] = metric_value
                
                if condition_met:
                    # Track condition violation for duration requirement
                    violation_key = f"{rule.rule_id}_{condition.metric_name}"
                    if violation_key not in self.condition_violations:
                        self.condition_violations[violation_key] = []
                    
                    self.condition_violations[violation_key].append(current_time)
                    
                    # Remove old violations outside duration window
                    cutoff_time = current_time - condition.duration_seconds
                    self.condition_violations[violation_key] = [
                        t for t in self.condition_violations[violation_key] if t > cutoff_time
                    ]
                    
                    # Check if condition has been violated for required duration
                    if (self.condition_violations[violation_key] and 
                        current_time - self.condition_violations[violation_key][0] >= condition.duration_seconds):
                        continue  # This condition is satisfied
                    else:
                        all_conditions_met = False
                        break
                else:
                    # Condition not met, clear violations
                    violation_key = f"{rule.rule_id}_{condition.metric_name}"
                    if violation_key in self.condition_violations:
                        del self.condition_violations[violation_key]
                    all_conditions_met = False
                    break
            
            # If all conditions met for required duration, trigger alert
            if all_conditions_met:
                alert = Alert(
                    alert_id=f"alert_{int(current_time)}_{rule.rule_id}",
                    rule_id=rule.rule_id,
                    title=rule.name,
                    description=rule.description,
                    severity=rule.severity,
                    status=AlertStatus.OPEN,
                    triggered_at=current_time,
                    resolved_at=None,
                    acknowledged_at=None,
                    acknowledged_by=None,
                    escalated_at=None,
                    trigger_values=trigger_values,
                    metadata=rule.metadata or {}
                )
                
                self.active_alerts.append(alert)
                new_alerts.append(alert)
                
                # Send notifications
                await self.notification_delivery.send_notification(alert, rule.notification_channels)
                
                # Clear condition violations after triggering
                for condition in rule.conditions:
                    violation_key = f"{rule.rule_id}_{condition.metric_name}"
                    if violation_key in self.condition_violations:
                        del self.condition_violations[violation_key]
        
        # Auto-resolve alerts if conditions are no longer met
        await self._auto_resolve_alerts(metrics)
        
        # Save state
        self.save_configuration()
        
        return new_alerts
    
    def _evaluate_condition(self, condition: AlertCondition, metric_value: float) -> bool:
        """Evaluate if a condition is met."""
        threshold = condition.threshold_value
        
        if condition.operator == ">":
            return metric_value > threshold
        elif condition.operator == "<":
            return metric_value < threshold
        elif condition.operator == ">=":
            return metric_value >= threshold
        elif condition.operator == "<=":
            return metric_value <= threshold
        elif condition.operator == "==":
            return abs(metric_value - threshold) < 0.001  # Float comparison
        elif condition.operator == "!=":
            return abs(metric_value - threshold) >= 0.001
        else:
            return False
    
    async def _auto_resolve_alerts(self, metrics: Dict[str, float]):
        """Auto-resolve alerts when conditions are no longer met."""
        current_time = time.time()
        
        for alert in self.active_alerts:
            if alert.status != AlertStatus.OPEN:
                continue
            
            # Find the rule for this alert
            rule = next((r for r in self.alert_rules if r.rule_id == alert.rule_id), None)
            if not rule or not rule.auto_resolve:
                continue
            
            # Check if all conditions are now resolved
            all_conditions_resolved = True
            
            for condition in rule.conditions:
                metric_value = metrics.get(condition.metric_name)
                if metric_value is None:
                    all_conditions_resolved = False
                    break
                
                # Check if condition is still violated
                if self._evaluate_condition(condition, metric_value):
                    all_conditions_resolved = False
                    break
            
            if all_conditions_resolved:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = current_time
                print(f"✅ Auto-resolved alert: {alert.title}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert."""
        alert = next((a for a in self.active_alerts if a.alert_id == alert_id), None)
        if not alert or alert.status != AlertStatus.OPEN:
            return False
        
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = time.time()
        alert.acknowledged_by = acknowledged_by
        
        self.save_configuration()
        return True
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an alert."""
        alert = next((a for a in self.active_alerts if a.alert_id == alert_id), None)
        if not alert or alert.status == AlertStatus.RESOLVED:
            return False
        
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = time.time()
        
        self.save_configuration()
        return True
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule."""
        # Check for duplicate rule IDs
        existing = next((r for r in self.alert_rules if r.rule_id == rule.rule_id), None)
        if existing:
            raise ValueError(f"Alert rule with ID {rule.rule_id} already exists")
        
        self.alert_rules.append(rule)
        self.save_configuration()
    
    def get_active_alerts(self, severity: AlertSeverity = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity."""
        active = [a for a in self.active_alerts if a.status == AlertStatus.OPEN]
        
        if severity:
            active = [a for a in active if a.severity == severity]
        
        return sorted(active, key=lambda a: a.triggered_at, reverse=True)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alerting system status."""
        active_alerts = self.get_active_alerts()
        
        severity_counts = {
            AlertSeverity.CRITICAL: len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            AlertSeverity.HIGH: len([a for a in active_alerts if a.severity == AlertSeverity.HIGH]),
            AlertSeverity.MEDIUM: len([a for a in active_alerts if a.severity == AlertSeverity.MEDIUM]),
            AlertSeverity.LOW: len([a for a in active_alerts if a.severity == AlertSeverity.LOW])
        }
        
        return {
            "total_active_alerts": len(active_alerts),
            "severity_breakdown": {sev.value: count for sev, count in severity_counts.items()},
            "total_rules": len(self.alert_rules),
            "notification_channels": len(self.notification_delivery.channels),
            "recent_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "title": alert.title,
                    "severity": alert.severity.value,
                    "status": alert.status.value,
                    "triggered_at": alert.triggered_at
                }
                for alert in sorted(self.active_alerts, key=lambda a: a.triggered_at, reverse=True)[:5]
            ]
        }
    
    async def start_monitoring(self, check_interval: int = 60):
        """Start continuous monitoring with specified check interval."""
        print(f"🚨 Starting alerting system monitoring (check interval: {check_interval}s)")
        
        while True:
            try:
                new_alerts = await self.check_alerts()
                if new_alerts:
                    print(f"⚠️  Triggered {len(new_alerts)} new alerts")
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                await asyncio.sleep(check_interval)
