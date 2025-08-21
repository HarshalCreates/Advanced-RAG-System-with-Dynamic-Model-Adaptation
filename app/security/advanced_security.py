"""Advanced security features including prompt injection detection."""
from __future__ import annotations

import re
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone
from pathlib import Path
import uuid


class ThreatType(Enum):
    """Types of security threats."""
    PROMPT_INJECTION = "prompt_injection"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    MALICIOUS_CODE = "malicious_code"
    DATA_EXFILTRATION = "data_exfiltration"
    DENIAL_OF_SERVICE = "denial_of_service"
    SOCIAL_ENGINEERING = "social_engineering"


class SecurityLevel(Enum):
    """Security alert levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityThreat:
    """Detected security threat."""
    threat_id: str
    threat_type: ThreatType
    severity: SecurityLevel
    description: str
    detected_content: str
    confidence: float
    mitigation_applied: str
    timestamp: float
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class SecurityRule:
    """Security detection rule."""
    rule_id: str
    name: str
    threat_type: ThreatType
    pattern: str
    severity: SecurityLevel
    action: str  # "block", "sanitize", "alert", "log"
    enabled: bool = True


class PromptInjectionDetector:
    """Detects prompt injection attacks in queries."""
    
    def __init__(self):
        # Prompt injection patterns
        self.injection_patterns = [
            # Direct instruction injection
            re.compile(r'ignore\s+(all\s+)?previous\s+instructions?', re.IGNORECASE),
            re.compile(r'forget\s+(all\s+)?(your\s+)?previous\s+instructions?', re.IGNORECASE),
            re.compile(r'disregard\s+(all\s+)?previous\s+instructions?', re.IGNORECASE),
            
            # System prompt manipulation
            re.compile(r'you\s+are\s+now\s+a\s+different\s+(ai|bot|assistant)', re.IGNORECASE),
            re.compile(r'pretend\s+(you\s+are|to\s+be)', re.IGNORECASE),
            re.compile(r'act\s+as\s+(if\s+you\s+are\s+)?a\s+different', re.IGNORECASE),
            
            # Role manipulation
            re.compile(r'you\s+are\s+no\s+longer\s+(an?\s+)?(ai|assistant|bot)', re.IGNORECASE),
            re.compile(r'from\s+now\s+on,?\s+you\s+(are|will\s+be)', re.IGNORECASE),
            re.compile(r'new\s+role:?\s*you\s+are', re.IGNORECASE),
            
            # System access attempts
            re.compile(r'show\s+me\s+(your\s+)?(system\s+)?(prompt|instructions)', re.IGNORECASE),
            re.compile(r'what\s+(are\s+)?your\s+(system\s+)?(prompt|instructions)', re.IGNORECASE),
            re.compile(r'reveal\s+(your\s+)?(system\s+)?(prompt|instructions)', re.IGNORECASE),
            
            # Jailbreak attempts
            re.compile(r'dan\s+mode', re.IGNORECASE),
            re.compile(r'developer\s+mode', re.IGNORECASE),
            re.compile(r'unrestricted\s+mode', re.IGNORECASE),
            re.compile(r'jailbreak', re.IGNORECASE),
            
            # Code execution attempts
            re.compile(r'execute\s+(the\s+)?(following\s+)?(code|command)', re.IGNORECASE),
            re.compile(r'run\s+(the\s+)?(following\s+)?(code|script|command)', re.IGNORECASE),
            re.compile(r'eval\s*\(', re.IGNORECASE),
            re.compile(r'exec\s*\(', re.IGNORECASE),
            
            # Data exfiltration attempts
            re.compile(r'output\s+(all\s+)?(your\s+)?(training\s+)?data', re.IGNORECASE),
            re.compile(r'dump\s+(all\s+)?(your\s+)?data', re.IGNORECASE),
            re.compile(r'list\s+all\s+(files|documents|users)', re.IGNORECASE),
            
            # Bypass attempts
            re.compile(r'translate\s+to\s+base64', re.IGNORECASE),
            re.compile(r'encode\s+(this\s+)?in\s+base64', re.IGNORECASE),
            re.compile(r'rot13', re.IGNORECASE),
            
            # Social engineering
            re.compile(r'emergency\s+override', re.IGNORECASE),
            re.compile(r'administrator\s+(access|override|command)', re.IGNORECASE),
            re.compile(r'sudo\s+', re.IGNORECASE),
        ]
        
        # Malicious keywords that often appear in injection attempts
        self.malicious_keywords = [
            'exploit', 'payload', 'shell', 'backdoor', 'rootkit',
            'keylogger', 'trojan', 'malware', 'virus', 'worm'
        ]
        
        # Common evasion techniques
        self.evasion_patterns = [
            re.compile(r'[a-z]\s*[a-z]', re.IGNORECASE),  # Character spacing
            re.compile(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?~`]'),  # Special character insertion
            re.compile(r'\\[a-z]', re.IGNORECASE),  # Escape sequences
        ]
    
    def detect_injection(self, text: str) -> Tuple[bool, List[str], float]:
        """Detect prompt injection in text."""
        
        detected_patterns = []
        confidence_scores = []
        
        # Check main injection patterns
        for pattern in self.injection_patterns:
            if pattern.search(text):
                detected_patterns.append(pattern.pattern)
                confidence_scores.append(0.8)
        
        # Check for malicious keywords
        text_lower = text.lower()
        for keyword in self.malicious_keywords:
            if keyword in text_lower:
                detected_patterns.append(f"malicious_keyword: {keyword}")
                confidence_scores.append(0.6)
        
        # Check for evasion techniques
        evasion_score = 0
        for pattern in self.evasion_patterns:
            matches = len(pattern.findall(text))
            if matches > 10:  # High frequency of evasion characters
                evasion_score += 0.3
        
        if evasion_score > 0.5:
            detected_patterns.append("evasion_techniques")
            confidence_scores.append(evasion_score)
        
        # Calculate overall confidence
        if confidence_scores:
            overall_confidence = min(max(confidence_scores), 1.0)
        else:
            overall_confidence = 0.0
        
        is_injection = len(detected_patterns) > 0
        
        return is_injection, detected_patterns, overall_confidence
    
    def sanitize_query(self, text: str) -> str:
        """Sanitize query by removing potential injection content."""
        
        sanitized = text
        
        # Remove detected injection patterns
        for pattern in self.injection_patterns:
            sanitized = pattern.sub('[FILTERED]', sanitized)
        
        # Remove malicious keywords
        for keyword in self.malicious_keywords:
            sanitized = re.sub(re.escape(keyword), '[FILTERED]', sanitized, flags=re.IGNORECASE)
        
        return sanitized


class InputValidator:
    """Validates and sanitizes user inputs."""
    
    def __init__(self):
        # SQL injection patterns
        self.sql_patterns = [
            re.compile(r'(\bUNION\b|\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bDROP\b)', re.IGNORECASE),
            re.compile(r'(\bOR\b|\bAND\b)\s+\d+\s*=\s*\d+', re.IGNORECASE),
            re.compile(r"';\s*(DROP|DELETE|INSERT|UPDATE)", re.IGNORECASE),
            re.compile(r'--\s*$', re.MULTILINE),
            re.compile(r'/\*.*?\*/', re.DOTALL),
        ]
        
        # XSS patterns
        self.xss_patterns = [
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),
            re.compile(r'<iframe[^>]*>', re.IGNORECASE),
            re.compile(r'<object[^>]*>', re.IGNORECASE),
            re.compile(r'<embed[^>]*>', re.IGNORECASE),
        ]
        
        # File path traversal
        self.path_traversal_patterns = [
            re.compile(r'\.\.[\\/]'),
            re.compile(r'[\\/]etc[\\/]passwd'),
            re.compile(r'[\\/]windows[\\/]system32'),
            re.compile(r'%2e%2e%2f', re.IGNORECASE),
            re.compile(r'%252e%252e%252f', re.IGNORECASE),
        ]
    
    def validate_input(self, text: str, input_type: str = "query") -> Tuple[bool, List[str]]:
        """Validate input for security threats."""
        
        threats = []
        
        # Check for SQL injection
        for pattern in self.sql_patterns:
            if pattern.search(text):
                threats.append("SQL injection pattern detected")
        
        # Check for XSS
        for pattern in self.xss_patterns:
            if pattern.search(text):
                threats.append("XSS pattern detected")
        
        # Check for path traversal
        for pattern in self.path_traversal_patterns:
            if pattern.search(text):
                threats.append("Path traversal pattern detected")
        
        # Additional validation based on input type
        if input_type == "filename":
            if any(char in text for char in '<>:"|?*'):
                threats.append("Invalid filename characters")
        
        is_valid = len(threats) == 0
        
        return is_valid, threats
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize input by removing malicious content."""
        
        sanitized = text
        
        # Remove SQL injection patterns
        for pattern in self.sql_patterns:
            sanitized = pattern.sub('', sanitized)
        
        # Remove XSS patterns
        for pattern in self.xss_patterns:
            sanitized = pattern.sub('', sanitized)
        
        # Remove path traversal patterns
        for pattern in self.path_traversal_patterns:
            sanitized = pattern.sub('', sanitized)
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', '\\0', '\\n', '\\r']
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        return sanitized.strip()


class RateLimiter:
    """Rate limiting to prevent abuse and DoS attacks."""
    
    def __init__(self):
        self.request_counts: Dict[str, List[float]] = {}
        self.blocked_ips: Dict[str, float] = {}  # IP -> blocked_until_timestamp
        
        # Rate limiting rules
        self.rules = {
            'requests_per_minute': 60,
            'requests_per_hour': 1000,
            'max_query_length': 10000,
            'max_file_size_mb': 50,
            'block_duration_minutes': 15
        }
    
    def is_rate_limited(self, identifier: str, request_type: str = "query") -> Tuple[bool, str]:
        """Check if identifier (IP, user_id) is rate limited."""
        
        current_time = time.time()
        
        # Check if IP is blocked
        if identifier in self.blocked_ips:
            if current_time < self.blocked_ips[identifier]:
                remaining = int(self.blocked_ips[identifier] - current_time)
                return True, f"Blocked for {remaining} more seconds"
            else:
                # Block expired
                del self.blocked_ips[identifier]
        
        # Clean old request timestamps
        if identifier in self.request_counts:
            minute_ago = current_time - 60
            hour_ago = current_time - 3600
            
            self.request_counts[identifier] = [
                ts for ts in self.request_counts[identifier] if ts > hour_ago
            ]
        else:
            self.request_counts[identifier] = []
        
        # Check rate limits
        recent_requests = [ts for ts in self.request_counts[identifier] if ts > current_time - 60]
        hourly_requests = self.request_counts[identifier]
        
        if len(recent_requests) >= self.rules['requests_per_minute']:
            self._block_identifier(identifier)
            return True, "Too many requests per minute"
        
        if len(hourly_requests) >= self.rules['requests_per_hour']:
            self._block_identifier(identifier)
            return True, "Too many requests per hour"
        
        # Record this request
        self.request_counts[identifier].append(current_time)
        
        return False, "OK"
    
    def _block_identifier(self, identifier: str):
        """Block an identifier for the configured duration."""
        block_duration = self.rules['block_duration_minutes'] * 60
        self.blocked_ips[identifier] = time.time() + block_duration
    
    def validate_content_size(self, content: str, content_type: str) -> Tuple[bool, str]:
        """Validate content size limits."""
        
        if content_type == "query" and len(content) > self.rules['max_query_length']:
            return False, f"Query too long (max {self.rules['max_query_length']} characters)"
        
        return True, "OK"


class SecurityManager:
    """Main security management system."""
    
    def __init__(self, storage_path: str = "./data/security"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.threats_file = self.storage_path / "security_threats.json"
        self.threats: List[SecurityThreat] = []
        
        # Security components
        self.prompt_detector = PromptInjectionDetector()
        self.input_validator = InputValidator()
        self.rate_limiter = RateLimiter()
        
        # Security rules
        self.security_rules = self._create_default_rules()
        
        self.load_threats()
    
    def _create_default_rules(self) -> List[SecurityRule]:
        """Create default security rules."""
        return [
            SecurityRule(
                rule_id="prompt_injection_critical",
                name="Critical Prompt Injection",
                threat_type=ThreatType.PROMPT_INJECTION,
                pattern="high_confidence_injection",
                severity=SecurityLevel.CRITICAL,
                action="block"
            ),
            SecurityRule(
                rule_id="sql_injection",
                name="SQL Injection",
                threat_type=ThreatType.SQL_INJECTION,
                pattern="sql_patterns",
                severity=SecurityLevel.HIGH,
                action="sanitize"
            ),
            SecurityRule(
                rule_id="xss_attack",
                name="Cross-Site Scripting",
                threat_type=ThreatType.XSS,
                pattern="xss_patterns",
                severity=SecurityLevel.HIGH,
                action="sanitize"
            ),
            SecurityRule(
                rule_id="rate_limit_violation",
                name="Rate Limit Violation",
                threat_type=ThreatType.DENIAL_OF_SERVICE,
                pattern="rate_exceeded",
                severity=SecurityLevel.MEDIUM,
                action="block"
            )
        ]
    
    def load_threats(self):
        """Load security threats from storage."""
        if self.threats_file.exists():
            try:
                with open(self.threats_file, 'r') as f:
                    data = json.load(f)
                    for threat_data in data[-1000:]:  # Keep last 1000 threats
                        threat_data['threat_type'] = ThreatType(threat_data['threat_type'])
                        threat_data['severity'] = SecurityLevel(threat_data['severity'])
                        threat = SecurityThreat(**threat_data)
                        self.threats.append(threat)
            except Exception as e:
                print(f"Failed to load security threats: {e}")
    
    def save_threats(self):
        """Save security threats to storage."""
        try:
            threats_data = []
            for threat in self.threats[-1000:]:  # Keep last 1000 threats
                data = asdict(threat)
                data['threat_type'] = threat.threat_type.value
                data['severity'] = threat.severity.value
                threats_data.append(data)
            
            with open(self.threats_file, 'w') as f:
                json.dump(threats_data, f, indent=2)
        except Exception as e:
            print(f"Failed to save security threats: {e}")
    
    def analyze_query_security(self, query: str, user_id: str = None, 
                             ip_address: str = None) -> Tuple[bool, str, Optional[SecurityThreat]]:
        """Analyze query for security threats."""
        
        # Check rate limiting
        identifier = ip_address or user_id or "anonymous"
        is_limited, limit_reason = self.rate_limiter.is_rate_limited(identifier)
        if is_limited:
            threat = self._create_threat(
                threat_type=ThreatType.DENIAL_OF_SERVICE,
                severity=SecurityLevel.MEDIUM,
                description=f"Rate limit exceeded: {limit_reason}",
                detected_content=query[:100],
                confidence=1.0,
                mitigation="request_blocked",
                user_id=user_id,
                ip_address=ip_address
            )
            return False, limit_reason, threat
        
        # Validate content size
        is_valid_size, size_reason = self.rate_limiter.validate_content_size(query, "query")
        if not is_valid_size:
            threat = self._create_threat(
                threat_type=ThreatType.DENIAL_OF_SERVICE,
                severity=SecurityLevel.MEDIUM,
                description=f"Content size violation: {size_reason}",
                detected_content=query[:100],
                confidence=1.0,
                mitigation="request_blocked",
                user_id=user_id,
                ip_address=ip_address
            )
            return False, size_reason, threat
        
        # Check for prompt injection
        is_injection, injection_patterns, injection_confidence = self.prompt_detector.detect_injection(query)
        if is_injection and injection_confidence > 0.7:
            threat = self._create_threat(
                threat_type=ThreatType.PROMPT_INJECTION,
                severity=SecurityLevel.HIGH if injection_confidence > 0.8 else SecurityLevel.MEDIUM,
                description=f"Prompt injection detected: {', '.join(injection_patterns)}",
                detected_content=query,
                confidence=injection_confidence,
                mitigation="query_blocked",
                user_id=user_id,
                ip_address=ip_address
            )
            return False, "Prompt injection detected - query blocked", threat
        
        # Check for other input validation issues
        is_valid_input, input_threats = self.input_validator.validate_input(query)
        if not is_valid_input:
            threat_type = ThreatType.SQL_INJECTION if "SQL" in str(input_threats) else ThreatType.XSS
            threat = self._create_threat(
                threat_type=threat_type,
                severity=SecurityLevel.HIGH,
                description=f"Input validation failed: {', '.join(input_threats)}",
                detected_content=query,
                confidence=0.9,
                mitigation="query_sanitized",
                user_id=user_id,
                ip_address=ip_address
            )
            # For input validation, we sanitize rather than block
            return True, "Query sanitized due to security concerns", threat
        
        return True, "Query passed security analysis", None
    
    def sanitize_query(self, query: str) -> str:
        """Sanitize query to remove security threats."""
        
        # Apply prompt injection sanitization
        sanitized = self.prompt_detector.sanitize_query(query)
        
        # Apply input validation sanitization
        sanitized = self.input_validator.sanitize_input(sanitized)
        
        return sanitized
    
    def _create_threat(self, threat_type: ThreatType, severity: SecurityLevel,
                      description: str, detected_content: str, confidence: float,
                      mitigation: str, user_id: str = None, ip_address: str = None) -> SecurityThreat:
        """Create and record a security threat."""
        
        threat = SecurityThreat(
            threat_id=str(uuid.uuid4()),
            threat_type=threat_type,
            severity=severity,
            description=description,
            detected_content=detected_content[:500],  # Limit stored content
            confidence=confidence,
            mitigation_applied=mitigation,
            timestamp=time.time(),
            user_id=user_id,
            ip_address=ip_address,
            metadata={}
        )
        
        self.threats.append(threat)
        self.save_threats()
        
        return threat
    
    def get_security_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get security statistics for the last N hours."""
        
        cutoff_time = time.time() - (hours * 3600)
        recent_threats = [t for t in self.threats if t.timestamp > cutoff_time]
        
        # Group by threat type
        threat_counts = {}
        for threat_type in ThreatType:
            threat_counts[threat_type.value] = len([t for t in recent_threats if t.threat_type == threat_type])
        
        # Group by severity
        severity_counts = {}
        for severity in SecurityLevel:
            severity_counts[severity.value] = len([t for t in recent_threats if t.severity == severity])
        
        return {
            "time_period_hours": hours,
            "total_threats": len(recent_threats),
            "threat_types": threat_counts,
            "severity_levels": severity_counts,
            "blocked_ips": len(self.rate_limiter.blocked_ips),
            "security_rules": len(self.security_rules),
            "recent_threats": [
                {
                    "threat_id": t.threat_id,
                    "type": t.threat_type.value,
                    "severity": t.severity.value,
                    "description": t.description,
                    "timestamp": t.timestamp,
                    "mitigation": t.mitigation_applied
                }
                for t in sorted(recent_threats, key=lambda x: x.timestamp, reverse=True)[:10]
            ]
        }
    
    def get_threat_by_id(self, threat_id: str) -> Optional[SecurityThreat]:
        """Get threat by ID."""
        return next((t for t in self.threats if t.threat_id == threat_id), None)
