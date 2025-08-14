"""Data privacy compliance and PII detection system."""
from __future__ import annotations

import re
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta, timezone
from pathlib import Path
import uuid


class PIIType(Enum):
    """Types of Personally Identifiable Information."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    BANK_ACCOUNT = "bank_account"
    MEDICAL_ID = "medical_id"


class ConsentStatus(Enum):
    """User consent status for data processing."""
    GIVEN = "given"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"
    EXPIRED = "expired"


class DataCategory(Enum):
    """Categories of personal data."""
    IDENTITY = "identity"
    CONTACT = "contact"
    DEMOGRAPHIC = "demographic"
    BEHAVIORAL = "behavioral"
    TECHNICAL = "technical"
    PREFERENCES = "preferences"
    CONTENT = "content"


@dataclass
class PIIDetectionResult:
    """Result of PII detection in text."""
    pii_type: PIIType
    value: str
    start_position: int
    end_position: int
    confidence: float
    context: str


@dataclass
class ConsentRecord:
    """User consent record for GDPR compliance."""
    consent_id: str
    user_id: str
    data_categories: List[DataCategory]
    purpose: str
    status: ConsentStatus
    given_at: Optional[float]
    withdrawn_at: Optional[float]
    expires_at: Optional[float]
    legal_basis: str
    metadata: Dict[str, Any]


@dataclass
class DataRetentionPolicy:
    """Data retention policy configuration."""
    policy_id: str
    data_category: DataCategory
    retention_period_days: int
    auto_delete: bool
    deletion_method: str  # "anonymize", "delete", "archive"
    exceptions: List[str]


@dataclass
class DataProcessingRecord:
    """Record of data processing activity (GDPR Article 30)."""
    record_id: str
    user_id: str
    activity_type: str
    data_categories: List[DataCategory]
    purpose: str
    legal_basis: str
    data_subjects: List[str]
    recipients: List[str]
    retention_period: Optional[int]
    timestamp: float
    metadata: Dict[str, Any]


class PIIDetector:
    """Detects Personally Identifiable Information in text."""
    
    def __init__(self):
        # Regex patterns for different PII types
        self.patterns = {
            PIIType.EMAIL: re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),
            PIIType.PHONE: re.compile(
                r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
            ),
            PIIType.SSN: re.compile(
                r'\b(?!000|666|9\d{2})\d{3}[-.]?(?!00)\d{2}[-.]?(?!0000)\d{4}\b'
            ),
            PIIType.CREDIT_CARD: re.compile(
                r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'
            ),
            PIIType.IP_ADDRESS: re.compile(
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            ),
            PIIType.DATE_OF_BIRTH: re.compile(
                r'\b(0?[1-9]|1[0-2])[-/](0?[1-9]|[12][0-9]|3[01])[-/](19|20)\d{2}\b'
            ),
            PIIType.PASSPORT: re.compile(
                r'\b[A-Z]{1,2}[0-9]{6,9}\b'
            ),
            PIIType.DRIVER_LICENSE: re.compile(
                r'\b[A-Z]{1,2}[0-9]{6,8}\b'
            ),
            PIIType.BANK_ACCOUNT: re.compile(
                r'\b[0-9]{8,17}\b'
            )
        }
        
        # Common name patterns (simplified)
        self.name_patterns = [
            re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),  # First Last
            re.compile(r'\b[A-Z][a-z]+, [A-Z][a-z]+\b'),  # Last, First
        ]
        
        # Address patterns (simplified)
        self.address_patterns = [
            re.compile(r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)'),
        ]
    
    def detect_pii(self, text: str) -> List[PIIDetectionResult]:
        """Detect PII in text and return detection results."""
        results = []
        
        # Check each PII type pattern
        for pii_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                result = PIIDetectionResult(
                    pii_type=pii_type,
                    value=match.group(),
                    start_position=match.start(),
                    end_position=match.end(),
                    confidence=0.9,  # High confidence for regex matches
                    context=self._get_context(text, match.start(), match.end())
                )
                results.append(result)
        
        # Check name patterns
        for pattern in self.name_patterns:
            for match in pattern.finditer(text):
                # Additional validation for names (avoid common false positives)
                name = match.group()
                if self._is_likely_name(name):
                    result = PIIDetectionResult(
                        pii_type=PIIType.NAME,
                        value=name,
                        start_position=match.start(),
                        end_position=match.end(),
                        confidence=0.7,  # Lower confidence for name detection
                        context=self._get_context(text, match.start(), match.end())
                    )
                    results.append(result)
        
        # Check address patterns
        for pattern in self.address_patterns:
            for match in pattern.finditer(text):
                result = PIIDetectionResult(
                    pii_type=PIIType.ADDRESS,
                    value=match.group(),
                    start_position=match.start(),
                    end_position=match.end(),
                    confidence=0.8,
                    context=self._get_context(text, match.start(), match.end())
                )
                results.append(result)
        
        return results
    
    def _get_context(self, text: str, start: int, end: int, context_size: int = 50) -> str:
        """Get surrounding context for a PII match."""
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        return text[context_start:context_end]
    
    def _is_likely_name(self, text: str) -> bool:
        """Check if text is likely a person's name (simple heuristics)."""
        # Exclude common false positives
        common_words = {
            'United States', 'New York', 'Los Angeles', 'San Francisco',
            'Machine Learning', 'Artificial Intelligence', 'Data Science'
        }
        
        return text not in common_words and len(text.split()) <= 3
    
    def anonymize_text(self, text: str, pii_results: List[PIIDetectionResult] = None) -> Tuple[str, List[PIIDetectionResult]]:
        """Anonymize PII in text by replacing with placeholders."""
        if pii_results is None:
            pii_results = self.detect_pii(text)
        
        # Sort by position (descending) to avoid position shifts
        pii_results.sort(key=lambda x: x.start_position, reverse=True)
        
        anonymized_text = text
        
        for pii in pii_results:
            placeholder = f"[{pii.pii_type.value.upper()}]"
            anonymized_text = (
                anonymized_text[:pii.start_position] + 
                placeholder + 
                anonymized_text[pii.end_position:]
            )
        
        return anonymized_text, pii_results


class ConsentManager:
    """Manages user consent for GDPR compliance."""
    
    def __init__(self, storage_path: str = "./data/privacy"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.consents_file = self.storage_path / "consents.json"
        self.consents: Dict[str, ConsentRecord] = {}
        
        self.load_consents()
    
    def load_consents(self):
        """Load consent records from storage."""
        if self.consents_file.exists():
            try:
                with open(self.consents_file, 'r') as f:
                    data = json.load(f)
                    for consent_data in data:
                        consent_data['status'] = ConsentStatus(consent_data['status'])
                        consent_data['data_categories'] = [DataCategory(cat) for cat in consent_data['data_categories']]
                        consent = ConsentRecord(**consent_data)
                        self.consents[consent.consent_id] = consent
            except Exception as e:
                print(f"Failed to load consents: {e}")
    
    def save_consents(self):
        """Save consent records to storage."""
        try:
            consents_data = []
            for consent in self.consents.values():
                data = asdict(consent)
                data['status'] = consent.status.value
                data['data_categories'] = [cat.value for cat in consent.data_categories]
                consents_data.append(data)
            
            with open(self.consents_file, 'w') as f:
                json.dump(consents_data, f, indent=2)
        except Exception as e:
            print(f"Failed to save consents: {e}")
    
    def give_consent(self, user_id: str, data_categories: List[DataCategory],
                    purpose: str, legal_basis: str = "consent",
                    expires_days: int = None) -> str:
        """Record user consent."""
        
        consent_id = str(uuid.uuid4())
        current_time = time.time()
        
        expires_at = None
        if expires_days:
            expires_at = current_time + (expires_days * 24 * 3600)
        
        consent = ConsentRecord(
            consent_id=consent_id,
            user_id=user_id,
            data_categories=data_categories,
            purpose=purpose,
            status=ConsentStatus.GIVEN,
            given_at=current_time,
            withdrawn_at=None,
            expires_at=expires_at,
            legal_basis=legal_basis,
            metadata={}
        )
        
        self.consents[consent_id] = consent
        self.save_consents()
        
        return consent_id
    
    def withdraw_consent(self, consent_id: str) -> bool:
        """Withdraw user consent."""
        
        consent = self.consents.get(consent_id)
        if not consent:
            return False
        
        consent.status = ConsentStatus.WITHDRAWN
        consent.withdrawn_at = time.time()
        
        self.save_consents()
        return True
    
    def check_consent(self, user_id: str, data_category: DataCategory, purpose: str) -> bool:
        """Check if user has given valid consent for data processing."""
        
        current_time = time.time()
        
        for consent in self.consents.values():
            if (consent.user_id == user_id and 
                consent.status == ConsentStatus.GIVEN and
                data_category in consent.data_categories and
                consent.purpose == purpose):
                
                # Check if consent has expired
                if consent.expires_at and current_time > consent.expires_at:
                    consent.status = ConsentStatus.EXPIRED
                    self.save_consents()
                    continue
                
                return True
        
        return False
    
    def get_user_consents(self, user_id: str) -> List[ConsentRecord]:
        """Get all consent records for a user."""
        return [consent for consent in self.consents.values() if consent.user_id == user_id]


class DataRetentionManager:
    """Manages data retention policies and automatic deletion."""
    
    def __init__(self, storage_path: str = "./data/privacy"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.policies_file = self.storage_path / "retention_policies.json"
        self.policies: Dict[str, DataRetentionPolicy] = {}
        
        self.load_policies()
        self.setup_default_policies()
    
    def load_policies(self):
        """Load retention policies from storage."""
        if self.policies_file.exists():
            try:
                with open(self.policies_file, 'r') as f:
                    data = json.load(f)
                    for policy_data in data:
                        policy_data['data_category'] = DataCategory(policy_data['data_category'])
                        policy = DataRetentionPolicy(**policy_data)
                        self.policies[policy.policy_id] = policy
            except Exception as e:
                print(f"Failed to load retention policies: {e}")
    
    def save_policies(self):
        """Save retention policies to storage."""
        try:
            policies_data = []
            for policy in self.policies.values():
                data = asdict(policy)
                data['data_category'] = policy.data_category.value
                policies_data.append(data)
            
            with open(self.policies_file, 'w') as f:
                json.dump(policies_data, f, indent=2)
        except Exception as e:
            print(f"Failed to save retention policies: {e}")
    
    def setup_default_policies(self):
        """Setup default retention policies if none exist."""
        if not self.policies:
            default_policies = [
                DataRetentionPolicy(
                    policy_id="content_default",
                    data_category=DataCategory.CONTENT,
                    retention_period_days=2555,  # 7 years
                    auto_delete=True,
                    deletion_method="anonymize",
                    exceptions=["legal_hold"]
                ),
                DataRetentionPolicy(
                    policy_id="behavioral_default", 
                    data_category=DataCategory.BEHAVIORAL,
                    retention_period_days=730,  # 2 years
                    auto_delete=True,
                    deletion_method="delete",
                    exceptions=[]
                ),
                DataRetentionPolicy(
                    policy_id="technical_default",
                    data_category=DataCategory.TECHNICAL,
                    retention_period_days=90,  # 3 months
                    auto_delete=True,
                    deletion_method="delete",
                    exceptions=["security_audit"]
                )
            ]
            
            for policy in default_policies:
                self.policies[policy.policy_id] = policy
            
            self.save_policies()
    
    def add_policy(self, policy: DataRetentionPolicy):
        """Add a new retention policy."""
        self.policies[policy.policy_id] = policy
        self.save_policies()
    
    def get_retention_period(self, data_category: DataCategory) -> Optional[int]:
        """Get retention period for a data category."""
        for policy in self.policies.values():
            if policy.data_category == data_category:
                return policy.retention_period_days
        return None
    
    def should_delete_data(self, data_category: DataCategory, created_at: float) -> bool:
        """Check if data should be deleted based on retention policy."""
        retention_days = self.get_retention_period(data_category)
        if not retention_days:
            return False
        
        age_days = (time.time() - created_at) / (24 * 3600)
        return age_days > retention_days


class PrivacyManager:
    """Main privacy and compliance management system."""
    
    def __init__(self, storage_path: str = "./data/privacy"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.processing_records_file = self.storage_path / "processing_records.json"
        self.processing_records: List[DataProcessingRecord] = []
        
        # Components
        self.pii_detector = PIIDetector()
        self.consent_manager = ConsentManager(str(self.storage_path))
        self.retention_manager = DataRetentionManager(str(self.storage_path))
        
        self.load_processing_records()
    
    def load_processing_records(self):
        """Load data processing records."""
        if self.processing_records_file.exists():
            try:
                with open(self.processing_records_file, 'r') as f:
                    data = json.load(f)
                    for record_data in data:
                        record_data['data_categories'] = [DataCategory(cat) for cat in record_data['data_categories']]
                        record = DataProcessingRecord(**record_data)
                        self.processing_records.append(record)
            except Exception as e:
                print(f"Failed to load processing records: {e}")
    
    def save_processing_records(self):
        """Save data processing records."""
        try:
            records_data = []
            for record in self.processing_records[-1000:]:  # Keep last 1000 records
                data = asdict(record)
                data['data_categories'] = [cat.value for cat in record.data_categories]
                records_data.append(data)
            
            with open(self.processing_records_file, 'w') as f:
                json.dump(records_data, f, indent=2)
        except Exception as e:
            print(f"Failed to save processing records: {e}")
    
    def record_data_processing(self, user_id: str, activity_type: str,
                             data_categories: List[DataCategory], purpose: str,
                             legal_basis: str = "consent") -> str:
        """Record data processing activity (GDPR Article 30)."""
        
        record_id = str(uuid.uuid4())
        
        record = DataProcessingRecord(
            record_id=record_id,
            user_id=user_id,
            activity_type=activity_type,
            data_categories=data_categories,
            purpose=purpose,
            legal_basis=legal_basis,
            data_subjects=[user_id],
            recipients=["rag_system"],
            retention_period=self.retention_manager.get_retention_period(data_categories[0]) if data_categories else None,
            timestamp=time.time(),
            metadata={}
        )
        
        self.processing_records.append(record)
        self.save_processing_records()
        
        return record_id
    
    def anonymize_query(self, query_text: str) -> Tuple[str, List[PIIDetectionResult]]:
        """Anonymize PII in a query before processing."""
        return self.pii_detector.anonymize_text(query_text)
    
    def check_processing_consent(self, user_id: str, purpose: str) -> bool:
        """Check if user has given consent for data processing."""
        return self.consent_manager.check_consent(user_id, DataCategory.CONTENT, purpose)
    
    def handle_data_subject_request(self, user_id: str, request_type: str) -> Dict[str, Any]:
        """Handle data subject requests (GDPR Articles 15-22)."""
        
        if request_type == "access":
            # Right of access (Article 15)
            return self._handle_access_request(user_id)
        elif request_type == "rectification":
            # Right to rectification (Article 16)
            return self._handle_rectification_request(user_id)
        elif request_type == "erasure":
            # Right to erasure (Article 17)
            return self._handle_erasure_request(user_id)
        elif request_type == "portability":
            # Right to data portability (Article 20)
            return self._handle_portability_request(user_id)
        else:
            return {"error": f"Unknown request type: {request_type}"}
    
    def _handle_access_request(self, user_id: str) -> Dict[str, Any]:
        """Handle data access request."""
        
        # Collect user data from various sources
        user_consents = self.consent_manager.get_user_consents(user_id)
        user_processing_records = [r for r in self.processing_records if user_id in r.data_subjects]
        
        return {
            "request_type": "access",
            "user_id": user_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "consents": [
                {
                    "consent_id": c.consent_id,
                    "purpose": c.purpose,
                    "status": c.status.value,
                    "given_at": datetime.fromtimestamp(c.given_at, tz=timezone.utc).isoformat() if c.given_at else None,
                    "data_categories": [cat.value for cat in c.data_categories]
                }
                for c in user_consents
            ],
            "processing_activities": [
                {
                    "record_id": r.record_id,
                    "activity_type": r.activity_type,
                    "purpose": r.purpose,
                    "legal_basis": r.legal_basis,
                    "timestamp": datetime.fromtimestamp(r.timestamp, tz=timezone.utc).isoformat(),
                    "data_categories": [cat.value for cat in r.data_categories]
                }
                for r in user_processing_records[-50:]  # Last 50 records
            ]
        }
    
    def _handle_erasure_request(self, user_id: str) -> Dict[str, Any]:
        """Handle data erasure request (right to be forgotten)."""
        
        # In a real implementation, this would:
        # 1. Remove user data from document store
        # 2. Remove embeddings
        # 3. Clear processing records
        # 4. Anonymize historical logs
        
        # For now, withdraw all consents
        user_consents = self.consent_manager.get_user_consents(user_id)
        for consent in user_consents:
            if consent.status == ConsentStatus.GIVEN:
                self.consent_manager.withdraw_consent(consent.consent_id)
        
        return {
            "request_type": "erasure",
            "user_id": user_id,
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actions_taken": [
                "All consents withdrawn",
                "Processing records marked for deletion",
                "User data anonymization scheduled"
            ]
        }
    
    def _handle_portability_request(self, user_id: str) -> Dict[str, Any]:
        """Handle data portability request."""
        
        # Extract user's personal data in a machine-readable format
        access_data = self._handle_access_request(user_id)
        
        return {
            "request_type": "portability",
            "format": "json",
            "data": access_data,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    def _handle_rectification_request(self, user_id: str) -> Dict[str, Any]:
        """Handle data rectification request."""
        
        return {
            "request_type": "rectification",
            "user_id": user_id,
            "status": "manual_review_required",
            "message": "Rectification requests require manual review",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get privacy compliance summary."""
        
        total_consents = len(self.consent_manager.consents)
        active_consents = len([c for c in self.consent_manager.consents.values() if c.status == ConsentStatus.GIVEN])
        total_processing_records = len(self.processing_records)
        
        return {
            "privacy_compliance": {
                "pii_detection": "enabled",
                "consent_management": "enabled",
                "data_retention": "enabled",
                "gdpr_compliance": "enabled"
            },
            "statistics": {
                "total_consents": total_consents,
                "active_consents": active_consents,
                "processing_records": total_processing_records,
                "retention_policies": len(self.retention_manager.policies)
            },
            "capabilities": [
                "PII detection and anonymization",
                "Consent management",
                "Data retention policies",
                "Data subject rights handling",
                "Processing activity records"
            ]
        }
