"""Document version tracking and change detection."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

import difflib


@dataclass
class DocumentVersion:
    """Represents a version of a document."""
    version_id: str
    document_id: str
    filename: str
    content_hash: str
    timestamp: float
    size_bytes: int
    change_summary: str
    metadata: Dict[str, Any]
    previous_version: Optional[str] = None


@dataclass
class ChangeDetection:
    """Represents detected changes between document versions."""
    added_sections: List[str]
    removed_sections: List[str]
    modified_sections: List[str]
    change_ratio: float
    significant_changes: bool
    summary: str


class DocumentVersionManager:
    """Manages document versions and tracks changes."""
    
    def __init__(self, storage_path: str = "./data/versions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.versions_file = self.storage_path / "versions.json"
        self.versions: Dict[str, List[DocumentVersion]] = {}
        self.load_versions()
    
    def load_versions(self):
        """Load version history from storage."""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r') as f:
                    data = json.load(f)
                    self.versions = {}
                    for doc_id, version_list in data.items():
                        self.versions[doc_id] = [
                            DocumentVersion(**v) for v in version_list
                        ]
            except Exception as e:
                print(f"Failed to load versions: {e}")
                self.versions = {}
    
    def save_versions(self):
        """Save version history to storage."""
        try:
            data = {}
            for doc_id, version_list in self.versions.items():
                data[doc_id] = [asdict(v) for v in version_list]
            
            with open(self.versions_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to save versions: {e}")
    
    def add_version(self, document_id: str, filename: str, content: str, metadata: Dict[str, Any] = None) -> DocumentVersion:
        """Add a new version of a document."""
        # Calculate content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Check if this exact version already exists
        if document_id in self.versions:
            for version in self.versions[document_id]:
                if version.content_hash == content_hash:
                    return version  # Same content, return existing version
        
        # Create new version
        version_id = f"{document_id}_v{int(time.time())}"
        previous_version = None
        
        if document_id in self.versions and self.versions[document_id]:
            previous_version = self.versions[document_id][-1].version_id
        
        # Detect changes if there's a previous version
        change_summary = "Initial version"
        if previous_version:
            prev_content = self.get_version_content(document_id, previous_version)
            if prev_content:
                change_detection = self.detect_changes(prev_content, content)
                change_summary = change_detection.summary
        
        new_version = DocumentVersion(
            version_id=version_id,
            document_id=document_id,
            filename=filename,
            content_hash=content_hash,
            timestamp=time.time(),
            size_bytes=len(content.encode()),
            change_summary=change_summary,
            metadata=metadata or {},
            previous_version=previous_version
        )
        
        # Add to versions list
        if document_id not in self.versions:
            self.versions[document_id] = []
        self.versions[document_id].append(new_version)
        
        # Store content
        self.store_version_content(version_id, content)
        
        # Save to disk
        self.save_versions()
        
        return new_version
    
    def store_version_content(self, version_id: str, content: str):
        """Store version content to disk."""
        content_file = self.storage_path / f"{version_id}.txt"
        try:
            with open(content_file, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"Failed to store content for {version_id}: {e}")
    
    def get_version_content(self, document_id: str, version_id: str) -> Optional[str]:
        """Retrieve content for a specific version."""
        content_file = self.storage_path / f"{version_id}.txt"
        if content_file.exists():
            try:
                with open(content_file, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"Failed to read content for {version_id}: {e}")
        return None
    
    def get_document_versions(self, document_id: str) -> List[DocumentVersion]:
        """Get all versions of a document."""
        return self.versions.get(document_id, [])
    
    def get_latest_version(self, document_id: str) -> Optional[DocumentVersion]:
        """Get the latest version of a document."""
        versions = self.get_document_versions(document_id)
        return versions[-1] if versions else None
    
    def detect_changes(self, old_content: str, new_content: str) -> ChangeDetection:
        """Detect changes between two versions of content."""
        old_lines = old_content.splitlines()
        new_lines = new_content.splitlines()
        
        # Use difflib to get detailed differences
        differ = difflib.unified_diff(old_lines, new_lines, lineterm='')
        diff_lines = list(differ)
        
        added_sections = []
        removed_sections = []
        modified_sections = []
        
        # Parse diff to categorize changes
        current_section = []
        change_type = None
        
        for line in diff_lines:
            if line.startswith('+++') or line.startswith('---') or line.startswith('@@'):
                continue
            elif line.startswith('+'):
                if change_type != 'added':
                    if current_section and change_type:
                        self._categorize_section(current_section, change_type, added_sections, removed_sections, modified_sections)
                    current_section = []
                    change_type = 'added'
                current_section.append(line[1:])
            elif line.startswith('-'):
                if change_type != 'removed':
                    if current_section and change_type:
                        self._categorize_section(current_section, change_type, added_sections, removed_sections, modified_sections)
                    current_section = []
                    change_type = 'removed'
                current_section.append(line[1:])
            else:
                if current_section and change_type:
                    self._categorize_section(current_section, change_type, added_sections, removed_sections, modified_sections)
                current_section = []
                change_type = None
        
        # Handle last section
        if current_section and change_type:
            self._categorize_section(current_section, change_type, added_sections, removed_sections, modified_sections)
        
        # Calculate change ratio
        total_lines = max(len(old_lines), len(new_lines))
        changed_lines = len(added_sections) + len(removed_sections) + len(modified_sections)
        change_ratio = changed_lines / total_lines if total_lines > 0 else 0.0
        
        # Determine if changes are significant
        significant_changes = (
            change_ratio > 0.1 or  # More than 10% change
            len(added_sections) > 5 or
            len(removed_sections) > 5
        )
        
        # Generate summary
        summary = self._generate_change_summary(added_sections, removed_sections, modified_sections, change_ratio)
        
        return ChangeDetection(
            added_sections=added_sections,
            removed_sections=removed_sections,
            modified_sections=modified_sections,
            change_ratio=change_ratio,
            significant_changes=significant_changes,
            summary=summary
        )
    
    def _categorize_section(self, section: List[str], change_type: str, added: List[str], removed: List[str], modified: List[str]):
        """Categorize a changed section."""
        section_text = '\n'.join(section)
        if change_type == 'added':
            added.append(section_text)
        elif change_type == 'removed':
            removed.append(section_text)
        else:
            modified.append(section_text)
    
    def _generate_change_summary(self, added: List[str], removed: List[str], modified: List[str], ratio: float) -> str:
        """Generate a human-readable change summary."""
        changes = []
        
        if added:
            changes.append(f"{len(added)} section(s) added")
        if removed:
            changes.append(f"{len(removed)} section(s) removed")
        if modified:
            changes.append(f"{len(modified)} section(s) modified")
        
        if not changes:
            return "No significant changes detected"
        
        summary = ", ".join(changes)
        summary += f" ({ratio:.1%} of document changed)"
        
        return summary
    
    def get_version_diff(self, document_id: str, version1_id: str, version2_id: str) -> Optional[str]:
        """Get a diff between two versions."""
        content1 = self.get_version_content(document_id, version1_id)
        content2 = self.get_version_content(document_id, version2_id)
        
        if not content1 or not content2:
            return None
        
        diff = difflib.unified_diff(
            content1.splitlines(keepends=True),
            content2.splitlines(keepends=True),
            fromfile=f"Version {version1_id}",
            tofile=f"Version {version2_id}"
        )
        
        return ''.join(diff)
    
    def cleanup_old_versions(self, document_id: str, keep_count: int = 10):
        """Remove old versions, keeping only the specified number."""
        versions = self.get_document_versions(document_id)
        
        if len(versions) <= keep_count:
            return
        
        # Keep the most recent versions
        to_remove = versions[:-keep_count]
        
        for version in to_remove:
            # Remove content file
            content_file = self.storage_path / f"{version.version_id}.txt"
            content_file.unlink(missing_ok=True)
        
        # Update versions list
        self.versions[document_id] = versions[-keep_count:]
        self.save_versions()
    
    def get_version_stats(self, document_id: str) -> Dict[str, Any]:
        """Get statistics about document versions."""
        versions = self.get_document_versions(document_id)
        
        if not versions:
            return {}
        
        total_versions = len(versions)
        latest_version = versions[-1]
        first_version = versions[0]
        
        # Calculate time span
        time_span = latest_version.timestamp - first_version.timestamp
        
        # Calculate average change frequency
        avg_change_frequency = time_span / (total_versions - 1) if total_versions > 1 else 0
        
        # Calculate size evolution
        sizes = [v.size_bytes for v in versions]
        size_growth = (sizes[-1] - sizes[0]) / sizes[0] if sizes[0] > 0 else 0
        
        return {
            'total_versions': total_versions,
            'first_version_date': datetime.fromtimestamp(first_version.timestamp, tz=timezone.utc).isoformat(),
            'latest_version_date': datetime.fromtimestamp(latest_version.timestamp, tz=timezone.utc).isoformat(),
            'time_span_days': time_span / 86400,  # Convert to days
            'avg_change_frequency_hours': avg_change_frequency / 3600,  # Convert to hours
            'size_growth_ratio': size_growth,
            'current_size_bytes': latest_version.size_bytes,
            'has_significant_changes': any('significant' in v.change_summary.lower() for v in versions)
        }
