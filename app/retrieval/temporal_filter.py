"""Enhanced temporal filtering with time-aware ranking."""
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import dateutil.parser
    from dateutil.relativedelta import relativedelta
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False


class TemporalQueryType(Enum):
    """Types of temporal queries."""
    HISTORICAL = "historical"           # Past events, "when was"
    CURRENT = "current"                 # Current state, "what is now"
    FUTURE = "future"                   # Future predictions, "what will"
    RANGE = "range"                     # Time range queries, "between X and Y"
    RECENT = "recent"                   # Recent events, "latest", "recent"
    COMPARATIVE = "comparative"         # Compare across time periods
    TIMELINE = "timeline"              # Chronological sequence
    DURATION = "duration"              # How long something took
    FREQUENCY = "frequency"            # How often something occurs


@dataclass
class TemporalExtraction:
    """Extracted temporal information from query."""
    query_type: TemporalQueryType
    time_expressions: List[str]
    extracted_dates: List[datetime]
    time_range: Optional[Tuple[datetime, datetime]]
    relative_time: Optional[str]  # "last week", "recent", etc.
    temporal_keywords: List[str]
    confidence: float


@dataclass
class DocumentTemporalInfo:
    """Temporal information extracted from a document."""
    document_id: str
    creation_date: Optional[datetime]
    modification_date: Optional[datetime]
    extracted_dates: List[datetime]
    temporal_keywords: List[str]
    content_time_period: Optional[Tuple[datetime, datetime]]
    temporal_relevance_score: float


class TemporalQueryAnalyzer:
    """Analyzes queries for temporal intent and extracts time information."""
    
    def __init__(self):
        self.dateutil_available = DATEUTIL_AVAILABLE
        
        # Temporal keywords by category
        self.temporal_keywords = {
            'past': ['when', 'history', 'historical', 'past', 'previous', 'before', 'earlier', 'old', 'ancient'],
            'present': ['now', 'current', 'currently', 'today', 'present', 'existing', 'modern'],
            'future': ['future', 'will', 'predict', 'forecast', 'upcoming', 'planned', 'expected'],
            'recent': ['recent', 'latest', 'new', 'fresh', 'updated', 'newest', 'just', 'recently'],
            'range': ['between', 'from', 'to', 'during', 'within', 'through', 'period'],
            'frequency': ['often', 'always', 'never', 'sometimes', 'usually', 'frequently', 'rarely'],
            'duration': ['long', 'short', 'duration', 'time', 'lasted', 'took', 'spent']
        }
        
        # Date pattern regexes
        self.date_patterns = [
            # ISO format: YYYY-MM-DD
            re.compile(r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b'),
            # US format: MM/DD/YYYY
            re.compile(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b'),
            # European format: DD/MM/YYYY
            re.compile(r'\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b'),
            # Year only
            re.compile(r'\b(19|20)\d{2}\b'),
            # Month Year
            re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b', re.IGNORECASE),
            # Relative dates
            re.compile(r'\b(last|past|next)\s+(week|month|year|day)\b', re.IGNORECASE),
            re.compile(r'\b(\d+)\s+(days?|weeks?|months?|years?)\s+(ago|from now)\b', re.IGNORECASE),
        ]
        
        # Time range patterns
        self.range_patterns = [
            re.compile(r'\bbetween\s+(.+?)\s+and\s+(.+?)\b', re.IGNORECASE),
            re.compile(r'\bfrom\s+(.+?)\s+to\s+(.+?)\b', re.IGNORECASE),
            re.compile(r'\bduring\s+(.+?)\b', re.IGNORECASE),
            re.compile(r'\bin\s+(\d{4})\b'),  # "in 2020"
        ]
        
        if not self.dateutil_available:
            print("Warning: dateutil not available. Date parsing will be limited.")
    
    def analyze_temporal_query(self, query: str) -> TemporalExtraction:
        """Analyze query for temporal intent and extract time information."""
        
        query_lower = query.lower()
        
        # Detect temporal query type
        query_type = self._classify_temporal_query(query_lower)
        
        # Extract time expressions
        time_expressions = self._extract_time_expressions(query)
        
        # Extract specific dates
        extracted_dates = self._extract_dates(query)
        
        # Extract time ranges
        time_range = self._extract_time_range(query)
        
        # Extract relative time expressions
        relative_time = self._extract_relative_time(query_lower)
        
        # Find temporal keywords
        temporal_keywords = self._find_temporal_keywords(query_lower)
        
        # Calculate confidence based on extracted information
        confidence = self._calculate_temporal_confidence(
            query_type, time_expressions, extracted_dates, 
            time_range, relative_time, temporal_keywords
        )
        
        return TemporalExtraction(
            query_type=query_type,
            time_expressions=time_expressions,
            extracted_dates=extracted_dates,
            time_range=time_range,
            relative_time=relative_time,
            temporal_keywords=temporal_keywords,
            confidence=confidence
        )
    
    def _classify_temporal_query(self, query: str) -> TemporalQueryType:
        """Classify the type of temporal query."""
        
        # Check for specific patterns
        if any(word in query for word in ['when was', 'history', 'historical']):
            return TemporalQueryType.HISTORICAL
        
        if any(word in query for word in ['now', 'current', 'currently', 'today']):
            return TemporalQueryType.CURRENT
        
        if any(word in query for word in ['will', 'future', 'predict', 'forecast']):
            return TemporalQueryType.FUTURE
        
        if any(word in query for word in ['between', 'from', 'during']):
            return TemporalQueryType.RANGE
        
        if any(word in query for word in ['recent', 'latest', 'new']):
            return TemporalQueryType.RECENT
        
        if any(word in query for word in ['compare', 'over time', 'timeline']):
            return TemporalQueryType.COMPARATIVE
        
        if any(word in query for word in ['how long', 'duration', 'took']):
            return TemporalQueryType.DURATION
        
        if any(word in query for word in ['how often', 'frequency', 'always']):
            return TemporalQueryType.FREQUENCY
        
        # Default to historical if any temporal keywords found
        for keywords in self.temporal_keywords.values():
            if any(word in query for word in keywords):
                return TemporalQueryType.HISTORICAL
        
        return TemporalQueryType.CURRENT  # Default
    
    def _extract_time_expressions(self, query: str) -> List[str]:
        """Extract time expressions from query."""
        
        expressions = []
        
        # Find date patterns
        for pattern in self.date_patterns:
            matches = pattern.findall(query)
            for match in matches:
                if isinstance(match, tuple):
                    expressions.append(' '.join(str(m) for m in match))
                else:
                    expressions.append(str(match))
        
        # Find relative time expressions
        relative_patterns = [
            r'\blast\s+\w+',
            r'\bnext\s+\w+',
            r'\b\d+\s+\w+\s+ago',
            r'\brecent\w*',
            r'\btoday\b',
            r'\byesterday\b',
            r'\btomorrow\b'
        ]
        
        for pattern in relative_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            expressions.extend(matches)
        
        return list(set(expressions))
    
    def _extract_dates(self, query: str) -> List[datetime]:
        """Extract specific dates from query."""
        
        dates = []
        
        if not self.dateutil_available:
            return dates
        
        # Try to parse with dateutil
        time_expressions = self._extract_time_expressions(query)
        
        for expr in time_expressions:
            try:
                parsed_date = dateutil.parser.parse(expr, fuzzy=True)
                dates.append(parsed_date)
            except (ValueError, TypeError):
                # Try manual parsing for common formats
                manual_date = self._manual_date_parsing(expr)
                if manual_date:
                    dates.append(manual_date)
        
        return dates
    
    def _manual_date_parsing(self, expr: str) -> Optional[datetime]:
        """Manual date parsing for common formats."""
        
        try:
            # Year only
            year_match = re.match(r'^(19|20)\d{2}$', expr)
            if year_match:
                return datetime(int(expr), 1, 1)
            
            # Relative dates
            if 'last year' in expr.lower():
                return datetime.now() - relativedelta(years=1)
            elif 'last month' in expr.lower():
                return datetime.now() - relativedelta(months=1)
            elif 'last week' in expr.lower():
                return datetime.now() - timedelta(weeks=1)
            
        except ValueError:
            pass
        
        return None
    
    def _extract_time_range(self, query: str) -> Optional[Tuple[datetime, datetime]]:
        """Extract time ranges from query."""
        
        for pattern in self.range_patterns:
            match = pattern.search(query)
            if match:
                if len(match.groups()) >= 2:
                    start_str, end_str = match.groups()[:2]
                    
                    try:
                        if self.dateutil_available:
                            start_date = dateutil.parser.parse(start_str, fuzzy=True)
                            end_date = dateutil.parser.parse(end_str, fuzzy=True)
                            return (start_date, end_date)
                    except (ValueError, TypeError):
                        pass
        
        return None
    
    def _extract_relative_time(self, query: str) -> Optional[str]:
        """Extract relative time expressions."""
        
        relative_patterns = [
            'last week', 'last month', 'last year',
            'recent', 'recently', 'latest',
            'past few', 'current', 'now'
        ]
        
        for pattern in relative_patterns:
            if pattern in query:
                return pattern
        
        return None
    
    def _find_temporal_keywords(self, query: str) -> List[str]:
        """Find temporal keywords in query."""
        
        found_keywords = []
        
        for category, keywords in self.temporal_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    found_keywords.append(keyword)
        
        return found_keywords
    
    def _calculate_temporal_confidence(self, query_type: TemporalQueryType,
                                     time_expressions: List[str],
                                     extracted_dates: List[datetime],
                                     time_range: Optional[Tuple[datetime, datetime]],
                                     relative_time: Optional[str],
                                     temporal_keywords: List[str]) -> float:
        """Calculate confidence in temporal analysis."""
        
        confidence = 0.0
        
        # Base confidence from temporal keywords
        confidence += min(0.3, len(temporal_keywords) * 0.1)
        
        # Boost for specific dates
        confidence += min(0.3, len(extracted_dates) * 0.15)
        
        # Boost for time expressions
        confidence += min(0.2, len(time_expressions) * 0.1)
        
        # Boost for time range
        if time_range:
            confidence += 0.2
        
        # Boost for relative time
        if relative_time:
            confidence += 0.15
        
        # Query type specific adjustments
        if query_type in [TemporalQueryType.CURRENT, TemporalQueryType.RECENT]:
            confidence += 0.1
        
        return min(1.0, confidence)


class DocumentTemporalAnalyzer:
    """Analyzes documents for temporal information."""
    
    def __init__(self):
        self.query_analyzer = TemporalQueryAnalyzer()
    
    def analyze_document_temporal_info(self, document_id: str, content: str, 
                                     metadata: Dict[str, Any]) -> DocumentTemporalInfo:
        """Analyze document for temporal information."""
        
        # Extract creation/modification dates from metadata
        creation_date = self._parse_date_from_metadata(metadata, 'creation_date')
        modification_date = self._parse_date_from_metadata(metadata, 'modification_date')
        
        # Extract dates from content
        extracted_dates = self._extract_dates_from_content(content)
        
        # Find temporal keywords
        temporal_keywords = self._find_temporal_keywords_in_content(content)
        
        # Determine content time period
        content_time_period = self._determine_content_time_period(extracted_dates)
        
        # Calculate temporal relevance score
        temporal_relevance_score = self._calculate_temporal_relevance(
            creation_date, modification_date, extracted_dates, temporal_keywords
        )
        
        return DocumentTemporalInfo(
            document_id=document_id,
            creation_date=creation_date,
            modification_date=modification_date,
            extracted_dates=extracted_dates,
            temporal_keywords=temporal_keywords,
            content_time_period=content_time_period,
            temporal_relevance_score=temporal_relevance_score
        )
    
    def _parse_date_from_metadata(self, metadata: Dict[str, Any], key: str) -> Optional[datetime]:
        """Parse date from metadata."""
        
        date_value = metadata.get(key)
        if not date_value:
            return None
        
        if isinstance(date_value, datetime):
            return date_value
        
        if isinstance(date_value, str):
            try:
                if DATEUTIL_AVAILABLE:
                    return dateutil.parser.parse(date_value)
                else:
                    # Basic parsing
                    return datetime.fromisoformat(date_value.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                pass
        
        return None
    
    def _extract_dates_from_content(self, content: str) -> List[datetime]:
        """Extract dates mentioned in document content."""
        
        # Use the same logic as query analyzer
        temp_extraction = self.query_analyzer.analyze_temporal_query(content[:2000])  # Limit for performance
        return temp_extraction.extracted_dates
    
    def _find_temporal_keywords_in_content(self, content: str) -> List[str]:
        """Find temporal keywords in document content."""
        
        content_lower = content.lower()
        found_keywords = []
        
        for category, keywords in self.query_analyzer.temporal_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    found_keywords.append(keyword)
        
        return list(set(found_keywords))
    
    def _determine_content_time_period(self, dates: List[datetime]) -> Optional[Tuple[datetime, datetime]]:
        """Determine the time period covered by document content."""
        
        if len(dates) < 2:
            return None
        
        # Return range from earliest to latest date
        sorted_dates = sorted(dates)
        return (sorted_dates[0], sorted_dates[-1])
    
    def _calculate_temporal_relevance(self, creation_date: Optional[datetime],
                                    modification_date: Optional[datetime],
                                    extracted_dates: List[datetime],
                                    temporal_keywords: List[str]) -> float:
        """Calculate overall temporal relevance score for document."""
        
        score = 0.0
        
        # Base score for having temporal information
        if creation_date or modification_date:
            score += 0.2
        
        # Score for dates in content
        score += min(0.4, len(extracted_dates) * 0.1)
        
        # Score for temporal keywords
        score += min(0.3, len(temporal_keywords) * 0.05)
        
        # Recency bonus
        if modification_date:
            days_old = (datetime.now() - modification_date).days
            if days_old < 30:
                score += 0.1
            elif days_old < 365:
                score += 0.05
        
        return min(1.0, score)


class TemporalRankingAdjuster:
    """Adjusts ranking based on temporal relevance."""
    
    def __init__(self):
        self.query_analyzer = TemporalQueryAnalyzer()
        self.doc_analyzer = DocumentTemporalAnalyzer()
    
    def adjust_ranking_for_temporal_query(self, query: str, 
                                        results: List[Tuple[str, float]], 
                                        document_temporal_info: Dict[str, DocumentTemporalInfo]) -> List[Tuple[str, float]]:
        """Adjust ranking based on temporal relevance."""
        
        # Analyze query for temporal intent
        temporal_extraction = self.query_analyzer.analyze_temporal_query(query)
        
        if temporal_extraction.confidence < 0.3:
            # Not a temporal query, return unchanged
            return results
        
        adjusted_results = []
        
        for doc_id, score in results:
            doc_temporal = document_temporal_info.get(doc_id)
            
            if not doc_temporal:
                # No temporal info available, keep original score
                adjusted_results.append((doc_id, score))
                continue
            
            # Calculate temporal adjustment factor
            temporal_boost = self._calculate_temporal_boost(
                temporal_extraction, doc_temporal
            )
            
            # Apply temporal boost
            adjusted_score = score * (1.0 + temporal_boost)
            adjusted_results.append((doc_id, adjusted_score))
        
        # Re-sort by adjusted scores
        adjusted_results.sort(key=lambda x: x[1], reverse=True)
        
        return adjusted_results
    
    def _calculate_temporal_boost(self, query_temporal: TemporalExtraction, 
                                doc_temporal: DocumentTemporalInfo) -> float:
        """Calculate temporal boost factor for a document."""
        
        boost = 0.0
        
        if query_temporal.query_type == TemporalQueryType.RECENT:
            # Boost recent documents
            if doc_temporal.modification_date:
                days_old = (datetime.now() - doc_temporal.modification_date).days
                if days_old < 7:
                    boost += 0.3
                elif days_old < 30:
                    boost += 0.2
                elif days_old < 90:
                    boost += 0.1
        
        elif query_temporal.query_type == TemporalQueryType.HISTORICAL:
            # Boost older documents or documents with historical content
            if doc_temporal.content_time_period:
                start_date, _ = doc_temporal.content_time_period
                if start_date.year < 2000:
                    boost += 0.2
                elif start_date.year < 2010:
                    boost += 0.1
        
        elif query_temporal.query_type == TemporalQueryType.RANGE:
            # Boost documents that fall within the query time range
            if query_temporal.time_range and doc_temporal.content_time_period:
                query_start, query_end = query_temporal.time_range
                doc_start, doc_end = doc_temporal.content_time_period
                
                # Check for overlap
                if (doc_start <= query_end and doc_end >= query_start):
                    boost += 0.25
        
        elif query_temporal.query_type == TemporalQueryType.CURRENT:
            # Boost recently modified documents
            if doc_temporal.modification_date:
                days_old = (datetime.now() - doc_temporal.modification_date).days
                if days_old < 30:
                    boost += 0.2
                elif days_old < 180:
                    boost += 0.1
        
        # Date matching boost
        if query_temporal.extracted_dates and doc_temporal.extracted_dates:
            for query_date in query_temporal.extracted_dates:
                for doc_date in doc_temporal.extracted_dates:
                    # Boost for exact year match
                    if query_date.year == doc_date.year:
                        boost += 0.15
                        # Extra boost for exact date match
                        if query_date.date() == doc_date.date():
                            boost += 0.1
        
        # Temporal relevance boost
        boost += doc_temporal.temporal_relevance_score * 0.1
        
        # Apply confidence weighting
        boost *= query_temporal.confidence
        
        return min(0.5, boost)  # Cap the boost at 50%
    
    def filter_by_time_range(self, results: List[Tuple[str, float]],
                           document_temporal_info: Dict[str, DocumentTemporalInfo],
                           time_range: Tuple[datetime, datetime]) -> List[Tuple[str, float]]:
        """Filter results to only include documents within a time range."""
        
        filtered_results = []
        start_date, end_date = time_range
        
        for doc_id, score in results:
            doc_temporal = document_temporal_info.get(doc_id)
            
            if not doc_temporal:
                continue
            
            # Check if document falls within time range
            include_doc = False
            
            # Check modification date
            if doc_temporal.modification_date:
                if start_date <= doc_temporal.modification_date <= end_date:
                    include_doc = True
            
            # Check content time period
            if doc_temporal.content_time_period and not include_doc:
                doc_start, doc_end = doc_temporal.content_time_period
                if (doc_start <= end_date and doc_end >= start_date):
                    include_doc = True
            
            # Check extracted dates
            if doc_temporal.extracted_dates and not include_doc:
                for doc_date in doc_temporal.extracted_dates:
                    if start_date <= doc_date <= end_date:
                        include_doc = True
                        break
            
            if include_doc:
                filtered_results.append((doc_id, score))
        
        return filtered_results


class TemporalFilterManager:
    """Main manager for temporal filtering functionality."""
    
    def __init__(self):
        self.query_analyzer = TemporalQueryAnalyzer()
        self.doc_analyzer = DocumentTemporalAnalyzer()
        self.ranking_adjuster = TemporalRankingAdjuster()
        
        # Cache for document temporal information
        self.doc_temporal_cache: Dict[str, DocumentTemporalInfo] = {}
    
    def analyze_and_cache_document(self, document_id: str, content: str, 
                                 metadata: Dict[str, Any]) -> DocumentTemporalInfo:
        """Analyze document for temporal info and cache it."""
        
        doc_temporal = self.doc_analyzer.analyze_document_temporal_info(
            document_id, content, metadata
        )
        
        self.doc_temporal_cache[document_id] = doc_temporal
        return doc_temporal
    
    def apply_temporal_filtering(self, query: str, results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Apply temporal filtering and ranking adjustments to search results."""
        
        # Analyze query for temporal intent
        temporal_extraction = self.query_analyzer.analyze_temporal_query(query)
        
        if temporal_extraction.confidence < 0.3:
            # Not a temporal query
            return results
        
        # Apply temporal ranking adjustments
        adjusted_results = self.ranking_adjuster.adjust_ranking_for_temporal_query(
            query, results, self.doc_temporal_cache
        )
        
        # Apply time range filtering if specified
        if temporal_extraction.time_range:
            adjusted_results = self.ranking_adjuster.filter_by_time_range(
                adjusted_results, self.doc_temporal_cache, temporal_extraction.time_range
            )
        
        return adjusted_results
    
    def get_temporal_explanation(self, query: str) -> Dict[str, Any]:
        """Get explanation of temporal analysis for query."""
        
        temporal_extraction = self.query_analyzer.analyze_temporal_query(query)
        
        return {
            "query": query,
            "temporal_analysis": {
                "query_type": temporal_extraction.query_type.value,
                "confidence": temporal_extraction.confidence,
                "time_expressions": temporal_extraction.time_expressions,
                "extracted_dates": [d.isoformat() for d in temporal_extraction.extracted_dates],
                "time_range": [d.isoformat() for d in temporal_extraction.time_range] if temporal_extraction.time_range else None,
                "relative_time": temporal_extraction.relative_time,
                "temporal_keywords": temporal_extraction.temporal_keywords
            },
            "filtering_applied": temporal_extraction.confidence >= 0.3
        }
    
    def get_temporal_statistics(self) -> Dict[str, Any]:
        """Get statistics about temporal information in cached documents."""
        
        if not self.doc_temporal_cache:
            return {"message": "No temporal information cached"}
        
        docs = list(self.doc_temporal_cache.values())
        
        # Count documents by temporal characteristics
        has_creation_date = sum(1 for d in docs if d.creation_date)
        has_modification_date = sum(1 for d in docs if d.modification_date)
        has_extracted_dates = sum(1 for d in docs if d.extracted_dates)
        has_temporal_keywords = sum(1 for d in docs if d.temporal_keywords)
        
        # Calculate average temporal relevance
        avg_temporal_relevance = sum(d.temporal_relevance_score for d in docs) / len(docs)
        
        # Find time period coverage
        all_dates = []
        for doc in docs:
            if doc.creation_date:
                all_dates.append(doc.creation_date)
            if doc.modification_date:
                all_dates.append(doc.modification_date)
            all_dates.extend(doc.extracted_dates)
        
        time_coverage = None
        if all_dates:
            sorted_dates = sorted(all_dates)
            time_coverage = {
                "earliest": sorted_dates[0].isoformat(),
                "latest": sorted_dates[-1].isoformat(),
                "span_years": (sorted_dates[-1] - sorted_dates[0]).days / 365.25
            }
        
        return {
            "total_documents": len(docs),
            "documents_with_creation_date": has_creation_date,
            "documents_with_modification_date": has_modification_date,
            "documents_with_extracted_dates": has_extracted_dates,
            "documents_with_temporal_keywords": has_temporal_keywords,
            "average_temporal_relevance": round(avg_temporal_relevance, 3),
            "time_coverage": time_coverage
        }
