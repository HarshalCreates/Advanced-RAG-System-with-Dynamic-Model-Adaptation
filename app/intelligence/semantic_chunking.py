"""Semantic boundary detection for intelligent document chunking."""
from __future__ import annotations

import re
import math
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

try:
    import spacy
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False


@dataclass
class SemanticChunk:
    """Represents a semantically coherent chunk of text."""
    id: str
    text: str
    start_pos: int
    end_pos: int
    semantic_score: float
    topic_keywords: List[str]
    section_title: Optional[str] = None
    chunk_type: str = "content"  # content, header, list, table, etc.
    metadata: Dict[str, Any] = None


@dataclass
class ChunkingConfig:
    """Configuration for semantic chunking."""
    min_chunk_size: int = 100
    max_chunk_size: int = 1000
    overlap_size: int = 50
    similarity_threshold: float = 0.3
    preserve_sections: bool = True
    preserve_paragraphs: bool = True
    preserve_sentences: bool = False


class SemanticChunker:
    """Advanced semantic chunking using multiple strategies."""
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self.nlp = None
        self.sentence_model = None
        
        # Initialize models if available
        if ADVANCED_AVAILABLE:
            try:
                # Load spaCy model
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                try:
                    # Fallback to smaller model
                    self.nlp = spacy.load("en_core_web_md")
                except OSError:
                    print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            
            try:
                # Load sentence transformer for semantic similarity
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Could not load sentence transformer: {e}")
        
        # Text patterns for structure detection
        self.section_patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown headers
            r'^([A-Z][A-Z\s]{2,}):?\s*$',  # ALL CAPS headers
            r'^\d+\.?\s+([A-Z].*)',  # Numbered sections
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):?\s*$',  # Title case
        ]
        
        self.list_patterns = [
            r'^\s*[-*•]\s+',  # Bullet points
            r'^\s*\d+\.\s+',  # Numbered lists
            r'^\s*[a-zA-Z]\.\s+',  # Lettered lists
        ]
        
        # Semantic boundary indicators
        self.transition_words = {
            'contrast': ['however', 'but', 'nevertheless', 'on the other hand', 'conversely'],
            'continuation': ['furthermore', 'moreover', 'additionally', 'also', 'likewise'],
            'conclusion': ['therefore', 'thus', 'consequently', 'in conclusion', 'finally'],
            'example': ['for example', 'for instance', 'such as', 'namely', 'specifically']
        }
    
    def chunk_document(self, text: str, document_id: str = "doc") -> List[SemanticChunk]:
        """Chunk document using semantic boundary detection."""
        
        # Step 1: Preprocess and structure analysis
        structured_text = self._analyze_document_structure(text)
        
        # Step 2: Identify semantic boundaries
        boundaries = self._find_semantic_boundaries(structured_text)
        
        # Step 3: Create chunks respecting boundaries
        chunks = self._create_semantic_chunks(structured_text, boundaries, document_id)
        
        # Step 4: Post-process chunks (merge/split if needed)
        optimized_chunks = self._optimize_chunks(chunks)
        
        return optimized_chunks
    
    def _analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure and identify components."""
        lines = text.split('\n')
        
        structure = {
            'original_text': text,
            'lines': lines,
            'sections': [],
            'paragraphs': [],
            'lists': [],
            'line_types': []
        }
        
        current_section = None
        current_paragraph = []
        in_list = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if not stripped:
                # Empty line - paragraph boundary
                if current_paragraph:
                    structure['paragraphs'].append({
                        'text': '\n'.join(current_paragraph),
                        'start_line': i - len(current_paragraph),
                        'end_line': i - 1,
                        'section': current_section
                    })
                    current_paragraph = []
                in_list = False
                structure['line_types'].append('empty')
                continue
            
            # Check for section headers
            section_match = self._detect_section_header(stripped)
            if section_match:
                # Finish current paragraph
                if current_paragraph:
                    structure['paragraphs'].append({
                        'text': '\n'.join(current_paragraph),
                        'start_line': i - len(current_paragraph),
                        'end_line': i - 1,
                        'section': current_section
                    })
                    current_paragraph = []
                
                current_section = section_match
                structure['sections'].append({
                    'title': section_match,
                    'line': i,
                    'level': self._get_header_level(stripped)
                })
                structure['line_types'].append('header')
                in_list = False
                continue
            
            # Check for list items
            if self._is_list_item(stripped):
                if not in_list:
                    # Starting new list
                    if current_paragraph:
                        structure['paragraphs'].append({
                            'text': '\n'.join(current_paragraph),
                            'start_line': i - len(current_paragraph),
                            'end_line': i - 1,
                            'section': current_section
                        })
                        current_paragraph = []
                
                structure['lists'].append({
                    'text': stripped,
                    'line': i,
                    'section': current_section
                })
                structure['line_types'].append('list')
                in_list = True
                continue
            
            # Regular text line
            current_paragraph.append(line)
            structure['line_types'].append('text')
            in_list = False
        
        # Handle final paragraph
        if current_paragraph:
            structure['paragraphs'].append({
                'text': '\n'.join(current_paragraph),
                'start_line': len(lines) - len(current_paragraph),
                'end_line': len(lines) - 1,
                'section': current_section
            })
        
        return structure
    
    def _detect_section_header(self, line: str) -> Optional[str]:
        """Detect if line is a section header."""
        for pattern in self.section_patterns:
            match = re.match(pattern, line)
            if match:
                return match.group(1) if match.groups() else line.strip()
        
        # Additional heuristics
        if len(line) < 100 and line.strip():  # Short lines might be headers
            # Check if line has title case or is all caps
            words = line.split()
            if len(words) <= 8:  # Reasonable header length
                if (line.isupper() or 
                    all(word[0].isupper() for word in words if word.isalpha())):
                    return line.strip()
        
        return None
    
    def _get_header_level(self, line: str) -> int:
        """Determine header level (1-6)."""
        # Markdown style
        if line.startswith('#'):
            return min(6, line.count('#'))
        
        # Number-based (1.1.1 etc.)
        if re.match(r'^\d+\.', line):
            return line.count('.') + 1
        
        # Default levels based on formatting
        if line.isupper():
            return 1
        elif line.strip().endswith(':'):
            return 2
        else:
            return 3
    
    def _is_list_item(self, line: str) -> bool:
        """Check if line is a list item."""
        return any(re.match(pattern, line) for pattern in self.list_patterns)
    
    def _find_semantic_boundaries(self, structured_text: Dict[str, Any]) -> List[int]:
        """Find semantic boundaries in the text."""
        boundaries = set()
        text = structured_text['original_text']
        lines = structured_text['lines']
        
        # Method 1: Structure-based boundaries
        for section in structured_text['sections']:
            boundaries.add(section['line'])
        
        for paragraph in structured_text['paragraphs']:
            if self.config.preserve_paragraphs:
                boundaries.add(paragraph['start_line'])
        
        # Method 2: Sentence-based boundaries (if enabled)
        if self.config.preserve_sentences and self.nlp:
            doc = self.nlp(text)
            char_to_line = self._create_char_to_line_mapping(text)
            
            for sent in doc.sents:
                line_num = char_to_line.get(sent.start, 0)
                boundaries.add(line_num)
        
        # Method 3: Semantic similarity boundaries
        if self.sentence_model:
            semantic_boundaries = self._find_similarity_boundaries(structured_text)
            boundaries.update(semantic_boundaries)
        
        # Method 4: Transition word boundaries
        transition_boundaries = self._find_transition_boundaries(lines)
        boundaries.update(transition_boundaries)
        
        return sorted(list(boundaries))
    
    def _create_char_to_line_mapping(self, text: str) -> Dict[int, int]:
        """Create mapping from character position to line number."""
        char_to_line = {}
        char_pos = 0
        
        for line_num, line in enumerate(text.split('\n')):
            for _ in range(len(line) + 1):  # +1 for newline
                char_to_line[char_pos] = line_num
                char_pos += 1
        
        return char_to_line
    
    def _find_similarity_boundaries(self, structured_text: Dict[str, Any]) -> List[int]:
        """Find boundaries based on semantic similarity."""
        boundaries = []
        paragraphs = structured_text['paragraphs']
        
        if not paragraphs or not self.sentence_model:
            return boundaries
        
        # Get embeddings for each paragraph
        paragraph_texts = [p['text'] for p in paragraphs]
        if len(paragraph_texts) < 2:
            return boundaries
        
        try:
            embeddings = self.sentence_model.encode(paragraph_texts)
            
            # Calculate similarities between adjacent paragraphs
            for i in range(len(embeddings) - 1):
                similarity = cosine_similarity(
                    embeddings[i].reshape(1, -1), 
                    embeddings[i + 1].reshape(1, -1)
                )[0][0]
                
                # If similarity is below threshold, it's a boundary
                if similarity < self.config.similarity_threshold:
                    boundaries.append(paragraphs[i + 1]['start_line'])
                    
        except Exception as e:
            print(f"Similarity boundary detection failed: {e}")
        
        return boundaries
    
    def _find_transition_boundaries(self, lines: List[str]) -> List[int]:
        """Find boundaries based on transition words."""
        boundaries = []
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Check for transition words at the beginning of sentences
            for transition_type, words in self.transition_words.items():
                for word in words:
                    if line_lower.startswith(word):
                        boundaries.append(i)
                        break
        
        return boundaries
    
    def _create_semantic_chunks(self, structured_text: Dict[str, Any], 
                              boundaries: List[int], document_id: str) -> List[SemanticChunk]:
        """Create chunks based on semantic boundaries."""
        chunks = []
        lines = structured_text['lines']
        
        if not boundaries:
            boundaries = [0]
        
        # Ensure we have start and end boundaries
        if 0 not in boundaries:
            boundaries.insert(0, 0)
        if len(lines) - 1 not in boundaries:
            boundaries.append(len(lines) - 1)
        
        # Create chunks between boundaries
        for i in range(len(boundaries) - 1):
            start_line = boundaries[i]
            end_line = boundaries[i + 1]
            
            chunk_lines = lines[start_line:end_line + 1]
            chunk_text = '\n'.join(chunk_lines).strip()
            
            if not chunk_text or len(chunk_text) < self.config.min_chunk_size:
                continue
            
            # Determine chunk characteristics
            chunk_type = self._determine_chunk_type(chunk_lines, structured_text)
            topic_keywords = self._extract_topic_keywords(chunk_text)
            section_title = self._find_section_title(start_line, structured_text)
            semantic_score = self._calculate_semantic_score(chunk_text)
            
            chunk = SemanticChunk(
                id=f"{document_id}_semantic_{i:04d}",
                text=chunk_text,
                start_pos=start_line,
                end_pos=end_line,
                semantic_score=semantic_score,
                topic_keywords=topic_keywords,
                section_title=section_title,
                chunk_type=chunk_type,
                metadata={
                    'line_count': len(chunk_lines),
                    'word_count': len(chunk_text.split()),
                    'char_count': len(chunk_text)
                }
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _determine_chunk_type(self, chunk_lines: List[str], structured_text: Dict[str, Any]) -> str:
        """Determine the type of content in the chunk."""
        # Check if chunk contains headers
        for line in chunk_lines:
            if self._detect_section_header(line.strip()):
                return "section_header"
        
        # Check if chunk is primarily list items
        list_lines = sum(1 for line in chunk_lines if self._is_list_item(line.strip()))
        if list_lines > len(chunk_lines) * 0.5:
            return "list"
        
        # Check for table-like content
        if self._looks_like_table('\n'.join(chunk_lines)):
            return "table"
        
        return "content"
    
    def _looks_like_table(self, text: str) -> bool:
        """Heuristic to detect table-like content."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return False
        
        # Check for consistent separators
        separators = ['|', '\t', '  ']
        for sep in separators:
            sep_counts = [line.count(sep) for line in lines]
            if len(set(sep_counts)) <= 2 and max(sep_counts) > 1:  # Consistent separator usage
                return True
        
        return False
    
    def _extract_topic_keywords(self, text: str) -> List[str]:
        """Extract key topic words from chunk text."""
        keywords = []
        
        if self.nlp:
            try:
                doc = self.nlp(text)
                
                # Extract named entities
                keywords.extend([ent.text.lower() for ent in doc.ents])
                
                # Extract important nouns and adjectives
                for token in doc:
                    if (token.pos_ in ['NOUN', 'PROPN'] and 
                        not token.is_stop and 
                        len(token.text) > 2):
                        keywords.append(token.lemma_.lower())
                
            except Exception:
                pass
        
        # Fallback: simple frequency-based extraction
        if not keywords:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            word_freq = defaultdict(int)
            for word in words:
                if word not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use']:
                    word_freq[word] += 1
            
            # Get top frequent words
            keywords = [word for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]]
        
        return keywords[:10]  # Limit to top 10
    
    def _find_section_title(self, start_line: int, structured_text: Dict[str, Any]) -> Optional[str]:
        """Find the section title that applies to this chunk."""
        # Find the most recent section header before this chunk
        applicable_section = None
        for section in structured_text['sections']:
            if section['line'] <= start_line:
                applicable_section = section['title']
            else:
                break
        
        return applicable_section
    
    def _calculate_semantic_score(self, text: str) -> float:
        """Calculate semantic coherence score for the chunk."""
        # Simple heuristic based on various factors
        score = 0.5  # Base score
        
        # Factor 1: Length appropriateness
        text_length = len(text)
        if self.config.min_chunk_size <= text_length <= self.config.max_chunk_size:
            score += 0.2
        elif text_length < self.config.min_chunk_size * 0.5:
            score -= 0.2
        elif text_length > self.config.max_chunk_size * 2:
            score -= 0.1
        
        # Factor 2: Sentence completeness
        sentences = re.split(r'[.!?]+', text)
        complete_sentences = sum(1 for s in sentences if len(s.strip()) > 10)
        if complete_sentences > 0:
            score += min(0.2, complete_sentences * 0.05)
        
        # Factor 3: Coherence indicators (repeated words, consistent terminology)
        words = text.lower().split()
        if len(words) > 10:
            unique_words = len(set(words))
            coherence_ratio = 1 - (unique_words / len(words))
            score += coherence_ratio * 0.1
        
        return max(0.0, min(1.0, score))
    
    def _optimize_chunks(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Optimize chunks by merging small ones and splitting large ones."""
        optimized = []
        
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            
            # Handle small chunks
            if len(chunk.text) < self.config.min_chunk_size and i < len(chunks) - 1:
                # Try to merge with next chunk
                next_chunk = chunks[i + 1]
                merged_text = chunk.text + "\n\n" + next_chunk.text
                
                if len(merged_text) <= self.config.max_chunk_size * 1.2:  # Allow slight overflow
                    # Merge chunks
                    merged_chunk = SemanticChunk(
                        id=f"{chunk.id}_merged",
                        text=merged_text,
                        start_pos=chunk.start_pos,
                        end_pos=next_chunk.end_pos,
                        semantic_score=(chunk.semantic_score + next_chunk.semantic_score) / 2,
                        topic_keywords=list(set(chunk.topic_keywords + next_chunk.topic_keywords)),
                        section_title=chunk.section_title or next_chunk.section_title,
                        chunk_type=chunk.chunk_type if chunk.chunk_type != "content" else next_chunk.chunk_type,
                        metadata={
                            'merged': True,
                            'original_chunks': [chunk.id, next_chunk.id],
                            'word_count': len(merged_text.split()),
                            'char_count': len(merged_text)
                        }
                    )
                    optimized.append(merged_chunk)
                    i += 2  # Skip next chunk as it's been merged
                    continue
            
            # Handle large chunks
            if len(chunk.text) > self.config.max_chunk_size:
                split_chunks = self._split_large_chunk(chunk)
                optimized.extend(split_chunks)
            else:
                optimized.append(chunk)
            
            i += 1
        
        return optimized
    
    def _split_large_chunk(self, chunk: SemanticChunk) -> List[SemanticChunk]:
        """Split a large chunk into smaller semantic pieces."""
        # Simple sentence-based splitting
        sentences = re.split(r'(?<=[.!?])\s+', chunk.text)
        
        if len(sentences) <= 1:
            return [chunk]  # Can't split further
        
        split_chunks = []
        current_text = ""
        chunk_count = 0
        
        for sentence in sentences:
            if len(current_text + sentence) > self.config.max_chunk_size and current_text:
                # Create chunk from current text
                split_chunk = SemanticChunk(
                    id=f"{chunk.id}_split_{chunk_count:02d}",
                    text=current_text.strip(),
                    start_pos=chunk.start_pos,
                    end_pos=chunk.end_pos,
                    semantic_score=chunk.semantic_score * 0.9,  # Slightly lower for split chunks
                    topic_keywords=self._extract_topic_keywords(current_text),
                    section_title=chunk.section_title,
                    chunk_type=chunk.chunk_type,
                    metadata={
                        'split_from': chunk.id,
                        'split_index': chunk_count,
                        'word_count': len(current_text.split()),
                        'char_count': len(current_text)
                    }
                )
                split_chunks.append(split_chunk)
                
                current_text = sentence
                chunk_count += 1
            else:
                current_text += " " + sentence if current_text else sentence
        
        # Add remaining text as final chunk
        if current_text.strip():
            split_chunk = SemanticChunk(
                id=f"{chunk.id}_split_{chunk_count:02d}",
                text=current_text.strip(),
                start_pos=chunk.start_pos,
                end_pos=chunk.end_pos,
                semantic_score=chunk.semantic_score * 0.9,
                topic_keywords=self._extract_topic_keywords(current_text),
                section_title=chunk.section_title,
                chunk_type=chunk.chunk_type,
                metadata={
                    'split_from': chunk.id,
                    'split_index': chunk_count,
                    'word_count': len(current_text.split()),
                    'char_count': len(current_text)
                }
            )
            split_chunks.append(split_chunk)
        
        return split_chunks if split_chunks else [chunk]
