"""Simple text chunking without advanced dependencies."""
from __future__ import annotations

import re
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class SimpleChunk:
    """Represents a simple chunk of text."""
    id: str
    text: str
    start_pos: int
    end_pos: int
    chunk_type: str = "content"


class SimpleChunker:
    """Simple text chunking without advanced dependencies."""
    
    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 1000, overlap_size: int = 50):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        
        # Simple patterns for structure detection
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
    
    def chunk_document(self, text: str, document_id: str = "doc") -> List[SimpleChunk]:
        """Chunk document using simple strategies."""
        if not text.strip():
            return []
        
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_start = 0
        chunk_id = 0
        
        for i, line in enumerate(lines):
            current_chunk.append(line)
            current_text = '\n'.join(current_chunk)
            
            # Check if we should create a chunk
            should_chunk = False
            
            # Chunk on section boundaries
            if self._is_section_boundary(line):
                should_chunk = True
            
            # Chunk on size limits
            elif len(current_text) >= self.max_chunk_size:
                should_chunk = True
            
            # Chunk on paragraph breaks (double newlines)
            elif line.strip() == '' and len(current_chunk) > 1:
                should_chunk = True
            
            # Create chunk if needed
            if should_chunk and len(current_text.strip()) >= self.min_chunk_size:
                chunk = SimpleChunk(
                    id=f"{document_id}__chunk_{chunk_id:04d}",
                    text=current_text.strip(),
                    start_pos=current_start,
                    end_pos=current_start + len(current_text),
                    chunk_type="content"
                )
                chunks.append(chunk)
                chunk_id += 1
                
                # Start new chunk with overlap
                if self.overlap_size > 0 and len(current_chunk) > 1:
                    # Keep last few lines for overlap
                    overlap_lines = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                    current_chunk = overlap_lines
                    current_start = current_start + len('\n'.join(current_chunk[:-2])) + 2
                else:
                    current_chunk = []
                    current_start = i + 1
        
        # Add final chunk if it has content
        if current_chunk and len('\n'.join(current_chunk).strip()) >= self.min_chunk_size:
            final_text = '\n'.join(current_chunk).strip()
            chunk = SimpleChunk(
                id=f"{document_id}__chunk_{chunk_id:04d}",
                text=final_text,
                start_pos=current_start,
                end_pos=current_start + len(final_text),
                chunk_type="content"
            )
            chunks.append(chunk)
        
        # If no chunks were created, create one from the entire text
        if not chunks and text.strip():
            chunk = SimpleChunk(
                id=f"{document_id}__chunk_0000",
                text=text.strip(),
                start_pos=0,
                end_pos=len(text),
                chunk_type="content"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _is_section_boundary(self, line: str) -> bool:
        """Check if a line represents a section boundary."""
        line = line.strip()
        
        # Check section patterns
        for pattern in self.section_patterns:
            if re.match(pattern, line):
                return True
        
        # Check for list patterns
        for pattern in self.list_patterns:
            if re.match(pattern, line):
                return True
        
        return False
    
    def chunk_text_simple(self, text: str) -> List[str]:
        """Simple text chunking that returns just the text chunks."""
        chunks = self.chunk_document(text)
        return [chunk.text for chunk in chunks]


# Create global instance
simple_chunker = SimpleChunker()
