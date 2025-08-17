"""Advanced table and figure extraction from documents."""
from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from PIL import Image

try:
    import camelot
    import tabula
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False


class TableExtractor:
    """Extracts tables from PDFs using multiple methods."""
    
    def __init__(self):
        self.methods = []
        if CAMELOT_AVAILABLE:
            self.methods = ["camelot", "tabula", "fallback"]
        else:
            self.methods = ["fallback"]
    
    def extract_tables_from_pdf(self, pdf_content: bytes, filename: str) -> List[Dict[str, Any]]:
        """Extract all tables from a PDF with metadata."""
        tables = []
        
        # Save to temp file for camelot/tabula
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(pdf_content)
            tmp_path = tmp.name
        
        try:
            # Method 1: Camelot (lattice-based)
            if "camelot" in self.methods:
                tables.extend(self._extract_with_camelot(tmp_path, filename))
            
            # Method 2: Tabula (stream-based)
            if "tabula" in self.methods:
                tables.extend(self._extract_with_tabula(tmp_path, filename))
            
            # Method 3: Fallback with pandas read attempts
            if not tables or "fallback" in self.methods:
                tables.extend(self._extract_fallback(pdf_content, filename))
                
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        
        return tables
    
    def _extract_with_camelot(self, pdf_path: str, filename: str) -> List[Dict[str, Any]]:
        """Extract tables using Camelot."""
        tables = []
        try:
            # Lattice method for tables with borders
            lattice_tables = camelot.read_pdf(pdf_path, flavor='lattice', pages='all')
            for i, table in enumerate(lattice_tables):
                if len(table.df) > 1 and len(table.df.columns) > 1:  # Valid table
                    tables.append({
                        'id': f"{filename}_camelot_lattice_{i}",
                        'page': table.page,
                        'method': 'camelot_lattice',
                        'confidence': table.accuracy,
                        'dataframe': table.df,
                        'csv': table.df.to_csv(index=False),
                        'shape': table.df.shape,
                        'bbox': table._bbox if hasattr(table, '_bbox') else None
                    })
            
            # Stream method for tables without borders
            stream_tables = camelot.read_pdf(pdf_path, flavor='stream', pages='all')
            for i, table in enumerate(stream_tables):
                if len(table.df) > 1 and len(table.df.columns) > 1:
                    tables.append({
                        'id': f"{filename}_camelot_stream_{i}",
                        'page': table.page,
                        'method': 'camelot_stream',
                        'confidence': table.accuracy,
                        'dataframe': table.df,
                        'csv': table.df.to_csv(index=False),
                        'shape': table.df.shape,
                        'bbox': table._bbox if hasattr(table, '_bbox') else None
                    })
        except Exception as e:
            print(f"Camelot extraction failed: {e}")
        
        return tables
    
    def _extract_with_tabula(self, pdf_path: str, filename: str) -> List[Dict[str, Any]]:
        """Extract tables using Tabula."""
        tables = []
        try:
            # Extract all tables
            dfs = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            for i, df in enumerate(dfs):
                if len(df) > 1 and len(df.columns) > 1:
                    tables.append({
                        'id': f"{filename}_tabula_{i}",
                        'page': i + 1,  # Approximation
                        'method': 'tabula',
                        'confidence': 0.8,  # Default confidence
                        'dataframe': df,
                        'csv': df.to_csv(index=False),
                        'shape': df.shape,
                        'bbox': None
                    })
        except Exception as e:
            print(f"Tabula extraction failed: {e}")
        
        return tables
    
    def _extract_fallback(self, pdf_content: bytes, filename: str) -> List[Dict[str, Any]]:
        """Fallback table extraction using basic text parsing."""
        tables = []
        try:
            # Try to find table-like structures in text
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(pdf_content))
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text:
                    potential_tables = self._parse_text_tables(text)
                    for i, table_data in enumerate(potential_tables):
                        tables.append({
                            'id': f"{filename}_fallback_p{page_num}_{i}",
                            'page': page_num,
                            'method': 'text_parsing',
                            'confidence': 0.5,
                            'dataframe': table_data,
                            'csv': table_data.to_csv(index=False),
                            'shape': table_data.shape,
                            'bbox': None
                        })
        except Exception as e:
            print(f"Fallback extraction failed: {e}")
        
        return tables
    
    def _parse_text_tables(self, text: str) -> List[pd.DataFrame]:
        """Parse potential tables from raw text."""
        tables = []
        lines = text.split('\n')
        
        # Look for lines with multiple columns (tabs, multiple spaces, pipes)
        potential_table_lines = []
        for line in lines:
            # Check for table indicators
            if any(sep in line for sep in ['\t', '|', '  ']):
                # Split by common separators
                if '\t' in line:
                    parts = line.split('\t')
                elif '|' in line:
                    parts = line.split('|')
                else:
                    parts = line.split()
                
                if len(parts) >= 2:  # At least 2 columns
                    potential_table_lines.append([p.strip() for p in parts])
        
        # Group consecutive table lines
        if len(potential_table_lines) >= 2:
            try:
                df = pd.DataFrame(potential_table_lines[1:], columns=potential_table_lines[0])
                if len(df) > 0:
                    tables.append(df)
            except:
                pass
        
        return tables


class FigureExtractor:
    """Extracts and analyzes figures from documents."""
    
    def __init__(self):
        self.figure_types = ['chart', 'graph', 'diagram', 'image', 'plot']
    
    def extract_figures_from_pdf(self, pdf_content: bytes, filename: str) -> List[Dict[str, Any]]:
        """Enhanced figure extraction from PDF pages with image detection."""
        figures = []
        
        try:
            from pdf2image import convert_from_bytes
            import pytesseract
            import numpy as np
            
            # Convert with higher DPI for better image quality
            images = convert_from_bytes(pdf_content, dpi=200, first_page=None, last_page=None)
            
            for page_num, image in enumerate(images, 1):
                # Extract text with multiple OCR methods
                ocr_text = self._enhanced_ocr_extraction(image)
                
                # Detect if page contains significant non-text content (images/figures)
                has_visual_content = self._detect_visual_content(image)
                
                # Look for figure indicators
                figure_info = self._analyze_figure_content(image, ocr_text, page_num, filename)
                
                # Enhanced detection for pages with images
                if figure_info or has_visual_content:
                    if not figure_info:
                        # Create figure info for visual content without explicit captions
                        figure_info = {
                            'id': f"{filename}_visual_p{page_num}",
                            'page': page_num,
                            'type': 'visual_content',
                            'caption': '',
                            'description': ocr_text[:800] if ocr_text else 'Visual content detected',
                            'confidence': 0.6,
                            'image_size': image.size,
                            'text_length': len(ocr_text),
                            'has_significant_visual': has_visual_content
                        }
                    
                    figures.append(figure_info)
                    
        except Exception as e:
            print(f"Enhanced figure extraction failed: {e}")
        
        return figures
    
    def _enhanced_ocr_extraction(self, image) -> str:
        """Enhanced OCR with multiple configurations for better text extraction."""
        try:
            import pytesseract
            
            # Try multiple OCR configurations
            configs = [
                '--psm 6',  # Uniform block of text
                '--psm 3',  # Fully automatic page segmentation
                '--psm 11', # Sparse text
                '--psm 12'  # Sparse text, OSD
            ]
            
            best_text = ""
            max_length = 0
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(image, config=config)
                    if len(text) > max_length:
                        max_length = len(text)
                        best_text = text
                except:
                    continue
            
            return best_text
        except:
            return ""
    
    def _detect_visual_content(self, image) -> bool:
        """Detect if image contains significant non-text visual content."""
        try:
            import numpy as np
            from PIL import ImageFilter, ImageStat
            
            # Convert to grayscale for analysis
            gray_image = image.convert('L')
            
            # Apply edge detection to find non-text elements
            edges = gray_image.filter(ImageFilter.FIND_EDGES)
            
            # Calculate edge density
            edge_array = np.array(edges)
            edge_density = np.count_nonzero(edge_array) / edge_array.size
            
            # Calculate variance in pixel values (high variance suggests images/graphics)
            variance = ImageStat.Stat(gray_image).var[0]
            
            # Heuristics for visual content detection
            has_high_edge_density = edge_density > 0.1
            has_high_variance = variance > 1000
            
            return has_high_edge_density or has_high_variance
            
        except Exception:
            # Fallback: assume visual content if page is large enough
            return image.size[0] * image.size[1] > 500000  # Roughly A4 size or larger
    
    def _analyze_figure_content(self, image: Image.Image, ocr_text: str, page_num: int, filename: str) -> Optional[Dict[str, Any]]:
        """Analyze image content to determine if it contains a figure."""
        # Look for figure indicators in text
        figure_keywords = ['figure', 'fig', 'chart', 'graph', 'diagram', 'plot', 'table']
        
        text_lower = ocr_text.lower()
        has_figure_keywords = any(keyword in text_lower for keyword in figure_keywords)
        
        # Simple heuristic: if image has significant text or figure keywords
        if len(ocr_text.strip()) > 50 or has_figure_keywords:
            return {
                'id': f"{filename}_figure_p{page_num}",
                'page': page_num,
                'type': 'mixed_content',
                'caption': self._extract_caption(ocr_text),
                'description': ocr_text[:500],  # First 500 chars
                'confidence': 0.7 if has_figure_keywords else 0.4,
                'image_size': image.size,
                'text_length': len(ocr_text)
            }
        
        return None
    
    def _extract_caption(self, text: str) -> str:
        """Extract potential figure caption from text."""
        lines = text.strip().split('\n')
        
        # Look for lines starting with "Figure", "Fig", "Table", etc.
        for line in lines:
            line_lower = line.lower().strip()
            if any(line_lower.startswith(prefix) for prefix in ['figure', 'fig.', 'table']):
                return line.strip()
        
        # Fallback: return first non-empty line
        for line in lines:
            if line.strip():
                return line.strip()[:200]  # Limit length
        
        return ""


class DocumentStructureAnalyzer:
    """Analyzes document structure and hierarchy."""
    
    def __init__(self):
        self.heading_indicators = ['#', 'chapter', 'section', 'part', 'abstract', 'introduction', 'conclusion']
    
    def analyze_structure(self, text: str, filename: str) -> Dict[str, Any]:
        """Analyze document structure and extract hierarchy."""
        lines = text.split('\n')
        
        structure = {
            'filename': filename,
            'sections': [],
            'hierarchy': [],
            'metadata': {
                'has_toc': False,
                'has_abstract': False,
                'estimated_sections': 0,
                'text_length': len(text)
            }
        }
        
        current_section = None
        section_stack = []
        
        for line_num, line in enumerate(lines):
            line_stripped = line.strip()
            
            if not line_stripped:
                continue
            
            # Detect headings
            heading_level = self._detect_heading_level(line_stripped)
            
            if heading_level > 0:
                # Create new section
                section = {
                    'title': line_stripped,
                    'level': heading_level,
                    'line_number': line_num,
                    'content': '',
                    'subsections': []
                }
                
                # Manage hierarchy
                if heading_level == 1:
                    structure['sections'].append(section)
                    section_stack = [section]
                else:
                    # Find parent section
                    while section_stack and section_stack[-1]['level'] >= heading_level:
                        section_stack.pop()
                    
                    if section_stack:
                        section_stack[-1]['subsections'].append(section)
                    else:
                        structure['sections'].append(section)
                    
                    section_stack.append(section)
                
                current_section = section
                structure['metadata']['estimated_sections'] += 1
            
            elif current_section:
                # Add content to current section
                current_section['content'] += line + '\n'
        
        # Post-process metadata
        text_lower = text.lower()
        structure['metadata']['has_abstract'] = 'abstract' in text_lower
        structure['metadata']['has_toc'] = any(indicator in text_lower for indicator in ['table of contents', 'contents'])
        
        return structure
    
    def _detect_heading_level(self, line: str) -> int:
        """Detect if line is a heading and return its level."""
        line_lower = line.lower()
        
        # Method 1: Markdown-style headers
        if line.startswith('#'):
            return line.count('#')
        
        # Method 2: All caps (likely heading)
        if line.isupper() and len(line) > 3:
            return 1
        
        # Method 3: Numbered sections
        import re
        if re.match(r'^\d+\.?\s+', line):
            dots = line.count('.')
            return min(dots + 1, 3)
        
        # Method 4: Common heading words
        heading_words = ['chapter', 'section', 'part', 'abstract', 'introduction', 'conclusion', 'references']
        for word in heading_words:
            if line_lower.startswith(word):
                return 1
        
        # Method 5: Short lines (potential headings)
        if len(line) < 80 and len(line.split()) <= 8:
            # Check if it looks like a title
            if any(char.isupper() for char in line) and not line.endswith('.'):
                return 2
        
        return 0
