"""Code snippet extraction and programming language detection."""
from __future__ import annotations

import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter

try:
    import pygments
    from pygments.lexers import get_lexer_by_name, guess_lexer, get_all_lexers
    from pygments.util import ClassNotFound
    from pygments.token import Token
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False

try:
    import ast
    import keyword
    AST_AVAILABLE = True
except ImportError:
    AST_AVAILABLE = False


@dataclass
class CodeSnippet:
    """Represents an extracted code snippet."""
    snippet_id: str
    raw_text: str
    cleaned_code: str
    language: str
    confidence: float
    syntax_valid: bool
    complexity_score: float
    keywords: List[str]
    functions: List[str]
    classes: List[str]
    imports: List[str]
    variables: List[str]
    comments: List[str]
    documentation: str
    context: str
    page_number: Optional[int]
    start_line: int
    end_line: int
    metadata: Dict[str, Any]


@dataclass
class CodeExtractionResult:
    """Result of code extraction from document."""
    total_snippets: int
    snippets_by_language: Dict[str, int]
    extracted_snippets: List[CodeSnippet]
    language_statistics: Dict[str, Any]
    extraction_metadata: Dict[str, Any]


class ProgrammingLanguageDetector:
    """Detects programming languages from code snippets."""
    
    def __init__(self):
        self.pygments_available = PYGMENTS_AVAILABLE
        
        # Language patterns and signatures
        self.language_signatures = {
            'python': {
                'keywords': ['def', 'class', 'import', 'from', 'if', 'elif', 'else', 'for', 'while', 'try', 'except', 'with', 'lambda'],
                'patterns': [r'def\s+\w+\s*\(', r'class\s+\w+', r'import\s+\w+', r'from\s+\w+\s+import'],
                'file_extensions': ['.py', '.pyw'],
                'indicators': ['print(', '__name__', '__main__', '#!/usr/bin/env python']
            },
            'javascript': {
                'keywords': ['function', 'var', 'let', 'const', 'if', 'else', 'for', 'while', 'return', 'class', 'extends'],
                'patterns': [r'function\s+\w+\s*\(', r'var\s+\w+', r'let\s+\w+', r'const\s+\w+', r'=>'],
                'file_extensions': ['.js', '.jsx', '.ts', '.tsx'],
                'indicators': ['console.log', 'document.', 'window.', 'JSON.']
            },
            'java': {
                'keywords': ['public', 'private', 'protected', 'class', 'interface', 'static', 'void', 'int', 'String'],
                'patterns': [r'public\s+class\s+\w+', r'public\s+static\s+void\s+main', r'@Override'],
                'file_extensions': ['.java'],
                'indicators': ['System.out.', 'public static void main', 'import java.']
            },
            'cpp': {
                'keywords': ['#include', 'class', 'namespace', 'using', 'template', 'public:', 'private:', 'protected:'],
                'patterns': [r'#include\s*<[\w./]+>', r'class\s+\w+', r'namespace\s+\w+', r'std::'],
                'file_extensions': ['.cpp', '.cc', '.cxx', '.h', '.hpp'],
                'indicators': ['std::', '#include', 'cout <<', 'cin >>']
            },
            'c': {
                'keywords': ['#include', 'int', 'char', 'float', 'double', 'void', 'struct', 'typedef'],
                'patterns': [r'#include\s*<[\w./]+>', r'int\s+main\s*\(', r'printf\s*\('],
                'file_extensions': ['.c', '.h'],
                'indicators': ['printf(', 'scanf(', 'malloc(', '#include <stdio.h>']
            },
            'csharp': {
                'keywords': ['using', 'namespace', 'class', 'public', 'private', 'static', 'void', 'string'],
                'patterns': [r'using\s+System', r'namespace\s+\w+', r'public\s+class\s+\w+'],
                'file_extensions': ['.cs'],
                'indicators': ['Console.WriteLine', 'using System', 'public static void Main']
            },
            'php': {
                'keywords': ['<?php', 'function', 'class', 'public', 'private', 'protected', '$'],
                'patterns': [r'<\?php', r'function\s+\w+\s*\(', r'\$\w+'],
                'file_extensions': ['.php'],
                'indicators': ['<?php', 'echo ', '$_GET', '$_POST']
            },
            'ruby': {
                'keywords': ['def', 'class', 'module', 'end', 'if', 'unless', 'while', 'until'],
                'patterns': [r'def\s+\w+', r'class\s+\w+', r'module\s+\w+'],
                'file_extensions': ['.rb'],
                'indicators': ['puts ', 'require ', 'gem ']
            },
            'go': {
                'keywords': ['package', 'import', 'func', 'var', 'const', 'type', 'struct', 'interface'],
                'patterns': [r'package\s+\w+', r'func\s+\w+\s*\(', r'import\s*\('],
                'file_extensions': ['.go'],
                'indicators': ['package main', 'func main()', 'fmt.Println']
            },
            'rust': {
                'keywords': ['fn', 'let', 'mut', 'struct', 'enum', 'impl', 'trait', 'use'],
                'patterns': [r'fn\s+\w+\s*\(', r'let\s+\w+', r'struct\s+\w+'],
                'file_extensions': ['.rs'],
                'indicators': ['fn main()', 'println!', 'use std::']
            },
            'sql': {
                'keywords': ['SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP'],
                'patterns': [r'SELECT\s+.*\s+FROM', r'INSERT\s+INTO', r'CREATE\s+TABLE'],
                'file_extensions': ['.sql'],
                'indicators': ['SELECT ', 'FROM ', 'WHERE ', 'INSERT INTO']
            },
            'html': {
                'keywords': ['<!DOCTYPE', '<html>', '<head>', '<body>', '<div>', '<span>', '<script>'],
                'patterns': [r'<!DOCTYPE\s+html>', r'<html.*?>', r'<\w+.*?>'],
                'file_extensions': ['.html', '.htm'],
                'indicators': ['<!DOCTYPE', '<html>', '<head>', '<body>']
            },
            'css': {
                'keywords': ['color:', 'background:', 'margin:', 'padding:', 'font-', 'border:'],
                'patterns': [r'\w+\s*\{', r'[\w-]+\s*:', r'@media', r'@import'],
                'file_extensions': ['.css'],
                'indicators': ['{', '}', ':', ';', '@media']
            },
            'bash': {
                'keywords': ['#!/bin/bash', 'if', 'then', 'else', 'fi', 'for', 'while', 'do', 'done'],
                'patterns': [r'#!/bin/bash', r'if\s*\[', r'for\s+\w+\s+in'],
                'file_extensions': ['.sh', '.bash'],
                'indicators': ['#!/bin/bash', 'echo ', '$', '&&', '||']
            }
        }
        
        if not self.pygments_available:
            print("Warning: Pygments not available. Language detection will use pattern matching only.")
    
    def detect_language(self, code: str, context: str = "") -> Tuple[str, float]:
        """Detect programming language of code snippet."""
        
        if not code.strip():
            return "unknown", 0.0
        
        # Try Pygments first if available
        if self.pygments_available:
            try:
                lexer = guess_lexer(code)
                return lexer.name.lower(), 0.9
            except ClassNotFound:
                pass
        
        # Fallback to pattern-based detection
        return self._pattern_based_detection(code, context)
    
    def _pattern_based_detection(self, code: str, context: str = "") -> Tuple[str, float]:
        """Detect language using pattern matching."""
        
        code_lower = code.lower()
        context_lower = context.lower()
        
        language_scores = {}
        
        for lang, signatures in self.language_signatures.items():
            score = 0.0
            
            # Check keywords
            for keyword in signatures['keywords']:
                if keyword.lower() in code_lower:
                    score += 1.0
            
            # Check patterns
            for pattern in signatures['patterns']:
                if re.search(pattern, code, re.IGNORECASE):
                    score += 2.0
            
            # Check indicators
            for indicator in signatures['indicators']:
                if indicator.lower() in code_lower:
                    score += 1.5
            
            # Check file extensions in context
            for ext in signatures['file_extensions']:
                if ext in context_lower:
                    score += 3.0
            
            # Normalize score
            max_possible = len(signatures['keywords']) + len(signatures['patterns']) * 2 + len(signatures['indicators']) * 1.5
            if max_possible > 0:
                language_scores[lang] = score / max_possible
        
        if language_scores:
            best_lang = max(language_scores, key=language_scores.get)
            confidence = min(1.0, language_scores[best_lang])
            return best_lang, confidence
        
        return "unknown", 0.0


class CodePatternMatcher:
    """Matches and extracts code patterns from text."""
    
    def __init__(self):
        # Code block patterns
        self.code_block_patterns = [
            # Markdown fenced code blocks
            re.compile(r'```(\w+)?\n(.*?)```', re.DOTALL),
            re.compile(r'~~~(\w+)?\n(.*?)~~~', re.DOTALL),
            # Indented code blocks (4+ spaces)
            re.compile(r'^(    .+?)(?=\n(?:    |\S|\Z))', re.MULTILINE | re.DOTALL),
            # HTML <code> and <pre> blocks
            re.compile(r'<code[^>]*>(.*?)</code>', re.DOTALL | re.IGNORECASE),
            re.compile(r'<pre[^>]*>(.*?)</pre>', re.DOTALL | re.IGNORECASE),
            # LaTeX verbatim
            re.compile(r'\\\\begin\{verbatim\}(.*?)\\\\end\{verbatim\}', re.DOTALL),
            re.compile(r'\\\\begin\{lstlisting\}(?:\[.*?\])?(.*?)\\\\end\{lstlisting\}', re.DOTALL),
        ]
        
        # Inline code patterns
        self.inline_code_patterns = [
            re.compile(r'`([^`\n]+)`'),  # Markdown inline code
            re.compile(r'<code>([^<]+)</code>', re.IGNORECASE),  # HTML inline code
        ]
        
        # Function/method patterns
        self.function_patterns = {
            'python': re.compile(r'def\s+(\w+)\s*\([^)]*\):'),
            'javascript': re.compile(r'function\s+(\w+)\s*\([^)]*\)|(\w+)\s*=\s*\([^)]*\)\s*=>'),
            'java': re.compile(r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\([^)]*\)'),
            'cpp': re.compile(r'\w+\s+(\w+)\s*\([^)]*\)\s*\{'),
            'c': re.compile(r'\w+\s+(\w+)\s*\([^)]*\)\s*\{'),
        }
        
        # Class patterns
        self.class_patterns = {
            'python': re.compile(r'class\s+(\w+)(?:\([^)]*\))?:'),
            'javascript': re.compile(r'class\s+(\w+)(?:\s+extends\s+\w+)?'),
            'java': re.compile(r'(?:public|private)?\s*class\s+(\w+)'),
            'cpp': re.compile(r'class\s+(\w+)'),
        }
        
        # Import patterns
        self.import_patterns = {
            'python': re.compile(r'(?:from\s+[\w.]+\s+)?import\s+([\w., ]+)'),
            'javascript': re.compile(r'import\s+(?:\{[^}]*\}|\w+)\s+from\s+[\'"][^\'"]+[\'"]'),
            'java': re.compile(r'import\s+([\w.]+);'),
            'cpp': re.compile(r'#include\s*[<"]([\w./]+)[>"]'),
        }
    
    def extract_code_blocks(self, text: str, page_number: Optional[int] = None) -> List[Dict[str, Any]]:
        """Extract code blocks from text."""
        
        code_blocks = []
        
        for pattern in self.code_block_patterns:
            matches = pattern.finditer(text)
            
            for match in matches:
                if len(match.groups()) >= 2:
                    # Fenced code block with language
                    language_hint = match.group(1) if match.group(1) else ""
                    code_content = match.group(2)
                elif len(match.groups()) == 1:
                    # Single group (like indented code or HTML)
                    language_hint = ""
                    code_content = match.group(1)
                else:
                    # Full match
                    language_hint = ""
                    code_content = match.group(0)
                
                # Clean up code content
                cleaned_code = self._clean_code_content(code_content)
                
                if cleaned_code and len(cleaned_code.strip()) > 10:  # Minimum code length
                    # Extract context around code block
                    start_pos = max(0, match.start() - 200)
                    end_pos = min(len(text), match.end() + 200)
                    context = text[start_pos:end_pos]
                    
                    # Calculate line numbers
                    text_before = text[:match.start()]
                    start_line = text_before.count('\n') + 1
                    end_line = start_line + code_content.count('\n')
                    
                    code_info = {
                        'raw_text': match.group(0),
                        'cleaned_code': cleaned_code,
                        'language_hint': language_hint,
                        'position': (match.start(), match.end()),
                        'context': context,
                        'page_number': page_number,
                        'start_line': start_line,
                        'end_line': end_line,
                        'block_type': 'code_block'
                    }
                    code_blocks.append(code_info)
        
        return code_blocks
    
    def extract_inline_code(self, text: str, page_number: Optional[int] = None) -> List[Dict[str, Any]]:
        """Extract inline code snippets."""
        
        inline_code = []
        
        for pattern in self.inline_code_patterns:
            matches = pattern.finditer(text)
            
            for match in matches:
                code_content = match.group(1)
                cleaned_code = self._clean_code_content(code_content)
                
                if cleaned_code and len(cleaned_code.strip()) > 3:  # Minimum inline code length
                    # Extract context
                    start_pos = max(0, match.start() - 100)
                    end_pos = min(len(text), match.end() + 100)
                    context = text[start_pos:end_pos]
                    
                    # Calculate line number
                    text_before = text[:match.start()]
                    line_number = text_before.count('\n') + 1
                    
                    code_info = {
                        'raw_text': match.group(0),
                        'cleaned_code': cleaned_code,
                        'language_hint': "",
                        'position': (match.start(), match.end()),
                        'context': context,
                        'page_number': page_number,
                        'start_line': line_number,
                        'end_line': line_number,
                        'block_type': 'inline_code'
                    }
                    inline_code.append(code_info)
        
        return inline_code
    
    def _clean_code_content(self, code: str) -> str:
        """Clean and normalize code content."""
        
        # Remove leading/trailing whitespace
        code = code.strip()
        
        # Remove HTML entities
        code = re.sub(r'&lt;', '<', code)
        code = re.sub(r'&gt;', '>', code)
        code = re.sub(r'&amp;', '&', code)
        code = re.sub(r'&quot;', '"', code)
        
        # Normalize line endings
        code = re.sub(r'\r\n', '\n', code)
        code = re.sub(r'\r', '\n', code)
        
        return code


class CodeAnalyzer:
    """Analyzes code snippets for semantic content."""
    
    def __init__(self):
        self.ast_available = AST_AVAILABLE
        self.pygments_available = PYGMENTS_AVAILABLE
        
        if not self.ast_available:
            print("Warning: AST module not available. Python code analysis will be limited.")
    
    def analyze_code(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code snippet for semantic content."""
        
        analysis = {
            'syntax_valid': False,
            'complexity_score': 0.0,
            'keywords': [],
            'functions': [],
            'classes': [],
            'imports': [],
            'variables': [],
            'comments': [],
            'documentation': "",
            'metadata': {}
        }
        
        try:
            # Language-specific analysis
            if language == 'python' and self.ast_available:
                analysis.update(self._analyze_python_code(code))
            elif self.pygments_available:
                analysis.update(self._analyze_with_pygments(code, language))
            else:
                analysis.update(self._analyze_with_patterns(code, language))
            
            # General analysis
            analysis['comments'] = self._extract_comments(code, language)
            analysis['complexity_score'] = self._calculate_complexity(code, language)
            
        except Exception as e:
            analysis['metadata']['analysis_error'] = str(e)
        
        return analysis
    
    def _analyze_python_code(self, code: str) -> Dict[str, Any]:
        """Analyze Python code using AST."""
        
        analysis = {
            'syntax_valid': False,
            'functions': [],
            'classes': [],
            'imports': [],
            'variables': [],
            'keywords': [],
            'documentation': ""
        }
        
        try:
            # Parse with AST
            tree = ast.parse(code)
            analysis['syntax_valid'] = True
            
            # Extract information from AST
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis['functions'].append(node.name)
                    # Extract docstring
                    if (node.body and isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Str)):
                        if not analysis['documentation']:
                            analysis['documentation'] = node.body[0].value.s
                
                elif isinstance(node, ast.ClassDef):
                    analysis['classes'].append(node.name)
                
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        analysis['imports'].append(node.module)
                
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    analysis['variables'].append(node.id)
            
            # Get Python keywords used
            for word in code.split():
                if keyword.iskeyword(word):
                    analysis['keywords'].append(word)
            
            # Remove duplicates
            analysis['functions'] = list(set(analysis['functions']))
            analysis['classes'] = list(set(analysis['classes']))
            analysis['imports'] = list(set(analysis['imports']))
            analysis['variables'] = list(set(analysis['variables']))
            analysis['keywords'] = list(set(analysis['keywords']))
            
        except SyntaxError:
            analysis['syntax_valid'] = False
        except Exception as e:
            analysis['metadata'] = {'ast_error': str(e)}
        
        return analysis
    
    def _analyze_with_pygments(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code using Pygments tokenization."""
        
        analysis = {
            'syntax_valid': True,  # Assume valid if Pygments can tokenize
            'keywords': [],
            'functions': [],
            'variables': [],
            'metadata': {}
        }
        
        try:
            lexer = get_lexer_by_name(language)
            tokens = list(lexer.get_tokens(code))
            
            keywords = []
            names = []
            
            for token_type, value in tokens:
                if token_type in Token.Keyword:
                    keywords.append(value)
                elif token_type in Token.Name:
                    names.append(value)
            
            analysis['keywords'] = list(set(keywords))
            analysis['variables'] = list(set(names))
            
            # Try to identify functions (simple heuristic)
            if language in ['python', 'javascript', 'java', 'cpp']:
                analysis['functions'] = self._extract_functions_simple(code, language)
            
        except Exception as e:
            analysis['metadata']['pygments_error'] = str(e)
        
        return analysis
    
    def _analyze_with_patterns(self, code: str, language: str) -> Dict[str, Any]:
        """Fallback analysis using regex patterns."""
        
        analysis = {
            'syntax_valid': True,  # Assume valid for pattern analysis
            'functions': [],
            'classes': [],
            'imports': [],
            'keywords': [],
            'variables': []
        }
        
        # Extract functions using patterns
        analysis['functions'] = self._extract_functions_simple(code, language)
        
        # Extract classes
        analysis['classes'] = self._extract_classes_simple(code, language)
        
        # Extract imports
        analysis['imports'] = self._extract_imports_simple(code, language)
        
        return analysis
    
    def _extract_functions_simple(self, code: str, language: str) -> List[str]:
        """Extract function names using simple patterns."""
        
        functions = []
        pattern_matcher = CodePatternMatcher()
        
        if language in pattern_matcher.function_patterns:
            pattern = pattern_matcher.function_patterns[language]
            matches = pattern.findall(code)
            
            for match in matches:
                if isinstance(match, tuple):
                    # Handle multiple groups
                    func_name = next((name for name in match if name), None)
                    if func_name:
                        functions.append(func_name)
                else:
                    functions.append(match)
        
        return list(set(functions))
    
    def _extract_classes_simple(self, code: str, language: str) -> List[str]:
        """Extract class names using simple patterns."""
        
        classes = []
        pattern_matcher = CodePatternMatcher()
        
        if language in pattern_matcher.class_patterns:
            pattern = pattern_matcher.class_patterns[language]
            matches = pattern.findall(code)
            classes.extend(matches)
        
        return list(set(classes))
    
    def _extract_imports_simple(self, code: str, language: str) -> List[str]:
        """Extract import statements using simple patterns."""
        
        imports = []
        pattern_matcher = CodePatternMatcher()
        
        if language in pattern_matcher.import_patterns:
            pattern = pattern_matcher.import_patterns[language]
            matches = pattern.findall(code)
            imports.extend(matches)
        
        return list(set(imports))
    
    def _extract_comments(self, code: str, language: str) -> List[str]:
        """Extract comments from code."""
        
        comments = []
        
        # Common comment patterns
        comment_patterns = {
            'python': [re.compile(r'#(.*)$', re.MULTILINE)],
            'javascript': [re.compile(r'//(.*)$', re.MULTILINE), re.compile(r'/\*(.*?)\*/', re.DOTALL)],
            'java': [re.compile(r'//(.*)$', re.MULTILINE), re.compile(r'/\*(.*?)\*/', re.DOTALL)],
            'cpp': [re.compile(r'//(.*)$', re.MULTILINE), re.compile(r'/\*(.*?)\*/', re.DOTALL)],
            'c': [re.compile(r'/\*(.*?)\*/', re.DOTALL)],
            'html': [re.compile(r'<!--(.*?)-->', re.DOTALL)],
            'css': [re.compile(r'/\*(.*?)\*/', re.DOTALL)],
            'bash': [re.compile(r'#(.*)$', re.MULTILINE)]
        }
        
        if language in comment_patterns:
            for pattern in comment_patterns[language]:
                matches = pattern.findall(code)
                comments.extend([match.strip() for match in matches if match.strip()])
        
        return comments
    
    def _calculate_complexity(self, code: str, language: str) -> float:
        """Calculate code complexity score."""
        
        complexity = 0.0
        
        # Base complexity from code length
        complexity += len(code) * 0.001
        
        # Line count
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        complexity += len(lines) * 0.1
        
        # Control flow complexity
        control_keywords = ['if', 'else', 'elif', 'for', 'while', 'switch', 'case', 'try', 'catch', 'except']
        for keyword in control_keywords:
            complexity += code.lower().count(keyword) * 0.3
        
        # Function/method complexity
        function_indicators = ['def ', 'function ', 'public ', 'private ', 'protected ']
        for indicator in function_indicators:
            complexity += code.lower().count(indicator) * 0.5
        
        # Nesting complexity (rough estimate)
        indent_levels = []
        for line in lines:
            if line:
                leading_spaces = len(line) - len(line.lstrip())
                indent_levels.append(leading_spaces)
        
        if indent_levels:
            max_indent = max(indent_levels)
            complexity += max_indent * 0.1
        
        # Normalize to 0-10 scale
        return min(10.0, complexity)


class CodeSnippetExtractor:
    """Main class for extracting and analyzing code snippets."""
    
    def __init__(self):
        self.pattern_matcher = CodePatternMatcher()
        self.language_detector = ProgrammingLanguageDetector()
        self.analyzer = CodeAnalyzer()
        self.snippet_cache: Dict[str, CodeSnippet] = {}
    
    def extract_code_from_text(self, text: str, filename: str = "", 
                              page_number: Optional[int] = None) -> CodeExtractionResult:
        """Extract and analyze code snippets from text."""
        
        # Extract code blocks and inline code
        code_blocks = self.pattern_matcher.extract_code_blocks(text, page_number)
        inline_code = self.pattern_matcher.extract_inline_code(text, page_number)
        
        all_code_snippets = code_blocks + inline_code
        extracted_snippets = []
        snippets_by_language = {}
        
        for i, code_info in enumerate(all_code_snippets):
            # Generate unique ID
            snippet_id = self._generate_snippet_id(code_info['cleaned_code'], filename, i)
            
            # Skip if already processed
            if snippet_id in self.snippet_cache:
                extracted_snippets.append(self.snippet_cache[snippet_id])
                continue
            
            # Detect language
            language_hint = code_info.get('language_hint', '')
            detected_language, confidence = self.language_detector.detect_language(
                code_info['cleaned_code'], 
                code_info['context'] + ' ' + language_hint
            )
            
            # Analyze the code
            analysis = self.analyzer.analyze_code(code_info['cleaned_code'], detected_language)
            
            # Create CodeSnippet object
            snippet = CodeSnippet(
                snippet_id=snippet_id,
                raw_text=code_info['raw_text'],
                cleaned_code=code_info['cleaned_code'],
                language=detected_language,
                confidence=confidence,
                syntax_valid=analysis['syntax_valid'],
                complexity_score=analysis['complexity_score'],
                keywords=analysis['keywords'],
                functions=analysis['functions'],
                classes=analysis['classes'],
                imports=analysis['imports'],
                variables=analysis['variables'],
                comments=analysis['comments'],
                documentation=analysis.get('documentation', ''),
                context=code_info['context'],
                page_number=code_info['page_number'],
                start_line=code_info['start_line'],
                end_line=code_info['end_line'],
                metadata={
                    'block_type': code_info['block_type'],
                    'language_hint': language_hint,
                    **analysis.get('metadata', {})
                }
            )
            
            extracted_snippets.append(snippet)
            self.snippet_cache[snippet_id] = snippet
            
            # Count by language
            language = snippet.language
            snippets_by_language[language] = snippets_by_language.get(language, 0) + 1
        
        # Calculate language statistics
        language_statistics = self._calculate_language_statistics(extracted_snippets)
        
        # Create extraction metadata
        metadata = {
            'filename': filename,
            'text_length': len(text),
            'page_number': page_number,
            'extraction_method': 'pattern_matching',
            'pygments_available': self.language_detector.pygments_available,
            'total_code_blocks': len(code_blocks),
            'total_inline_code': len(inline_code)
        }
        
        return CodeExtractionResult(
            total_snippets=len(extracted_snippets),
            snippets_by_language=snippets_by_language,
            extracted_snippets=extracted_snippets,
            language_statistics=language_statistics,
            extraction_metadata=metadata
        )
    
    def _generate_snippet_id(self, code: str, filename: str, index: int) -> str:
        """Generate unique ID for a code snippet."""
        content = f"{filename}_{code}_{index}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _calculate_language_statistics(self, snippets: List[CodeSnippet]) -> Dict[str, Any]:
        """Calculate statistics for extracted code snippets."""
        
        if not snippets:
            return {}
        
        # Language distribution
        languages = [snippet.language for snippet in snippets]
        language_counts = Counter(languages)
        
        # Complexity by language
        complexity_by_lang = {}
        for snippet in snippets:
            lang = snippet.language
            if lang not in complexity_by_lang:
                complexity_by_lang[lang] = []
            complexity_by_lang[lang].append(snippet.complexity_score)
        
        # Average complexity per language
        avg_complexity_by_lang = {}
        for lang, complexities in complexity_by_lang.items():
            avg_complexity_by_lang[lang] = sum(complexities) / len(complexities)
        
        # Most common functions/keywords
        all_functions = []
        all_keywords = []
        for snippet in snippets:
            all_functions.extend(snippet.functions)
            all_keywords.extend(snippet.keywords)
        
        function_counts = Counter(all_functions)
        keyword_counts = Counter(all_keywords)
        
        return {
            'language_distribution': dict(language_counts),
            'most_common_language': language_counts.most_common(1)[0][0] if language_counts else 'unknown',
            'average_complexity_by_language': avg_complexity_by_lang,
            'total_functions': len(all_functions),
            'unique_functions': len(set(all_functions)),
            'most_common_functions': dict(function_counts.most_common(10)),
            'most_common_keywords': dict(keyword_counts.most_common(10)),
            'syntax_valid_ratio': sum(1 for s in snippets if s.syntax_valid) / len(snippets)
        }
    
    def search_code_snippets(self, query: str, snippets: List[CodeSnippet], 
                            search_type: str = "content") -> List[Tuple[CodeSnippet, float]]:
        """Search code snippets based on query."""
        
        results = []
        query_lower = query.lower()
        
        for snippet in snippets:
            score = 0.0
            
            if search_type == "content":
                # Search in code content
                if query_lower in snippet.cleaned_code.lower():
                    score += 0.9
                
            elif search_type == "language":
                # Search by programming language
                if query_lower == snippet.language.lower():
                    score += 1.0
                
            elif search_type == "function":
                # Search in function names
                if any(query_lower in func.lower() or func.lower() in query_lower 
                      for func in snippet.functions):
                    score += 0.8
                
            elif search_type == "keyword":
                # Search in keywords
                if any(query_lower in keyword.lower() for keyword in snippet.keywords):
                    score += 0.6
                
            elif search_type == "comment":
                # Search in comments
                if any(query_lower in comment.lower() for comment in snippet.comments):
                    score += 0.5
                
            elif search_type == "context":
                # Search in surrounding context
                if query_lower in snippet.context.lower():
                    score += 0.4
            
            # Boost for exact matches
            if query_lower == snippet.cleaned_code.lower():
                score = 1.0
            
            if score > 0:
                results.append((snippet, score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def get_code_summary(self, snippets: List[CodeSnippet]) -> Dict[str, Any]:
        """Get summary statistics for extracted code snippets."""
        
        if not snippets:
            return {"total": 0, "message": "No code snippets found"}
        
        statistics = self._calculate_language_statistics(snippets)
        
        # Additional summary information
        total_lines = sum(snippet.end_line - snippet.start_line + 1 for snippet in snippets)
        avg_complexity = sum(snippet.complexity_score for snippet in snippets) / len(snippets)
        
        return {
            "total_snippets": len(snippets),
            "total_lines_of_code": total_lines,
            "average_complexity": round(avg_complexity, 2),
            "languages_found": list(statistics['language_distribution'].keys()),
            "most_common_language": statistics['most_common_language'],
            **statistics
        }
