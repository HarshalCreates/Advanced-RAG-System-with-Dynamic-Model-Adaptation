"""Mathematical formula extraction and semantic understanding."""
from __future__ import annotations

import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    import sympy
    from sympy.parsing.latex import parse_latex
    from sympy import latex, simplify, expand
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from io import BytesIO
    import base64
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class MathFormula:
    """Represents an extracted mathematical formula."""
    formula_id: str
    raw_text: str
    latex_code: str
    formula_type: str  # "equation", "expression", "inequality", "system"
    variables: List[str]
    constants: List[str]
    operations: List[str]
    complexity_score: float
    semantic_representation: Optional[str]
    normalized_form: Optional[str]
    page_number: Optional[int]
    context: str
    confidence: float


@dataclass
class MathExtractionResult:
    """Result of mathematical formula extraction."""
    total_formulas: int
    formulas_by_type: Dict[str, int]
    extracted_formulas: List[MathFormula]
    extraction_metadata: Dict[str, Any]


class LaTeXPatternMatcher:
    """Matches and extracts LaTeX mathematical formulas."""
    
    def __init__(self):
        # Common LaTeX math environment patterns
        self.math_patterns = [
            # Inline math: $...$
            re.compile(r'\$([^$]+)\$'),
            # Display math: $$...$$
            re.compile(r'\$\$([^$]+)\$\$'),
            # LaTeX environments
            re.compile(r'\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}', re.DOTALL),
            re.compile(r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}', re.DOTALL),
            re.compile(r'\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}', re.DOTALL),
            re.compile(r'\\begin\{multline\*?\}(.*?)\\end\{multline\*?\}', re.DOTALL),
            # Brackets
            re.compile(r'\\\\[(.*?)\\\\]', re.DOTALL),
            re.compile(r'\\\\((.*?)\\\\)', re.DOTALL),
        ]
        
        # Mathematical symbols and operations
        self.operation_patterns = {
            'arithmetic': ['+', '-', '*', '/', '^', '_'],
            'calculus': ['\\\\int', '\\\\sum', '\\\\prod', '\\\\lim', '\\\\frac', '\\\\partial'],
            'algebra': ['\\\\sqrt', '\\\\log', '\\\\ln', '\\\\exp', '\\\\sin', '\\\\cos', '\\\\tan'],
            'logic': ['\\\\forall', '\\\\exists', '\\\\land', '\\\\lor', '\\\\neg', '\\\\implies'],
            'set_theory': ['\\\\cup', '\\\\cap', '\\\\subset', '\\\\supset', '\\\\in', '\\\\notin'],
            'geometry': ['\\\\angle', '\\\\triangle', '\\\\parallel', '\\\\perp'],
            'probability': ['\\\\Pr', '\\\\mathbb\{P\}', '\\\\mathbb\{E\}', '\\\\text\{Var\}']
        }
        
        # Variable patterns
        self.variable_pattern = re.compile(r'\\?([a-zA-Z](?:_\{[^}]+\}|_[a-zA-Z0-9])?(?:\^\{[^}]+\}|\^[a-zA-Z0-9])?)')
        
        # Number patterns
        self.number_pattern = re.compile(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')
    
    def extract_latex_formulas(self, text: str, page_number: Optional[int] = None) -> List[Dict[str, Any]]:
        """Extract LaTeX formulas from text."""
        formulas = []
        
        for pattern in self.math_patterns:
            matches = pattern.finditer(text)
            
            for match in matches:
                formula_text = match.group(1) if match.groups() else match.group(0)
                
                # Clean up the formula
                cleaned_formula = self._clean_latex_formula(formula_text)
                
                if cleaned_formula and len(cleaned_formula.strip()) > 1:
                    # Extract context around the formula
                    start_pos = max(0, match.start() - 100)
                    end_pos = min(len(text), match.end() + 100)
                    context = text[start_pos:end_pos]
                    
                    formula_info = {
                        'raw_text': match.group(0),
                        'latex_code': cleaned_formula,
                        'position': (match.start(), match.end()),
                        'context': context,
                        'page_number': page_number
                    }
                    formulas.append(formula_info)
        
        return formulas
    
    def _clean_latex_formula(self, formula: str) -> str:
        """Clean and normalize LaTeX formula."""
        # Remove extra whitespace
        formula = re.sub(r'\s+', ' ', formula.strip())
        
        # Remove common LaTeX formatting that doesn't affect math content
        formula = re.sub(r'\\text\{([^}]+)\}', r'\1', formula)
        formula = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', formula)
        
        # Ensure proper spacing around operators
        formula = re.sub(r'([+\-=])([^\s])', r'\1 \2', formula)
        formula = re.sub(r'([^\s])([+\-=])', r'\1 \2', formula)
        
        return formula.strip()


class MathFormulaAnalyzer:
    """Analyzes mathematical formulas for semantic content."""
    
    def __init__(self):
        self.latex_matcher = LaTeXPatternMatcher()
        self.sympy_available = SYMPY_AVAILABLE
        
        if not self.sympy_available:
            print("Warning: SymPy not available. Mathematical analysis will be limited.")
    
    def analyze_formula(self, latex_code: str, context: str = "") -> Dict[str, Any]:
        """Analyze a mathematical formula for semantic content."""
        
        analysis = {
            'variables': [],
            'constants': [],
            'operations': [],
            'formula_type': 'expression',
            'complexity_score': 0.0,
            'semantic_representation': None,
            'normalized_form': None,
            'confidence': 0.5
        }
        
        # Extract variables
        variables = self._extract_variables(latex_code)
        analysis['variables'] = list(set(variables))
        
        # Extract constants/numbers
        constants = self._extract_constants(latex_code)
        analysis['constants'] = list(set(constants))
        
        # Identify operations
        operations = self._identify_operations(latex_code)
        analysis['operations'] = operations
        
        # Determine formula type
        analysis['formula_type'] = self._classify_formula_type(latex_code, context)
        
        # Calculate complexity
        analysis['complexity_score'] = self._calculate_complexity(latex_code, variables, operations)
        
        # Try SymPy analysis if available
        if self.sympy_available:
            sympy_analysis = self._sympy_analysis(latex_code)
            analysis.update(sympy_analysis)
        
        return analysis
    
    def _extract_variables(self, latex_code: str) -> List[str]:
        """Extract variable names from LaTeX formula."""
        variables = []
        
        # Find variable patterns
        matches = self.latex_matcher.variable_pattern.findall(latex_code)
        
        for match in matches:
            # Skip common function names and constants
            if match not in ['sin', 'cos', 'tan', 'log', 'ln', 'exp', 'pi', 'e', 'infty']:
                variables.append(match)
        
        return variables
    
    def _extract_constants(self, latex_code: str) -> List[str]:
        """Extract numerical constants from LaTeX formula."""
        constants = []
        
        # Find number patterns
        matches = self.latex_matcher.number_pattern.findall(latex_code)
        constants.extend(matches)
        
        # Find common mathematical constants
        math_constants = ['\\\\pi', '\\\\e', '\\\\infty', '\\\\phi', '\\\\gamma']
        for const in math_constants:
            if const in latex_code:
                constants.append(const.replace('\\\\', ''))
        
        return constants
    
    def _identify_operations(self, latex_code: str) -> List[str]:
        """Identify mathematical operations in the formula."""
        operations = []
        
        for category, ops in self.latex_matcher.operation_patterns.items():
            for op in ops:
                if op in latex_code:
                    operations.append(f"{category}:{op.replace('\\\\', '')}")
        
        return operations
    
    def _classify_formula_type(self, latex_code: str, context: str = "") -> str:
        """Classify the type of mathematical formula."""
        
        # Check for equations (contains =)
        if '=' in latex_code:
            if '\\\\leq' in latex_code or '\\\\geq' in latex_code or '<' in latex_code or '>' in latex_code:
                return 'inequality'
            else:
                return 'equation'
        
        # Check for systems (multiple equations)
        if '\\\\begin{align}' in latex_code or '\\\\begin{cases}' in latex_code:
            return 'system'
        
        # Check for definitions (context analysis)
        context_lower = context.lower()
        if any(word in context_lower for word in ['define', 'definition', 'let', 'where']):
            return 'definition'
        
        # Check for theorems/lemmas
        if any(word in context_lower for word in ['theorem', 'lemma', 'proposition', 'corollary']):
            return 'theorem'
        
        # Default to expression
        return 'expression'
    
    def _calculate_complexity(self, latex_code: str, variables: List[str], operations: List[str]) -> float:
        """Calculate complexity score for the formula."""
        
        complexity = 0.0
        
        # Base complexity from formula length
        complexity += len(latex_code) * 0.01
        
        # Variable complexity
        complexity += len(variables) * 0.5
        
        # Operation complexity
        operation_weights = {
            'arithmetic': 0.1,
            'algebra': 0.3,
            'calculus': 0.8,
            'logic': 0.4,
            'set_theory': 0.3,
            'geometry': 0.2,
            'probability': 0.6
        }
        
        for op in operations:
            category = op.split(':')[0]
            weight = operation_weights.get(category, 0.2)
            complexity += weight
        
        # Nested structures increase complexity
        nesting_indicators = ['\\\\frac', '\\\\sqrt', '\\\\sum', '\\\\int', '^{', '_{']
        for indicator in nesting_indicators:
            complexity += latex_code.count(indicator) * 0.2
        
        # Normalize to 0-10 scale
        return min(10.0, complexity)
    
    def _sympy_analysis(self, latex_code: str) -> Dict[str, Any]:
        """Perform SymPy-based mathematical analysis."""
        
        analysis = {
            'semantic_representation': None,
            'normalized_form': None,
            'confidence': 0.5
        }
        
        try:
            # Clean LaTeX for SymPy
            cleaned_latex = self._prepare_for_sympy(latex_code)
            
            # Parse with SymPy
            expr = parse_latex(cleaned_latex)
            
            # Get semantic representation
            analysis['semantic_representation'] = str(expr)
            
            # Get normalized form
            try:
                normalized = simplify(expr)
                analysis['normalized_form'] = latex(normalized)
            except:
                analysis['normalized_form'] = latex(expr)
            
            analysis['confidence'] = 0.9
            
        except Exception as e:
            # Fallback analysis
            analysis['semantic_representation'] = f"Parse error: {str(e)[:100]}"
            analysis['confidence'] = 0.2
        
        return analysis
    
    def _prepare_for_sympy(self, latex_code: str) -> str:
        """Prepare LaTeX code for SymPy parsing."""
        
        # Basic cleanup for SymPy compatibility
        cleaned = latex_code
        
        # Replace common LaTeX constructs
        replacements = {
            '\\\\cdot': '*',
            '\\\\times': '*',
            '\\\\div': '/',
            '\\\\pm': '+',
            '\\\\mp': '-',
            '\\\\neq': '!=',
            '\\\\leq': '<=',
            '\\\\geq': '>=',
            '\\\\ll': '<<',
            '\\\\gg': '>>'
        }
        
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        return cleaned


class MathFormulaExtractor:
    """Main class for extracting and analyzing mathematical formulas."""
    
    def __init__(self):
        self.latex_matcher = LaTeXPatternMatcher()
        self.analyzer = MathFormulaAnalyzer()
        self.formula_cache: Dict[str, MathFormula] = {}
    
    def extract_formulas_from_text(self, text: str, filename: str = "", 
                                  page_number: Optional[int] = None) -> MathExtractionResult:
        """Extract and analyze mathematical formulas from text."""
        
        # Extract raw LaTeX formulas
        raw_formulas = self.latex_matcher.extract_latex_formulas(text, page_number)
        
        extracted_formulas = []
        formulas_by_type = {}
        
        for i, formula_info in enumerate(raw_formulas):
            # Generate unique ID
            formula_id = self._generate_formula_id(formula_info['latex_code'], filename, i)
            
            # Skip if already processed
            if formula_id in self.formula_cache:
                extracted_formulas.append(self.formula_cache[formula_id])
                continue
            
            # Analyze the formula
            analysis = self.analyzer.analyze_formula(
                formula_info['latex_code'], 
                formula_info['context']
            )
            
            # Create MathFormula object
            formula = MathFormula(
                formula_id=formula_id,
                raw_text=formula_info['raw_text'],
                latex_code=formula_info['latex_code'],
                formula_type=analysis['formula_type'],
                variables=analysis['variables'],
                constants=analysis['constants'],
                operations=analysis['operations'],
                complexity_score=analysis['complexity_score'],
                semantic_representation=analysis['semantic_representation'],
                normalized_form=analysis['normalized_form'],
                page_number=formula_info['page_number'],
                context=formula_info['context'],
                confidence=analysis['confidence']
            )
            
            extracted_formulas.append(formula)
            self.formula_cache[formula_id] = formula
            
            # Count by type
            formula_type = formula.formula_type
            formulas_by_type[formula_type] = formulas_by_type.get(formula_type, 0) + 1
        
        # Create extraction metadata
        metadata = {
            'filename': filename,
            'text_length': len(text),
            'page_number': page_number,
            'extraction_method': 'latex_pattern_matching',
            'sympy_available': self.analyzer.sympy_available,
            'average_complexity': sum(f.complexity_score for f in extracted_formulas) / len(extracted_formulas) if extracted_formulas else 0.0
        }
        
        return MathExtractionResult(
            total_formulas=len(extracted_formulas),
            formulas_by_type=formulas_by_type,
            extracted_formulas=extracted_formulas,
            extraction_metadata=metadata
        )
    
    def _generate_formula_id(self, latex_code: str, filename: str, index: int) -> str:
        """Generate unique ID for a formula."""
        content = f"{filename}_{latex_code}_{index}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def search_formulas(self, query: str, formulas: List[MathFormula], 
                       search_type: str = "semantic") -> List[Tuple[MathFormula, float]]:
        """Search formulas based on query."""
        
        results = []
        query_lower = query.lower()
        
        for formula in formulas:
            score = 0.0
            
            if search_type == "semantic":
                # Search in semantic representation
                if formula.semantic_representation and query_lower in formula.semantic_representation.lower():
                    score += 0.8
                
                # Search in variables
                if any(var.lower() in query_lower or query_lower in var.lower() for var in formula.variables):
                    score += 0.6
                
                # Search in operations
                if any(query_lower in op.lower() for op in formula.operations):
                    score += 0.4
                
            elif search_type == "latex":
                # Search in LaTeX code
                if query_lower in formula.latex_code.lower():
                    score += 0.9
                
            elif search_type == "context":
                # Search in context
                if query_lower in formula.context.lower():
                    score += 0.7
            
            elif search_type == "type":
                # Search by formula type
                if query_lower in formula.formula_type.lower():
                    score += 1.0
            
            # Boost score for exact matches
            if query_lower == formula.latex_code.lower():
                score = 1.0
            
            if score > 0:
                results.append((formula, score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def get_formula_summary(self, formulas: List[MathFormula]) -> Dict[str, Any]:
        """Get summary statistics for extracted formulas."""
        
        if not formulas:
            return {"total": 0, "message": "No formulas found"}
        
        # Type distribution
        type_counts = {}
        for formula in formulas:
            type_counts[formula.formula_type] = type_counts.get(formula.formula_type, 0) + 1
        
        # Variable frequency
        all_variables = []
        for formula in formulas:
            all_variables.extend(formula.variables)
        
        var_frequency = {}
        for var in all_variables:
            var_frequency[var] = var_frequency.get(var, 0) + 1
        
        # Most common variables
        common_vars = sorted(var_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Complexity distribution
        complexities = [f.complexity_score for f in formulas]
        
        return {
            "total_formulas": len(formulas),
            "formula_types": type_counts,
            "most_common_variables": common_vars,
            "complexity_stats": {
                "min": min(complexities) if complexities else 0,
                "max": max(complexities) if complexities else 0,
                "average": sum(complexities) / len(complexities) if complexities else 0
            },
            "high_complexity_formulas": len([f for f in formulas if f.complexity_score > 5.0]),
            "unique_variables": len(set(all_variables))
        }


class MathFormulaRenderer:
    """Renders mathematical formulas for visualization."""
    
    def __init__(self):
        self.matplotlib_available = MATPLOTLIB_AVAILABLE
        
        if not self.matplotlib_available:
            print("Warning: Matplotlib not available. Formula rendering disabled.")
    
    def render_formula_to_image(self, latex_code: str, formula_id: str) -> Optional[str]:
        """Render LaTeX formula to base64 image."""
        
        if not self.matplotlib_available:
            return None
        
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.axis('off')
            
            # Render LaTeX
            ax.text(0.5, 0.5, f'${latex_code}$', transform=ax.transAxes, 
                   fontsize=16, ha='center', va='center')
            
            # Save to bytes
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', 
                       dpi=150, transparent=True)
            buffer.seek(0)
            
            # Convert to base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            plt.close(fig)
            
            return image_base64
            
        except Exception as e:
            print(f"Failed to render formula {formula_id}: {e}")
            return None
    
    def create_formula_summary_visualization(self, formulas: List[MathFormula]) -> Optional[str]:
        """Create visualization of formula statistics."""
        
        if not self.matplotlib_available or not formulas:
            return None
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Formula types distribution
            type_counts = {}
            for formula in formulas:
                type_counts[formula.formula_type] = type_counts.get(formula.formula_type, 0) + 1
            
            if type_counts:
                ax1.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
                ax1.set_title('Formula Types Distribution')
            
            # Complexity distribution
            complexities = [f.complexity_score for f in formulas]
            if complexities:
                ax2.hist(complexities, bins=10, alpha=0.7, color='skyblue')
                ax2.set_xlabel('Complexity Score')
                ax2.set_ylabel('Count')
                ax2.set_title('Complexity Distribution')
            
            # Variable frequency (top 10)
            all_variables = []
            for formula in formulas:
                all_variables.extend(formula.variables)
            
            var_frequency = {}
            for var in all_variables:
                var_frequency[var] = var_frequency.get(var, 0) + 1
            
            if var_frequency:
                top_vars = sorted(var_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
                vars_names, vars_counts = zip(*top_vars) if top_vars else ([], [])
                
                ax3.bar(vars_names, vars_counts)
                ax3.set_xlabel('Variables')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Most Common Variables')
                ax3.tick_params(axis='x', rotation=45)
            
            # Operations frequency
            all_operations = []
            for formula in formulas:
                all_operations.extend([op.split(':')[1] for op in formula.operations])
            
            op_frequency = {}
            for op in all_operations:
                op_frequency[op] = op_frequency.get(op, 0) + 1
            
            if op_frequency:
                top_ops = sorted(op_frequency.items(), key=lambda x: x[1], reverse=True)[:8]
                ops_names, ops_counts = zip(*top_ops) if top_ops else ([], [])
                
                ax4.bar(ops_names, ops_counts)
                ax4.set_xlabel('Operations')
                ax4.set_ylabel('Frequency')
                ax4.set_title('Most Common Operations')
                ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            plt.close(fig)
            
            return image_base64
            
        except Exception as e:
            print(f"Failed to create formula summary visualization: {e}")
            return None
