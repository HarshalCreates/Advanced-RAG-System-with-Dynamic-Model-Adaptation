"""Query expansion and rewriting mechanisms for improved retrieval."""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

try:
    import nltk
    from nltk.corpus import wordnet
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


@dataclass
class ExpandedQuery:
    """Represents an expanded query with multiple variations."""
    original: str
    expansions: List[str]
    synonyms: Dict[str, List[str]]
    related_terms: List[str]
    boosted_terms: List[str]
    confidence: float


class QueryExpander:
    """Expands queries using multiple strategies."""
    
    def __init__(self):
        self.nltk_available = NLTK_AVAILABLE
        self.domain_synonyms = self._load_domain_synonyms()
        self.expansion_cache = {}
        
        if self.nltk_available:
            try:
                # Download required NLTK data if not present
                nltk.download('punkt', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('stopwords', quiet=True)
                self.stop_words = set(stopwords.words('english'))
            except:
                self.nltk_available = False
                self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        else:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    
    def _load_domain_synonyms(self) -> Dict[str, List[str]]:
        """Load domain-specific synonym dictionary."""
        return {
            # Technology terms
            'ai': ['artificial intelligence', 'machine learning', 'neural networks', 'deep learning'],
            'ml': ['machine learning', 'artificial intelligence', 'predictive modeling'],
            'data': ['information', 'dataset', 'records', 'statistics'],
            'algorithm': ['method', 'procedure', 'approach', 'technique'],
            'model': ['framework', 'system', 'approach', 'methodology'],
            
            # Business terms
            'revenue': ['income', 'earnings', 'profit', 'sales'],
            'customer': ['client', 'user', 'consumer', 'buyer'],
            'strategy': ['plan', 'approach', 'method', 'tactic'],
            'analysis': ['examination', 'study', 'evaluation', 'assessment'],
            'performance': ['results', 'metrics', 'outcomes', 'effectiveness'],
            
            # Academic terms
            'research': ['study', 'investigation', 'analysis', 'examination'],
            'paper': ['article', 'publication', 'document', 'study'],
            'findings': ['results', 'conclusions', 'discoveries', 'outcomes'],
            'methodology': ['method', 'approach', 'procedure', 'technique'],
            'literature': ['publications', 'papers', 'articles', 'research']
        }
    
    def expand_query(self, query: str, max_expansions: int = 5) -> ExpandedQuery:
        """Expand a query using multiple strategies."""
        # Check cache first
        cache_key = f"{query}_{max_expansions}"
        if cache_key in self.expansion_cache:
            return self.expansion_cache[cache_key]
        
        original_query = query.strip()
        expansions = []
        synonyms = {}
        related_terms = []
        boosted_terms = []
        
        # Strategy 1: Domain-specific synonyms
        domain_expanded = self._expand_with_domain_synonyms(original_query)
        expansions.extend(domain_expanded[:max_expansions//2])
        
        # Strategy 2: WordNet synonyms (if available)
        if self.nltk_available:
            wordnet_expanded, wordnet_synonyms = self._expand_with_wordnet(original_query)
            expansions.extend(wordnet_expanded[:max_expansions//2])
            synonyms.update(wordnet_synonyms)
        
        # Strategy 3: Query reformulation
        reformulated = self._reformulate_query(original_query)
        expansions.extend(reformulated[:2])
        
        # Strategy 4: Identify key terms for boosting
        boosted_terms = self._identify_key_terms(original_query)
        
        # Strategy 5: Add related terms
        related_terms = self._find_related_terms(original_query)
        
        # Remove duplicates and limit expansions
        unique_expansions = []
        seen = set([original_query.lower()])
        
        for expansion in expansions:
            if expansion.lower() not in seen:
                unique_expansions.append(expansion)
                seen.add(expansion.lower())
                
                if len(unique_expansions) >= max_expansions:
                    break
        
        # Calculate confidence based on expansion quality
        confidence = self._calculate_expansion_confidence(
            original_query, unique_expansions, synonyms, related_terms
        )
        
        result = ExpandedQuery(
            original=original_query,
            expansions=unique_expansions,
            synonyms=synonyms,
            related_terms=related_terms,
            boosted_terms=boosted_terms,
            confidence=confidence
        )
        
        # Cache result
        self.expansion_cache[cache_key] = result
        return result
    
    def _expand_with_domain_synonyms(self, query: str) -> List[str]:
        """Expand query using domain-specific synonyms."""
        words = self._tokenize_and_clean(query)
        expansions = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.domain_synonyms:
                for synonym in self.domain_synonyms[word_lower]:
                    # Replace word with synonym
                    expanded = query.replace(word, synonym)
                    if expanded != query:
                        expansions.append(expanded)
        
        return expansions
    
    def _expand_with_wordnet(self, query: str) -> Tuple[List[str], Dict[str, List[str]]]:
        """Expand query using WordNet synonyms."""
        if not self.nltk_available:
            return [], {}
        
        try:
            tokens = word_tokenize(query)
            pos_tags = pos_tag(tokens)
            
            expansions = []
            synonyms = {}
            
            for word, pos in pos_tags:
                if word.lower() in self.stop_words or len(word) < 3:
                    continue
                
                # Convert POS tag to WordNet format
                wn_pos = self._get_wordnet_pos(pos)
                if not wn_pos:
                    continue
                
                # Get synonyms from WordNet
                word_synonyms = set()
                for synset in wordnet.synsets(word, pos=wn_pos):
                    for lemma in synset.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        if synonym.lower() != word.lower():
                            word_synonyms.add(synonym)
                
                if word_synonyms:
                    synonyms[word] = list(word_synonyms)[:3]  # Limit to top 3
                    
                    # Create expanded queries
                    for synonym in list(word_synonyms)[:2]:  # Top 2 synonyms
                        expanded = query.replace(word, synonym)
                        if expanded != query:
                            expansions.append(expanded)
            
            return expansions, synonyms
            
        except Exception as e:
            print(f"WordNet expansion failed: {e}")
            return [], {}
    
    def _reformulate_query(self, query: str) -> List[str]:
        """Reformulate query using different phrasings."""
        reformulations = []
        
        # Pattern 1: Question to statement
        if query.endswith('?'):
            statement = query.rstrip('?')
            reformulations.append(statement)
            reformulations.append(f"information about {statement}")
        
        # Pattern 2: Add context words
        if not any(word in query.lower() for word in ['what', 'how', 'why', 'when', 'where']):
            reformulations.append(f"what is {query}")
            reformulations.append(f"how to {query}")
        
        # Pattern 3: Add domain context
        if len(query.split()) <= 3:  # Short queries
            reformulations.append(f"{query} definition")
            reformulations.append(f"{query} explanation")
            reformulations.append(f"{query} examples")
        
        # Pattern 4: Paraphrasing
        paraphrases = self._generate_paraphrases(query)
        reformulations.extend(paraphrases)
        
        return reformulations
    
    def _generate_paraphrases(self, query: str) -> List[str]:
        """Generate paraphrases of the query."""
        paraphrases = []
        
        # Simple pattern-based paraphrasing
        patterns = [
            (r'find (.+)', r'search for \1'),
            (r'what is (.+)', r'definition of \1'),
            (r'how to (.+)', r'method for \1'),
            (r'(.+) analysis', r'analyze \1'),
            (r'(.+) comparison', r'compare \1'),
        ]
        
        for pattern, replacement in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                paraphrase = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
                paraphrases.append(paraphrase)
        
        return paraphrases
    
    def _identify_key_terms(self, query: str) -> List[str]:
        """Identify key terms that should be boosted in search."""
        words = self._tokenize_and_clean(query)
        key_terms = []
        
        # Heuristics for identifying key terms
        for word in words:
            # Skip stop words and short words
            if word.lower() in self.stop_words or len(word) < 3:
                continue
            
            # Boost proper nouns, technical terms, and domain-specific words
            if (word[0].isupper() or  # Proper noun
                word.lower() in self.domain_synonyms or  # Domain term
                len(word) > 6):  # Longer words often more specific
                key_terms.append(word)
        
        return key_terms
    
    def _find_related_terms(self, query: str) -> List[str]:
        """Find terms related to the query."""
        words = self._tokenize_and_clean(query)
        related = []
        
        # Find related terms from domain knowledge
        for word in words:
            word_lower = word.lower()
            
            # Add synonyms as related terms
            if word_lower in self.domain_synonyms:
                related.extend(self.domain_synonyms[word_lower][:2])
            
            # Add domain-specific related terms
            related_map = {
                'data': ['statistics', 'information', 'analytics'],
                'model': ['prediction', 'training', 'validation'],
                'algorithm': ['computation', 'optimization', 'efficiency'],
                'performance': ['metrics', 'benchmarks', 'evaluation'],
                'research': ['methodology', 'findings', 'hypothesis']
            }
            
            if word_lower in related_map:
                related.extend(related_map[word_lower])
        
        return list(set(related))  # Remove duplicates
    
    def _calculate_expansion_confidence(self, original: str, expansions: List[str], 
                                      synonyms: Dict[str, List[str]], related: List[str]) -> float:
        """Calculate confidence score for the expansion quality."""
        base_confidence = 0.5
        
        # Boost confidence based on number of quality expansions
        if expansions:
            base_confidence += min(0.3, len(expansions) * 0.1)
        
        # Boost confidence based on synonym quality
        if synonyms:
            base_confidence += min(0.2, len(synonyms) * 0.05)
        
        # Boost confidence based on related terms
        if related:
            base_confidence += min(0.1, len(related) * 0.02)
        
        # Penalty for very short or very long queries
        word_count = len(original.split())
        if word_count < 2 or word_count > 10:
            base_confidence *= 0.8
        
        return min(0.95, base_confidence)
    
    def _tokenize_and_clean(self, text: str) -> List[str]:
        """Tokenize and clean text."""
        if self.nltk_available:
            try:
                tokens = word_tokenize(text)
                return [token for token in tokens if token.isalpha()]
            except:
                pass
        
        # Fallback tokenization
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        return words
    
    def _get_wordnet_pos(self, treebank_tag: str) -> Optional[str]:
        """Convert TreeBank POS tag to WordNet POS tag."""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None


class QueryRewriter:
    """Rewrites queries for better retrieval performance."""
    
    def __init__(self):
        self.expander = QueryExpander()
        self.rewrite_patterns = self._load_rewrite_patterns()
    
    def _load_rewrite_patterns(self) -> List[Tuple[str, str, float]]:
        """Load query rewrite patterns."""
        return [
            # (pattern, replacement, confidence)
            (r"what is the (.+) of (.+)", r"\1 \2", 0.8),
            (r"how does (.+) work", r"\1 mechanism operation", 0.9),
            (r"difference between (.+) and (.+)", r"\1 vs \2 comparison", 0.85),
            (r"benefits of (.+)", r"\1 advantages pros", 0.8),
            (r"problems with (.+)", r"\1 issues disadvantages", 0.8),
            (r"(.+) tutorial", r"how to \1 guide", 0.7),
            (r"(.+) example", r"\1 instance case study", 0.75),
        ]
    
    def rewrite_query(self, query: str) -> List[Tuple[str, float]]:
        """Rewrite query using learned patterns."""
        rewrites = [(query, 1.0)]  # Original query with full confidence
        
        query_lower = query.lower()
        
        # Apply rewrite patterns
        for pattern, replacement, confidence in self.rewrite_patterns:
            match = re.search(pattern, query_lower)
            if match:
                rewritten = re.sub(pattern, replacement, query_lower)
                rewrites.append((rewritten, confidence))
        
        # Add expanded queries
        expanded = self.expander.expand_query(query, max_expansions=3)
        for expansion in expanded.expansions:
            rewrites.append((expansion, expanded.confidence * 0.8))
        
        # Remove duplicates and sort by confidence
        unique_rewrites = {}
        for rewrite, conf in rewrites:
            if rewrite not in unique_rewrites or unique_rewrites[rewrite] < conf:
                unique_rewrites[rewrite] = conf
        
        return sorted(unique_rewrites.items(), key=lambda x: x[1], reverse=True)
