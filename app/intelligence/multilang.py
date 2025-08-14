"""Multi-language detection and cross-language retrieval capabilities."""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

try:
    from langdetect import detect, detect_langs, LangDetectException
    import spacy
    from transformers import AutoModel, AutoTokenizer
    MULTILANG_AVAILABLE = True
except ImportError:
    MULTILANG_AVAILABLE = False

try:
    from app.intelligence.translation_services import RealTranslationManager, TranslationConfig
    REAL_TRANSLATION_AVAILABLE = True
except (ImportError, AttributeError) as e:
    REAL_TRANSLATION_AVAILABLE = False
    print(f"Warning: Real translation services not available: {e}")


@dataclass
class LanguageInfo:
    """Language detection result."""
    language: str
    confidence: float
    script: Optional[str] = None
    region: Optional[str] = None


@dataclass
class CrossLangQuery:
    """Cross-language query with translations."""
    original: str
    original_lang: str
    translations: Dict[str, str]  # lang_code -> translated_query
    confidence: float


class LanguageDetector:
    """Detects language of text with confidence scoring."""
    
    def __init__(self):
        self.available = MULTILANG_AVAILABLE
        if not self.available:
            print("Warning: Multi-language support not available. Install langdetect and spacy.")
    
    def detect_language(self, text: str) -> LanguageInfo:
        """Detect language of text with confidence."""
        if not self.available or not text.strip():
            return LanguageInfo(language="en", confidence=0.5)
        
        try:
            # Primary detection
            detected = detect(text)
            
            # Get confidence scores for all detected languages
            lang_probs = detect_langs(text)
            
            # Find confidence for detected language
            confidence = 0.5
            for lang_prob in lang_probs:
                if lang_prob.lang == detected:
                    confidence = lang_prob.prob
                    break
            
            return LanguageInfo(
                language=detected,
                confidence=confidence,
                script=self._detect_script(text),
                region=None  # Could be enhanced with region detection
            )
            
        except LangDetectException:
            # Fallback for very short texts or detection failures
            return LanguageInfo(language="en", confidence=0.3)
    
    def _detect_script(self, text: str) -> Optional[str]:
        """Detect script (Latin, Cyrillic, Arabic, etc.)."""
        # Simple script detection based on Unicode ranges
        scripts = {
            'latin': range(0x0000, 0x024F),
            'cyrillic': range(0x0400, 0x04FF),
            'arabic': range(0x0600, 0x06FF),
            'chinese': range(0x4E00, 0x9FFF),
            'japanese_hiragana': range(0x3040, 0x309F),
            'japanese_katakana': range(0x30A0, 0x30FF),
            'korean': range(0xAC00, 0xD7AF),
        }
        
        char_counts = {script: 0 for script in scripts}
        
        for char in text:
            char_code = ord(char)
            for script, char_range in scripts.items():
                if char_code in char_range:
                    char_counts[script] += 1
                    break
        
        # Return most common script
        if char_counts:
            max_script = max(char_counts, key=char_counts.get)
            if char_counts[max_script] > 0:
                return max_script
        
        return "latin"  # Default fallback


class CrossLanguageRetriever:
    """Handles cross-language retrieval and query translation."""
    
    def __init__(self):
        self.detector = LanguageDetector()
        self.translation_cache = {}  # Simple in-memory cache
        self.multilingual_models = {}
        
        # Initialize real translation services
        if REAL_TRANSLATION_AVAILABLE:
            try:
                self.translation_manager = RealTranslationManager()
            except Exception as e:
                print(f"Failed to initialize real translation services: {e}")
                self.translation_manager = None
        else:
            self.translation_manager = None
        
        # Load multilingual sentence transformer if available
        if MULTILANG_AVAILABLE:
            try:
                from sentence_transformers import SentenceTransformer
                self.multilingual_encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            except:
                self.multilingual_encoder = None
        else:
            self.multilingual_encoder = None
    
    def prepare_cross_lang_query(self, query: str, target_languages: List[str] = None) -> CrossLangQuery:
        """Prepare query for cross-language retrieval."""
        # Detect query language
        lang_info = self.detector.detect_language(query)
        
        if target_languages is None:
            target_languages = ['en', 'es', 'fr', 'de', 'zh', 'ja', 'ar']  # Common languages
        
        # Remove source language from targets
        target_languages = [lang for lang in target_languages if lang != lang_info.language]
        
        # Translate query (mock implementation - would use real translation service)
        translations = {}
        for target_lang in target_languages:
            translated = self._translate_text(query, lang_info.language, target_lang)
            if translated:
                translations[target_lang] = translated
        
        return CrossLangQuery(
            original=query,
            original_lang=lang_info.language,
            translations=translations,
            confidence=lang_info.confidence
        )
    
    def _translate_text(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Translate text between languages using real translation services."""
        # Create cache key
        cache_key = hashlib.md5(f"{text}_{source_lang}_{target_lang}".encode()).hexdigest()
        
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        # Use real translation services if available
        if self.translation_manager:
            try:
                import asyncio
                # Run async translation in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        self.translation_manager.translate(text, target_lang, source_lang)
                    )
                    translated = result.translated_text
                    self.translation_cache[cache_key] = translated
                    return translated
                finally:
                    loop.close()
            except Exception as e:
                print(f"Real translation failed: {e}")
        
        # Fallback to mock translation
        if source_lang != target_lang:
            translated = f"[{target_lang.upper()}] {text}"  # Placeholder
            self.translation_cache[cache_key] = translated
            return translated
        
        return text
    
    def get_multilingual_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Get embeddings using multilingual model."""
        if not self.multilingual_encoder:
            return None
        
        try:
            embeddings = self.multilingual_encoder.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            print(f"Multilingual embedding failed: {e}")
            return None
    
    def expand_query_multilingually(self, query: str, num_expansions: int = 3) -> List[str]:
        """Expand query using multilingual synonyms and variations."""
        expansions = [query]  # Original query
        
        # Detect language
        lang_info = self.detector.detect_language(query)
        
        # Add language-specific expansions
        if lang_info.language == 'en':
            # English expansions
            expansions.extend(self._get_english_expansions(query, num_expansions))
        elif lang_info.language == 'es':
            # Spanish expansions
            expansions.extend(self._get_spanish_expansions(query, num_expansions))
        # Add more languages as needed
        
        # Add cross-language translations
        cross_lang_query = self.prepare_cross_lang_query(query, ['en', 'es', 'fr'])
        expansions.extend(cross_lang_query.translations.values())
        
        return list(set(expansions))  # Remove duplicates
    
    def _get_english_expansions(self, query: str, num_expansions: int) -> List[str]:
        """Get English query expansions."""
        # Simple synonym expansion (in production, use WordNet, word2vec, etc.)
        synonyms = {
            'find': ['search', 'locate', 'discover'],
            'document': ['file', 'paper', 'text'],
            'information': ['data', 'details', 'facts'],
            'analyze': ['examine', 'study', 'evaluate'],
            'summary': ['overview', 'abstract', 'synopsis']
        }
        
        expansions = []
        words = query.lower().split()
        
        for word in words:
            if word in synonyms and len(expansions) < num_expansions:
                for synonym in synonyms[word][:num_expansions - len(expansions)]:
                    expanded_query = query.replace(word, synonym)
                    expansions.append(expanded_query)
        
        return expansions
    
    def _get_spanish_expansions(self, query: str, num_expansions: int) -> List[str]:
        """Get Spanish query expansions."""
        # Spanish synonyms
        synonyms = {
            'buscar': ['encontrar', 'localizar', 'hallar'],
            'documento': ['archivo', 'papel', 'texto'],
            'información': ['datos', 'detalles', 'hechos'],
            'analizar': ['examinar', 'estudiar', 'evaluar']
        }
        
        expansions = []
        words = query.lower().split()
        
        for word in words:
            if word in synonyms and len(expansions) < num_expansions:
                for synonym in synonyms[word][:num_expansions - len(expansions)]:
                    expanded_query = query.replace(word, synonym)
                    expansions.append(expanded_query)
        
        return expansions


class DocumentLanguageAnalyzer:
    """Analyzes language characteristics of documents."""
    
    def __init__(self):
        self.detector = LanguageDetector()
    
    def analyze_document_languages(self, text: str, chunk_size: int = 1000) -> Dict[str, Any]:
        """Analyze language distribution in a document."""
        # Split into chunks for analysis
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        language_distribution = {}
        chunk_languages = []
        
        for chunk in chunks:
            if chunk.strip():
                lang_info = self.detector.detect_language(chunk)
                language_distribution[lang_info.language] = language_distribution.get(lang_info.language, 0) + 1
                chunk_languages.append({
                    'language': lang_info.language,
                    'confidence': lang_info.confidence,
                    'chunk_length': len(chunk)
                })
        
        # Determine primary language
        primary_language = max(language_distribution, key=language_distribution.get) if language_distribution else 'en'
        
        # Calculate confidence
        total_chunks = len(chunk_languages)
        primary_chunks = language_distribution.get(primary_language, 0)
        primary_confidence = primary_chunks / total_chunks if total_chunks > 0 else 0.0
        
        return {
            'primary_language': primary_language,
            'primary_confidence': primary_confidence,
            'language_distribution': language_distribution,
            'is_multilingual': len(language_distribution) > 1,
            'chunk_analysis': chunk_languages,
            'total_chunks': total_chunks
        }
    
    def suggest_retrieval_strategy(self, doc_analysis: Dict[str, Any], query_lang: str) -> Dict[str, Any]:
        """Suggest retrieval strategy based on document and query languages."""
        doc_lang = doc_analysis['primary_language']
        is_cross_lang = doc_lang != query_lang
        
        strategy = {
            'use_translation': is_cross_lang,
            'use_multilingual_embeddings': doc_analysis['is_multilingual'] or is_cross_lang,
            'boost_factor': 1.0,
            'recommended_methods': []
        }
        
        if is_cross_lang:
            strategy['recommended_methods'].append('translate_query')
            strategy['boost_factor'] = 0.8  # Slightly lower confidence for cross-language
        
        if doc_analysis['is_multilingual']:
            strategy['recommended_methods'].append('multilingual_embeddings')
            strategy['recommended_methods'].append('language_aware_chunking')
        
        if doc_analysis['primary_confidence'] < 0.7:
            strategy['recommended_methods'].append('multiple_language_processing')
        
        return strategy
