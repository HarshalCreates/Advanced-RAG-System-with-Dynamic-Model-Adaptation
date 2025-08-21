"""Real translation services integration for cross-language capabilities."""
from __future__ import annotations

import hashlib
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import asyncio

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    from googletrans import Translator as GoogleTranslator
    GOOGLETRANS_AVAILABLE = True
except (ImportError, AttributeError) as e:
    GOOGLETRANS_AVAILABLE = False
    print(f"Warning: Google Translate not available: {e}")


class TranslationService(Enum):
    """Available translation services."""
    GOOGLE_TRANSLATE = "google_translate"
    AZURE_TRANSLATOR = "azure_translator"
    AWS_TRANSLATE = "aws_translate"
    DEEPL = "deepl"
    LIBRE_TRANSLATE = "libre_translate"
    MOCK = "mock"  # For testing/fallback


@dataclass
class TranslationResult:
    """Result of a translation operation."""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float
    service_used: TranslationService
    translation_time_ms: int
    cached: bool = False


@dataclass
class TranslationConfig:
    """Configuration for translation services."""
    primary_service: TranslationService = TranslationService.GOOGLE_TRANSLATE
    fallback_services: List[TranslationService] = None
    api_keys: Dict[str, str] = None
    cache_enabled: bool = True
    cache_ttl_hours: int = 24
    max_text_length: int = 5000
    request_timeout_seconds: int = 10
    
    def __post_init__(self):
        if self.fallback_services is None:
            self.fallback_services = [TranslationService.MOCK]
        if self.api_keys is None:
            self.api_keys = {}


class BaseTranslationProvider:
    """Base class for translation providers."""
    
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.service_name = TranslationService.MOCK
    
    async def translate(self, text: str, target_language: str, 
                       source_language: str = "auto") -> TranslationResult:
        """Translate text from source to target language."""
        raise NotImplementedError
    
    async def detect_language(self, text: str) -> str:
        """Detect the language of the given text."""
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """Check if the translation service is available."""
        return True


class GoogleTranslateProvider(BaseTranslationProvider):
    """Google Translate provider using googletrans library."""
    
    def __init__(self, config: TranslationConfig):
        super().__init__(config)
        self.service_name = TranslationService.GOOGLE_TRANSLATE
        self.translator = None
        
        if GOOGLETRANS_AVAILABLE:
            try:
                self.translator = GoogleTranslator()
            except Exception as e:
                print(f"Failed to initialize Google Translator: {e}")
    
    def is_available(self) -> bool:
        return GOOGLETRANS_AVAILABLE and self.translator is not None
    
    async def translate(self, text: str, target_language: str, 
                       source_language: str = "auto") -> TranslationResult:
        """Translate using Google Translate."""
        
        if not self.is_available():
            raise Exception("Google Translate not available")
        
        start_time = time.time()
        
        try:
            # Truncate text if too long
            if len(text) > self.config.max_text_length:
                text = text[:self.config.max_text_length]
            
            # Perform translation
            result = self.translator.translate(text, dest=target_language, src=source_language)
            
            translation_time = int((time.time() - start_time) * 1000)
            
            return TranslationResult(
                original_text=text,
                translated_text=result.text,
                source_language=result.src,
                target_language=target_language,
                confidence=0.9,  # Google Translate doesn't provide confidence
                service_used=self.service_name,
                translation_time_ms=translation_time
            )
            
        except Exception as e:
            raise Exception(f"Google Translate error: {str(e)}")
    
    async def detect_language(self, text: str) -> str:
        """Detect language using Google Translate."""
        
        if not self.is_available():
            return "en"  # Default fallback
        
        try:
            result = self.translator.detect(text)
            return result.lang
        except Exception:
            return "en"


class AzureTranslatorProvider(BaseTranslationProvider):
    """Azure Translator provider using REST API."""
    
    def __init__(self, config: TranslationConfig):
        super().__init__(config)
        self.service_name = TranslationService.AZURE_TRANSLATOR
        self.api_key = config.api_keys.get("azure_translator_key", "")
        self.region = config.api_keys.get("azure_translator_region", "global")
        self.base_url = "https://api.cognitive.microsofttranslator.com"
    
    def is_available(self) -> bool:
        return AIOHTTP_AVAILABLE and bool(self.api_key)
    
    async def translate(self, text: str, target_language: str, 
                       source_language: str = "auto") -> TranslationResult:
        """Translate using Azure Translator."""
        
        if not self.is_available():
            raise Exception("Azure Translator not available")
        
        start_time = time.time()
        
        # Truncate text if too long
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length]
        
        headers = {
            'Ocp-Apim-Subscription-Key': self.api_key,
            'Ocp-Apim-Subscription-Region': self.region,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(hashlib.md5(text.encode()).hexdigest())
        }
        
        params = {
            'api-version': '3.0',
            'to': target_language
        }
        
        if source_language != "auto":
            params['from'] = source_language
        
        body = [{'text': text}]
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_seconds)) as session:
                async with session.post(
                    f"{self.base_url}/translate",
                    params=params,
                    headers=headers,
                    json=body
                ) as response:
                    
                    if response.status != 200:
                        raise Exception(f"Azure Translator API error: {response.status}")
                    
                    result = await response.json()
                    
                    if not result or not result[0].get('translations'):
                        raise Exception("Invalid response from Azure Translator")
                    
                    translation = result[0]['translations'][0]
                    detected_language = result[0].get('detectedLanguage', {}).get('language', source_language)
                    confidence = result[0].get('detectedLanguage', {}).get('score', 0.9)
                    
                    translation_time = int((time.time() - start_time) * 1000)
                    
                    return TranslationResult(
                        original_text=text,
                        translated_text=translation['text'],
                        source_language=detected_language,
                        target_language=target_language,
                        confidence=confidence,
                        service_used=self.service_name,
                        translation_time_ms=translation_time
                    )
                    
        except Exception as e:
            raise Exception(f"Azure Translator error: {str(e)}")
    
    async def detect_language(self, text: str) -> str:
        """Detect language using Azure Translator."""
        
        if not self.is_available():
            return "en"
        
        headers = {
            'Ocp-Apim-Subscription-Key': self.api_key,
            'Ocp-Apim-Subscription-Region': self.region,
            'Content-type': 'application/json'
        }
        
        params = {'api-version': '3.0'}
        body = [{'text': text[:1000]}]  # Limit text for detection
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_seconds)) as session:
                async with session.post(
                    f"{self.base_url}/detect",
                    params=params,
                    headers=headers,
                    json=body
                ) as response:
                    
                    if response.status != 200:
                        return "en"
                    
                    result = await response.json()
                    
                    if result and result[0].get('language'):
                        return result[0]['language']
                    
                    return "en"
                    
        except Exception:
            return "en"


class DeepLProvider(BaseTranslationProvider):
    """DeepL provider using REST API."""
    
    def __init__(self, config: TranslationConfig):
        super().__init__(config)
        self.service_name = TranslationService.DEEPL
        self.api_key = config.api_keys.get("deepl_api_key", "")
        self.is_pro = config.api_keys.get("deepl_is_pro", False)
        self.base_url = "https://api.deepl.com" if self.is_pro else "https://api-free.deepl.com"
    
    def is_available(self) -> bool:
        return AIOHTTP_AVAILABLE and bool(self.api_key)
    
    async def translate(self, text: str, target_language: str, 
                       source_language: str = "auto") -> TranslationResult:
        """Translate using DeepL."""
        
        if not self.is_available():
            raise Exception("DeepL not available")
        
        start_time = time.time()
        
        # Truncate text if too long
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length]
        
        headers = {
            'Authorization': f'DeepL-Auth-Key {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'text': [text],
            'target_lang': target_language.upper()
        }
        
        if source_language != "auto":
            data['source_lang'] = source_language.upper()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_seconds)) as session:
                async with session.post(
                    f"{self.base_url}/v2/translate",
                    headers=headers,
                    json=data
                ) as response:
                    
                    if response.status != 200:
                        raise Exception(f"DeepL API error: {response.status}")
                    
                    result = await response.json()
                    
                    if not result.get('translations'):
                        raise Exception("Invalid response from DeepL")
                    
                    translation = result['translations'][0]
                    
                    translation_time = int((time.time() - start_time) * 1000)
                    
                    return TranslationResult(
                        original_text=text,
                        translated_text=translation['text'],
                        source_language=translation.get('detected_source_language', source_language).lower(),
                        target_language=target_language.lower(),
                        confidence=0.95,  # DeepL is generally high quality
                        service_used=self.service_name,
                        translation_time_ms=translation_time
                    )
                    
        except Exception as e:
            raise Exception(f"DeepL error: {str(e)}")


class LibreTranslateProvider(BaseTranslationProvider):
    """LibreTranslate provider (open source alternative)."""
    
    def __init__(self, config: TranslationConfig):
        super().__init__(config)
        self.service_name = TranslationService.LIBRE_TRANSLATE
        self.api_key = config.api_keys.get("libretranslate_api_key", "")
        self.base_url = config.api_keys.get("libretranslate_url", "https://libretranslate.de")
    
    def is_available(self) -> bool:
        return AIOHTTP_AVAILABLE
    
    async def translate(self, text: str, target_language: str, 
                       source_language: str = "auto") -> TranslationResult:
        """Translate using LibreTranslate."""
        
        if not self.is_available():
            raise Exception("LibreTranslate not available")
        
        start_time = time.time()
        
        # Truncate text if too long
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length]
        
        data = {
            'q': text,
            'source': source_language,
            'target': target_language,
            'format': 'text'
        }
        
        if self.api_key:
            data['api_key'] = self.api_key
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_seconds)) as session:
                async with session.post(
                    f"{self.base_url}/translate",
                    json=data
                ) as response:
                    
                    if response.status != 200:
                        raise Exception(f"LibreTranslate API error: {response.status}")
                    
                    result = await response.json()
                    
                    if not result.get('translatedText'):
                        raise Exception("Invalid response from LibreTranslate")
                    
                    translation_time = int((time.time() - start_time) * 1000)
                    
                    return TranslationResult(
                        original_text=text,
                        translated_text=result['translatedText'],
                        source_language=source_language,
                        target_language=target_language,
                        confidence=0.8,  # Moderate confidence for open source
                        service_used=self.service_name,
                        translation_time_ms=translation_time
                    )
                    
        except Exception as e:
            raise Exception(f"LibreTranslate error: {str(e)}")


class MockTranslationProvider(BaseTranslationProvider):
    """Mock translation provider for testing and fallback."""
    
    def __init__(self, config: TranslationConfig):
        super().__init__(config)
        self.service_name = TranslationService.MOCK
    
    async def translate(self, text: str, target_language: str, 
                       source_language: str = "auto") -> TranslationResult:
        """Mock translation - adds language prefix."""
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        translated_text = f"[{target_language.upper()}] {text}"
        
        return TranslationResult(
            original_text=text,
            translated_text=translated_text,
            source_language=source_language if source_language != "auto" else "en",
            target_language=target_language,
            confidence=0.5,
            service_used=self.service_name,
            translation_time_ms=100
        )
    
    async def detect_language(self, text: str) -> str:
        """Mock language detection."""
        return "en"


class TranslationCache:
    """Simple in-memory cache for translations."""
    
    def __init__(self, ttl_hours: int = 24):
        self.cache: Dict[str, Tuple[TranslationResult, float]] = {}
        self.ttl_seconds = ttl_hours * 3600
    
    def _generate_key(self, text: str, target_lang: str, source_lang: str) -> str:
        """Generate cache key for translation."""
        content = f"{source_lang}:{target_lang}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, target_lang: str, source_lang: str) -> Optional[TranslationResult]:
        """Get cached translation if available and not expired."""
        
        key = self._generate_key(text, target_lang, source_lang)
        
        if key in self.cache:
            result, timestamp = self.cache[key]
            
            if time.time() - timestamp < self.ttl_seconds:
                # Mark as cached
                result.cached = True
                return result
            else:
                # Remove expired entry
                del self.cache[key]
        
        return None
    
    def set(self, text: str, target_lang: str, source_lang: str, result: TranslationResult):
        """Cache translation result."""
        
        key = self._generate_key(text, target_lang, source_lang)
        self.cache[key] = (result, time.time())
    
    def clear_expired(self):
        """Remove expired entries from cache."""
        
        current_time = time.time()
        expired_keys = []
        
        for key, (_, timestamp) in self.cache.items():
            if current_time - timestamp >= self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        
        total_entries = len(self.cache)
        current_time = time.time()
        
        expired_count = sum(
            1 for _, timestamp in self.cache.values()
            if current_time - timestamp >= self.ttl_seconds
        )
        
        return {
            "total_entries": total_entries,
            "active_entries": total_entries - expired_count,
            "expired_entries": expired_count,
            "cache_size_mb": sum(len(str(result)) for result, _ in self.cache.values()) / (1024 * 1024)
        }


class RealTranslationManager:
    """Main manager for real translation services."""
    
    def __init__(self, config: TranslationConfig = None):
        self.config = config or TranslationConfig()
        self.cache = TranslationCache(self.config.cache_ttl_hours) if self.config.cache_enabled else None
        
        # Initialize providers
        self.providers = {
            TranslationService.GOOGLE_TRANSLATE: GoogleTranslateProvider(self.config),
            TranslationService.AZURE_TRANSLATOR: AzureTranslatorProvider(self.config),
            TranslationService.DEEPL: DeepLProvider(self.config),
            TranslationService.LIBRE_TRANSLATE: LibreTranslateProvider(self.config),
            TranslationService.MOCK: MockTranslationProvider(self.config),
        }
        
        # Get available providers
        self.available_providers = {
            service: provider for service, provider in self.providers.items()
            if provider.is_available()
        }
        
        print(f"Translation services available: {list(self.available_providers.keys())}")
    
    async def translate(self, text: str, target_language: str, 
                       source_language: str = "auto") -> TranslationResult:
        """Translate text with fallback support."""
        
        # Check cache first
        if self.cache:
            cached_result = self.cache.get(text, target_language, source_language)
            if cached_result:
                return cached_result
        
        # Try primary service first
        if self.config.primary_service in self.available_providers:
            try:
                result = await self.available_providers[self.config.primary_service].translate(
                    text, target_language, source_language
                )
                
                # Cache the result
                if self.cache:
                    self.cache.set(text, target_language, source_language, result)
                
                return result
                
            except Exception as e:
                print(f"Primary translation service {self.config.primary_service} failed: {e}")
        
        # Try fallback services
        for fallback_service in self.config.fallback_services:
            if fallback_service in self.available_providers:
                try:
                    result = await self.available_providers[fallback_service].translate(
                        text, target_language, source_language
                    )
                    
                    # Cache the result
                    if self.cache:
                        self.cache.set(text, target_language, source_language, result)
                    
                    return result
                    
                except Exception as e:
                    print(f"Fallback translation service {fallback_service} failed: {e}")
        
        # If all services fail, use mock
        mock_provider = self.providers[TranslationService.MOCK]
        return await mock_provider.translate(text, target_language, source_language)
    
    async def translate_batch(self, texts: List[str], target_language: str, 
                             source_language: str = "auto") -> List[TranslationResult]:
        """Translate multiple texts concurrently."""
        
        # Create translation tasks
        tasks = [
            self.translate(text, target_language, source_language)
            for text in texts
        ]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result
                mock_provider = self.providers[TranslationService.MOCK]
                error_result = await mock_provider.translate(texts[i], target_language, source_language)
                final_results.append(error_result)
            else:
                final_results.append(result)
        
        return final_results
    
    async def detect_language(self, text: str) -> str:
        """Detect language of text using available services."""
        
        # Try primary service first
        if self.config.primary_service in self.available_providers:
            try:
                return await self.available_providers[self.config.primary_service].detect_language(text)
            except Exception as e:
                print(f"Language detection failed with {self.config.primary_service}: {e}")
        
        # Try fallback services
        for fallback_service in self.config.fallback_services:
            if fallback_service in self.available_providers:
                try:
                    return await self.available_providers[fallback_service].detect_language(text)
                except Exception as e:
                    print(f"Language detection failed with {fallback_service}: {e}")
        
        # Default fallback
        return "en"
    
    def get_available_services(self) -> List[TranslationService]:
        """Get list of available translation services."""
        return list(self.available_providers.keys())
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about translation services."""
        
        return {
            "primary_service": self.config.primary_service.value,
            "fallback_services": [s.value for s in self.config.fallback_services],
            "available_services": [s.value for s in self.available_providers.keys()],
            "cache_enabled": self.config.cache_enabled,
            "cache_stats": self.cache.get_stats() if self.cache else None,
            "max_text_length": self.config.max_text_length,
            "request_timeout": self.config.request_timeout_seconds
        }
    
    def update_config(self, new_config: TranslationConfig):
        """Update translation configuration."""
        self.config = new_config
        
        # Reinitialize providers with new config
        for service, provider in self.providers.items():
            self.providers[service] = type(provider)(new_config)
        
        # Update available providers
        self.available_providers = {
            service: provider for service, provider in self.providers.items()
            if provider.is_available()
        }
        
        # Update cache if needed
        if new_config.cache_enabled and not self.cache:
            self.cache = TranslationCache(new_config.cache_ttl_hours)
        elif not new_config.cache_enabled:
            self.cache = None
    
    def clear_cache(self):
        """Clear translation cache."""
        if self.cache:
            self.cache.cache.clear()
    
    async def test_services(self) -> Dict[str, Any]:
        """Test all available translation services."""
        
        test_text = "Hello, how are you today?"
        target_lang = "es"  # Spanish
        
        results = {}
        
        for service, provider in self.available_providers.items():
            try:
                start_time = time.time()
                result = await provider.translate(test_text, target_lang)
                test_time = int((time.time() - start_time) * 1000)
                
                results[service.value] = {
                    "status": "success",
                    "translated_text": result.translated_text,
                    "confidence": result.confidence,
                    "response_time_ms": test_time
                }
                
            except Exception as e:
                results[service.value] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        return {
            "test_text": test_text,
            "target_language": target_lang,
            "results": results
        }
