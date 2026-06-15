import logging

from config import LLM_PROVIDER
from base import LLMProvider
from claude_provider import ClaudeProvider
from gemini_provider import GeminiProvider

logger = logging.getLogger(__name__)

_provider_instance: LLMProvider | None = None


def get_provider() -> LLMProvider:
    """Aktif LLM sağlayıcısının (singleton) örneğini döndürür.

    LLM_PROVIDER ortam değişkenine göre "claude" veya "gemini" seçilir.
    """
    global _provider_instance
    if _provider_instance is None:
        if LLM_PROVIDER == "gemini":
            logger.info("LLM sağlayıcısı: Gemini")
            _provider_instance = GeminiProvider()
        else:
            logger.info("LLM sağlayıcısı: Claude")
            _provider_instance = ClaudeProvider()
    return _provider_instance
