from abc import ABC, abstractmethod
from typing import Dict


class LLMProvider(ABC):
    """Tüm LLM sağlayıcılarının uyması gereken arayüz."""

    @abstractmethod
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Verilen metni analiz eder ve her etiket için 0.0-1.0 arası
        bir confidence skoru döndürür.

        Örnek dönüş: {"akici_ve_surukleyici": 0.82, "mizah": 0.10, ...}
        """
        raise NotImplementedError

    @abstractmethod
    def is_healthy(self) -> bool:
        """Sağlayıcının kullanıma hazır olup olmadığını kontrol eder
        (örn. API key tanımlı mı)."""
        raise NotImplementedError
