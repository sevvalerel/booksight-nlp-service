import json
import logging
import re
from typing import Dict

from config import LABELS

logger = logging.getLogger(__name__)


def parse_label_scores(raw_text: str) -> Dict[str, float]:
    """LLM yanıtından JSON'u ayıklar ve etiket skorlarına dönüştürür.

    LLM bazen JSON'u ```json ... ``` bloğu içinde veya öncesinde/sonrasında
    ek açıklama metniyle döndürebilir; bu fonksiyon yanıt içindeki ilk
    geçerli JSON objesini bulup ayıklar.
    """
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        logger.error("LLM yanıtından JSON ayıklanamadı: %s", raw_text)
        raise ValueError("LLM yanıtı geçerli JSON içermiyor")

    data = json.loads(match.group(0))

    scores = {}
    for label in LABELS:
        value = data.get(label, 0.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        scores[label] = max(0.0, min(1.0, value))

    return scores
