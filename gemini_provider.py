import logging
from typing import Dict

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL, build_prompt
from llm.base import LLMProvider
from llm.utils import parse_label_scores

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    def __init__(self):
        self._configured = bool(GEMINI_API_KEY)
        self.client = genai.Client(api_key=GEMINI_API_KEY) if self._configured else None
        self.model = GEMINI_MODEL

    def is_healthy(self) -> bool:
        return self._configured

    def analyze(self, text: str) -> Dict[str, float]:
        if not self.client:
            raise RuntimeError("GEMINI_API_KEY tanımlı değil")

        prompt = build_prompt(text)

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=512,
                temperature=0.0,
            ),
        )

        raw_text = response.text
        return parse_label_scores(raw_text)
