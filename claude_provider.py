import logging
from typing import Dict

from anthropic import Anthropic

from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, build_prompt
from llm.base import LLMProvider
from llm.utils import parse_label_scores

logger = logging.getLogger(__name__)


class ClaudeProvider(LLMProvider):
    def __init__(self):
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
        self.model = ANTHROPIC_MODEL

    def is_healthy(self) -> bool:
        return self.client is not None

    def analyze(self, text: str) -> Dict[str, float]:
        if not self.client:
            raise RuntimeError("ANTHROPIC_API_KEY tanımlı değil")

        prompt = build_prompt(text)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = "".join(
            block.text for block in response.content if block.type == "text"
        )

        return parse_label_scores(raw_text)
