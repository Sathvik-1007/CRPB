from __future__ import annotations
import os
from typing import List, Dict, Any
from openai import OpenAI


class LLM:
    def __init__(self, model: str | None = None):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model or os.environ.get("CRPB_OPENAI_MODEL", "gpt-4o-mini")

    def complete(self, system: str, messages: List[Dict[str, str]], temperature: float = 0.2, model: str | None = None) -> str:
        resp = self.client.chat.completions.create(
            model=model or self.model,
            messages=[{"role": "system", "content": system}] + messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""
