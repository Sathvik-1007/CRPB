from __future__ import annotations
import os
import time
from typing import List, Dict, Any
from openai import OpenAI
from ..config import DEFAULT_MODEL


class LLM:
    def __init__(self, model: str | None = None):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model or DEFAULT_MODEL

    def complete(self, system: str, messages: List[Dict[str, str]], temperature: float = 0.2, model: str | None = None) -> str:
        retries = int(os.environ.get("CRPB_LLM_RETRIES", "2"))
        backoff = float(os.environ.get("CRPB_LLM_RETRY_BACKOFF", "1.0"))
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                start = time.time()
                resp = self.client.chat.completions.create(
                    model=model or self.model,
                    messages=[{"role": "system", "content": system}] + messages,
                    temperature=temperature,
                )
                _ = time.time() - start  # elapsed for potential future logging
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                if attempt < retries:
                    time.sleep(backoff * (2 ** attempt))
                else:
                    break
        raise RuntimeError(f"LLM request failed after {retries + 1} attempts: {last_err}")
