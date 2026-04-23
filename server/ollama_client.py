# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Round 2 — thin wrapper around the Ollama Cloud chat API.

Reads OLLAMA_API_KEY and OLLAMA_HOST from the environment.
Graceful degradation: every public method returns None when the API key
is missing or the HTTP request fails. The adversarial designer (which is
the only caller today) must handle None and fall back to procedural
random_incident scenarios.

Deliberately minimal. Does not:
  - do retries (one attempt, then None)
  - stream (entire response is captured in one call)
  - implement any of the /api/generate endpoints (only /api/chat)
"""

import json
import logging
import os
from typing import Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class OllamaClient:
    """Minimal Ollama Cloud chat client."""

    DEFAULT_HOST = "https://ollama.com"
    DEFAULT_TIMEOUT = 60.0

    def __init__(
        self,
        model: str = "gpt-oss:120b",
        host: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST") or self.DEFAULT_HOST
        self.api_key = api_key if api_key is not None else os.environ.get("OLLAMA_API_KEY")
        self.timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT

        if not self.api_key:
            logger.warning(
                "OllamaClient: no OLLAMA_API_KEY set — generate/generate_json "
                "will return None. Set OLLAMA_API_KEY in .env to enable the "
                "adversarial designer."
            )

    # --- public API -----------------------------------------------------------

    def generate(
        self,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Optional[str]:
        """Plain-text completion via /api/chat. None on any failure."""
        if not self.api_key:
            return None

        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.host}/api/chat", json=payload, headers=headers
                )
                resp.raise_for_status()
                data = resp.json()
                return data["message"]["content"]
        except Exception as e:
            logger.error("OllamaClient.generate failed (%s): %s", type(e).__name__, e)
            return None

    def generate_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.5,
        max_tokens: int = 2048,
    ) -> Optional[Dict]:
        """JSON-parsed completion. Strips markdown fences; None on any failure."""
        system_json = (
            system.rstrip()
            + "\n\nReturn valid JSON only. No markdown code fences. No prose."
        )
        content = self.generate(system_json, user, temperature, max_tokens)
        if content is None:
            return None

        content = _strip_code_fence(content.strip())
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error("OllamaClient.generate_json: invalid JSON (%s). First 300 chars: %r",
                         e, content[:300])
            return None


def _strip_code_fence(text: str) -> str:
    """Remove a leading/trailing markdown code fence (```json ... ```) if present."""
    if text.startswith("```"):
        # Drop leading fence + optional language tag.
        text = text[3:]
        if text.startswith("json"):
            text = text[4:]
        text = text.lstrip("\n")
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()
