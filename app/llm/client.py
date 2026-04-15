"""Thin OpenAI-compatible LLM/VLM client using httpx.

Supports text-only and vision (image) requests. Never raises into callers — returns None on any failure.
"""

from __future__ import annotations

import base64
import json
import logging
from io import BytesIO
from typing import Any

import httpx
from PIL import Image

logger = logging.getLogger("loci.llm")


class LlmClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 8.0,
        api_key: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, body: dict[str, Any]) -> dict[str, Any] | None:
        """Send a request to the chat completions endpoint. Returns parsed JSON or None."""
        try:
            url = f"{self.base_url}/v1/chat/completions"
            response = httpx.post(
                url, json=body, headers=self._headers(), timeout=self.timeout,
            )
            if response.status_code != 200:
                logger.warning("LLM request failed: status=%d body=%s", response.status_code, response.text[:200])
                return None
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            logger.debug("LLM raw response (first 500): %s", content[:500])
            # Strip markdown code fences that some models wrap JSON in
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)
            # Try to find JSON in the response if direct parse fails
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON object from mixed text
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    logger.debug("Extracting JSON from position %d-%d", start, end)
                    return json.loads(content[start:end])
                logger.warning("Could not parse JSON from LLM response: %s", content[:300])
                return None
        except httpx.TimeoutException:
            logger.warning("LLM request timed out after %.1fs", self.timeout)
            return None
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.warning("LLM request error: %s", exc)
            return None
        except Exception as exc:
            logger.warning("LLM unexpected error: %s", exc)
            return None

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Text-only chat completion. Returns parsed JSON or None."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
        }
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema},
            }
        return self._post(body)

    def vision_extract_json(
        self,
        system: str,
        user_text: str,
        image: Image.Image,
        *,
        schema: dict[str, Any] | None = None,
        detail: str = "high",
    ) -> dict[str, Any] | None:
        """Vision chat completion — sends an image + text prompt, returns parsed JSON or None."""
        try:
            buf = BytesIO()
            image.convert("RGB").save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as exc:
            logger.warning("Failed to encode image for VLM: %s", exc)
            return None

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                                "detail": detail,
                            },
                        },
                    ],
                },
            ],
            "temperature": 0.0,
        }
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema},
            }
        return self._post(body)
