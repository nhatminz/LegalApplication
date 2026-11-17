from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from uuid import uuid4

import httpx

from .config import Config
from .scoring import LABEL_SCHEMA
from .thinking import thinking_parameters

log = logging.getLogger(__name__)

class APIError(RuntimeError):
    def __init__(self, message: str, metadata: dict | None = None):
        super().__init__(message)
        self.metadata = metadata or {}


class IncompleteResponse(APIError):
    pass


@dataclass
class Completion:
    content: str
    metadata: dict


class APIClient:
    def __init__(self, config: Config, role: str, transport: httpx.AsyncBaseTransport | None = None):
        self.config, self.role = config, role
        self.model = getattr(config, f"{role}_model")
        self.base_url = getattr(config, f"{role}_base_url").rstrip("/")
        self.key = getattr(config, f"{role}_api_key")
        self.semaphore = asyncio.Semaphore(getattr(config, f"{role}_concurrency"))
        self.http = httpx.AsyncClient(timeout=config.timeout_seconds, transport=transport, trust_env=False)

    def body(self, messages: list[dict]) -> dict:
        result = {"model": self.model, "messages": messages, "stream": False,
                  "max_tokens": self.config.max_output_tokens}
        result.update(thinking_parameters(getattr(self.config, f"{self.role}_thinking_mode")))
        if self.role == "judge":
            result["response_format"] = (
                {"type": "json_schema", "json_schema": {"name": "legal_application_labels", "strict": True, "schema": LABEL_SCHEMA}}
                if self.config.judge_response_format == "json_schema" else {"type": "json_object"}
            )
        return result

    async def complete(self, messages: list[dict], *, doc_name: str) -> Completion:
        body = self.body(messages)
        request_id = uuid4().hex
        start = time.monotonic()
        attempts = []
        for attempt in range(self.config.api_max_retries + 1):
            status, retry_after = None, 0.0
            log.info("API start role=%s doc_name=%s request=%s attempt=%d", self.role, doc_name, request_id, attempt + 1)
            try:
                async with self.semaphore:
                    response = await self.http.post(self.base_url + "/chat/completions", json=body,
                                                    headers={"Authorization": "Bearer " + self.key,
                                                             "X-Request-ID": request_id})
                status = response.status_code
                if status in {408, 409, 429} or status >= 500:
                    try:
                        retry_after = max(0.0, min(float(response.headers.get("retry-after", 0)), 60.0))
                    except ValueError:
                        pass
                    raise httpx.HTTPStatusError("retryable HTTP status", request=response.request, response=response)
                if response.is_error:
                    suffix = " (check json_schema/reasoning-parser support; no automatic format fallback)" if self.role == "judge" and status == 400 else ""
                    raise APIError(f"{self.role} HTTP {status}{suffix}", {"request_id": request_id, "attempts": attempts})
                data = response.json()
                choice = data["choices"][0]
                content = choice["message"].get("content")
                metadata = {"request_id": request_id, "api_response_id": data.get("id"),
                            "response_model": data.get("model"), "finish_reason": choice.get("finish_reason"),
                            "usage": data.get("usage"), "http_attempts": attempt + 1,
                            "retry_errors": attempts, "latency_seconds": time.monotonic() - start,
                            "actual_thinking_parameters": {k: body[k] for k in ["chat_template_kwargs"] if k in body}}
                if choice.get("finish_reason") not in {"stop", None}:
                    raise IncompleteResponse(f"{self.role} incomplete response: finish_reason={choice.get('finish_reason')}", metadata)
                if not isinstance(content, str) or not content.strip():
                    raise IncompleteResponse(f"{self.role} returned empty final content", metadata)
                # Preserve message.content verbatim. Never read/forward message.reasoning_content.
                return Completion(content, metadata)
            except (httpx.TransportError, httpx.HTTPStatusError, ValueError, KeyError, IndexError, TypeError) as exc:
                category = f"HTTP {status}" if isinstance(exc, httpx.HTTPStatusError) else type(exc).__name__
                attempts.append({"attempt": attempt + 1, "error": category})
                log.warning("API retry role=%s doc_name=%s request=%s attempt=%d error=%s", self.role, doc_name, request_id, attempt + 1, category)
                if attempt == self.config.api_max_retries:
                    raise APIError(f"{self.role} exhausted API retries: {category}", {"request_id": request_id, "attempts": attempts}) from None
                delay = max(retry_after, min(60.0, self.config.retry_backoff * 2 ** attempt))
                if delay:
                    await asyncio.sleep(delay + random.uniform(0, min(delay / 4, 1)))
        raise AssertionError("unreachable")

    async def close(self) -> None:
        await self.http.aclose()

