"""Ollama provider implementations (raw httpx, no SDK)."""

from __future__ import annotations

import logging

import httpx

from core.config import config
from services.providers.base import (
    EmbeddingProvider,
    LLMProvider,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
_DEFAULT_LLM_MODEL = "llama3"


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def _classify_http_error(exc: httpx.HTTPStatusError) -> ProviderError:
    """Map an HTTP status error from the Ollama REST API to a provider exception."""
    code = exc.response.status_code
    if code == 429:
        return ProviderRateLimitError(str(exc))
    if code in (401, 403):
        return ProviderAuthError(str(exc))
    return ProviderError(str(exc))


def _classify_network_error(
    exc: httpx.TimeoutException | httpx.ConnectError | httpx.NetworkError,
) -> ProviderTimeoutError | ProviderConnectionError:
    """Map network-level httpx exceptions to provider exceptions."""
    if isinstance(exc, httpx.TimeoutException):
        return ProviderTimeoutError(str(exc))
    return ProviderConnectionError(str(exc))


# ---------------------------------------------------------------------------
# Embedding provider
# ---------------------------------------------------------------------------


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by the Ollama REST API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self._model = model or config.EMBEDDING_MODEL or _DEFAULT_EMBEDDING_MODEL
        self._base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """POST to ``/api/embed`` for batch embedding."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self._base_url}/api/embed",
                    json={"model": self._model, "input": texts},
                    timeout=60.0,
                )
                response.raise_for_status()
                return response.json()["embeddings"]

        except httpx.HTTPStatusError as exc:
            raise _classify_http_error(exc) from exc
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as exc:
            raise _classify_network_error(exc) from exc


# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------


class OllamaLLMProvider(LLMProvider):
    """LLM provider backed by the Ollama REST API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self._model = model or config.LLM_MODEL or _DEFAULT_LLM_MODEL
        self._base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")

    async def generate(
        self,
        prompt: str,
        *,
        system_instruction: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        """POST to ``/api/generate`` for non-streaming text generation."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": self._model,
                        "prompt": prompt,
                        "system": system_instruction,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_output_tokens,
                        },
                    },
                    timeout=120.0,
                )
                response.raise_for_status()
                return response.json().get("response", "No response.")

        except httpx.HTTPStatusError as exc:
            raise _classify_http_error(exc) from exc
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as exc:
            raise _classify_network_error(exc) from exc
