"""Unit tests for provider implementations — mock the SDK/HTTP calls."""

import pytest

from services.providers.base import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)


# ---------------------------------------------------------------------------
# Gemini error classification
# ---------------------------------------------------------------------------


class TestGeminiErrorClassification:
    """Verify _classify_gemini_error maps status codes correctly."""

    def _classify(self, code, status="", message=""):
        """Build a fake APIError and classify it."""
        from services.providers.gemini import _classify_gemini_error

        class _FakeAPIError(Exception):
            pass

        exc = _FakeAPIError(message)
        exc.code = code
        exc.status = status
        return _classify_gemini_error(exc)

    def test_429_is_rate_limit(self):
        result = self._classify(429)
        assert isinstance(result, ProviderRateLimitError)

    def test_resource_exhausted_status_is_rate_limit(self):
        result = self._classify(503, status="RESOURCE_EXHAUSTED")
        assert isinstance(result, ProviderRateLimitError)

    def test_quota_in_message_is_rate_limit(self):
        result = self._classify(400, message="Quota exceeded for this project")
        assert isinstance(result, ProviderRateLimitError)

    def test_401_is_auth(self):
        result = self._classify(401)
        assert isinstance(result, ProviderAuthError)

    def test_403_is_auth(self):
        result = self._classify(403)
        assert isinstance(result, ProviderAuthError)

    def test_400_with_api_key_message_is_auth(self):
        result = self._classify(400, message="Invalid API key provided")
        assert isinstance(result, ProviderAuthError)

    def test_400_generic_is_provider_error(self):
        result = self._classify(400, message="Malformed request")
        assert isinstance(result, ProviderError)
        assert not isinstance(result, ProviderAuthError)

    def test_500_is_provider_error(self):
        result = self._classify(500, message="Internal server error")
        assert isinstance(result, ProviderError)


# ---------------------------------------------------------------------------
# OpenAI error classification
# ---------------------------------------------------------------------------


try:
    import openai as _openai  # noqa: F401
    _has_openai = True
except ImportError:
    _has_openai = False


@pytest.mark.skipif(not _has_openai, reason="openai package not installed")
class TestOpenAIErrorClassification:
    """Verify _classify_openai_error maps OpenAI SDK exceptions correctly."""

    @staticmethod
    def _fake_response(status_code: int):
        """Build a minimal httpx.Response that OpenAI exceptions accept."""
        import httpx

        return httpx.Response(status_code, request=httpx.Request("POST", "https://api.openai.com"))

    def test_rate_limit(self):
        import openai
        from services.providers.openai import _classify_openai_error

        exc = openai.RateLimitError(
            message="Rate limit exceeded",
            response=self._fake_response(429),
            body=None,
        )
        result = _classify_openai_error(exc)
        assert isinstance(result, ProviderRateLimitError)

    def test_auth_error(self):
        import openai
        from services.providers.openai import _classify_openai_error

        exc = openai.AuthenticationError(
            message="Invalid API key",
            response=self._fake_response(401),
            body=None,
        )
        result = _classify_openai_error(exc)
        assert isinstance(result, ProviderAuthError)

    def test_timeout(self):
        import openai
        from services.providers.openai import _classify_openai_error

        exc = openai.APITimeoutError(request=None)
        result = _classify_openai_error(exc)
        assert isinstance(result, ProviderTimeoutError)

    def test_connection_error(self):
        import openai
        from services.providers.openai import _classify_openai_error

        exc = openai.APIConnectionError(request=None)
        result = _classify_openai_error(exc)
        assert isinstance(result, ProviderConnectionError)


# ---------------------------------------------------------------------------
# Ollama error classification
# ---------------------------------------------------------------------------


class TestOllamaErrorClassification:
    """Verify Ollama error mappers classify httpx errors correctly."""

    def test_http_429_is_rate_limit(self):
        import httpx
        from services.providers.ollama import _classify_http_error

        response = httpx.Response(429, request=httpx.Request("POST", "http://test"))
        exc = httpx.HTTPStatusError("rate limited", request=response.request, response=response)
        result = _classify_http_error(exc)
        assert isinstance(result, ProviderRateLimitError)

    def test_http_401_is_auth(self):
        import httpx
        from services.providers.ollama import _classify_http_error

        response = httpx.Response(401, request=httpx.Request("POST", "http://test"))
        exc = httpx.HTTPStatusError("unauthorized", request=response.request, response=response)
        result = _classify_http_error(exc)
        assert isinstance(result, ProviderAuthError)

    def test_http_500_is_provider_error(self):
        import httpx
        from services.providers.ollama import _classify_http_error

        response = httpx.Response(500, request=httpx.Request("POST", "http://test"))
        exc = httpx.HTTPStatusError("server error", request=response.request, response=response)
        result = _classify_http_error(exc)
        assert isinstance(result, ProviderError)

    def test_timeout_is_provider_timeout(self):
        import httpx
        from services.providers.ollama import _classify_network_error

        exc = httpx.ReadTimeout("timed out")
        result = _classify_network_error(exc)
        assert isinstance(result, ProviderTimeoutError)

    def test_connect_error_is_connection(self):
        import httpx
        from services.providers.ollama import _classify_network_error

        exc = httpx.ConnectError("refused")
        result = _classify_network_error(exc)
        assert isinstance(result, ProviderConnectionError)
