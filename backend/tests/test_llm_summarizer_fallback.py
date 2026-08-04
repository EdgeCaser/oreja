"""
Dependency-light unit tests for llm_summarizer.py's Ollama client, with
urllib.request.urlopen mocked so no network/Ollama process is needed.

llm_summarizer.py is pure stdlib (json, urllib) - no stubbing required.
"""

import json
import urllib.error
from io import BytesIO
from unittest.mock import patch

import pytest

from llm_summarizer import (
    summarize_with_llm,
    get_status,
    is_available,
    OllamaError,
)


SEGMENTS = [
    {"start": 0.0, "end": 2.0, "text": "Let's start the planning meeting.", "speaker": "Alice"},
    {"start": 2.5, "end": 5.0, "text": "I'll own the mobile fixes.", "speaker": "Bob"},
]


class FakeResponse:
    """Stand-in for the context-manager urllib.request.urlopen() returns."""

    def __init__(self, payload: dict, status: int = 200):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# Failure paths: Ollama not running / unreachable / broken
# ---------------------------------------------------------------------------

class TestSummarizeWithLlmFallback:
    def test_connection_refused_raises_ollama_error(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError(ConnectionRefusedError("refused")),
        ):
            with pytest.raises(OllamaError):
                summarize_with_llm(SEGMENTS)

    def test_timeout_raises_ollama_error(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            with pytest.raises(OllamaError):
                summarize_with_llm(SEGMENTS)

    def test_http_error_raises_ollama_error(self):
        http_err = urllib.error.HTTPError(
            url="http://127.0.0.1:11434/api/tags", code=500, msg="Internal Error",
            hdrs=None, fp=BytesIO(b""),
        )
        with patch("urllib.request.urlopen", side_effect=http_err):
            with pytest.raises(OllamaError):
                summarize_with_llm(SEGMENTS)

    def test_malformed_json_raises_ollama_error(self):
        class BadJsonResponse(FakeResponse):
            def read(self):
                return b"{not valid json"

        with patch("urllib.request.urlopen", return_value=BadJsonResponse({})):
            with pytest.raises(OllamaError):
                summarize_with_llm(SEGMENTS)

    def test_no_segments_raises_without_any_network_call(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            with pytest.raises(OllamaError, match="No segments"):
                summarize_with_llm([])
            mock_urlopen.assert_not_called()

    def test_segments_with_no_text_raise_without_reachability_probe(self):
        blank_segments = [{"start": 0.0, "end": 1.0, "text": "   ", "speaker": "A"}]
        with patch("urllib.request.urlopen") as mock_urlopen:
            with pytest.raises(OllamaError, match="no text content"):
                summarize_with_llm(blank_segments)
            mock_urlopen.assert_not_called()

    def test_generate_returns_no_text_raises(self):
        tags_response = FakeResponse({"models": [{"name": "llama3.2"}]})
        generate_response = FakeResponse({"response": ""})  # empty text

        with patch("urllib.request.urlopen", side_effect=[tags_response, generate_response]):
            with pytest.raises(OllamaError, match="no usable text"):
                summarize_with_llm(SEGMENTS)

    def test_generate_response_missing_field_raises(self):
        tags_response = FakeResponse({"models": []})
        generate_response = FakeResponse({"unexpected_key": "value"})

        with patch("urllib.request.urlopen", side_effect=[tags_response, generate_response]):
            with pytest.raises(OllamaError):
                summarize_with_llm(SEGMENTS)


# ---------------------------------------------------------------------------
# Success path (still fully mocked - verifies request/response wiring)
# ---------------------------------------------------------------------------

class TestSummarizeWithLlmSuccess:
    def test_successful_generation_returns_expected_shape(self):
        tags_response = FakeResponse({"models": [{"name": "llama3.2"}]})
        generate_response = FakeResponse({"response": "Meeting summary text."})

        with patch("urllib.request.urlopen", side_effect=[tags_response, generate_response]):
            result = summarize_with_llm(SEGMENTS, style="meeting_minutes")

        assert result["engine"] == "ollama-llama3.2"
        assert result["style"] == "meeting_minutes"
        assert result["model"] == "llama3.2"
        assert result["text"] == "Meeting summary text."
        assert "generated_at" in result

    def test_custom_model_name_is_used_in_engine(self):
        tags_response = FakeResponse({"models": []})
        generate_response = FakeResponse({"response": "Custom model output."})

        with patch("urllib.request.urlopen", side_effect=[tags_response, generate_response]):
            result = summarize_with_llm(SEGMENTS, model="mistral")

        assert result["engine"] == "ollama-mistral"
        assert result["model"] == "mistral"


# ---------------------------------------------------------------------------
# get_status / is_available - never raise, always return a dict/bool
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_status_when_ollama_unreachable(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no route")):
            status = get_status()

        assert status["available"] is False
        assert "error" in status

    def test_status_when_ollama_reachable(self):
        response = FakeResponse({"models": [{"name": "llama3.2"}, {"name": "mistral"}]})
        with patch("urllib.request.urlopen", return_value=response):
            status = get_status()

        assert status["available"] is True
        assert set(status["models"]) == {"llama3.2", "mistral"}

    def test_is_available_false_on_failure(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
            assert is_available() is False

    def test_is_available_true_on_success(self):
        response = FakeResponse({"models": []})
        with patch("urllib.request.urlopen", return_value=response):
            assert is_available() is True

    def test_get_status_never_raises_on_garbage_response(self):
        class BadResponse(FakeResponse):
            def read(self):
                return b"not json at all"

        with patch("urllib.request.urlopen", return_value=BadResponse({})):
            status = get_status()  # must not raise

        assert status["available"] is False
