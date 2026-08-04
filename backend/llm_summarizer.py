#!/usr/bin/env python3
"""
Optional local-LLM summarizer backed by Ollama.

Talks to a locally-running Ollama server (https://ollama.com) over its HTTP
API using ONLY the Python standard library (urllib) - no new pip dependency,
and nothing ever leaves the machine: Ollama itself is local-only, same as
every other model this project uses.

Usage:
    from llm_summarizer import summarize_with_llm, OllamaError

    try:
        result = summarize_with_llm(segments, style="meeting_minutes")
        # result = {"engine": "ollama-llama3.2", "style": "meeting_minutes",
        #           "model": "llama3.2", "text": "...", "generated_at": "..."}
    except OllamaError:
        # fall back to the extractive summarizer

Configuration (environment variables):
    OREJA_OLLAMA_URL             base URL, default http://127.0.0.1:11434
    OREJA_OLLAMA_MODEL           model name, default llama3.2
    OREJA_OLLAMA_CONNECT_TIMEOUT seconds, default 2 - used only for the quick
                                 reachability probe, so a stopped/absent
                                 Ollama fails fast instead of blocking a
                                 request for a long time.
    OREJA_OLLAMA_GENERATE_TIMEOUT seconds, default 60 - the actual generation
                                 call gets a much longer budget since a real
                                 model response legitimately takes a while.

Design note: summarize_with_llm() never raises anything but OllamaError.
Every failure mode (Ollama not running, DNS/connection failure, HTTP error,
malformed JSON, empty transcript, timeout) is normalized into OllamaError so
callers can do a single `except OllamaError` to decide to fall back.
"""

import json
import logging
import os
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = os.getenv("OREJA_OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OREJA_OLLAMA_MODEL", "llama3.2")
# Short timeout used only to detect "Ollama isn't running" quickly, so a
# summarization request can fall back to the extractive engine without a
# long hang.
CONNECT_TIMEOUT_SECONDS = float(os.getenv("OREJA_OLLAMA_CONNECT_TIMEOUT", "2"))
# Once we know Ollama is reachable, actually generating a summary can
# legitimately take much longer than 2s.
GENERATE_TIMEOUT_SECONDS = float(os.getenv("OREJA_OLLAMA_GENERATE_TIMEOUT", "60"))


class OllamaError(Exception):
    """Raised for any failure talking to Ollama - unreachable, timed out,
    returned an error, or returned something we couldn't parse. Callers
    should catch this (or plain Exception) and fall back to the extractive
    summarizer; the transcript itself is always more important than the
    fancier summary."""


STYLE_DESCRIPTIONS: Dict[str, str] = {
    "meeting_minutes": "Structured meeting minutes: summary, key discussion points, decisions, and action items.",
    "action_items": "A focused checklist of action items / follow-ups, each with an owner when the transcript makes one clear.",
    "decisions": "Just the decisions and conclusions reached, with brief context for each.",
    "summary": "A short, plain-language paragraph summarizing what the conversation was about.",
}

_STYLE_INSTRUCTIONS: Dict[str, str] = {
    "meeting_minutes": (
        "Write concise, well-structured meeting minutes for the transcript below. "
        "Use these sections, omitting any that have no content:\n"
        "Summary: a 2-4 sentence overview of what was discussed.\n"
        "Key Points: a bullet list of the main topics and points raised.\n"
        "Decisions: a bullet list of decisions or conclusions reached.\n"
        "Action Items: a bullet list of tasks or follow-ups, naming the responsible speaker when the transcript makes it clear."
    ),
    "action_items": (
        "Read the transcript below and extract ONLY the action items - concrete tasks, "
        "commitments, or follow-ups anyone agreed to do. Output a bullet list, one action "
        "item per line, naming the responsible speaker when the transcript makes it clear. "
        "If there are no action items, say so in one sentence."
    ),
    "decisions": (
        "Read the transcript below and extract ONLY the decisions or conclusions that were "
        "reached. Output a bullet list, one decision per line, with a short clause of context "
        "for each. If no decisions were reached, say so in one sentence."
    ),
    "summary": (
        "Write a short, plain-language summary (3-5 sentences) of what this conversation was about."
    ),
}


def _format_transcript(segments: List[Dict[str, Any]]) -> str:
    """Render speaker-labeled segments as `[MM:SS] Speaker: text` lines."""
    lines = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = seg.get("speaker") or "Unknown"
        start = seg.get("start") or 0
        try:
            total_seconds = max(0, int(float(start)))
        except (TypeError, ValueError):
            total_seconds = 0
        minutes, seconds = divmod(total_seconds, 60)
        lines.append(f"[{minutes:02d}:{seconds:02d}] {speaker}: {text}")
    return "\n".join(lines)


def _build_prompt(transcript_text: str, style: str) -> str:
    instructions = _STYLE_INSTRUCTIONS.get(style, _STYLE_INSTRUCTIONS["meeting_minutes"])
    return (
        f"{instructions}\n\n"
        "Only use information present in the transcript - do not invent names, "
        "numbers, or facts that were not said. Respond with plain text only, "
        "no preamble like \"Here is the summary\".\n\n"
        "Transcript:\n"
        f"{transcript_text}"
    )


def _http_get_json(url: str, timeout: float) -> Dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raise OllamaError(f"Ollama returned HTTP {e.code} for GET {url}") from e
    except urllib.error.URLError as e:
        raise OllamaError(f"Could not reach Ollama at {url}: {e.reason}") from e
    except OSError as e:
        raise OllamaError(f"Could not reach Ollama at {url}: {e}") from e

    try:
        return json.loads(body) if body else {}
    except json.JSONDecodeError as e:
        raise OllamaError(f"Ollama returned invalid JSON from {url}: {e}") from e


def _http_post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        raise OllamaError(f"Ollama returned HTTP {e.code} for POST {url}: {detail[:200]}") from e
    except urllib.error.URLError as e:
        raise OllamaError(f"Could not reach Ollama at {url}: {e.reason}") from e
    except OSError as e:
        raise OllamaError(f"Could not reach Ollama at {url}: {e}") from e

    try:
        return json.loads(body) if body else {}
    except json.JSONDecodeError as e:
        raise OllamaError(f"Ollama returned invalid JSON from {url}: {e}") from e


def get_status(url: Optional[str] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
    """
    Non-raising reachability probe, for status/discovery endpoints.
    Returns {"available": bool, "url": str, "models": [...]} on success, or
    {"available": False, "url": str, "error": str} on failure.
    """
    base_url = (url or DEFAULT_OLLAMA_URL).rstrip("/")
    try:
        data = _http_get_json(f"{base_url}/api/tags", timeout or CONNECT_TIMEOUT_SECONDS)
        models = [m.get("name") for m in data.get("models", []) if isinstance(m, dict) and m.get("name")]
        return {
            "available": True,
            "url": base_url,
            "default_model": DEFAULT_OLLAMA_MODEL,
            "models": models,
        }
    except Exception as e:
        return {"available": False, "url": base_url, "default_model": DEFAULT_OLLAMA_MODEL, "error": str(e)}


def is_available(url: Optional[str] = None, timeout: Optional[float] = None) -> bool:
    """Quick boolean reachability check."""
    return get_status(url, timeout).get("available", False)


def summarize_with_llm(
    segments: List[Dict[str, Any]],
    style: str = "meeting_minutes",
    model: Optional[str] = None,
    url: Optional[str] = None,
    connect_timeout: Optional[float] = None,
    request_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Summarize a speaker-labeled transcript with a local Ollama model.

    Args:
        segments: list of {"start", "end", "text", "speaker", ...} dicts, the
            same shape POST /transcribe returns segments in.
        style: one of "meeting_minutes", "action_items", "decisions", "summary".
        model: Ollama model name, defaults to OREJA_OLLAMA_MODEL / "llama3.2".
        url: Ollama base URL, defaults to OREJA_OLLAMA_URL / http://127.0.0.1:11434.
        connect_timeout: seconds for the fast reachability probe (default 2s).
        request_timeout: seconds for the actual generation call (default 60s).

    Returns:
        {"engine": "ollama-<model>", "style": style, "model": model,
         "text": <generated text>, "generated_at": <ISO timestamp>}

    Raises:
        OllamaError on ANY failure - unreachable server, HTTP error, timeout,
        malformed response, or an empty/unusable transcript. Callers should
        catch this and fall back to extractive summarization.
    """
    if not segments:
        raise OllamaError("No segments to summarize")

    model = model or DEFAULT_OLLAMA_MODEL
    base_url = (url or DEFAULT_OLLAMA_URL).rstrip("/")
    connect_timeout = connect_timeout if connect_timeout is not None else CONNECT_TIMEOUT_SECONDS
    request_timeout = request_timeout if request_timeout is not None else GENERATE_TIMEOUT_SECONDS

    transcript_text = _format_transcript(segments)
    if not transcript_text.strip():
        raise OllamaError("Transcript has no text content to summarize")

    # Fast reachability probe first (short timeout) so a stopped/absent
    # Ollama fails fast rather than hanging the request for up to
    # request_timeout seconds before falling back.
    _http_get_json(f"{base_url}/api/tags", connect_timeout)

    prompt = _build_prompt(transcript_text, style)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }

    data = _http_post_json(f"{base_url}/api/generate", payload, request_timeout)

    text = data.get("response")
    if not isinstance(text, str) or not text.strip():
        raise OllamaError(f"Ollama returned no usable text (model={model!r})")

    return {
        "engine": f"ollama-{model}",
        "style": style,
        "model": model,
        "text": text.strip(),
        "generated_at": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    # Tiny manual smoke test: `python3 llm_summarizer.py`
    demo_segments = [
        {"start": 0.0, "end": 3.0, "text": "Good morning everyone, let's start the planning meeting.", "speaker": "Alice"},
        {"start": 3.5, "end": 7.0, "text": "I think we should prioritize the mobile performance fixes.", "speaker": "Bob"},
        {"start": 7.5, "end": 10.0, "text": "Agreed. I'll own that and report back Friday.", "speaker": "Bob"},
    ]
    print(f"Checking Ollama at {DEFAULT_OLLAMA_URL} ...")
    status = get_status()
    print(status)
    if status.get("available"):
        try:
            result = summarize_with_llm(demo_segments, style="meeting_minutes")
            print(result)
        except OllamaError as e:
            print(f"summarize_with_llm failed: {e}")
    else:
        print("Ollama not reachable - skipping generation demo.")
