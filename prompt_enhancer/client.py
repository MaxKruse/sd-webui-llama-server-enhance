"""Single chat-completion requests against llama-server.

Sends prompts via the OpenAI-compatible /v1/chat/completions endpoint,
extracts the ``content`` field only (a ``reasoning_content`` field, when
present, is never read - its thinking tokens still count in
usage.completion_tokens, so reported token numbers include them),
and returns a ChatResult with timing stats. The request path is sync
(urllib); the shared retry policy is also used by the aiohttp batch variant
in ``prompt_enhancer.batching``. Server lifecycle (spawn/health/kill) lives
in ``prompt_enhancer.server``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from threading import Lock
from urllib import error, request

import aiohttp

from prompt_enhancer.server import (
    _discover_model_name,
    _kill_server,
    _ServerProcess,
    _warmup_chat_endpoint,
    spawn_healthy_server,
)

logger = logging.getLogger(__name__)

# Seconds to wait for a single chat completion request before giving up on it.
DEFAULT_REQUEST_TIMEOUT = 120

# Max retries for chat completion requests (transient server errors).
_MAX_CHAT_RETRIES = 5

_print_lock = Lock()


class ChatResult:
    """Result from a single chat completion, including timing stats."""
    __slots__ = ("content", "completion_tokens", "prompt_ms", "predicted_ms", "error")

    def __init__(
        self,
        content: str | None = None,
        *,
        completion_tokens: int = 0,
        prompt_ms: float = 0.0,
        predicted_ms: float = 0.0,
        error: str | None = None,
    ):
        self.content = content
        self.completion_tokens = completion_tokens
        self.prompt_ms = prompt_ms
        self.predicted_ms = predicted_ms
        self.error = error

    @property
    def generation_tps(self) -> float:
        """Tokens per second for token generation."""
        return self.completion_tokens / (self.predicted_ms / 1000) if self.predicted_ms > 0 else 0.0

    @property
    def total_ms(self) -> float:
        """Total inference time in milliseconds."""
        return self.prompt_ms + self.predicted_ms


def _completion_payload(model_name: str, system_prompt: str, user_prompt: str) -> dict:
    """Build the /v1/chat/completions request body (shared by sync and async)."""
    return {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }


def _is_transient_error(exc: BaseException) -> bool:
    """Return True for errors that a retry is likely to resolve.

    HTTP 4xx (except 408/429) are deterministic client errors (context
    overflow, unknown model, bad request) - retrying them only burns time.
    """
    if isinstance(exc, error.HTTPError):
        return exc.code in (408, 429) or exc.code >= 500
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status in (408, 429) or exc.status >= 500
    return isinstance(exc, (error.URLError, aiohttp.ClientError, json.JSONDecodeError, ConnectionError, TimeoutError, OSError))


def _retry_backoff(attempt: int) -> float:
    """Backoff (seconds) after a failed 1-based attempt: 3, 6, 12, 24 (capped 30).

    A single request can take 16-30 s, so wide spacing is needed so later
    attempts land after the first batch frees its slots.
    """
    return min(3 * (2 ** (attempt - 1)), 30)


def _clean_http_error(status: int, body: str) -> str:
    """One-line human-readable description of an HTTP error response.

    llama-server returns OpenAI-style error bodies: ``{"error": {"message": ...}}``
    (newer builds) or ``{"error": "message"}`` (older). Dumping the raw JSON
    into the console on every retry attempt is noise, so the message is
    extracted; unparseable bodies fall back to a short raw truncation.
    """
    try:
        data = json.loads(body)
    except ValueError:
        data = None
    message = None
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            candidate = err.get("message")
            if isinstance(candidate, str):
                message = candidate
        elif isinstance(err, str):
            message = err
    if message and message.strip():
        return f"HTTP {status}: {message.strip()[:300]}"
    raw = body.strip()[:200]
    return f"HTTP {status}: {raw}" if raw else f"HTTP {status}: <empty error body>"


def _describe_error(exc: BaseException) -> str:
    """One-line description of a chat-completion failure.

    For HTTP errors the response body is parsed into the server's one-line
    message (see _clean_http_error) because the raw JSON body is noise.
    """
    if isinstance(exc, error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = "<unreadable>"
        return _clean_http_error(exc.code, body)
    return f"{type(exc).__name__}: {exc}"


def _send_chat_request(
    base_url: str,
    payload: dict,
    request_timeout: float,
) -> tuple[ChatResult | None, BaseException | None]:
    """Single urllib attempt against /v1/chat/completions (no retry).

    Returns (ChatResult, None) on success or (None, exception) on failure.
    Inference params (max_tokens, temperature, top_p) are omitted so the
    server uses its own defaults.
    """
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer no-key",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=request_timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]

            # Token counts come from usage; timings (prompt_ms/predicted_ms)
            # from llama-server's timings block.
            usage = result.get("usage", {})
            completion_tokens = usage.get("completion_tokens", usage.get("predicted_n", 0))
            timings = result.get("timings", {})
            return ChatResult(
                content=content.strip() if content else None,
                completion_tokens=completion_tokens,
                prompt_ms=timings.get("prompt_ms", 0),
                predicted_ms=timings.get("predicted_ms", 0),
            ), None
    except Exception as exc:
        return None, exc


def _chat_completion(
    base_url: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    *,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> ChatResult:
    """Send a single chat completion request with the shared retry policy.

    Retries up to _MAX_CHAT_RETRIES times, but only on transient errors
    (connection failures, timeouts, 408/429/5xx). Deterministic 4xx errors
    fail immediately without retrying.
    """
    payload = _completion_payload(model_name, system_prompt, user_prompt)

    last_error: str | None = None
    for attempt in range(1, _MAX_CHAT_RETRIES + 1):
        chat_result, exc = _send_chat_request(base_url, payload, request_timeout)
        if chat_result is not None:
            return chat_result

        last_error = _describe_error(exc)
        logger.error("Chat completion request failed (attempt %d/%d): %s", attempt, _MAX_CHAT_RETRIES, last_error)
        print(f"  Chat completion request failed (attempt {attempt}/{_MAX_CHAT_RETRIES}): {last_error}")

        if not _is_transient_error(exc):
            logger.error("Error is not transient - not retrying")
            break

        # Retry with exponential backoff (skip sleep on last attempt).
        if attempt < _MAX_CHAT_RETRIES:
            backoff = _retry_backoff(attempt)
            logger.info("Retrying in %.1fs...", backoff)
            print(f"  Retrying in {backoff:.0f}s...")
            time.sleep(backoff)

    # Retries exhausted (or a non-transient error)
    error_msg = last_error or "all retries failed"
    logger.error("Chat completion request failed: %s", error_msg)
    print(f"  Chat completion request failed: {error_msg}")
    return ChatResult(error=error_msg)


def chat_completion_request(
    sp: _ServerProcess,
    system_prompt: str,
    user_prompt: str,
    *,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    label: str = "Enhanced prompt",
) -> ChatResult:
    """Discover the model, warm up the chat endpoint, and send one chat
    completion against an already-running server.

    The caller owns the server lifecycle (spawn_server / _kill_server), so
    several prompts can share a single model load.
    """
    # Discover the loaded model name
    model_name = _discover_model_name(sp.base_url)
    logger.info("Discovered model: %s", model_name)

    # Warm up the chat endpoint - /health returns 200 before the chat handler
    # and parallel KV slots are fully initialized. Without this, the first real
    # request can get HTTP 400 (especially with --parallel N --no-kv-unified).
    _warmup_chat_endpoint(sp.base_url, model_name, num_slots=1)

    logger.info("llama-server user prompt: %s", user_prompt)
    result = _chat_completion(
        sp.base_url, model_name, system_prompt, user_prompt,
        request_timeout=request_timeout,
    )

    if result.content:
        logger.info("%s (%d chars): %s", label, len(result.content), result.content)
        print(
            f"  Result ({len(result.content)} chars, {result.completion_tokens} tokens, "
            f"{result.generation_tps:.1f} tok/s, {result.total_ms:.0f}ms total): "
            f"{result.content[:200]}{'...' if len(result.content) > 200 else ''}"
        )
    else:
        logger.warning("Empty or failed response from server")
        stderr_tail = sp.stderr_tail(2000)
        if stderr_tail.strip():
            logger.warning("Server stderr:\n%s", stderr_tail)

    return result


def enhance_prompt(
    server_path: str,
    model_path: str,
    system_prompt: str,
    user_prompt: str,
    extra_flags: str = "",
    *,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    startup_timeout: float | None = None,
) -> ChatResult:
    """Enhance a prompt by spawning a temporary llama-server instance.

    1. Start llama-server on a free port (output pipes drained continuously).
    2. Wait for /health to confirm the server is ready.
    3. Send the prompt via /v1/chat/completions.
    4. Kill the server (always, even on failure).
    5. Return a ChatResult with content and timing stats.
    """
    if not Path(model_path).is_file():
        logger.error("Model file not found: %s", model_path)
        return ChatResult()

    logger.info("llama-server user prompt: %s", user_prompt)
    sp = spawn_healthy_server(
        server_path, model_path, extra_flags, startup_timeout=startup_timeout,
    )
    if sp is None:
        return ChatResult()

    try:
        return chat_completion_request(
            sp, system_prompt, user_prompt, request_timeout=request_timeout,
        )
    finally:
        _kill_server(sp)
