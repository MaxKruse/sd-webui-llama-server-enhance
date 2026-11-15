"""llama-server HTTP client for prompt enhancement.

Spawns a temporary llama-server instance, sends prompts via the OpenAI-compatible
/v1/chat/completions endpoint, extracts the `content` field (reasoning_content is
stripped server-side), then kills the server.

Supports both single-prompt and batch-prompt modes. In batch mode requests run
in strict buckets of ``max_concurrent``: a whole bucket must finish before the
next one starts, so the server's parallel slots stay busy without unbounded
overlap.
"""

from __future__ import annotations

import json
import logging
import random
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from threading import Lock
from urllib import error, request

import aiohttp
import asyncio

logger = logging.getLogger(__name__)

# How long to wait for the server to become healthy (model load time).
_SERVER_STARTUP_TIMEOUT = 60

# Timeout for individual health-check polls.
_HTTP_TIMEOUT = 5

# Seconds to wait for a single chat completion request before giving up on it.
DEFAULT_REQUEST_TIMEOUT = 120

# Max retries for chat completion requests (transient server errors).
_MAX_CHAT_RETRIES = 5

# Port range to probe for a free port.
_PORT_RANGE = range(49152, 65536)


def _find_free_port() -> int:
    """Find a free TCP port by probing random ports in the ephemeral range.

    Binds a socket to confirm the port is available, then returns it.
    There's a tiny race window between releasing the socket and
    llama-server binding, but it's negligible in practice.
    """
    candidates = random.sample(_PORT_RANGE, 100)
    for port in candidates:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    # Fallback: let the OS pick
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _user_set_ctx(flags_list: list[str]) -> bool:
    """Check if the user already specified a context size in their flags."""
    for token in flags_list:
        if token in ("-c", "--ctx-size"):
            return True
        if token.startswith("--ctx-size="):
            return True
    return False


def _build_command(
    server_path: str,
    model_path: str,
    port: int,
    extra_flags: str,
) -> list[str]:
    """Build the full llama-server command list."""
    import shlex

    user_flags = shlex.split(extra_flags) if extra_flags else []

    cmd = [
        server_path,
        "-m", model_path,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--no-ui",       # disable web UI for speed
        "--no-warmup",   # skip empty warmup run
    ]

    # Only set a default context size if the user didn't specify one
    if not _user_set_ctx(user_flags):
        cmd.extend(["-c", "32000"])

    cmd.extend(user_flags)

    return cmd


def _wait_for_server(
    base_url: str,
    timeout: int = _SERVER_STARTUP_TIMEOUT,
    proc: subprocess.Popen | None = None,
) -> bool:
    """Poll /health until the server responds, the timeout is reached, or the
    process exits early (a crashed server can never become healthy)."""
    deadline = time.monotonic() + timeout
    url = f"{base_url}/health"

    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            logger.error("Server process exited early (code=%s) - aborting health wait", proc.returncode)
            return False
        try:
            with request.urlopen(url, timeout=_HTTP_TIMEOUT) as resp:
                if resp.status == 200:
                    return True
        except (error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.5)

    return False


_DRAIN_MAX_BYTES = 256 * 1024


class _PipeDrainer(threading.Thread):
    """Continuously reads a subprocess pipe into a bounded buffer.

    llama-server logs its startup and every task to stderr. Windows anonymous
    pipes only buffer 4 KB, so without a reader the server's own threads block
    on write once the buffer fills: /health keeps answering (it never logs) but
    chat requests hang. Draining keeps the server unblocked and retains the
    last _DRAIN_MAX_BYTES of output for diagnostics.
    """

    def __init__(self, name: str, stream, on_chunk=None):
        super().__init__(name=name, daemon=True)
        self._stream = stream
        self._chunks: list[bytes] = []
        self._size = 0
        self._lock = Lock()
        # Optional callback invoked with each decoded chunk (e.g. to stream
        # output into a log file). Must be thread-safe and quick.
        self._on_chunk = on_chunk

    def run(self) -> None:
        while True:
            chunk = self._stream.read1(65536)
            if not chunk:
                break
            with self._lock:
                self._chunks.append(chunk)
                self._size += len(chunk)
                while self._size > _DRAIN_MAX_BYTES and len(self._chunks) > 1:
                    self._size -= len(self._chunks.pop(0))
            if self._on_chunk is not None:
                try:
                    self._on_chunk(chunk.decode("utf-8", errors="replace"))
                except Exception:
                    logger.exception("on_chunk callback failed (continuing to drain)")

    def tail(self, max_chars: int = 3000) -> str:
        """Return the tail of the captured output as text."""
        with self._lock:
            data = b"".join(self._chunks)
        return data.decode("utf-8", errors="replace")[-max_chars:]


class _ServerProcess:
    """A spawned llama-server with drained output pipes and lifecycle helpers."""

    def __init__(self, proc: subprocess.Popen, port: int, on_stderr=None):
        self.proc = proc
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self._stdout_drainer = _PipeDrainer("llama-server-stdout", proc.stdout)
        self._stderr_drainer = _PipeDrainer("llama-server-stderr", proc.stderr, on_chunk=on_stderr)

    @property
    def pid(self) -> int:
        return self.proc.pid

    def start_drainers(self) -> None:
        self._stdout_drainer.start()
        self._stderr_drainer.start()

    def stop_drainers(self) -> None:
        # The process is gone at this point, so pipes hit EOF and joins return fast.
        self._stdout_drainer.join(timeout=2)
        self._stderr_drainer.join(timeout=2)

    def stdout_tail(self, max_chars: int = 3000) -> str:
        return self._stdout_drainer.tail(max_chars)

    def stderr_tail(self, max_chars: int = 3000) -> str:
        return self._stderr_drainer.tail(max_chars)


def spawn_server(
    server_path: str,
    model_path: str,
    port: int,
    extra_flags: str,
    on_stderr=None,
) -> _ServerProcess:
    """Start a llama-server subprocess with continuously drained output pipes.

    Raises FileNotFoundError if the binary is not found, or an OSError
    subclass if the process cannot be created.
    """
    cmd = _build_command(server_path, model_path, port, extra_flags)
    logger.info("llama-server command: %s", " ".join(cmd))

    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
    )
    logger.info("llama-server started (pid=%s, port=%s)", proc.pid, port)
    print(f"  Started llama-server (pid={proc.pid}, port={port})")

    sp = _ServerProcess(proc, port, on_stderr=on_stderr)
    sp.start_drainers()
    return sp


def spawn_healthy_server(
    server_path: str,
    model_path: str,
    extra_flags: str,
    *,
    startup_timeout: float | None = None,
    on_stderr=None,
    on_debug=None,
    label: str = "",
) -> _ServerProcess | None:
    """Start a llama-server and wait until it is healthy.

    Single spawn-wait-kill path shared by the WebUI integration,
    ``enhance_prompt`` and the test scripts.

    Progress is printed to stdout. ``on_stderr`` (optional callable)
    receives decoded stderr chunks as they arrive; ``on_debug`` (optional
    callable) receives the same diagnostic lines for a debug log sink, and
    when absent the failure stderr tail goes to ``logger`` instead.
    ``label`` is appended to the "Running llama-server" line (e.g. " (test)").

    Returns the server handle, or None on failure (the server is already
    killed in that case).
    """
    timeout = float(_SERVER_STARTUP_TIMEOUT if startup_timeout is None else startup_timeout)
    port = _find_free_port()
    model_name = Path(model_path).name

    def _emit(line: str):
        if on_debug is not None:
            on_debug(line)

    print(f"\n  Running llama-server (model: {model_name}, port: {port}){label}...")
    _emit(f"  Running llama-server (model: {model_name}, port: {port}){label}...")

    try:
        sp = spawn_server(server_path, model_path, port, extra_flags, on_stderr=on_stderr)
    except FileNotFoundError:
        logger.error("llama-server not found at: %s", server_path)
        print(f"  FAILED: llama-server not found at: {server_path}")
        _emit(f"  → llama-server not found at {server_path}")
        return None
    except Exception:
        logger.exception("Failed to start llama-server")
        print("  FAILED: could not start llama-server")
        _emit("  → failed to start llama-server")
        return None

    print(f"  Waiting for server to start (timeout: {timeout:.0f}s)...")
    _emit(f"  Waiting for server to start (timeout: {timeout:.0f}s)...")
    start = time.monotonic()

    if not _wait_for_server(sp.base_url, timeout=timeout, proc=sp.proc):
        elapsed = time.monotonic() - start
        print(f"  FAILED: Server did not become healthy within {timeout:.0f}s ({elapsed:.1f}s elapsed)")
        logger.error("Server did not become healthy within %.0fs", timeout)
        _emit(f"  → server health check failed after {elapsed:.1f}s")
        stderr_tail = sp.stderr_tail(3000)
        if stderr_tail.strip():
            _emit(f"  llama-server stderr (last 3000 chars):\n{stderr_tail}")
            print(f"  Server stderr (last 500 chars):\n{stderr_tail[-500:]}")
            if on_debug is None:
                logger.warning("llama-server stderr (last 2000 chars):\n%s", stderr_tail[-2000:])
        _kill_server(sp)
        return None

    elapsed = time.monotonic() - start
    print(f"  Server ready on port {port} in {elapsed:.1f}s")
    _emit(f"  Server ready on port {port} in {elapsed:.1f}s")
    return sp


_print_lock = Lock()


class ChatResult:
    """Result from a single chat completion, including timing stats."""
    __slots__ = ("content", "prompt_tokens", "completion_tokens", "prompt_ms", "predicted_ms", "error")

    def __init__(
        self,
        content: str | None = None,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        prompt_ms: float = 0.0,
        predicted_ms: float = 0.0,
        error: str | None = None,
    ):
        self.content = content
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_ms = prompt_ms
        self.predicted_ms = predicted_ms
        self.error = error

    @property
    def prompt_tps(self) -> float:
        """Tokens per second for prompt processing."""
        return self.prompt_tokens / (self.prompt_ms / 1000) if self.prompt_ms > 0 else 0.0

    @property
    def generation_tps(self) -> float:
        """Tokens per second for token generation."""
        return self.completion_tokens / (self.predicted_ms / 1000) if self.predicted_ms > 0 else 0.0

    @property
    def total_ms(self) -> float:
        """Total inference time in milliseconds."""
        return self.prompt_ms + self.predicted_ms

    def __bool__(self):
        return bool(self.content)


def _discover_model_name(base_url: str) -> str:
    """Fetch the currently loaded model name from /v1/models.

    Prefers the model whose status is "loaded". Falls back to the first
    model in the list, then to "llama" if nothing is available.
    Strips trailing .gguf extension and full filesystem paths — some
    llama-server builds reject model names that include the extension
    or the full path in chat completions.
    """
    def _clean_name(raw: str) -> str:
        """Strip .gguf extension and filesystem path, returning just the model filename."""
        p = Path(raw)
        name = p.stem  # strips .gguf if present
        return name or "llama"

    try:
        with request.urlopen(f"{base_url}/v1/models", timeout=_HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            # Prefer the loaded model
            for m in models:
                status = m.get("status", {})
                if isinstance(status, dict) and status.get("value") == "loaded":
                    return _clean_name(m.get("id", "llama"))
            # Fallback: first model
            if models:
                return _clean_name(models[0].get("id", "llama"))
    except (error.URLError, json.JSONDecodeError, KeyError):
        pass
    return "llama"


def _discover_slot_count(base_url: str, model_name: str) -> int:
    """Query /slots to determine how many parallel slots the server has.

    Returns the number of slots (parallelism). Falls back to 1 if the
    endpoint is unavailable or returns an error.
    """
    try:
        with request.urlopen(
            f"{base_url}/slots?model={model_name}",
            timeout=_HTTP_TIMEOUT,
        ) as resp:
            slots = json.loads(resp.read().decode("utf-8"))
            if isinstance(slots, list):
                count = len(slots)
                logger.info("Discovered %d slot(s) from /slots", count)
                return max(count, 1)
    except (error.URLError, error.HTTPError, json.JSONDecodeError, OSError):
        pass
    logger.info("/slots unavailable, defaulting to 1 slot")
    return 1


def _warmup_chat_endpoint(base_url: str, model_name: str, num_slots: int = 2):
    """Send minimal chat completions to warm up the chat handler and parallel slots.

    With --no-warmup + --parallel N, /health returns 200 before the chat
    endpoint is fully ready. The first real request can get HTTP 400.
    This sends tiny requests (1 token each) sequentially to ensure multiple
    slots are initialized — critical when --no-kv-unified is used because
    each slot has its own KV cache.
    """
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }

    data = json.dumps(payload).encode("utf-8")

    deadline = time.monotonic() + 30  # 30s max total for warmup
    for slot_num in range(num_slots):
        slot_start = time.monotonic()
        if slot_start > deadline:
            logger.warning("Warmup deadline exceeded before slot %d — proceeding", slot_num)
            break

        slot_ok = False
        while time.monotonic() < deadline:
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
                with request.urlopen(req, timeout=15) as resp:
                    resp.read()  # consume response
                    logger.info("Chat endpoint warmup slot %d successful", slot_num)
                    slot_ok = True
                    break
            except error.HTTPError as exc:
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    body = ""
                logger.warning(
                    "Warmup slot %d got HTTP %s — retrying (%s)",
                    slot_num,
                    exc.code,
                    body[:200],
                )
                time.sleep(1)
            except (error.URLError, OSError):
                time.sleep(0.5)

        if not slot_ok:
            logger.warning("Warmup slot %d did not succeed — proceeding anyway", slot_num)

    logger.info("Chat endpoint warmup complete (warmed %d slot(s))", num_slots)


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


def _describe_error(exc: BaseException) -> str:
    """One-line description of a chat-completion failure.

    Shared by the sync (urllib) and async (aiohttp) retry loops. For
    HTTP errors the response body is included (truncated) because it
    usually carries the real reason (context overflow, bad request, ...).
    """
    if isinstance(exc, error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = "<unreadable>"
        return f"HTTP {exc.code}: {body[:500]}"
    return f"{type(exc).__name__}: {exc}"


def _chat_completion(
    base_url: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    *,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> ChatResult:
    """Send a single chat completion request and return a ChatResult.

    Uses the OpenAI-compatible /v1/chat/completions endpoint.
    The server automatically strips reasoning_content from content.
    Inference params (max_tokens, temperature, top_p) are omitted so the
    server uses its own defaults.

    Retries up to _MAX_CHAT_RETRIES times, but only on transient errors
    (connection failures, timeouts, truncated responses, 408/429/5xx).
    Deterministic 4xx errors fail immediately without retrying.
    """
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    last_error: str | None = None
    for attempt in range(1, _MAX_CHAT_RETRIES + 1):
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

                # Extract token counts from usage
                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", usage.get("prompt_n", 0))
                completion_tokens = usage.get("completion_tokens", usage.get("predicted_n", 0))

                # Extract timing from the timings block (llama-server)
                timings = result.get("timings", {})
                prompt_ms = timings.get("prompt_ms", 0)
                predicted_ms = timings.get("predicted_ms", 0)

                return ChatResult(
                    content=content.strip() if content else None,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    prompt_ms=prompt_ms,
                    predicted_ms=predicted_ms,
                )
        except Exception as exc:
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


async def _chat_completion_async(
    session: aiohttp.ClientSession,
    base_url: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    index: int,
    *,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> tuple[int, ChatResult]:
    """Send a single chat completion request via aiohttp and return (index, ChatResult).

    Retries up to _MAX_CHAT_RETRIES times, but only on transient errors
    (connection failures, timeouts, truncated responses, 408/429/5xx).
    Deterministic 4xx errors fail immediately without retrying.
    """
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    last_error: str | None = None
    chat_result = ChatResult()

    for attempt in range(1, _MAX_CHAT_RETRIES + 1):
        http_error_body = ""
        failed_exc: BaseException | None = None
        try:
            async with session.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer no-key"},
                timeout=aiohttp.ClientTimeout(total=request_timeout),
            ) as resp:
                try:
                    resp.raise_for_status()
                except aiohttp.ClientResponseError:
                    # Response is still open: read the body for diagnostics.
                    # (ClientResponseError does not expose it.)
                    try:
                        http_error_body = (await resp.read()).decode("utf-8", errors="replace")[:500]
                    except Exception:
                        http_error_body = "<unreadable>"
                    raise
                result = await resp.json(content_type=None)
                content = result["choices"][0]["message"]["content"]

                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", usage.get("prompt_n", 0))
                completion_tokens = usage.get("completion_tokens", usage.get("predicted_n", 0))

                timings = result.get("timings", {})
                prompt_ms = timings.get("prompt_ms", 0)
                predicted_ms = timings.get("predicted_ms", 0)

                chat_result = ChatResult(
                    content=content.strip() if content else None,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    prompt_ms=prompt_ms,
                    predicted_ms=predicted_ms,
                )
                break  # Success
        except aiohttp.ClientResponseError as exc:
            failed_exc = exc
            last_error = f"HTTP {exc.status}: {http_error_body}"
        except Exception as exc:
            failed_exc = exc
            last_error = _describe_error(exc)
        else:
            continue  # unreachable: the success path breaks above

        # A request failed on this attempt - log, decide on retry.
        logger.error("Chat completion request failed (attempt %d/%d): %s", attempt, _MAX_CHAT_RETRIES, last_error)
        with _print_lock:
            print(f"  Chat completion request failed (attempt {attempt}/{_MAX_CHAT_RETRIES}): {last_error}")

        if not _is_transient_error(failed_exc):
            logger.error("Error is not transient - not retrying")
            break

        # Retry with exponential backoff (skip sleep on last attempt).
        if attempt < _MAX_CHAT_RETRIES:
            backoff = _retry_backoff(attempt)
            logger.info("Retrying in %.1fs...", backoff)
            with _print_lock:
                print(f"  Retrying in {backoff:.0f}s...")
            await asyncio.sleep(backoff)
    if not chat_result.content:
        chat_result.error = last_error or "all retries failed"

    # Final status print
    with _print_lock:
        if chat_result.content:
            tps = chat_result.generation_tps
            print(f"  Prompt {index + 1} response received ({len(chat_result.content)} chars, {chat_result.completion_tokens} tokens, {tps:.1f} tok/s)")
        else:
            reason = last_error or "unknown"
            print(f"  Prompt {index + 1}: empty/failed response ({reason})")

    return index, chat_result


def _plan_buckets(prompts: list[tuple[int, str]], bucket_size: int) -> list[list[tuple[int, str]]]:
    """Split prompts into consecutive buckets of bucket_size (last may be shorter)."""
    bucket_size = max(1, bucket_size)
    return [prompts[i:i + bucket_size] for i in range(0, len(prompts), bucket_size)]


def _variation_reference_block(previous_texts: list[str]) -> str:
    """Build the system-prompt addendum that references the previous bucket's prompts.

    The LLM sees what the previous bucket already produced, so it can vary
    mood, lighting, environment, pose, and composition instead of repeating
    the same scene. Whitespace is collapsed so each reference stays on one
    numbered line.
    """
    numbered = "\n".join(
        f"{n}. {' '.join(text.split())}" for n, text in enumerate(previous_texts, start=1)
    )
    return (
        "\n\n## Previous Batch Reference\n"
        "The following prompts were already enhanced and generated in the immediately preceding batch of this job:\n"
        f"{numbered}\n"
        "Do not repeat any of them exactly, and avoid near-duplicates of their mood, lighting, "
        "environment, pose, action, and composition. Introduce fresh variation while staying "
        "faithful to the user's original intent in the current request."
    )


def _batch_chat_completions(
    base_url: str,
    model_name: str,
    system_prompt: str,
    prompts: list[tuple[int, str]],
    *,
    max_concurrent: int | None = None,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    vary_across_buckets: bool = False,
) -> list[tuple[int, ChatResult]]:
    """Send multiple chat completion requests concurrently using aiohttp.

    Requests run in strict buckets of max_concurrent: a whole bucket must
    finish before the next bucket starts. With vary_across_buckets, every
    bucket after the first receives the previous bucket's prompts (enhanced
    text where available, original after a failure) as a "do not repeat,
    vary" reference appended to the system prompt.

    Args:
        base_url: Server base URL.
        model_name: Model id from /v1/models.
        system_prompt: Shared system prompt for all requests.
        prompts: List of (original_index, user_prompt) tuples.
        max_concurrent: Bucket size / max simultaneous requests. Defaults to len(prompts) (single bucket).
        request_timeout: Seconds to wait for a single request before retrying.
        vary_across_buckets: Append each bucket the previous bucket's prompts as a variation reference.

    Returns:
        List of (original_index, ChatResult) tuples, in input order.
    """
    bucket_size = max_concurrent if max_concurrent else max(len(prompts), 1)
    buckets = _plan_buckets(prompts, bucket_size)

    async def _run() -> list[tuple[int, ChatResult]]:
        connector = aiohttp.TCPConnector(limit=bucket_size, force_close=False)
        all_results: list[tuple[int, ChatResult]] = []
        previous_texts: list[str] = []

        async with aiohttp.ClientSession(connector=connector) as session:
            for bucket_num, bucket in enumerate(buckets):
                has_reference = vary_across_buckets and bool(previous_texts)
                bucket_system = (
                    system_prompt + _variation_reference_block(previous_texts)
                    if has_reference
                    else system_prompt
                )
                logger.info(
                    "Bucket %d/%d: %d prompt(s)%s",
                    bucket_num + 1, len(buckets), len(bucket),
                    " (with previous-bucket reference)" if has_reference else "",
                )
                results = await asyncio.gather(
                    *[
                        _chat_completion_async(
                            session, base_url, model_name, bucket_system, prompt, idx,
                            request_timeout=request_timeout,
                        )
                        for idx, prompt in bucket
                    ]
                )
                all_results.extend(results)
                # What the next bucket should reference: the enhanced text
                # actually used, or the original prompt where enhancement failed.
                previous_texts = [
                    chat_result.content or original
                    for (_, original), (_, chat_result) in zip(bucket, results)
                ]

        return all_results

    return asyncio.run(_run())


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


def _kill_server(sp: _ServerProcess | None):
    """Terminate the server subprocess, then stop the output drainers."""
    if sp is None:
        return

    pid = sp.pid
    logger.info("Killing llama-server (pid=%s)...", pid)
    print(f"  Killing llama-server (pid={pid})...")
    try:
        sp.proc.terminate()
        try:
            sp.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("llama-server (pid=%s) did not terminate, sending SIGKILL", pid)
            sp.proc.kill()
            sp.proc.wait(timeout=3)
    except Exception:
        logger.exception("Error stopping llama-server")
    finally:
        sp.stop_drainers()
        logger.info("llama-server (pid=%s) stopped.", pid)
        print(f"  llama-server (pid={pid}) stopped.")
