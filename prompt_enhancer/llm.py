"""llama-server HTTP client for prompt enhancement.

Spawns a temporary llama-server instance, sends prompts via the OpenAI-compatible
/v1/chat/completions endpoint, extracts the `content` field (reasoning_content is
stripped server-side), then kills the server.

Supports both single-prompt and batch-prompt modes. In batch mode all requests are
fired concurrently so the server can process them in parallel.
"""

from __future__ import annotations

import json
import logging
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from threading import Lock
from urllib import request, error

import aiohttp
import asyncio

logger = logging.getLogger(__name__)

# How long to wait for the server to become healthy (model load time).
_SERVER_STARTUP_TIMEOUT = 60

# Timeout for individual HTTP requests.
_HTTP_TIMEOUT = 5

# Port range to probe for a free port.
_PORT_RANGE = range(49152, 65536)


def _find_free_port() -> int:
    """Find a free TCP port by probing random ports in the ephemeral range.

    Binds a socket to confirm the port is available, then returns it.
    There's a tiny race window between releasing the socket and
    llama-server binding, but it's negligible in practice.
    """
    candidates = random.sample(list(_PORT_RANGE), min(100, len(_PORT_RANGE)))
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


def _wait_for_server(base_url: str, timeout: int = _SERVER_STARTUP_TIMEOUT) -> bool:
    """Poll /health until the server responds or timeout is reached."""
    deadline = time.monotonic() + timeout
    url = f"{base_url}/health"

    while time.monotonic() < deadline:
        try:
            with request.urlopen(url, timeout=_HTTP_TIMEOUT) as resp:
                if resp.status == 200:
                    return True
        except (error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.5)

    return False


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
    Strips trailing .gguf extension — some llama-server builds reject
    model names that include the file extension in chat completions.
    """
    try:
        with request.urlopen(f"{base_url}/v1/models", timeout=_HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            # Prefer the loaded model
            for m in models:
                status = m.get("status", {})
                if isinstance(status, dict) and status.get("value") == "loaded":
                    name = m.get("id", "llama")
                    return name[:-5] if name.lower().endswith(".gguf") else name
            # Fallback: first model
            if models:
                name = models[0].get("id", "llama")
                return name[:-5] if name.lower().endswith(".gguf") else name
    except (error.URLError, json.JSONDecodeError, KeyError):
        pass
    return "llama"


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


def _chat_completion(
    base_url: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
) -> ChatResult:
    """Send a single chat completion request and return a ChatResult.

    Uses the OpenAI-compatible /v1/chat/completions endpoint.
    The server automatically strips reasoning_content from content.
    Inference params (max_tokens, temperature, top_p) are omitted so the
    server uses its own defaults.
    """
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

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
        with request.urlopen(req, timeout=120) as resp:
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
    except error.HTTPError as exc:
        # Capture the server's error response body for debugging
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = "<unreadable>"
        error_msg = f"HTTP {exc.code}: {body[:500]}"
        logger.error("Chat completion request failed: %s", error_msg)
        print(f"  Chat completion request failed: HTTP {exc.code}: {body[:200]}")
        return ChatResult(error=error_msg)
    except (error.URLError, json.JSONDecodeError, KeyError, IndexError) as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
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
) -> tuple[int, ChatResult]:
    """Send a single chat completion request via aiohttp and return (index, ChatResult)."""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer no-key"},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
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
    except aiohttp.ClientResponseError as exc:
        error_msg = f"HTTP {exc.status}: {exc.message}"
        logger.error("Chat completion request failed: %s", error_msg)
        with _print_lock:
            print(f"  Chat completion request failed: {error_msg}")
        chat_result = ChatResult(error=error_msg)
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error("Chat completion request failed: %s", error_msg)
        with _print_lock:
            print(f"  Chat completion request failed: {error_msg}")
        chat_result = ChatResult(error=error_msg)

    with _print_lock:
        if chat_result.content:
            tps = chat_result.generation_tps
            print(f"  Prompt {index + 1} response received ({len(chat_result.content)} chars, {chat_result.completion_tokens} tokens, {tps:.1f} tok/s)")
        else:
            reason = chat_result.error if chat_result.error else "unknown"
            print(f"  Prompt {index + 1}: empty/failed response ({reason})")

    return index, chat_result


def _batch_chat_completions(
    base_url: str,
    model_name: str,
    system_prompt: str,
    prompts: list[tuple[int, str]],
) -> list[tuple[int, ChatResult]]:
    """Send multiple chat completion requests concurrently using aiohttp.

    Uses asyncio + aiohttp for true concurrent HTTP I/O (no GIL blocking).

    Args:
        base_url: Server base URL.
        model_name: Model id from /v1/models.
        system_prompt: Shared system prompt for all requests.
        prompts: List of (original_index, user_prompt) tuples.

    Returns:
        List of (original_index, ChatResult) tuples, ordered by index.
    """
    async def _run() -> list[tuple[int, ChatResult]]:
        connector = aiohttp.TCPConnector(limit=len(prompts), force_close=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                _chat_completion_async(session, base_url, model_name, system_prompt, prompt, idx)
                for idx, prompt in prompts
            ]
            return await asyncio.gather(*tasks)

    return asyncio.run(_run())


def enhance_prompt(
    server_path: str,
    model_path: str,
    system_prompt: str,
    user_prompt: str,
    extra_flags: str = "",
) -> ChatResult:
    """Enhance a prompt by spawning a temporary llama-server instance.

    1. Find a free port by probing the ephemeral range.
    2. Start llama-server on that port.
    3. Wait for /health to confirm the server is ready.
    4. Send the prompt via /v1/chat/completions.
    5. Kill the server.
    6. Return a ChatResult with content and timing stats.
    """
    if not Path(model_path).is_file():
        logger.error("Model file not found: %s", model_path)
        return ChatResult()

    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    cmd = _build_command(server_path, model_path, port, extra_flags)

    logger.info("llama-server command: %s", " ".join(cmd))
    logger.info("llama-server user prompt: %s", user_prompt)
    print(f"\n  Running llama-server (model: {Path(model_path).name}, port: {port})...")

    # Windows: prevent console window popup
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    server_proc: subprocess.Popen | None = None
    try:
        server_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        logger.info("llama-server started (pid=%s, port=%s)", server_proc.pid, port)
        print(f"  Started llama-server (pid={server_proc.pid}, port={port})")
    except FileNotFoundError:
        logger.error("llama-server not found at: %s", server_path)
        print(f"  FAILED: llama-server not found at: {server_path}")
        return ChatResult()
    except Exception:
        logger.exception("Failed to start llama-server")
        return ChatResult()

    # Wait for /health to confirm the server is fully ready
    print(f"  Waiting for server to start (timeout: {_SERVER_STARTUP_TIMEOUT}s)...")
    start = time.monotonic()

    if not _wait_for_server(base_url, timeout=_SERVER_STARTUP_TIMEOUT):
        elapsed = time.monotonic() - start
        print(f"  FAILED: Server did not become healthy within {_SERVER_STARTUP_TIMEOUT}s ({elapsed:.1f}s elapsed)")
        logger.error("Server did not become healthy within %ds", _SERVER_STARTUP_TIMEOUT)

        # Check if the process crashed and log stderr
        if server_proc.poll() is not None:
            stderr = server_proc.stderr.read().decode("utf-8", errors="replace") if server_proc.stderr else ""
            logger.error("Server process exited (code=%s). Stderr:\n%s", server_proc.returncode, stderr[-1000:])
        _kill_server(server_proc)
        return ChatResult()

    ready_time = time.monotonic()

    # Discover the loaded model name
    model_name = _discover_model_name(base_url)
    logger.info("Discovered model: %s", model_name)
    print(f"  Server ready on port {port} in {ready_time - start:.1f}s. Model: {model_name}")

    # Warm up the chat endpoint — /health returns 200 before the chat handler
    # and parallel KV slots are fully initialized. Without this, the first real
    # request can get HTTP 400 (especially with --parallel N --no-kv-unified).
    _warmup_chat_endpoint(base_url, model_name, num_slots=1)

    # Send the chat completion request
    result = _chat_completion(base_url, model_name, system_prompt, user_prompt)

    elapsed = time.monotonic() - ready_time

    # Always kill the server
    _kill_server(server_proc)

    if result.content:
        logger.info("Enhanced prompt (%d chars): %s", len(result.content), result.content)
        print(
            f"  Result ({len(result.content)} chars, {result.completion_tokens} tokens, "
            f"{result.generation_tps:.1f} tok/s, {result.total_ms:.0f}ms total, {elapsed:.1f}s wall): "
            f"{result.content[:200]}{'...' if len(result.content) > 200 else ''}"
        )
    else:
        logger.warning("Empty or failed response from server")
        print(f"  FAILED: empty or invalid response ({elapsed:.1f}s)")
        # Capture server stderr for diagnostics
        if server_proc and server_proc.stderr:
            try:
                stderr_output = server_proc.stderr.read().decode("utf-8", errors="replace")
                if stderr_output.strip():
                    logger.warning("Server stderr:\n%s", stderr_output[-2000:])
            except Exception:
                pass

    return result


def _kill_server(proc: subprocess.Popen | None):
    """Terminate the server subprocess and wait for it to exit."""
    if proc is None:
        return

    pid = proc.pid
    logger.info("Killing llama-server (pid=%s)...", pid)
    print(f"  Killing llama-server (pid={pid})...")
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("llama-server (pid=%s) did not terminate, sending SIGKILL", pid)
            proc.kill()
            proc.wait(timeout=3)
    except Exception:
        logger.exception("Error stopping llama-server")
    finally:
        logger.info("llama-server (pid=%s) stopped.", pid)
        print(f"  llama-server (pid={pid}) stopped.")
