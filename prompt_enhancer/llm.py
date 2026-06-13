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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from urllib import request, error

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


def _chat_completion(
    base_url: str,
    system_prompt: str,
    user_prompt: str,
) -> str | None:
    """Send a single chat completion request and return the assistant's content.

    Uses the OpenAI-compatible /v1/chat/completions endpoint.
    The server automatically strips reasoning_content from content.
    """
    payload = {
        "model": "llama",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 4096,
        "temperature": 0.8,
        "top_p": 0.95,
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
            return content.strip() if content else None
    except (error.URLError, json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.error("Chat completion request failed: %s", exc)
        return None


def _chat_completion_with_index(
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    index: int,
) -> tuple[int, str | None]:
    """Send a single chat completion and return (index, content)."""
    content = _chat_completion(base_url, system_prompt, user_prompt)
    with _print_lock:
        if content:
            print(f"  Prompt {index + 1} response received ({len(content)} chars)")
        else:
            print(f"  Prompt {index + 1}: empty/failed response")
    return index, content


def _batch_chat_completions(
    base_url: str,
    system_prompt: str,
    prompts: list[tuple[int, str]],
) -> list[tuple[int, str | None]]:
    """Send multiple chat completion requests concurrently.

    Args:
        base_url: Server base URL.
        system_prompt: Shared system prompt for all requests.
        prompts: List of (original_index, user_prompt) tuples.

    Returns:
        List of (original_index, content) tuples, ordered by completion.
    """
    results: list[tuple[int, str | None]] = [None] * len(prompts)  # type: ignore[assignment]

    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = {
            executor.submit(
                _chat_completion_with_index, base_url, system_prompt, prompt, idx
            ): idx
            for idx, prompt in prompts
        }
        for future in as_completed(futures):
            idx, content = future.result()
            results[idx] = (idx, content)

    return results


def enhance_prompt(
    server_path: str,
    model_path: str,
    system_prompt: str,
    user_prompt: str,
    extra_flags: str = "",
) -> str | None:
    """Enhance a prompt by spawning a temporary llama-server instance.

    1. Find a free port by probing the ephemeral range.
    2. Start llama-server on that port.
    3. Wait for /health to confirm the server is ready.
    4. Send the prompt via /v1/chat/completions.
    5. Kill the server.
    6. Return the enhanced prompt (or None on failure).
    """
    if not Path(model_path).is_file():
        logger.error("Model file not found: %s", model_path)
        return None

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
        return None
    except Exception:
        logger.exception("Failed to start llama-server")
        return None

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
        return None

    ready_time = time.monotonic()
    print(f"  Server ready on port {port} in {ready_time - start:.1f}s. Sending prompt...")

    # Send the chat completion request
    result = _chat_completion(base_url, system_prompt, user_prompt)

    elapsed = time.monotonic() - ready_time

    # Always kill the server
    _kill_server(server_proc)

    if result:
        logger.info("Enhanced prompt (%d chars): %s", len(result), result)
        print(f"  Result ({len(result)} chars, {elapsed:.1f}s): {result[:200]}{'...' if len(result) > 200 else ''}")
        return result
    else:
        logger.warning("Empty or failed response from server")
        print(f"  FAILED: empty or invalid response ({elapsed:.1f}s)")
        return None


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
