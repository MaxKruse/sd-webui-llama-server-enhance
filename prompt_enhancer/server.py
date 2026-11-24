"""llama-server subprocess lifecycle and server introspection.

Owns everything about the *process*: building the command line, finding a
free port, spawning with continuously drained output pipes, waiting for
/health, discovering the loaded model and parallel slot count, and killing
the server.

Host-independent (stdlib only). The HTTP client for chat completions lives
in ``prompt_enhancer.client``.
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
from threading import Lock, Thread
from urllib import error, request

logger = logging.getLogger(__name__)

# Timeout for individual health-check and discovery polls.
_HTTP_TIMEOUT = 5

# How long to wait for the server to become healthy (model load time).
_SERVER_STARTUP_TIMEOUT = 60

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
    timeout: float = _SERVER_STARTUP_TIMEOUT,
    proc: subprocess.Popen | None = None,
    heartbeat_interval: float = 10.0,
) -> bool:
    """Poll /health until the server responds, the timeout is reached, or the
    process exits early (a crashed server can never become healthy).

    Prints a heartbeat line every heartbeat_interval seconds so a slow model
    load (up to `timeout` seconds of console silence) looks alive instead of
    hung. Full load progress stays in the debug log - streaming all of the
    server's stderr to the console would be noise.
    """
    deadline = time.monotonic() + timeout
    start = time.monotonic()
    next_heartbeat = start + heartbeat_interval
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
        if time.monotonic() >= next_heartbeat:
            print(f"  ... still loading ({time.monotonic() - start:.0f}s / {timeout:.0f}s)")
            next_heartbeat += heartbeat_interval
        time.sleep(0.5)

    return False


_DRAIN_MAX_BYTES = 256 * 1024


class _PipeDrainer(Thread):
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


def _await_health(
    sp: _ServerProcess,
    timeout: float,
    emit,
    *,
    log_stderr_to_console: bool,
) -> bool:
    """Wait for /health and report the outcome to the console and ``emit``.

    On failure the server is killed and False is returned.
    ``log_stderr_to_console`` mirrors the stderr tail into ``logger`` when no
    debug sink is attached, so failures stay visible in the WebUI console.
    """
    print(f"  Waiting for server to start (timeout: {timeout:.0f}s)...")
    emit(f"  Waiting for server to start (timeout: {timeout:.0f}s)...")
    start = time.monotonic()

    if not _wait_for_server(sp.base_url, timeout=timeout, proc=sp.proc):
        elapsed = time.monotonic() - start
        print(f"  FAILED: Server did not become healthy within {timeout:.0f}s ({elapsed:.1f}s elapsed)")
        logger.error("Server did not become healthy within %.0fs", timeout)
        emit(f"  → server health check failed after {elapsed:.1f}s")
        stderr_tail = sp.stderr_tail(3000)
        if stderr_tail.strip():
            emit(f"  llama-server stderr (last 3000 chars):\n{stderr_tail}")
            print(f"  Server stderr (last 500 chars):\n{stderr_tail[-500:]}")
            if log_stderr_to_console:
                logger.warning("llama-server stderr (last 2000 chars):\n%s", stderr_tail[-2000:])
        _kill_server(sp)
        return False

    elapsed = time.monotonic() - start
    print(f"  Server ready on port {sp.port} in {elapsed:.1f}s")
    emit(f"  Server ready on port {sp.port} in {elapsed:.1f}s")
    return True


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

    if not _await_health(sp, timeout, _emit, log_stderr_to_console=on_debug is None):
        return None
    return sp


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


def _discover_model_name(base_url: str) -> str:
    """Fetch the currently loaded model name from /v1/models.

    Prefers the model whose status is "loaded". Falls back to the first
    model in the list, then to "llama" if nothing is available.
    Strips trailing .gguf extension and full filesystem paths - some
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


def _warmup_slot(base_url: str, data: bytes, slot_num: int, deadline: float) -> bool:
    """Warm up one parallel slot: retry a 1-token completion until it
    succeeds, or the overall warmup deadline is exceeded."""
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
                return True
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

    return False


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
        if time.monotonic() > deadline:
            logger.warning("Warmup deadline exceeded before slot %d — proceeding", slot_num)
            break
        if not _warmup_slot(base_url, data, slot_num, deadline):
            logger.warning("Warmup slot %d did not succeed — proceeding anyway", slot_num)

    logger.info("Chat endpoint warmup complete (warmed %d slot(s))", num_slots)
