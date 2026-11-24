"""Offline tests for scripts/prompt_enhancer.py (the host-dependent entry).

The entry is one of the three modules allowed to import the host
(gradio + modules.*), so the host boundary is stubbed in sys.modules and
the entry is loaded by file path - the same stubbing precedent as
test_install.py (host `launch`) and test_opts.py (host `modules.shared`).

Under test: the mutual exclusion between a generation's prompt enhancement
(process()) and the UI "Test server" button (_test_server) - exactly one
transient llama-server at a time - plus the process() happy path and the
skip/enable behavior, against a fake local llama-server (real HTTP client,
no model).

Run with the host venv's python (needs aiohttp):
    python test_entry.py
Exit code 0 on pass.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace

_EXT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXT_ROOT))

# ---------------------------------------------------------------------------
# Host boundary stubs (must be installed BEFORE the entry is loaded)
# ---------------------------------------------------------------------------

def _install_host_stubs() -> ModuleType:
    """Stub gradio + modules.* in sys.modules; returns the `shared` stub."""
    gradio = ModuleType("gradio")
    gradio.HTML = lambda *a, **k: None
    gradio.Dropdown = lambda *a, **k: None
    gradio.Checkbox = lambda *a, **k: None
    gradio.Button = lambda *a, **k: None
    gradio.Textbox = lambda *a, **k: None
    gradio.update = lambda *a, **k: None

    scripts_mod = ModuleType("modules.scripts")

    class Script:
        pass

    scripts_mod.Script = Script
    scripts_mod.AlwaysVisible = "ALWAYS_VISIBLE"

    callbacks_mod = ModuleType("modules.script_callbacks")
    callbacks_mod.on_ui_settings = lambda fn: None

    shared_mod = ModuleType("modules.shared")
    shared_mod.opts = SimpleNamespace()
    shared_mod.state = SimpleNamespace(job="")

    ui_mod = ModuleType("modules.ui_components")

    class InputAccordion:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    ui_mod.InputAccordion = InputAccordion

    processing_mod = ModuleType("modules.processing")

    class StableDiffusionProcessing:
        pass

    processing_mod.StableDiffusionProcessing = StableDiffusionProcessing

    modules_pkg = ModuleType("modules")
    modules_pkg.scripts = scripts_mod
    modules_pkg.script_callbacks = callbacks_mod
    modules_pkg.shared = shared_mod
    modules_pkg.ui_components = ui_mod
    modules_pkg.processing = processing_mod

    sys.modules.update({
        "gradio": gradio,
        "modules": modules_pkg,
        "modules.scripts": scripts_mod,
        "modules.script_callbacks": callbacks_mod,
        "modules.shared": shared_mod,
        "modules.ui_components": ui_mod,
        "modules.processing": processing_mod,
    })
    return shared_mod


_shared = _install_host_stubs()

_spec = importlib.util.spec_from_file_location(
    "entry_under_test", _EXT_ROOT / "scripts" / "prompt_enhancer.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

from prompt_enhancer.server import _WARMUP_PROMPT as WARMUP  # noqa: E402

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}{(' - ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# Fake llama-server: canned chat completions per user message; unregistered
# user messages get HTTP 400 so unexpected requests fail loudly.
# ---------------------------------------------------------------------------

class _FakeHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, status: int, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._send(200, {"data": [{"id": "fake-model.gguf"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        user = next(m["content"] for m in body["messages"] if m["role"] == "user")
        if user not in self.server.respond:
            self._send(400, {"error": f"unexpected user message: {user!r}"})
            return
        self._send(200, {
            "choices": [{"message": {"role": "assistant", "content": self.server.respond[user]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "timings": {"prompt_ms": 1.0, "predicted_ms": 1.0},
        })


def _start_fake_server(respond: dict):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHandler)
    server.respond = respond
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{server.server_address[1]}", server


class _FakeSp:
    """Stand-in for server._ServerProcess: the surface the entry touches."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.pid = 99999
        self.proc = SimpleNamespace(
            pid=99999, terminate=lambda: None, wait=lambda timeout=None: None
        )

    def stop_drainers(self):
        pass

    def stderr_tail(self, max_chars: int = 3000) -> str:
        return ""


# ---------------------------------------------------------------------------
# Entry-module patches: record instead of touching the real GPU, log, spawn.
# ---------------------------------------------------------------------------

_log_lines: list[str] = []


class _FakeLog:
    def write(self, msg: str):
        _log_lines.append(msg)


_mod = mod
_mod._DEBUG_LOG = _FakeLog()
_gpu_calls = {"n": 0}
_mod._free_gpu_memory = lambda: _gpu_calls.__setitem__("n", _gpu_calls["n"] + 1)
_spawn_calls = {"n": 0}
_sp_to_return: _FakeSp | None = None


def _fake_spawn(on_debug, label: str = ""):
    _spawn_calls["n"] += 1
    return _sp_to_return


_mod._opts.spawn_healthy_server = _fake_spawn
entry = _mod.LlamaServerEnhance()


def _set_opts(**kwargs):
    _shared.opts = SimpleNamespace(**kwargs)


def _set_state(job: str = ""):
    _shared.state = SimpleNamespace(job=job)


def _fake_p() -> SimpleNamespace:
    return SimpleNamespace(
        all_prompts=["a cat"],
        all_negative_prompts=["bad"],
        n_iter=1,
        batch_size=1,
        width=512,
        height=512,
        enable_hr=False,
    )


def _reset_counters():
    _log_lines.clear()
    _gpu_calls["n"] = 0
    _spawn_calls["n"] = 0


MODEL_FILE = str(_EXT_ROOT / "presets" / "anima.txt")  # any existing file


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_process_happy_path():
    print("\n[process] happy path: once mode enhances, writes back, kills server")
    base_url, server = _start_fake_server({WARMUP: "ok", "a cat": "enhanced cat"})
    try:
        _set_opts(
            forge_preset="anima",
            sd_model_checkpoint="",
            llama_enhance_model_path=MODEL_FILE,
            llama_enhance_server_path=sys.executable,
        )
        _set_state()
        _sp_to_return_set(base_url)
        _reset_counters()
        p = _fake_p()
        entry.process(p, True, "Once", "Auto (server slots)", False)
        check("positive enhanced and written back", p.all_prompts == ["enhanced cat"], repr(p.all_prompts))
        check("negative untouched (enhancement off)", p.all_negative_prompts == ["bad"], repr(p.all_negative_prompts))
        check("exactly one server spawned", _spawn_calls["n"] == 1, repr(_spawn_calls["n"]))
        check("GPU freed before the server", _gpu_calls["n"] == 1, repr(_gpu_calls["n"]))
    finally:
        server.shutdown()
        server.server_close()


def _sp_to_return_set(base_url: str):
    global _sp_to_return
    _sp_to_return = _FakeSp(base_url)


def test_process_skips_when_enhancement_in_progress():
    print("\n[process] lock held by another enhancement: skipped, nothing spawned")
    lock = getattr(mod, "_ENHANCE_LOCK", None)
    if lock is None:
        check("entry module owns a shared enhancement lock", False,
              "no _ENHANCE_LOCK on the entry module")
        return
    lock.acquire()
    try:
        _set_opts(
            forge_preset="anima",
            llama_enhance_model_path=MODEL_FILE,
            llama_enhance_server_path=sys.executable,
        )
        _reset_counters()
        p = _fake_p()
        entry.process(p, True, "Once", "Auto (server slots)", False)
        check("prompt untouched while an enhancement is running",
              p.all_prompts == ["a cat"], repr(p.all_prompts))
        check("no server spawned while an enhancement is running",
              _spawn_calls["n"] == 0, repr(_spawn_calls["n"]))
        check("GPU not freed for the skipped enhancement",
              _gpu_calls["n"] == 0, repr(_gpu_calls["n"]))
        check("skip announced in the debug log",
              any("skipped" in line for line in _log_lines), repr(_log_lines[-5:]))
    finally:
        lock.release()


def test_test_server_busy_while_enhancement_running():
    print("\n[_test_server] lock held by an enhancement: busy message, nothing spawned")
    lock = getattr(mod, "_ENHANCE_LOCK", None)
    if lock is None:
        check("entry module owns a shared enhancement lock", False,
              "no _ENHANCE_LOCK on the entry module")
        return
    base_url, server = _start_fake_server({WARMUP: "ok", "ping": "OK"})
    try:
        _set_opts(
            forge_preset="anima",
            llama_enhance_model_path=MODEL_FILE,
            llama_enhance_server_path=sys.executable,
        )
        _set_state()
        _sp_to_return_set(base_url)
        _reset_counters()
        lock.acquire()
        try:
            result = entry._test_server()
            check("busy message returned", "already running" in result, repr(result))
            check("no server spawned while an enhancement is running",
                  _spawn_calls["n"] == 0, repr(_spawn_calls["n"]))
        finally:
            lock.release()
    finally:
        server.shutdown()
        server.server_close()


def test_test_server_blocked_during_generation_job():
    print("\n[_test_server] a generation job is in progress: blocked (pre-existing guard)")
    base_url, server = _start_fake_server({WARMUP: "ok", "ping": "OK"})
    try:
        _set_opts(
            forge_preset="anima",
            llama_enhance_model_path=MODEL_FILE,
            llama_enhance_server_path=sys.executable,
        )
        _set_state(job="txt2img")
        _sp_to_return_set(base_url)
        _reset_counters()
        result = entry._test_server()
        check("generation-in-progress message returned", "in progress" in result, repr(result))
        check("no server spawned while a generation runs", _spawn_calls["n"] == 0,
              repr(_spawn_calls["n"]))
    finally:
        server.shutdown()
        server.server_close()


def test_process_disabled():
    print("\n[process] accordion closed: no GPU free, no spawn")
    _set_opts(forge_preset="anima", llama_enhance_model_path=MODEL_FILE)
    _reset_counters()
    p = _fake_p()
    entry.process(p, False, "Once", "Auto (server slots)", False)
    check("prompt untouched", p.all_prompts == ["a cat"], repr(p.all_prompts))
    check("no GPU free when disabled", _gpu_calls["n"] == 0, repr(_gpu_calls["n"]))
    check("no spawn when disabled", _spawn_calls["n"] == 0, repr(_spawn_calls["n"]))


def main():
    print("entry tests (host stubbed, real entry under test)")
    test_process_happy_path()
    test_process_skips_when_enhancement_in_progress()
    test_test_server_busy_while_enhancement_running()
    test_test_server_blocked_during_generation_job()
    test_process_disabled()
    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
