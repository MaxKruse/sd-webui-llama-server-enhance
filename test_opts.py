"""Standalone tests for prompt_enhancer/opts.py (host-dependent settings access).

opts.py is one of the two modules allowed to import the host's ``modules``
package, so the host boundary is stubbed in sys.modules before the import:
a ``modules`` package with a ``shared`` module whose ``shared.opts`` is a
plain namespace the tests populate per case. Everything under test (the
readers and their validation/fallback logic) is the real module.

Runs with the host venv's python (the package import needs aiohttp, like
test_batching.py):
    python test_opts.py
Exit code 0 on pass.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

_EXT_ROOT = Path(__file__).resolve().parent

# Stub the host boundary BEFORE importing the module under test.
_MODULES_STUB = types.ModuleType("modules")
_SHARED_STUB = types.ModuleType("modules.shared")
_SHARED_STUB.opts = SimpleNamespace()
_MODULES_STUB.shared = _SHARED_STUB
sys.modules["modules"] = _MODULES_STUB
sys.modules["modules.shared"] = _SHARED_STUB

from prompt_enhancer import opts as o  # noqa: E402
from prompt_enhancer.client import DEFAULT_REQUEST_TIMEOUT  # noqa: E402
from prompt_enhancer.server import _SERVER_STARTUP_TIMEOUT  # noqa: E402

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


def set_opts(**kwargs):
    _SHARED_STUB.opts = SimpleNamespace(**kwargs)


# ---------------------------------------------------------------------------
# read_request_timeout
# ---------------------------------------------------------------------------

def test_request_timeout():
    print("\n=== read_request_timeout ===")
    set_opts(llama_enhance_timeout=90)
    check("int value passes through", o.read_request_timeout() == 90.0,
          repr(o.read_request_timeout()))
    set_opts(llama_enhance_timeout="135")
    check("numeric string passes through", o.read_request_timeout() == 135.0,
          repr(o.read_request_timeout()))
    set_opts(llama_enhance_timeout=0)
    check("zero falls back to the default", o.read_request_timeout() == float(DEFAULT_REQUEST_TIMEOUT),
          repr(o.read_request_timeout()))
    set_opts(llama_enhance_timeout=-5)
    check("negative falls back to the default", o.read_request_timeout() == float(DEFAULT_REQUEST_TIMEOUT),
          repr(o.read_request_timeout()))
    set_opts(llama_enhance_timeout="abc")
    check("non-numeric falls back to the default", o.read_request_timeout() == float(DEFAULT_REQUEST_TIMEOUT),
          repr(o.read_request_timeout()))
    set_opts()
    check("missing option falls back to the default", o.read_request_timeout() == float(DEFAULT_REQUEST_TIMEOUT),
          repr(o.read_request_timeout()))


# ---------------------------------------------------------------------------
# read_startup_timeout
# ---------------------------------------------------------------------------

def test_startup_timeout():
    print("\n=== read_startup_timeout ===")
    set_opts(llama_enhance_startup_timeout=120)
    check("int value passes through", o.read_startup_timeout() == 120.0,
          repr(o.read_startup_timeout()))
    set_opts(llama_enhance_startup_timeout=-1)
    check("negative falls back to the default", o.read_startup_timeout() == float(_SERVER_STARTUP_TIMEOUT),
          repr(o.read_startup_timeout()))
    set_opts()
    check("missing option falls back to the default", o.read_startup_timeout() == float(_SERVER_STARTUP_TIMEOUT),
          repr(o.read_startup_timeout()))


# ---------------------------------------------------------------------------
# read_server_path / read_model_path / read_extra_flags
# ---------------------------------------------------------------------------

def test_path_and_flag_accessors():
    print("\n=== read_server_path / read_model_path / read_extra_flags ===")
    set_opts(llama_enhance_server_path="C:/tools/llama-server.exe")
    check("server path passes through", o.read_server_path() == "C:/tools/llama-server.exe",
          repr(o.read_server_path()))
    set_opts()
    check("missing server path falls back to the PATH name", o.read_server_path() == "llama-server",
          repr(o.read_server_path()))
    set_opts(llama_enhance_model_path="C:/models/m.gguf")
    check("model path passes through", o.read_model_path() == "C:/models/m.gguf",
          repr(o.read_model_path()))
    set_opts(llama_enhance_model_path="")
    check("empty model path -> ''", o.read_model_path() == "", repr(o.read_model_path()))
    set_opts()
    check("missing model path -> ''", o.read_model_path() == "", repr(o.read_model_path()))
    set_opts(llama_enhance_extra_flags="-ngl 99")
    check("extra flags pass through", o.read_extra_flags() == "-ngl 99",
          repr(o.read_extra_flags()))
    set_opts()
    check("missing extra flags -> ''", o.read_extra_flags() == "", repr(o.read_extra_flags()))


# ---------------------------------------------------------------------------
# spawn_healthy_server wrapper (delegates to the server.py helper)
# ---------------------------------------------------------------------------

def test_spawn_wrapper_delegates_to_server_helper():
    print("\n=== spawn_healthy_server wrapper ===")
    lines: list[str] = []
    # A real binary that exits immediately (the venv's python with a .txt
    # "model"): the wrapper must delegate to the server helper, which then
    # observes the early exit and returns None - not raise.
    _SHARED_STUB.opts = SimpleNamespace(
        llama_enhance_server_path=sys.executable,
        llama_enhance_model_path=str(_EXT_ROOT / "presets" / "anima.txt"),
        llama_enhance_extra_flags="",
        llama_enhance_startup_timeout=2,
    )
    try:
        result = o.spawn_healthy_server(lines.append, label=" (test)")
    except TypeError as exc:
        check("wrapper delegates to the server helper (no TypeError)", False, repr(exc))
        return
    check("unspawnable server -> None (killed, no exception)", result is None, repr(result))
    check("server helper ran with the debug sink wired",
          any("Running llama-server" in line for line in lines), repr(lines[:3]))


def main():
    print("opts tests (host `modules` stubbed, real module under test)")
    test_request_timeout()
    test_startup_timeout()
    test_path_and_flag_accessors()
    test_spawn_wrapper_delegates_to_server_helper()
    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
