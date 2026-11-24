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
from types import SimpleNamespace

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


def main():
    print("opts tests (host `modules` stubbed, real module under test)")
    test_request_timeout()
    test_startup_timeout()
    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
