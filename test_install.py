"""Standalone tests for install.py (the startup dependency installer).

Runs with the host venv's python (needs `packaging`, which Forge installs
before extension installers run). The host `launch` module cannot be
imported outside the webui, so it is stubbed in sys.modules and install.py
is loaded by file path. The stub's run_pip records calls instead of
executing pip.

Usage: python test_install.py
Exit code 0 on pass.
"""

import importlib.util
import os
import sys
import tempfile
import types

EXT_ROOT = os.path.dirname(os.path.abspath(__file__))
INSTALL_PY = os.path.join(EXT_ROOT, "install.py")

checks = 0


def check(cond, msg):
    global checks
    checks += 1
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    print(f"ok {checks}: {msg}")


def load_install_module():
    """Load install.py with a stubbed `launch` module; returns (module, stub)."""
    check(os.path.isfile(INSTALL_PY), "install.py exists in the extension root")

    stub = types.ModuleType("launch")
    stub.run_pip_calls = []

    def run_pip(command, desc=None, live=None):
        stub.run_pip_calls.append((command, desc))
        return ""

    stub.run_pip = run_pip
    sys.modules["launch"] = stub

    spec = importlib.util.spec_from_file_location("enhance_install_under_test", INSTALL_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, stub


def main():
    mod, stub = load_install_module()

    # --- _parse_requirement: requirements.txt line -> pip spec (or None)
    check(mod._parse_requirement("aiohttp>=3.9,<4") == "aiohttp>=3.9,<4", "version-ranged spec is returned as-is")
    check(mod._parse_requirement("numpy==1.26.4") == "numpy==1.26.4", "pinned spec is returned as-is")
    check(mod._parse_requirement("") is None, "blank line -> None")
    check(mod._parse_requirement("   ") is None, "whitespace-only line -> None")
    check(mod._parse_requirement("# comment") is None, "comment line -> None")
    check(mod._parse_requirement("requests  # inline comment") == "requests", "inline comment is stripped")
    check(mod._parse_requirement("-r other-requirements.txt") is None, "pip option line -> None")

    # --- _satisfied: spec vs the real environment (no mocks)
    check(mod._satisfied("packaging>=0") is True, "installed `packaging` satisfies >=0")
    check(mod._satisfied("packaging<0") is False, "impossible upper bound -> unsatisfied")
    check(mod._satisfied("packaging>=9999") is False, "impossible lower bound -> unsatisfied")
    check(mod._satisfied("no-such-package-zzz>=1") is False, "missing package -> unsatisfied")

    # --- install(): end-to-end against a temp requirements file
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf8") as f:
        f.write("# comment line\n")
        f.write("packaging>=0\n")
        f.write("no-such-package-zzz>=1\n")
        req_path = f.name

    try:
        stub.run_pip_calls.clear()
        mod.REQUIREMENTS_FILE = req_path
        mod.install()
    finally:
        os.unlink(req_path)

    check(len(stub.run_pip_calls) == 1, "exactly one pip call: satisfied spec skipped, missing spec installed")
    if stub.run_pip_calls:
        command, desc = stub.run_pip_calls[0]
        check(command == 'install "no-such-package-zzz>=1"', f"missing spec is installed, quoted for the shell: {command!r}")
        check(desc is not None and "llama-server-enhance" in desc, "pip call carries a descriptive desc")

    # all satisfied -> no-op
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf8") as f:
        f.write("packaging>=0\n")
        req_path = f.name

    try:
        stub.run_pip_calls.clear()
        mod.REQUIREMENTS_FILE = req_path
        mod.install()
    finally:
        os.unlink(req_path)

    check(len(stub.run_pip_calls) == 0, "all-satisfied requirements file -> no pip calls")

    print(f"PASS: {checks} checks")


if __name__ == "__main__":
    main()
