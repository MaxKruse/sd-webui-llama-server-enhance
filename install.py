"""Startup dependency installer for sd-webui-llama-server-enhance.

Forge runs this file at startup: modules/launch_utils.py
run_extensions_installers() executes each extension's install.py with the
host venv's python, before extension scripts are loaded - which is what
makes the aiohttp dependency of prompt_enhancer/client.py available before
scripts/prompt_enhancer.py imports it.

It is a no-op once requirements.txt is satisfied, and re-checks on every
startup so a fresh install heals itself without manual steps.
"""

from __future__ import annotations

import os
import re
from importlib.metadata import PackageNotFoundError, version

from packaging.specifiers import SpecifierSet

import launch

REQUIREMENTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")

_NAME_SPLIT = re.compile(r"[\[<>=!~; ]")


def _parse_requirement(line: str) -> str | None:
    """Return the pip requirement spec from a requirements.txt line, or None for comments, blanks and pip options."""
    line = line.split("#", 1)[0].strip()
    if not line or line.startswith("-"):
        return None
    name = _NAME_SPLIT.split(line, maxsplit=1)[0].strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        return None
    return line


def _satisfied(spec: str) -> bool:
    """True if an installed distribution matches the requirement spec."""
    name = _NAME_SPLIT.split(spec, maxsplit=1)[0].strip()
    try:
        installed = version(name)
    except (PackageNotFoundError, ValueError):
        return False
    return SpecifierSet(spec.split(name, 1)[1]).contains(installed, prereleases=True)


def install() -> None:
    with open(REQUIREMENTS_FILE, "r", encoding="utf8") as file:
        specs = [spec for spec in map(_parse_requirement, file) if spec]

    for spec in specs:
        if _satisfied(spec):
            continue
        try:
            # Quoted: launch_utils.run() executes via shell=True, so an
            # unquoted '<'/'>' in the spec would be a shell redirect.
            launch.run_pip(f'install "{spec}"', "requirements for sd-webui-llama-server-enhance")
        except Exception as exc:
            print(f"Failed to install {spec}; the extension stays disabled until it is installed manually: {exc}")


if __name__ == "__main__":
    install()
