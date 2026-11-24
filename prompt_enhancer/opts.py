"""Host-dependent settings access for the LLM prompt enhancer.

Reads ``shared.opts`` defensively (getattr with defaults / ``or``
fallbacks): a missing option (other fork, DP extension absent) degrades
to a default instead of crashing the generation.

One of the two modules allowed to import host ``modules.*`` (the other is
``prompt_enhancer.settings``) - everything else in the package stays
host-independent so the logic can be tested standalone.
"""

from __future__ import annotations

import logging

from modules import shared
from prompt_enhancer.client import DEFAULT_REQUEST_TIMEOUT
from prompt_enhancer.prompts import assemble_system_prompt

# spawn_healthy_server is imported under an alias: the settings-backed
# wrapper below keeps the public name `spawn_healthy_server`, and a
# same-name import would be shadowed by that def - making the wrapper
# call itself (TypeError) instead of the server.py helper.
from prompt_enhancer.server import (
    _SERVER_STARTUP_TIMEOUT,
)
from prompt_enhancer.server import (
    spawn_healthy_server as _spawn_server_impl,
)

logger = logging.getLogger(__name__)


def _coerce_timeout(value, default: float, option_name: str) -> float:
    """Coerce a settings timeout to a positive float (seconds).

    Missing, non-numeric, zero or negative values fall back to the default
    with a warning - a non-positive timeout would make the wait loop expire
    immediately or silently change behavior (``or``-based defaults treat 0
    as "unset" but let negatives through).
    """
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid %s value %r - falling back to %ss", option_name, value, default,
        )
        return float(default)
    if seconds <= 0:
        logger.warning(
            "Invalid %s value %r (must be positive) - falling back to %ss",
            option_name, value, default,
        )
        return float(default)
    return seconds


def read_request_timeout() -> float:
    """Read the per-request LLM timeout from settings (seconds)."""
    return _coerce_timeout(
        getattr(shared.opts, "llama_enhance_timeout", DEFAULT_REQUEST_TIMEOUT),
        DEFAULT_REQUEST_TIMEOUT,
        "llama_enhance_timeout",
    )


def read_startup_timeout() -> float:
    """Read the server startup timeout from settings (seconds)."""
    return _coerce_timeout(
        getattr(shared.opts, "llama_enhance_startup_timeout", _SERVER_STARTUP_TIMEOUT),
        _SERVER_STARTUP_TIMEOUT,
        "llama_enhance_startup_timeout",
    )


def read_server_path() -> str:
    """The llama-server binary path; defaults to 'llama-server' from PATH."""
    return getattr(shared.opts, "llama_enhance_server_path", "") or "llama-server"


def read_model_path() -> str:
    """The configured .gguf model path ("" if unset)."""
    return getattr(shared.opts, "llama_enhance_model_path", "") or ""


def read_extra_flags() -> str:
    """Extra llama-server flags from settings."""
    return getattr(shared.opts, "llama_enhance_extra_flags", "") or ""


def read_dp_separators() -> tuple[str, str, str]:
    """Read the Dynamic Prompts separators (variant start/end, wildcard wrap)."""
    v_start = getattr(shared.opts, "dp_parser_variant_start", "{") or "{"
    v_end = getattr(shared.opts, "dp_parser_variant_end", "}") or "}"
    wc_wrap = getattr(shared.opts, "dp_parser_wildcard_wrap", "__") or "__"
    return v_start, v_end, wc_wrap


def build_system_prompts(
    preset_content: str,
    *,
    width: int,
    height: int,
    hires: bool,
    base_width: int,
    base_height: int,
    negative_instruction: str | None = None,
) -> tuple[str, str | None]:
    """Assemble the positive (and negative) system prompts for one generation.

    Appends the resolution context to the preset. No wildcard-preservation
    note: this extension runs after sd-dynamic-prompts (see metadata.ini),
    so the prompt it enhances never contains unexpanded wildcards.
    """
    return assemble_system_prompt(
        preset_content,
        width=width,
        height=height,
        hires=hires,
        base_width=base_width,
        base_height=base_height,
        negative_instruction=negative_instruction,
    )


def spawn_healthy_server(on_debug, label: str = ""):
    """Start llama-server and wait until it becomes healthy.

    Wires in the WebUI-specific bits: the user-configured startup timeout
    and the debug-log sink (server stderr streamed in as
    ``[llama-server]`` lines, plus the same diagnostic lines the helper
    prints). Returns the server handle, or None on failure (already killed).
    """

    def _on_stderr(chunk: str):
        for line in chunk.rstrip("\n").split("\n"):
            if line.strip():
                on_debug(f"  [llama-server] {line}")

    return _spawn_server_impl(
        read_server_path(),
        read_model_path(),
        read_extra_flags(),
        startup_timeout=read_startup_timeout(),
        on_stderr=_on_stderr,
        on_debug=on_debug,
        label=label,
    )
