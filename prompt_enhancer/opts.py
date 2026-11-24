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
from pathlib import Path

from modules import shared

from prompt_enhancer.client import DEFAULT_REQUEST_TIMEOUT
from prompt_enhancer.prompts import assemble_system_prompt, build_dp_wildcard_instruction
from prompt_enhancer.server import _SERVER_STARTUP_TIMEOUT, spawn_healthy_server

logger = logging.getLogger(__name__)


def read_request_timeout() -> float:
    """Read the per-request LLM timeout from settings (seconds)."""
    try:
        return float(
            getattr(shared.opts, "llama_enhance_timeout", DEFAULT_REQUEST_TIMEOUT)
            or DEFAULT_REQUEST_TIMEOUT
        )
    except (TypeError, ValueError):
        logger.warning(
            "Invalid llama_enhance_timeout value - falling back to %ss",
            DEFAULT_REQUEST_TIMEOUT,
        )
        return float(DEFAULT_REQUEST_TIMEOUT)


def read_startup_timeout() -> float:
    """Read the server startup timeout from settings (seconds)."""
    try:
        return float(
            getattr(shared.opts, "llama_enhance_startup_timeout", _SERVER_STARTUP_TIMEOUT)
            or _SERVER_STARTUP_TIMEOUT
        )
    except (TypeError, ValueError):
        logger.warning(
            "Invalid llama_enhance_startup_timeout value - falling back to %ss",
            _SERVER_STARTUP_TIMEOUT,
        )
        return float(_SERVER_STARTUP_TIMEOUT)


def read_server_path() -> str:
    """The llama-server binary path; defaults to 'llama-server' from PATH."""
    return shared.opts.llama_enhance_server_path or "llama-server"


def read_model_path() -> str:
    """The configured .gguf model path ("" if unset)."""
    return shared.opts.llama_enhance_model_path or ""


def read_extra_flags() -> str:
    """Extra llama-server flags from settings."""
    return shared.opts.llama_enhance_extra_flags or ""


def read_dp_separators() -> tuple[str, str, str]:
    """Read the Dynamic Prompts separators (variant start/end, wildcard wrap)."""
    v_start = getattr(shared.opts, "dp_parser_variant_start", "{") or "{"
    v_end = getattr(shared.opts, "dp_parser_variant_end", "}") or "}"
    wc_wrap = getattr(shared.opts, "dp_parser_wildcard_wrap", "__") or "__"
    return v_start, v_end, wc_wrap


def dynamic_prompts_installed(extensions_dir: Path) -> bool:
    """Check if the sd-dynamic-prompts extension is present."""
    return (extensions_dir / "sd-dynamic-prompts").is_dir()


def build_system_prompts(
    preset_content: str,
    *,
    width: int,
    height: int,
    hires: bool,
    base_width: int,
    base_height: int,
    extensions_dir: Path,
    negative_instruction: str | None = None,
) -> tuple[str, str | None]:
    """Assemble the positive (and negative) system prompts for one generation.

    Appends the resolution context to the preset, plus the DP
    wildcard-preservation note when sd-dynamic-prompts is installed.
    """
    dp_instruction = ""
    if dynamic_prompts_installed(extensions_dir):
        wc_wrap, v_start, v_end = read_dp_separators()
        dp_instruction = build_dp_wildcard_instruction(wc_wrap, v_start, v_end)

    return assemble_system_prompt(
        preset_content,
        width=width,
        height=height,
        hires=hires,
        base_width=base_width,
        base_height=base_height,
        dp_instruction=dp_instruction,
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

    return spawn_healthy_server(
        read_server_path(),
        read_model_path(),
        read_extra_flags(),
        startup_timeout=read_startup_timeout(),
        on_stderr=_on_stderr,
        on_debug=on_debug,
        label=label,
    )
