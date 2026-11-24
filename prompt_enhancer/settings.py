"""WebUI settings registration for the LLM prompt enhancer.

This file is intended for human reading and was written with the assistance of an LLM.
"""

from __future__ import annotations

from modules import shared
from prompt_enhancer.client import DEFAULT_REQUEST_TIMEOUT
from prompt_enhancer.server import _SERVER_STARTUP_TIMEOUT


def on_ui_settings():
    """Register settings in the WebUI Settings tab."""
    section = ("llama_enhance", "LLama Server Enhance")

    shared.opts.add_option(
        key="llama_enhance_server_path",
        info=shared.OptionInfo(
            "llama-server",
            label="llama-server path: Path to the llama-server binary (default: llama-server from PATH)",
            section=section,
        ),
    )
    shared.opts.add_option(
        key="llama_enhance_model_path",
        info=shared.OptionInfo(
            "",
            label="Model path: Full path to the .gguf model file",
            section=section,
        ),
    )
    shared.opts.add_option(
        key="llama_enhance_timeout",
        info=shared.OptionInfo(
            DEFAULT_REQUEST_TIMEOUT,
            label=f"Request timeout: Seconds to wait for a single LLM request before retrying (default: {DEFAULT_REQUEST_TIMEOUT})",
            section=section,
        ),
    )
    shared.opts.add_option(
        key="llama_enhance_startup_timeout",
        info=shared.OptionInfo(
            _SERVER_STARTUP_TIMEOUT,
            label=f"Server startup timeout: Seconds to wait for llama-server to load the model and become healthy (default: {_SERVER_STARTUP_TIMEOUT})",
            section=section,
        ),
    )
    shared.opts.add_option(
        key="llama_enhance_extra_flags",
        info=shared.OptionInfo(
            "",
            label="Inference flags: Extra flags passed to llama-server (e.g. -ngl 99 --temp 0.8 --top-p 0.9)",
            section=section,
        ),
    )
