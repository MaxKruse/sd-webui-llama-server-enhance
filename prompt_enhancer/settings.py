"""WebUI settings registration for the LLM prompt enhancer."""

from __future__ import annotations

from modules import shared

from prompt_enhancer.presets import AUTO_PRESET


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
        key="llama_enhance_extra_flags",
        info=shared.OptionInfo(
            "",
            label="Inference flags: Extra flags passed to llama-server (e.g. -ngl 99 --temp 0.8 --top-p 0.9)",
            section=section,
        ),
    )
    shared.opts.add_option(
        key="llama_enhance_preset",
        info=shared.OptionInfo(
            AUTO_PRESET,
            label="Default preset: Auto selects based on the Forge-Neo UI preset",
            section=section,
        ),
    )
