"""AlwaysVisible script that enhances prompts via llama-server."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import modules.scripts as scripts
from modules import script_callbacks, shared
from modules.ui_components import InputAccordion
from modules.processing import StableDiffusionProcessing

from prompt_enhancer import settings as _settings_module
from prompt_enhancer.llm import (
    _SERVER_STARTUP_TIMEOUT,
    _build_command,
    _find_free_port,
    _wait_for_server,
    _batch_chat_completions,
    _kill_server,
    enhance_prompt,
)
from prompt_enhancer.presets import load_preset, list_presets

logger = logging.getLogger(__name__)

# Resolve the extensions directory relative to this file
_EXTENSIONS_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Log file for debugging — written to the extension root so it's easy to find
_LOG_DIR = Path(__file__).resolve().parent.parent
_LOG_FILE = _LOG_DIR / "enhance_debug.log"


def _log_to_file(msg: str):
    """Append a timestamped message to the debug log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] {msg}\n"
    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)


def _dynamic_prompts_installed() -> bool:
    """Check if the sd-dynamic-prompts extension is present."""
    return (_EXTENSIONS_DIR / "sd-dynamic-prompts").is_dir()


def _build_resolution_instruction(width: int, height: int) -> str:
    """Build a resolution hint block for the LLM system prompt.

    Tells the LLM the target image dimensions and orientation so it can
    frame compositions, aspect ratios, and layout descriptions appropriately.
    """
    gcd = __import__("math").gcd(width, height)
    ratio_w, ratio_h = width // gcd, height // gcd
    orientation = "portrait" if height > width else "landscape" if width > height else "square"

    return (
        f"\n\n--- Image Resolution ---\n"
        f"Target resolution: {width}x{height} ({orientation}, {ratio_w}:{ratio_h} aspect ratio)\n"
        f"Frame your description to suit this orientation."
    )


def _build_dp_wildcard_instruction() -> str:
    """Build a wildcard-preservation instruction for the LLM system prompt.

    Reads the user's Dynamic Prompts settings (wildcard wrapper and variant
    brackets) and returns a block the LLM should append to its system prompt.
    Returns an empty string if sd-dynamic-prompts is not installed.
    """
    if not _dynamic_prompts_installed():
        return ""

    wc_wrap = getattr(shared.opts, "dp_parser_wildcard_wrap", "__") or "__"
    v_start = getattr(shared.opts, "dp_parser_variant_start", "{") or "{"
    v_end = getattr(shared.opts, "dp_parser_variant_end", "}") or "}"

    return (
        "\n\n--- Dynamic Prompts Wildcards (MUST be preserved verbatim) ---\n"
        "The user's prompt may contain Dynamic Prompts wildcards that will be "
        "expanded AFTER your enhancement. You MUST keep every wildcard token "
        "exactly as written — do NOT expand, replace, or remove them.\n\n"
        f"  Wildcard syntax: {wc_wrap}<name>{wc_wrap}  "
        "(e.g. __style__, __artist__)\n"
        f"  Variant syntax:  {v_start}option1|option2{v_end}  "
        f"(e.g. {v_start}red|blue|green{v_end})\n\n"
        "Your job is to enhance the NON-wildcard parts of the prompt while "
        f"leaving all {wc_wrap}...{wc_wrap} and {v_start}...{v_end} tokens untouched.\n"
    )


def _effective_system_prompt(
    preset_content: str,
    *,
    width: int = 0,
    height: int = 0,
) -> str:
    """Return the preset content with optional resolution and DP instructions appended."""
    result = preset_content

    # Append resolution info if dimensions provided
    if width > 0 and height > 0:
        result += _build_resolution_instruction(width, height)

    # Append Dynamic Prompts wildcard preservation note
    dp_instruction = _build_dp_wildcard_instruction()
    if dp_instruction:
        result += dp_instruction

    return result


class Script(scripts.Script):
    """Enhance positive prompts through a local LLM (llama-server)."""

    # Run AFTER sd-dynamic-prompts (priority 1000) so our enhanced prompt
    # is the FINAL value of p.all_prompts and isn't overwritten by DP.
    sorting_priority = 1001

    def title(self):
        return "LLama Server Enhance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        choices = list_presets()
        current = shared.opts.llama_enhance_preset
        value = current if current in choices else (choices[0] if choices else None)

        with InputAccordion(
            value=False,
            label="LLama Server Enhance",
            elem_id=self.elem_id("main-accordion"),
        ) as enable:
            preset_dropdown = gr.Dropdown(
                choices=choices,
                value=value,
                label="System prompt preset",
                info="Select a preset from presets/",
                interactive=True,
                elem_id=self.elem_id("preset"),
            )
            refresh_btn = gr.Button(
                value="\U0001f504",
                variant="tool",
                elem_id=self.elem_id("refresh-presets"),
            )
            enhance_mode = gr.Dropdown(
                choices=["Per image", "Once"],
                value="Per image",
                label="Enhance mode",
                info="Per image: enhance each prompt individually. Once: enhance one prompt and apply to all",
                interactive=True,
                elem_id=self.elem_id("enhance-mode"),
            )

            def on_refresh():
                choices = list_presets()
                return gr.update(
                    choices=choices,
                    value=choices[0] if choices else None,
                )

            refresh_btn.click(
                fn=on_refresh,
                inputs=[],
                outputs=[preset_dropdown],
            )

        return [enable, preset_dropdown, enhance_mode]

    def process(
        self,
        p: StableDiffusionProcessing,
        enable: bool,
        preset_name: str,
        enhance_mode: str,
    ):
        """Called once before any sampling. Enhance prompts here."""
        _log_to_file("=" * 72)
        _log_to_file(f"process() entered — enable={enable!r}, preset={preset_name!r}, mode={enhance_mode!r}")
        _log_to_file(f"  p.all_prompts (input)  = {p.all_prompts!r}")
        _log_to_file(f"  p.n_iter = {p.n_iter}, p.batch_size = {p.batch_size}")

        if not enable:
            _log_to_file("  → disabled, skipping")
            return

        if not preset_name or preset_name == "":
            _log_to_file("  → no preset selected, skipping")
            return

        preset_content = load_preset(preset_name)
        if not preset_content:
            logger.warning("Preset '%s' not found, skipping enhancement", preset_name)
            _log_to_file(f"  → preset '{preset_name}' not found, skipping")
            return

        # Build effective system prompt (preset + resolution + Dynamic Prompts wildcard note)
        system_prompt = _effective_system_prompt(
            preset_content,
            width=p.width,
            height=p.height,
        )
        _log_to_file(f"  system_prompt ({len(system_prompt)} chars): {system_prompt}")

        server_path = shared.opts.llama_enhance_server_path or "llama-server"
        model_path = shared.opts.llama_enhance_model_path
        if not model_path:
            logger.warning("Model path not set, skipping enhancement")
            _log_to_file("  → model path not set, skipping")
            return

        extra_flags = shared.opts.llama_enhance_extra_flags or ""

        # "Once" mode: enhance a single prompt and apply to all images
        if enhance_mode == "Once":
            original = p.all_prompts[0] if p.all_prompts else ""
            if not original.strip():
                _log_to_file("  → Once mode: original prompt empty, skipping")
                return

            _log_to_file(f"  → Once mode: enhancing prompt 0 ({len(original)} chars)")
            result = enhance_prompt(
                server_path=server_path,
                model_path=model_path,
                system_prompt=system_prompt,
                user_prompt=original,
                extra_flags=extra_flags,
            )

            if result:
                _log_to_file(f"  → Once mode: enhanced ({len(result)} chars): {result}")
                p.all_prompts = [result] * len(p.all_prompts)
                logger.info(
                    "Prompt enhanced (Once mode): %s → %s",
                    original[:60],
                    result[:60],
                )
                print(f"  Enhanced prompt (full): {result}")
            else:
                _log_to_file("  → Once mode: enhancement failed, keeping original")
                logger.info("Using original prompt (enhancement failed)")
                print("  Using original prompt (enhancement failed)")
            _log_to_file(f"  p.all_prompts (output) = {p.all_prompts!r}")
            return

        # "Per image" mode: start ONE server, send all prompts in parallel, collect all
        _log_to_file(f"  → Per image mode: {len(p.all_prompts)} prompt(s)")

        # Collect prompts that need enhancement (skip empty ones)
        prompts_to_enhance: list[tuple[int, str]] = []
        skipped_prompts: dict[int, str] = {}
        for idx, original in enumerate(p.all_prompts):
            if not original.strip():
                _log_to_file(f"  → prompt {idx}: empty, keeping as-is")
                skipped_prompts[idx] = original
            else:
                prompts_to_enhance.append((idx, original))

        # If nothing to enhance, keep originals
        if not prompts_to_enhance:
            _log_to_file("  → all prompts empty, skipping enhancement")
            _log_to_file(f"  p.all_prompts (output) = {p.all_prompts!r}")
            return

        # Start a single server instance
        port = _find_free_port()
        base_url = f"http://127.0.0.1:{port}"
        cmd = _build_command(server_path, model_path, port, extra_flags)

        logger.info("llama-server command: %s", " ".join(cmd))
        print(f"\n  Running llama-server (model: {Path(model_path).name}, port: {port}), {len(prompts_to_enhance)} prompt(s)...")

        # Windows: prevent console window popup
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

        server_proc: subprocess.Popen | None = None
        try:
            server_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creationflags,
            )
            logger.info("llama-server started (pid=%s, port=%s)", server_proc.pid, port)
            print(f"  Started llama-server (pid={server_proc.pid}, port={port})")
        except FileNotFoundError:
            logger.error("llama-server not found at: %s", server_path)
            print(f"  FAILED: llama-server not found at: {server_path}")
            _log_to_file(f"  → llama-server not found at {server_path}")
            _log_to_file(f"  p.all_prompts (output) = {p.all_prompts!r}")
            return
        except Exception:
            logger.exception("Failed to start llama-server")
            _log_to_file("  → failed to start llama-server")
            _log_to_file(f"  p.all_prompts (output) = {p.all_prompts!r}")
            return

        # Wait for server to become healthy
        print(f"  Waiting for server to start (timeout: {_SERVER_STARTUP_TIMEOUT}s)...")
        start = time.monotonic()

        if not _wait_for_server(base_url):
            elapsed = time.monotonic() - start
            print(f"  FAILED: Server did not become healthy within {_SERVER_STARTUP_TIMEOUT}s ({elapsed:.1f}s elapsed)")
            logger.error("Server did not become healthy within %ds", _SERVER_STARTUP_TIMEOUT)
            _log_to_file(f"  → server health check failed after {elapsed:.1f}s")
            _kill_server(server_proc)
            _log_to_file(f"  p.all_prompts (output) = {p.all_prompts!r}")
            return

        ready_time = time.monotonic()
        print(f"  Server ready on port {port} in {ready_time - start:.1f}s. Sending {len(prompts_to_enhance)} prompt(s) in parallel...")
        _log_to_file(f"  Server ready in {ready_time - start:.1f}s")

        # Send all prompts concurrently
        results = _batch_chat_completions(base_url, system_prompt, prompts_to_enhance)

        # Build enhanced prompt list preserving order
        result_map = {idx: content for idx, content in results}
        enhanced_prompts: list[str] = []
        for idx, original in enumerate(p.all_prompts):
            if idx in skipped_prompts:
                enhanced_prompts.append(original)
                continue

            result = result_map.get(idx)
            if result:
                _log_to_file(f"  → prompt {idx}: enhanced ({len(result)} chars): {result}")
                logger.info(
                    "Prompt %d enhanced: %s → %s",
                    idx + 1,
                    original[:60],
                    result[:60],
                )
                print(f"  Enhanced prompt {idx + 1} (full): {result}")
                enhanced_prompts.append(result)
            else:
                _log_to_file(f"  → prompt {idx}: enhancement failed, keeping original")
                logger.info("Prompt %d: using original (enhancement failed)", idx + 1)
                print(f"  Prompt {idx + 1}: using original (enhancement failed)")
                enhanced_prompts.append(original)

        # Kill the server
        _kill_server(server_proc)

        p.all_prompts = enhanced_prompts
        _log_to_file(f"  p.all_prompts (output) = {p.all_prompts!r}")


# Register settings callback on import
script_callbacks.on_ui_settings(_settings_module.on_ui_settings)
