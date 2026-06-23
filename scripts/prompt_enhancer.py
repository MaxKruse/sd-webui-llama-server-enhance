"""AlwaysVisible script that enhances prompts via llama-server.

This file was modified with the assistance of an LLM.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import gc

import gradio as gr
import modules.scripts as scripts
from modules import script_callbacks, shared
from modules.ui_components import InputAccordion
from modules.processing import StableDiffusionProcessing

from prompt_enhancer import settings as _settings_module
from prompt_enhancer.llm import (
    ChatResult,
    _SERVER_STARTUP_TIMEOUT,
    _build_command,
    _discover_model_name,
    _discover_slot_count,
    _find_free_port,
    _wait_for_server,
    _warmup_chat_endpoint,
    _batch_chat_completions,
    _kill_server,
    enhance_prompt,
)
from prompt_enhancer.presets import AUTO_PRESET, load_preset, resolve_preset

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


def _free_gpu_memory():
    """Unload all SD models from GPU to free VRAM for llama-server."""
    from backend import memory_management

    memory_management.unload_all_models()
    memory_management.soft_empty_cache(force=True)
    gc.collect()
    _log_to_file("  GPU memory freed for llama-server")


def _dynamic_prompts_installed() -> bool:
    """Check if the sd-dynamic-prompts extension is present."""
    return (_EXTENSIONS_DIR / "sd-dynamic-prompts").is_dir()


def _get_effective_dimensions(p) -> tuple[int, int]:
    """Get the final output dimensions, accounting for hires fix.

    Returns (width, height) of the final rendered image.
    Mirrors the logic in StableDiffusionProcessingTxt2Img.calculate_target_resolution().
    """
    if not getattr(p, "enable_hr", False):
        return (p.width, p.height)

    hr_resize_x = getattr(p, "hr_resize_x", 0)
    hr_resize_y = getattr(p, "hr_resize_y", 0)

    if hr_resize_x == 0 and hr_resize_y == 0:
        hr_scale = getattr(p, "hr_scale", 2.0)
        return (round(p.width * hr_scale), round(p.height * hr_scale))

    if hr_resize_y == 0:
        return (hr_resize_x, round(hr_resize_x * (p.height / p.width)))
    elif hr_resize_x == 0:
        return (round(hr_resize_y * (p.width / p.height)), hr_resize_y)

    return (hr_resize_x, hr_resize_y)


def _build_resolution_instruction(width: int, height: int, *, hires: bool = False, base_width: int = 0, base_height: int = 0) -> str:
    """Build a resolution hint block for the LLM system prompt.

    Tells the LLM the target image dimensions and orientation so it can
    frame compositions, aspect ratios, and layout descriptions appropriately.
    Includes hires fix info when applicable.
    """
    gcd = __import__("math").gcd(width, height)
    ratio_w, ratio_h = width // gcd, height // gcd
    orientation = "portrait" if height > width else "landscape" if width > height else "square"

    block = (
        f"\n\n--- Image Resolution ---\n"
        f"Final output resolution: {width}x{height} ({orientation}, {ratio_w}:{ratio_h} aspect ratio)\n"
        f"Frame your description to suit this orientation."
    )

    if hires and base_width > 0 and base_height > 0:
        base_gcd = __import__("math").gcd(base_width, base_height)
        base_ratio_w, base_ratio_h = base_width // base_gcd, base_height // base_gcd
        block += (
            f"\nBase generation resolution: {base_width}x{base_height} ({base_ratio_w}:{base_ratio_h}, hires fix upscaling to final)"
        )

    return block


def _build_negative_prompt_instruction() -> str:
    """Build a negative-prompt-enhancement instruction block for the LLM system prompt.

    Appended to the preset when the user checks 'Also enhance negative prompt'.
    The LLM receives the enhanced positive prompt for context and the original
    negative prompt to enhance.
    """
    return (
        "\n\n--- Negative Prompt Enhancement ---\n"
        "You are now enhancing a NEGATIVE prompt. The user message contains:\n"
        "  - 'POSITIVE: ' followed by the already-enhanced positive prompt (for context only)\n"
        "  - 'NEGATIVE: ' followed by the original negative prompt to enhance\n\n"
        "Your job: enhance the negative prompt to best complement the positive prompt.\n"
        "Take the positive prompt into account — target unwanted elements that would"
        "conflict with the desired output.\n\n"
        "Output ONLY the enhanced negative prompt as a single line of text.\n"
        "Do not include the positive prompt, explanations, or any other commentary.\n"
    )


def _build_negative_user_prompt(positive: str, negative: str) -> str:
    """Format a user message for the negative-prompt-enhancement call."""
    return f"POSITIVE: {positive}\n\nNEGATIVE: {negative}"


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


def _build_base_system_prompt(
    preset_content: str,
    *,
    width: int = 0,
    height: int = 0,
    hires: bool = False,
    base_width: int = 0,
    base_height: int = 0,
) -> str:
    """Return the preset content with optional resolution and DP instructions appended.

    This is the shared base used for BOTH positive and negative enhancement calls.
    """
    result = preset_content

    # Append resolution info if dimensions provided
    if width > 0 and height > 0:
        result += _build_resolution_instruction(
            width, height,
            hires=hires,
            base_width=base_width,
            base_height=base_height,
        )

    # Append Dynamic Prompts wildcard preservation note
    dp_instruction = _build_dp_wildcard_instruction()
    if dp_instruction:
        result += dp_instruction

    return result


def _effective_system_prompt(
    preset_content: str,
    *,
    width: int = 0,
    height: int = 0,
    hires: bool = False,
    base_width: int = 0,
    base_height: int = 0,
) -> str:
    """System prompt for the positive-prompt enhancement call."""
    return _build_base_system_prompt(
        preset_content,
        width=width,
        height=height,
        hires=hires,
        base_width=base_width,
        base_height=base_height,
    )


def _effective_negative_system_prompt(
    preset_content: str,
    *,
    width: int = 0,
    height: int = 0,
    hires: bool = False,
    base_width: int = 0,
    base_height: int = 0,
) -> str:
    """System prompt for the negative-prompt enhancement call.

    Same base as positive, plus the negative-enhancement instruction block.
    """
    return _build_base_system_prompt(
        preset_content,
        width=width,
        height=height,
        hires=hires,
        base_width=base_width,
        base_height=base_height,
    ) + _build_negative_prompt_instruction()


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
        with InputAccordion(
            value=False,
            label="LLama Server Enhance",
            elem_id=self.elem_id("main-accordion"),
        ) as enable:
            gr.HTML(
                value="<small>Preset is auto-selected based on the Forge-Neo UI preset "
                "(flux→flux-dev, zit→z-image-turbo, anima→anima).</small>",
                elem_id=self.elem_id("preset-info"),
            )
            enhance_mode = gr.Dropdown(
                choices=["Per image", "Once"],
                value="Per image",
                label="Enhance mode",
                info="Per image: enhance each prompt individually. Once: enhance one prompt and apply to all",
                interactive=True,
                elem_id=self.elem_id("enhance-mode"),
            )
            enhance_negative = gr.Checkbox(
                value=False,
                label="Also enhance negative prompt",
                info="Ask the LLM to enhance the negative prompt too (uses the preset's negative prompt guidance)",
                interactive=True,
                elem_id=self.elem_id("enhance-negative"),
            )

        return [enable, enhance_mode, enhance_negative]

    def process(
        self,
        p: StableDiffusionProcessing,
        enable: bool,
        enhance_mode: str,
        enhance_negative: bool,
    ):
        """Called once before any sampling. Enhance prompts here."""
        _log_to_file("=" * 72)
        _log_to_file(f"process() entered — enable={enable!r}, mode={enhance_mode!r}, enhance_negative={enhance_negative!r}")
        _log_to_file(f"  p.all_prompts (input)  = {p.all_prompts!r}")
        _log_to_file(f"  p.all_negative_prompts (input) = {p.all_negative_prompts!r}")
        _log_to_file(f"  p.n_iter = {p.n_iter}, p.batch_size = {p.batch_size}")

        if not enable:
            _log_to_file("  → disabled, skipping")
            return

        # Free GPU memory before starting llama-server so it has room for its model
        _free_gpu_memory()

        # Resolve preset — always auto-selected based on Forge-Neo UI preset
        forge_preset = getattr(shared.opts, "forge_preset", None)
        checkpoint_name = getattr(shared.opts, "sd_model_checkpoint", None)
        resolved = resolve_preset(
            AUTO_PRESET,
            forge_preset=forge_preset,
            checkpoint_name=checkpoint_name,
        )
        _log_to_file(f"  forge_preset={forge_preset!r}, checkpoint={checkpoint_name!r} → resolved={resolved!r}")

        if not resolved:
            _log_to_file("  → no preset resolved, skipping")
            return

        preset_content = load_preset(resolved)
        if not preset_content:
            logger.warning("Preset '%s' not found, skipping enhancement", resolved)
            _log_to_file(f"  → preset '{resolved}' not found, skipping")
            return

        # Determine effective output dimensions (final after hires fix)
        hires_enabled = getattr(p, "enable_hr", False)
        final_w, final_h = _get_effective_dimensions(p)
        base_w, base_h = p.width, p.height

        _log_to_file(
            f"  dimensions: base={base_w}x{base_h}, "
            f"hires={hires_enabled}, final={final_w}x{final_h}"
        )

        # Build system prompts (preset + resolution + Dynamic Prompts wildcard note)
        system_prompt = _effective_system_prompt(
            preset_content,
            width=final_w,
            height=final_h,
            hires=hires_enabled,
            base_width=base_w,
            base_height=base_h,
        )
        negative_system_prompt = _effective_negative_system_prompt(
            preset_content,
            width=final_w,
            height=final_h,
            hires=hires_enabled,
            base_width=base_w,
            base_height=base_h,
        ) if enhance_negative else None
        _log_to_file(f"  system_prompt ({len(system_prompt)} chars): {system_prompt[:200]}...")
        if negative_system_prompt:
            _log_to_file(f"  negative_system_prompt ({len(negative_system_prompt)} chars): {negative_system_prompt[:200]}...")

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

            original_negative = p.all_negative_prompts[0] if p.all_negative_prompts else ""

            # Step 1: Enhance positive prompt
            _log_to_file(f"  → Once mode: enhancing positive ({len(original)} chars)")
            pos_result: ChatResult = enhance_prompt(
                server_path=server_path,
                model_path=model_path,
                system_prompt=system_prompt,
                user_prompt=original,
                extra_flags=extra_flags,
            )

            if not pos_result.content:
                _log_to_file("  → Once mode: positive enhancement failed, keeping original")
                logger.info("Using original prompt (enhancement failed)")
                print("  Using original prompt (enhancement failed)")
                _log_to_file(f"  p.all_prompts (output) = {p.all_prompts!r}")
                _log_to_file(f"  p.all_negative_prompts (output) = {p.all_negative_prompts!r}")
                return

            enhanced_positive = pos_result.content
            _log_to_file(
                f"  → Once mode: positive enhanced ({len(enhanced_positive)} chars, "
                f"{pos_result.completion_tokens} tokens, {pos_result.generation_tps:.1f} tok/s, "
                f"{pos_result.total_ms:.0f}ms total)"
            )
            print(f"  Enhanced prompt (full): {enhanced_positive}")

            # Step 2: Enhance negative prompt (if enabled)
            enhanced_negative = original_negative
            if enhance_negative:
                _log_to_file(f"  → Once mode: enhancing negative ({len(original_negative)} chars)")
                neg_user_prompt = _build_negative_user_prompt(enhanced_positive, original_negative)
                neg_result: ChatResult = enhance_prompt(
                    server_path=server_path,
                    model_path=model_path,
                    system_prompt=negative_system_prompt,
                    user_prompt=neg_user_prompt,
                    extra_flags=extra_flags,
                )

                if neg_result.content:
                    enhanced_negative = neg_result.content
                    _log_to_file(
                        f"  → Once mode: negative enhanced ({len(enhanced_negative)} chars, "
                        f"{neg_result.completion_tokens} tokens, {neg_result.generation_tps:.1f} tok/s, "
                        f"{neg_result.total_ms:.0f}ms total)"
                    )
                    print(f"  Enhanced negative (full): {enhanced_negative}")

            p.all_prompts = [enhanced_positive] * len(p.all_prompts)
            p.all_negative_prompts = [enhanced_negative] * len(p.all_negative_prompts)
            logger.info("Prompt enhanced (Once mode): %s → %s", original[:60], enhanced_positive[:60])
            logger.info("Negative: %s → %s", original_negative[:60], enhanced_negative[:60])
            _log_to_file(f"  p.all_prompts (output) = {p.all_prompts!r}")
            _log_to_file(f"  p.all_negative_prompts (output) = {p.all_negative_prompts!r}")
            return

        # "Per image" mode: start ONE server, batch positives, then batch negatives
        _log_to_file(f"  → Per image mode: {len(p.all_prompts)} prompt(s)")

        # Collect prompts that need enhancement (skip empty ones)
        prompts_to_enhance: list[tuple[int, str]] = []
        skipped_indices: set[int] = set()
        for idx, original in enumerate(p.all_prompts):
            if not original.strip():
                _log_to_file(f"  → prompt {idx}: empty, keeping as-is")
                skipped_indices.add(idx)
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

        # Discover the loaded model name
        model_name = _discover_model_name(base_url)
        _log_to_file(f"  Discovered model: {model_name}")

        # Discover how many parallel slots the server has
        slot_count = _discover_slot_count(base_url, model_name)
        _log_to_file(f"  Discovered {slot_count} slot(s)")

        print(
            f"  Server ready on port {port} in {ready_time - start:.1f}s. "
            f"Model: {model_name}, Slots: {slot_count}"
        )
        _log_to_file(f"  Server ready in {ready_time - start:.1f}s")

        # Warm up the chat endpoint — /health returns 200 before parallel slots
        # are initialized, causing HTTP 400 on the first real request.
        # Warm up exactly as many slots as the server has.
        _warmup_chat_endpoint(base_url, model_name, num_slots=slot_count)

        # --- Step 1: Batch enhance positive prompts ---
        print(
            f"  Sending {len(prompts_to_enhance)} positive prompt(s) "
            f"({slot_count} concurrent) ..."
        )
        pos_results = _batch_chat_completions(
            base_url, model_name, system_prompt, prompts_to_enhance,
            max_concurrent=slot_count,
        )
        pos_result_map: dict[int, ChatResult] = {idx: result for idx, result in pos_results}

        # Build enhanced positive list + track which indices succeeded
        enhanced_prompts: list[str] = [""] * len(p.all_prompts)
        successful_indices: list[int] = []

        for idx, original in enumerate(p.all_prompts):
            if idx in skipped_indices:
                enhanced_prompts[idx] = original
                continue

            result = pos_result_map.get(idx)
            if result and result.content:
                _log_to_file(
                    f"  → prompt {idx}: positive enhanced ({len(result.content)} chars, "
                    f"{result.completion_tokens} tokens, {result.generation_tps:.1f} tok/s, "
                    f"{result.total_ms:.0f}ms total)"
                )
                logger.info("Prompt %d enhanced: %s → %s", idx + 1, original[:60], result.content[:60])
                print(f"  Enhanced prompt {idx + 1} (full): {result.content}")
                enhanced_prompts[idx] = result.content
                successful_indices.append(idx)
            else:
                error_detail = result.error if result and result.error else "no result"
                _log_to_file(f"  → prompt {idx}: positive enhancement failed ({error_detail}), keeping original")
                logger.info("Prompt %d: using original (enhancement failed: %s)", idx + 1, error_detail)
                print(f"  Prompt {idx + 1}: using original (enhancement failed: {error_detail})")
                enhanced_prompts[idx] = original

        # --- Step 2: Batch enhance negative prompts (if enabled) ---
        # Start with original negatives as baseline (pad with empty strings if needed)
        enhanced_negatives: list[str] = list(p.all_negative_prompts)
        while len(enhanced_negatives) < len(enhanced_prompts):
            enhanced_negatives.append("")

        if enhance_negative and successful_indices:
            neg_prompts: list[tuple[int, str]] = []
            for idx in successful_indices:
                original_negative = p.all_negative_prompts[idx] if idx < len(p.all_negative_prompts) else ""
                neg_user_prompt = _build_negative_user_prompt(enhanced_prompts[idx], original_negative)
                neg_prompts.append((idx, neg_user_prompt))

            print(
                f"  Sending {len(neg_prompts)} negative prompt(s) "
                f"({slot_count} concurrent) ..."
            )
            neg_results = _batch_chat_completions(
                base_url, model_name, negative_system_prompt, neg_prompts,
                max_concurrent=slot_count,
            )
            neg_result_map: dict[int, ChatResult] = {idx: result for idx, result in neg_results}

            for idx in successful_indices:
                original_negative = p.all_negative_prompts[idx] if idx < len(p.all_negative_prompts) else ""
                result = neg_result_map.get(idx)
                if result and result.content:
                    _log_to_file(
                        f"  → prompt {idx}: negative enhanced ({len(result.content)} chars, "
                        f"{result.completion_tokens} tokens, {result.generation_tps:.1f} tok/s, "
                        f"{result.total_ms:.0f}ms total)"
                    )
                    logger.info("Negative %d enhanced: %s → %s", idx + 1, original_negative[:60], result.content[:60])
                    print(f"  Enhanced negative {idx + 1} (full): {result.content}")
                    enhanced_negatives[idx] = result.content
                # else: keep original negative (already set as baseline)

        # Kill the server and capture stderr for diagnostics
        _kill_server(server_proc)
        if server_proc and server_proc.stderr:
            try:
                stderr_output = server_proc.stderr.read().decode("utf-8", errors="replace")
                stderr_tail = stderr_output[-3000:]
                if stderr_tail.strip():
                    _log_to_file(f"  llama-server stderr (last 3000 chars):\n{stderr_tail}")
            except Exception:
                pass

        p.all_prompts = enhanced_prompts
        p.all_negative_prompts = enhanced_negatives
        _log_to_file(f"  p.all_prompts (output) = {p.all_prompts!r}")
        _log_to_file(f"  p.all_negative_prompts (output) = {p.all_negative_prompts!r}")

    def postprocess(self, p, processed, enable, *args):
        """Called after all processing ends. Unload SD models to free VRAM."""
        if not enable:
            return
        _log_to_file("postprocess() — unloading SD models to free VRAM")
        _free_gpu_memory()


# Register settings callback on import
script_callbacks.on_ui_settings(_settings_module.on_ui_settings)
