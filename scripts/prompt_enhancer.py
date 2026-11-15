"""AlwaysVisible script that enhances prompts via llama-server.

This file was modified with the assistance of an LLM.
"""

from __future__ import annotations

import logging
import shutil
import threading
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
    DEFAULT_REQUEST_TIMEOUT,
    _SERVER_STARTUP_TIMEOUT,
    _ServerProcess,
    _discover_model_name,
    _discover_slot_count,
    _warmup_chat_endpoint,
    _batch_chat_completions,
    _kill_server,
    chat_completion_request,
    spawn_healthy_server,
)
from prompt_enhancer.presets import (
    FORGE_TO_PRESET_MAP,
    load_preset,
    negatives_ignored,
    resolve_auto_preset,
)

logger = logging.getLogger(__name__)

# Resolve the extensions directory relative to this file
_EXTENSIONS_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Log file for debugging - written to the extension root so it's easy to find
_LOG_DIR = Path(__file__).resolve().parent.parent
_LOG_FILE = _LOG_DIR / "enhance_debug.log"
_LOG_BACKUP = _LOG_FILE.with_name("enhance_debug.log.1")

# Rotate the debug log once it grows past this size (bytes).
_LOG_MAX_BYTES = 5 * 1024 * 1024

# Guards the log file against concurrent writers (main thread + pipe drainer).
_LOG_LOCK = threading.Lock()

# Guards the UI "Test server" button against concurrent runs.
_TEST_LOCK = threading.Lock()


def _rotate_log_if_needed():
    """Rotate the debug log to enhance_debug.log.1 once it passes the size limit."""
    try:
        if _LOG_FILE.exists() and _LOG_FILE.stat().st_size > _LOG_MAX_BYTES:
            if _LOG_BACKUP.exists():
                _LOG_BACKUP.unlink()
            _LOG_FILE.rename(_LOG_BACKUP)
    except OSError:
        logger.warning("Could not rotate debug log", exc_info=True)


def _log_to_file(msg: str):
    """Append a timestamped message to the debug log file (thread-safe)."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] {msg}\n"
    with _LOG_LOCK:
        _rotate_log_if_needed()
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)


def _read_request_timeout() -> float:
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


def _read_startup_timeout() -> float:
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


# Bucket-size dropdown choices for "Per image" mode. The Auto choice means
# "use the server's parallel slot count" (discovered from /slots at run time).
BUCKET_SIZE_AUTO = "Auto (server slots)"
BUCKET_SIZE_CHOICES = [BUCKET_SIZE_AUTO, "1", "2", "3", "4", "5", "6", "8", "10", "16"]


def _parse_bucket_size(raw) -> int:
    """Parse the bucket-size dropdown value; returns 0 for auto/invalid."""
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return 0
    return max(1, value)


def _spawn_healthy_server(
    server_path: str,
    model_path: str,
    extra_flags: str,
    label: str = "",
) -> _ServerProcess | None:
    """Start llama-server and wait until it becomes healthy.

    Thin wrapper over the host-independent ``spawn_healthy_server`` that wires
    in the WebUI-specific bits: the user-configured startup timeout and the
    debug-log sinks (server stderr streamed in as ``[llama-server]`` lines,
    plus the same diagnostic lines the helper prints).
    """

    def _on_stderr(chunk: str):
        for line in chunk.rstrip("\n").split("\n"):
            if line.strip():
                _log_to_file(f"  [llama-server] {line}")

    return spawn_healthy_server(
        server_path,
        model_path,
        extra_flags,
        startup_timeout=_read_startup_timeout(),
        on_stderr=_on_stderr,
        on_debug=_log_to_file,
        label=label,
    )


def _free_gpu_memory():
    """Unload all SD models from GPU to free VRAM for llama-server.

    Best-effort: the backend module is Forge-specific, so on other WebUI forks
    this logs a warning instead of crashing the whole generation.
    """
    try:
        from backend import memory_management

        memory_management.unload_all_models()
        memory_management.soft_empty_cache(force=True)
        gc.collect()
        _log_to_file("  GPU memory freed for llama-server")
    except Exception:
        logger.warning("Could not unload SD models from GPU (non-Forge install?)", exc_info=True)
        _log_to_file("  WARNING: could not free GPU memory (backend module unavailable)")


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


def _escape_accidental_dp_syntax(original: str, enhanced: str) -> str:
    """Escape variant pipes in LLM-generated text that wasn't in the original prompt.

    When the LLM generates text containing DP variant syntax (e.g. {red|blue})
    that wasn't present in the user's original prompt, those pipes would be
    misinterpreted by the Dynamic Prompts parser. This function detects such
    accidental syntax and replaces the pipes with commas.

    Handles configurable separators (variant_start, variant_end, wildcard_wrap)
    read from shared.opts.

    Args:
        original: The user's original prompt (before enhancement).
        enhanced: The LLM-enhanced prompt.

    Returns:
        The enhanced prompt with accidental variant pipes escaped.
    """
    v_start = getattr(shared.opts, "dp_parser_variant_start", "{") or "{"
    v_end = getattr(shared.opts, "dp_parser_variant_end", "}") or "}"
    wc_wrap = getattr(shared.opts, "dp_parser_wildcard_wrap", "__") or "__"

    # Collect all original variant blocks for comparison
    original_variants: set[str] = set()
    _extract_variant_blocks(original, v_start, v_end, original_variants)

    result: list[str] = []
    i = 0
    while i < len(enhanced):
        # Check for variant start at position i
        if enhanced[i:i + len(v_start)] == v_start:
            # Find the matching end (handles nested variants)
            end_pos = _find_matching_variant_end(enhanced, i + len(v_start), v_start, v_end)
            if end_pos is not None:
                full_block = enhanced[i:end_pos + len(v_end)]
                inner = full_block[len(v_start):-len(v_end)]

                # Skip if the variant contains wildcard syntax (e.g. {__style__|realistic})
                # The LLM was instructed to preserve wildcards, so variants containing
                # them are intentional and should be kept as-is.
                if _contains_wildcard(inner, wc_wrap):
                    result.append(full_block)
                    i = end_pos + len(v_end)
                    continue

                # If this exact block was in the original, preserve it
                if full_block in original_variants:
                    result.append(full_block)
                    i = end_pos + len(v_end)
                    continue

                # Accidental DP syntax - escape pipes inside
                inner_escaped = inner.replace("|", ",")
                result.append(f"{v_start}{inner_escaped}{v_end}")
                i = end_pos + len(v_end)
                continue

        result.append(enhanced[i])
        i += 1

    return "".join(result)


def _contains_wildcard(text: str, wc_wrap: str) -> bool:
    """Check if text contains a wildcard token (wc_wrap + name + wc_wrap)."""
    if not wc_wrap:
        return False
    # Look for pattern: wc_wrap ... wc_wrap where there's at least one char between
    wc_len = len(wc_wrap)
    if wc_len * 2 + 1 > len(text):
        return False
    first = text.find(wc_wrap)
    if first < 0:
        return False
    second = text.find(wc_wrap, first + wc_len)
    return second > first + wc_len  # At least one char between the wraps


def _extract_variant_blocks(text: str, v_start: str, v_end: str, out: set[str]):
    """Extract all variant blocks from text and add them to the out set."""
    i = 0
    while i < len(text):
        if text[i:i + len(v_start)] == v_start:
            end_pos = _find_matching_variant_end(text, i + len(v_start), v_start, v_end)
            if end_pos is not None:
                full_block = text[i:end_pos + len(v_end)]
                out.add(full_block)
                i = end_pos + len(v_end)
                continue
        i += 1


def _find_matching_variant_end(text: str, start: int, v_start: str, v_end: str) -> int | None:
    """Find the position of the matching variant_end, accounting for nesting.

    Returns the index of the first character of variant_end, or None if not found.

    When v_start == v_end (e.g. both are '###'), we check v_end FIRST so the
    closing token is matched before the opening token, keeping depth correct.
    """
    same_delim = v_start == v_end
    depth = 1
    i = start
    limit = len(text) - len(v_end)
    while i <= limit:
        if same_delim:
            # When delimiters are identical, check end first so we close before opening
            if text[i:i + len(v_end)] == v_end:
                depth -= 1
                if depth == 0:
                    return i
                i += len(v_end)
                continue
            # Also acts as start (nesting)
            if text[i:i + len(v_start)] == v_start:
                depth += 1
                i += len(v_start)
                continue
        else:
            if text[i:i + len(v_start)] == v_start:
                depth += 1
                i += len(v_start)
                continue
            if text[i:i + len(v_end)] == v_end:
                depth -= 1
                if depth == 0:
                    return i
                i += len(v_end)
                continue
        i += 1
    return None


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
        base_width=base_width,
        base_height=base_height,
    ) + _build_negative_prompt_instruction()


class LlamaServerEnhance(scripts.Script):
    """Enhance positive prompts through a local LLM (llama-server)."""

    # NOTE: sorting_priority only orders this always-visible UI accordion
    # section - it does NOT control process() call order. Forge invokes
    # process() in extension discovery order (alphabetical), so
    # sd-dynamic-prompts normally expands wildcards BEFORE we enhance.
    # To enhance raw prompts first, pin this script above "Script" in
    # Settings -> "Script process callback priority".
    sorting_priority = 999

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
            bucket_size = gr.Dropdown(
                choices=BUCKET_SIZE_CHOICES,
                value=BUCKET_SIZE_AUTO,
                label="Bucket size (Per image mode)",
                info="Prompts per bucket = max concurrent LLM requests. Auto = the server's parallel slot count.",
                interactive=True,
                elem_id=self.elem_id("bucket-size"),
            )
            enhance_negative = gr.Checkbox(
                value=False,
                label="Also enhance negative prompt",
                info="Ask the LLM to enhance the negative prompt too (uses the preset's negative prompt guidance; no-op for flux-dev, flux-krea-dev, z-image-turbo - these models ignore negative prompts)",
                interactive=True,
                elem_id=self.elem_id("enhance-negative"),
            )

            def _on_mode_change(mode: str):
                return gr.update(visible=mode != "Once")

            enhance_mode.change(fn=_on_mode_change, inputs=[enhance_mode], outputs=[bucket_size])

            test_button = gr.Button("Test server", variant="secondary")
            test_output = gr.Textbox(
                label="Test result",
                interactive=False,
                lines=6,
                elem_id=self.elem_id("test-result"),
            )
            test_button.click(fn=self._test_server, outputs=[test_output])

        return [enable, enhance_mode, bucket_size, enhance_negative]

    def _test_server(self) -> str:
        """UI button handler: validate the configured server without generating."""
        _log_to_file("=" * 72)
        _log_to_file("Test server requested from UI")

        if getattr(shared.state, "job", ""):
            msg = "A generation is in progress - please wait for it to finish before testing."
            _log_to_file(f"  → {msg}")
            return msg

        if not _TEST_LOCK.acquire(blocking=False):
            msg = "A server test is already running - please wait for it to finish."
            _log_to_file(f"  → {msg}")
            return msg

        try:
            return self._run_server_test()
        except Exception as exc:
            logger.exception("Server test failed")
            _log_to_file(f"  → test crashed: {exc}")
            return f"FAILED: unexpected error: {exc}"
        finally:
            _TEST_LOCK.release()

    def _run_server_test(self) -> str:
        """Check binary + model, then run a live spawn/health/chat test."""
        server_path = shared.opts.llama_enhance_server_path or "llama-server"
        model_path = shared.opts.llama_enhance_model_path or ""
        extra_flags = shared.opts.llama_enhance_extra_flags or ""

        lines: list[str] = []

        # 1. Binary check
        binary = str(Path(server_path).resolve()) if Path(server_path).is_file() else shutil.which(server_path)
        if not binary:
            msg = f"Binary: FAILED - {server_path!r} not found. Set 'llama-server path' in Settings -> LLama Server Enhance."
            _log_to_file(f"  → {msg}")
            return msg
        lines.append(f"Binary: OK ({binary})")
        _log_to_file(f"  binary: {binary}")

        # 2. Model file check
        if not model_path or not Path(model_path).is_file():
            msg = f"Model: FAILED - {model_path or '<not set>'!r} not found. Set 'Model path' in Settings -> LLama Server Enhance."
            _log_to_file(f"  → {msg}")
            return "\n".join([*lines, msg])
        lines.append(f"Model: OK ({Path(model_path).name})")
        _log_to_file(f"  model: {model_path}")

        # 3. Live test: spawn, health check, small chat request
        _log_to_file("  → live test: starting server")
        _free_gpu_memory()
        sp = _spawn_healthy_server(server_path, model_path, extra_flags, label=" (test)")
        if sp is None:
            msg = "Server: FAILED - did not become healthy. See enhance_debug.log for the server's stderr."
            _log_to_file(f"  → {msg}")
            return "\n".join([*lines, msg])

        try:
            self._test_chat(sp, lines)
        finally:
            _kill_server(sp)

        return "\n".join(lines)

    def _test_chat(self, sp: _ServerProcess, lines: list[str]) -> None:
        """Run one small chat request against a healthy server, appending results."""
        model_name = _discover_model_name(sp.base_url)
        lines.append(f"Server: OK (model id: {model_name})")
        _log_to_file(f"  model id: {model_name}")

        t0 = time.monotonic()
        result = chat_completion_request(
            sp,
            "Reply with exactly one word: OK",
            "ping",
            request_timeout=_read_request_timeout(),
            label="Test reply",
        )
        elapsed = time.monotonic() - t0
        if result.content:
            lines.append(
                f"Chat: OK ({elapsed:.1f}s, {result.completion_tokens} tokens, "
                f"{result.generation_tps:.1f} tok/s, reply: {result.content[:80]!r})"
            )
            _log_to_file(f"  chat OK in {elapsed:.1f}s: {result.content[:200]!r}")
        else:
            lines.append(f"Chat: FAILED - {result.error}")
            _log_to_file(f"  → chat failed: {result.error}")

    def process(
        self,
        p: StableDiffusionProcessing,
        enable: bool,
        enhance_mode: str,
        bucket_size: str,
        enhance_negative: bool,
    ):
        """Called once before any sampling. Enhance prompts here."""
        _log_to_file("=" * 72)
        _log_to_file(f"process() entered — enable={enable!r}, mode={enhance_mode!r}, bucket_size={bucket_size!r}, enhance_negative={enhance_negative!r}")
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
        resolved = resolve_auto_preset(
            forge_preset=forge_preset,
            checkpoint_name=checkpoint_name,
        )
        _log_to_file(f"  forge_preset={forge_preset!r}, checkpoint={checkpoint_name!r} → resolved={resolved!r}")

        if not resolved:
            mapped = ", ".join(sorted(set(FORGE_TO_PRESET_MAP.values())))
            print(f"  Llama Enhance: skipping - Forge UI preset '{forge_preset}' has no LLM preset (mapped: {mapped})")
            logger.warning(
                "LLama Enhance: skipping - Forge UI preset %r has no LLM preset (mapped: %s)",
                forge_preset, mapped,
            )
            _log_to_file(f"  → no preset resolved for forge_preset={forge_preset!r}, skipping")
            return

        preset_content = load_preset(resolved)
        if not preset_content:
            print(f"  Llama Enhance: skipping - preset '{resolved}' not found (look in the presets/ directory)")
            logger.warning("Preset '%s' not found, skipping enhancement", resolved)
            _log_to_file(f"  → preset '{resolved}' not found, skipping")
            return

        # Some presets (FLUX / Z-Image) have no negative-prompt conditioning;
        # enhancing a negative prompt against them would contradict the preset.
        skip_negatives = negatives_ignored(resolved)
        if enhance_negative and skip_negatives:
            print(f"  Llama Enhance: negative enhancement skipped - preset '{resolved}' ignores negative prompts (model has no negative-prompt conditioning)")
            _log_to_file(f"  → negative enhancement skipped: preset '{resolved}' ignores negative prompts")

        # Determine effective output dimensions (final after hires fix)
        hires_enabled = getattr(p, "enable_hr", False)
        final_w, final_h = _get_effective_dimensions(p)
        base_w, base_h = p.width, p.height

        _log_to_file(
            f"  dimensions: base={base_w}x{base_h}, "
            f"hires={hires_enabled}, final={final_w}x{final_h}"
        )

        # Build system prompts (preset + resolution + Dynamic Prompts wildcard note)
        system_prompt = _build_base_system_prompt(
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
            print("  Llama Enhance: skipping - no model path set (Settings > LLama Server Enhance > Model path)")
            logger.warning("Model path not set, skipping enhancement")
            _log_to_file("  → model path not set, skipping")
            return

        extra_flags = shared.opts.llama_enhance_extra_flags or ""
        request_timeout = _read_request_timeout()
        # "Once" mode: enhance a single prompt and apply to all images.
        # One shared server serves both the positive and negative requests,
        # so the model is loaded only once.
        if enhance_mode == "Once":
            original = p.all_prompts[0] if p.all_prompts else ""
            if not original.strip():
                _log_to_file("  → Once mode: original prompt empty, skipping")
                return

            original_negative = p.all_negative_prompts[0] if p.all_negative_prompts else ""

            sp = _spawn_healthy_server(server_path, model_path, extra_flags)
            if sp is None:
                _log_to_file(f"  p.all_prompts (output) = {p.all_prompts!r}")
                _log_to_file(f"  p.all_negative_prompts (output) = {p.all_negative_prompts!r}")
                return

            try:
                # Step 1: Enhance positive prompt
                _log_to_file(f"  → Once mode: enhancing positive ({len(original)} chars)")
                pos_result: ChatResult = chat_completion_request(
                    sp, system_prompt, original,
                    request_timeout=request_timeout,
                )

                if not pos_result.content:
                    _log_to_file("  → Once mode: positive enhancement failed, keeping original")
                    logger.info("Using original prompt (enhancement failed)")
                    print("  Using original prompt (enhancement failed)")
                    _log_to_file(f"  p.all_prompts (output) = {p.all_prompts!r}")
                    _log_to_file(f"  p.all_negative_prompts (output) = {p.all_negative_prompts!r}")
                    return

                enhanced_positive = pos_result.content

                # Escape accidental DP syntax in the enhanced prompt
                enhanced_positive = _escape_accidental_dp_syntax(original, enhanced_positive)

                _log_to_file(
                    f"  → Once mode: positive enhanced ({len(enhanced_positive)} chars, "
                    f"{pos_result.completion_tokens} tokens, {pos_result.generation_tps:.1f} tok/s, "
                    f"{pos_result.total_ms:.0f}ms total)"
                )
                print(f"  Enhanced prompt (full): {enhanced_positive}")

                # Step 2: Enhance negative prompt (if enabled and the preset
                # actually uses negative prompts)
                enhanced_negative = original_negative
                if enhance_negative and not skip_negatives:
                    _log_to_file(f"  → Once mode: enhancing negative ({len(original_negative)} chars)")
                    neg_user_prompt = _build_negative_user_prompt(enhanced_positive, original_negative)
                    neg_result: ChatResult = chat_completion_request(
                        sp, negative_system_prompt, neg_user_prompt,
                        request_timeout=request_timeout,
                        label="Enhanced negative prompt",
                    )

                    if neg_result.content:
                        enhanced_negative = neg_result.content

                        # Escape accidental DP syntax in the enhanced negative prompt
                        enhanced_negative = _escape_accidental_dp_syntax(original_negative, enhanced_negative)

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
            finally:
                _kill_server(sp)
            # Every Once-mode path exits here (failure paths return inside the
            # try). Without this return the success path would fall through
            # into Per-image mode and re-enhance the already-enhanced prompts
            # with a second server.
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

        # Start a single server instance (the helper drains the pipes,
        # streams stderr into the debug log, and waits for health)
        sp = _spawn_healthy_server(
            server_path, model_path, extra_flags,
            label=f", {len(prompts_to_enhance)} prompt(s)",
        )
        if sp is None:
            _log_to_file(f"  p.all_prompts (output) = {p.all_prompts!r}")
            return

        base_url = sp.base_url

        try:
            # Discover the loaded model name
            model_name = _discover_model_name(base_url)
            _log_to_file(f"  Discovered model: {model_name}")

            # Discover how many parallel slots the server has
            slot_count = _discover_slot_count(base_url, model_name)
            _log_to_file(f"  Discovered {slot_count} slot(s)")

            # Resolve the effective bucket size (max concurrent LLM requests
            # per bucket). The dropdown's Auto choice maps to the slot count.
            bucket = _parse_bucket_size(bucket_size)
            if bucket == 0:
                bucket = slot_count
                _log_to_file(f"  bucket size: auto → {slot_count} (server slots)")
            else:
                suffix = f" (server has {slot_count} slots - requests will queue)" if bucket > slot_count else ""
                _log_to_file(f"  bucket size: user choice {bucket}{suffix}")

            print(f"  Model: {model_name}, Slots: {slot_count}, Bucket: {bucket}")

            # Warm up the chat endpoint — /health returns 200 before parallel slots
            # are initialized, causing HTTP 400 on the first real request.
            # Warm up exactly as many slots as the server has.
            _warmup_chat_endpoint(base_url, model_name, num_slots=slot_count)

            # --- Step 1: Batch enhance positive prompts ---
            # Prompts run in strict buckets of slot_count. Each bucket after the
            # first receives the previous bucket's prompts as a "do not repeat,
            # vary" reference so similar inputs don't collapse into clones.
            num_buckets = (len(prompts_to_enhance) + bucket - 1) // bucket
            print(
                f"  Sending {len(prompts_to_enhance)} positive prompt(s) "
                f"({bucket} concurrent, {num_buckets} bucket(s)) ..."
            )
            _log_to_file(f"  → positive batch: {len(prompts_to_enhance)} prompt(s) in {num_buckets} bucket(s) of {bucket}, cross-bucket variance enabled")
            pos_results = _batch_chat_completions(
                base_url, model_name, system_prompt, prompts_to_enhance,
                max_concurrent=bucket,
                request_timeout=request_timeout,
                vary_across_buckets=True,
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

                    # Escape accidental DP syntax in the enhanced prompt
                    escaped_content = _escape_accidental_dp_syntax(original, result.content)

                    logger.info("Prompt %d enhanced: %s → %s", idx + 1, original[:60], escaped_content[:60])
                    print(f"  Enhanced prompt {idx + 1} (full): {escaped_content}")
                    enhanced_prompts[idx] = escaped_content
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

            if enhance_negative and not skip_negatives and successful_indices:
                neg_prompts: list[tuple[int, str]] = []
                for idx in successful_indices:
                    original_negative = p.all_negative_prompts[idx] if idx < len(p.all_negative_prompts) else ""
                    neg_user_prompt = _build_negative_user_prompt(enhanced_prompts[idx], original_negative)
                    neg_prompts.append((idx, neg_user_prompt))

                print(
                    f"  Sending {len(neg_prompts)} negative prompt(s) "
                    f"({bucket} concurrent) ..."
                )
                neg_results = _batch_chat_completions(
                    base_url, model_name, negative_system_prompt, neg_prompts,
                    max_concurrent=bucket,
                    request_timeout=request_timeout,
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

                        # Escape accidental DP syntax in the enhanced negative prompt
                        escaped_negative = _escape_accidental_dp_syntax(original_negative, result.content)

                        logger.info("Negative %d enhanced: %s → %s", idx + 1, original_negative[:60], escaped_negative[:60])
                        print(f"  Enhanced negative {idx + 1} (full): {escaped_negative}")
                        enhanced_negatives[idx] = escaped_negative
                    # else: keep original negative (already set as baseline)

            p.all_prompts = enhanced_prompts
            p.all_negative_prompts = enhanced_negatives
            _log_to_file(f"  p.all_prompts (output) = {p.all_prompts!r}")
            _log_to_file(f"  p.all_negative_prompts (output) = {p.all_negative_prompts!r}")
        finally:
            # Always kill the server: an unexpected exception above must not
            # leak a GPU-resident llama-server (or its port) until restart.
            _kill_server(sp)
            stderr_tail = sp.stderr_tail(3000)
            if stderr_tail.strip():
                _log_to_file(f"  llama-server stderr (last 3000 chars):\n{stderr_tail}")

    def postprocess(self, p, processed, enable, *args):
        """Called after all processing ends. Unload SD models to free VRAM."""
        if not enable:
            return
        _log_to_file("postprocess() — unloading SD models to free VRAM")
        _free_gpu_memory()


# Register settings callback on import
script_callbacks.on_ui_settings(_settings_module.on_ui_settings)
