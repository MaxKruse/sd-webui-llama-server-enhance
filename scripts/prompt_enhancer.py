"""AlwaysVisible script that enhances prompts via llama-server.

This file was modified with the assistance of an LLM.
"""

from __future__ import annotations

import gc
import logging
import shutil
import threading
import time
from pathlib import Path

import gradio as gr
import modules.scripts as scripts
from modules import script_callbacks, shared
from modules.ui_components import InputAccordion
from modules.processing import StableDiffusionProcessing

from prompt_enhancer import settings as _settings_module
from prompt_enhancer import opts as _opts
from prompt_enhancer.batching import _batch_chat_completions
from prompt_enhancer.client import (
    ChatResult,
    chat_completion_request,
)
from prompt_enhancer.escape import escape_accidental_dp_syntax
from prompt_enhancer.logfile import RotatingLog
from prompt_enhancer.negative import (
    build_negative_instruction,
    build_negative_user_prompt,
    validate_negative_output,
)
from prompt_enhancer.prompts import get_effective_dimensions
from prompt_enhancer.server import (
    _ServerProcess,
    _discover_model_name,
    _discover_slot_count,
    _kill_server,
    _warmup_chat_endpoint,
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

# Debug log - written to the extension root so it's easy to find.
# Rotates to enhance_debug.log.1 at 5MB (thread-safe: the WebUI main thread
# and the server's stderr pipe drainer write concurrently).
_LOG_DIR = Path(__file__).resolve().parent.parent
_LOG_FILE = _LOG_DIR / "enhance_debug.log"
_DEBUG_LOG = RotatingLog(_LOG_FILE)

# Guards the UI "Test server" button against concurrent runs.
_TEST_LOCK = threading.Lock()


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


def _spawn_healthy_server(label: str = ""):
    """Start llama-server (settings + debug log wired in) and wait for health."""
    return _opts.spawn_healthy_server(_DEBUG_LOG.write, label=label)


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
        _DEBUG_LOG.write("  GPU memory freed for llama-server")
    except Exception:
        logger.warning("Could not unload SD models from GPU (non-Forge install?)", exc_info=True)
        _DEBUG_LOG.write("  WARNING: could not free GPU memory (backend module unavailable)")


def _escape_accidental_dp_syntax(original: str, enhanced: str) -> str:
    """Escape variant pipes in LLM-generated text that wasn't in the original prompt.

    Reads the user's DP separators (variant_start, variant_end, wildcard_wrap)
    and delegates to prompt_enhancer.escape.
    """
    v_start = getattr(shared.opts, "dp_parser_variant_start", "{") or "{"
    v_end = getattr(shared.opts, "dp_parser_variant_end", "}") or "}"
    wc_wrap = getattr(shared.opts, "dp_parser_wildcard_wrap", "__") or "__"
    return escape_accidental_dp_syntax(original, enhanced, v_start=v_start, v_end=v_end, wc_wrap=wc_wrap)


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
        # The accordion's open state doubles as the enable flag (A1111
        # InputAccordion pattern): tell users explicitly, because a closed
        # accordion silently disables the extension for the next generation.
        gr.HTML(
            value="<small>LLM enhancement runs while this accordion is <b>open</b>. "
            "Closing it disables the extension for the next generation.</small>",
            elem_id=self.elem_id("enable-hint"),
        )
        with InputAccordion(
            value=False,
            label="LLama Server Enhance",
            elem_id=self.elem_id("main-accordion"),
        ) as enable:
            gr.HTML(
                value="<small>Preset is auto-selected from the Forge-Neo UI preset: "
                "flux→flux-dev (krea checkpoints→flux-krea-dev), zit→z-image-turbo, "
                "anima→anima. Other Forge presets have no LLM preset - enhancement is skipped.</small>",
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
        _DEBUG_LOG.write("=" * 72)
        _DEBUG_LOG.write("Test server requested from UI")

        if getattr(shared.state, "job", ""):
            msg = "A generation is in progress - please wait for it to finish before testing."
            _DEBUG_LOG.write(f"  → {msg}")
            return msg

        if not _TEST_LOCK.acquire(blocking=False):
            msg = "A server test is already running - please wait for it to finish."
            _DEBUG_LOG.write(f"  → {msg}")
            return msg

        try:
            return self._run_server_test()
        except Exception as exc:
            logger.exception("Server test failed")
            _DEBUG_LOG.write(f"  → test crashed: {exc}")
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
            _DEBUG_LOG.write(f"  → {msg}")
            return msg
        lines.append(f"Binary: OK ({binary})")
        _DEBUG_LOG.write(f"  binary: {binary}")

        # 2. Model file check
        if not model_path or not Path(model_path).is_file():
            msg = f"Model: FAILED - {model_path or '<not set>'!r} not found. Set 'Model path' in Settings -> LLama Server Enhance."
            _DEBUG_LOG.write(f"  → {msg}")
            return "\n".join([*lines, msg])
        lines.append(f"Model: OK ({Path(model_path).name})")
        _DEBUG_LOG.write(f"  model: {model_path}")

        # 3. Live test: spawn, health check, small chat request
        _DEBUG_LOG.write("  → live test: starting server")
        _free_gpu_memory()
        sp = _spawn_healthy_server(label=" (test)")
        if sp is None:
            msg = f"Server: FAILED - did not become healthy. See {_LOG_FILE} for the server's stderr."
            _DEBUG_LOG.write(f"  → {msg}")
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
        _DEBUG_LOG.write(f"  model id: {model_name}")

        t0 = time.monotonic()
        result = chat_completion_request(
            sp,
            "Reply with exactly one word: OK",
            "ping",
            request_timeout=_opts.read_request_timeout(),
            label="Test reply",
        )
        elapsed = time.monotonic() - t0
        if result.content:
            lines.append(
                f"Chat: OK ({elapsed:.1f}s, {result.completion_tokens} tokens, "
                f"{result.generation_tps:.1f} tok/s, reply: {result.content[:80]!r})"
            )
            _DEBUG_LOG.write(f"  chat OK in {elapsed:.1f}s: {result.content[:200]!r}")
        else:
            lines.append(f"Chat: FAILED - {result.error}")
            _DEBUG_LOG.write(f"  → chat failed: {result.error}")

    def process(
        self,
        p: StableDiffusionProcessing,
        enable: bool,
        enhance_mode: str,
        bucket_size: str,
        enhance_negative: bool,
    ):
        """Called once before any sampling. Enhance prompts here."""
        _DEBUG_LOG.write("=" * 72)
        _DEBUG_LOG.write(f"process() entered — enable={enable!r}, mode={enhance_mode!r}, bucket_size={bucket_size!r}, enhance_negative={enhance_negative!r}")
        _DEBUG_LOG.write(f"  p.all_prompts (input)  = {p.all_prompts!r}")
        _DEBUG_LOG.write(f"  p.all_negative_prompts (input) = {p.all_negative_prompts!r}")
        _DEBUG_LOG.write(f"  p.n_iter = {p.n_iter}, p.batch_size = {p.batch_size}")

        if not enable:
            _DEBUG_LOG.write("  → disabled, skipping")
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
        _DEBUG_LOG.write(f"  forge_preset={forge_preset!r}, checkpoint={checkpoint_name!r} → resolved={resolved!r}")

        if not resolved:
            mapped = ", ".join(sorted(set(FORGE_TO_PRESET_MAP.values())))
            print(f"  Llama Enhance: skipping - Forge UI preset '{forge_preset}' has no LLM preset (mapped: {mapped})")
            logger.warning(
                "LLama Enhance: skipping - Forge UI preset %r has no LLM preset (mapped: %s)",
                forge_preset, mapped,
            )
            _DEBUG_LOG.write(f"  → no preset resolved for forge_preset={forge_preset!r}, skipping")
            return

        preset_content = load_preset(resolved)
        if not preset_content:
            print(f"  Llama Enhance: skipping - preset '{resolved}' not found (look in the presets/ directory)")
            logger.warning("Preset '%s' not found, skipping enhancement", resolved)
            _DEBUG_LOG.write(f"  → preset '{resolved}' not found, skipping")
            return

        # Some presets (FLUX / Z-Image) have no negative-prompt conditioning;
        # enhancing a negative prompt against them would contradict the preset.
        skip_negatives = negatives_ignored(resolved)
        if enhance_negative and skip_negatives:
            print(f"  Llama Enhance: negative enhancement skipped - preset '{resolved}' ignores negative prompts (model has no negative-prompt conditioning)")
            _DEBUG_LOG.write(f"  → negative enhancement skipped: preset '{resolved}' ignores negative prompts")

        # Determine effective output dimensions (final after hires fix)
        hires_enabled = getattr(p, "enable_hr", False)
        final_w, final_h = get_effective_dimensions(p)
        base_w, base_h = p.width, p.height

        _DEBUG_LOG.write(
            f"  dimensions: base={base_w}x{base_h}, "
            f"hires={hires_enabled}, final={final_w}x{final_h}"
        )

        # Build system prompts (preset + resolution + Dynamic Prompts wildcard note)
        system_prompt, negative_system_prompt = _opts.build_system_prompts(
            preset_content,
            width=final_w,
            height=final_h,
            hires=hires_enabled,
            base_width=base_w,
            base_height=base_h,
            extensions_dir=_EXTENSIONS_DIR,
            negative_instruction=build_negative_instruction() if enhance_negative else None,
        )
        _DEBUG_LOG.write(f"  system_prompt ({len(system_prompt)} chars): {system_prompt[:200]}...")
        if negative_system_prompt:
            _DEBUG_LOG.write(f"  negative_system_prompt ({len(negative_system_prompt)} chars): {negative_system_prompt[:200]}...")

        model_path = _opts.read_model_path()
        if not model_path:
            print("  Llama Enhance: skipping - no model path set (Settings > LLama Server Enhance > Model path)")
            logger.warning("Model path not set, skipping enhancement")
            _DEBUG_LOG.write("  → model path not set, skipping")
            return

        request_timeout = _opts.read_request_timeout()
        # "Once" mode: enhance a single prompt and apply to all images.
        # One shared server serves both the positive and negative requests,
        # so the model is loaded only once.
        if enhance_mode == "Once":
            original = p.all_prompts[0] if p.all_prompts else ""
            if not original.strip():
                _DEBUG_LOG.write("  → Once mode: original prompt empty, skipping")
                return

            original_negative = p.all_negative_prompts[0] if p.all_negative_prompts else ""

            sp = _spawn_healthy_server()
            if sp is None:
                _DEBUG_LOG.write(f"  p.all_prompts (output) = {p.all_prompts!r}")
                _DEBUG_LOG.write(f"  p.all_negative_prompts (output) = {p.all_negative_prompts!r}")
                return

            try:
                # Step 1: Enhance positive prompt
                _DEBUG_LOG.write(f"  → Once mode: enhancing positive ({len(original)} chars)")
                pos_result: ChatResult = chat_completion_request(
                    sp, system_prompt, original,
                    request_timeout=request_timeout,
                )

                if not pos_result.content:
                    _DEBUG_LOG.write("  → Once mode: positive enhancement failed, keeping original")
                    logger.info("Using original prompt (enhancement failed)")
                    print("  Using original prompt (enhancement failed)")
                    _DEBUG_LOG.write(f"  p.all_prompts (output) = {p.all_prompts!r}")
                    _DEBUG_LOG.write(f"  p.all_negative_prompts (output) = {p.all_negative_prompts!r}")
                    return

                enhanced_positive = pos_result.content

                # Escape accidental DP syntax in the enhanced prompt
                enhanced_positive = _escape_accidental_dp_syntax(original, enhanced_positive)

                _DEBUG_LOG.write(
                    f"  → Once mode: positive enhanced ({len(enhanced_positive)} chars, "
                    f"{pos_result.completion_tokens} tokens, {pos_result.generation_tps:.1f} tok/s, "
                    f"{pos_result.total_ms:.0f}ms total)"
                )
                print(f"  Enhanced prompt (full): {enhanced_positive}")

                # Step 2: Enhance negative prompt (if enabled and the preset
                # actually uses negative prompts)
                enhanced_negative = original_negative
                if enhance_negative and not skip_negatives:
                    _DEBUG_LOG.write(f"  → Once mode: enhancing negative ({len(original_negative)} chars)")
                    neg_user_prompt = build_negative_user_prompt(enhanced_positive, original_negative)
                    neg_result: ChatResult = chat_completion_request(
                        sp, negative_system_prompt, neg_user_prompt,
                        request_timeout=request_timeout,
                        label="Enhanced negative prompt",
                    )

                    if neg_result.content:
                        cleaned_negative, reject_reason = validate_negative_output(
                            enhanced_positive, original_negative, neg_result.content
                        )
                        if cleaned_negative is None:
                            _DEBUG_LOG.write(f"  → Once mode: negative rejected ({reject_reason}), keeping original")
                            print(f"  Negative enhancement rejected ({reject_reason}) - keeping original")
                        else:
                            # Escape accidental DP syntax in the enhanced negative prompt
                            enhanced_negative = _escape_accidental_dp_syntax(original_negative, cleaned_negative)

                            _DEBUG_LOG.write(
                                f"  → Once mode: negative enhanced ({len(enhanced_negative)} chars, "
                                f"{neg_result.completion_tokens} tokens, {neg_result.generation_tps:.1f} tok/s, "
                                f"{neg_result.total_ms:.0f}ms total)"
                            )
                            print(f"  Enhanced negative (full): {enhanced_negative}")

                p.all_prompts = [enhanced_positive] * len(p.all_prompts)
                p.all_negative_prompts = [enhanced_negative] * len(p.all_negative_prompts)
                logger.info("Prompt enhanced (Once mode): %s → %s", original[:60], enhanced_positive[:60])
                logger.info("Negative: %s → %s", original_negative[:60], enhanced_negative[:60])
                _DEBUG_LOG.write(f"  p.all_prompts (output) = {p.all_prompts!r}")
                _DEBUG_LOG.write(f"  p.all_negative_prompts (output) = {p.all_negative_prompts!r}")
            finally:
                _kill_server(sp)
            # Every Once-mode path exits here (failure paths return inside the
            # try). Without this return the success path would fall through
            # into Per-image mode and re-enhance the already-enhanced prompts
            # with a second server.
            return

        # "Per image" mode: start ONE server, batch positives, then batch negatives
        _DEBUG_LOG.write(f"  → Per image mode: {len(p.all_prompts)} prompt(s)")

        # Collect prompts that need enhancement (skip empty ones)
        prompts_to_enhance: list[tuple[int, str]] = []
        skipped_indices: set[int] = set()
        for idx, original in enumerate(p.all_prompts):
            if not original.strip():
                _DEBUG_LOG.write(f"  → prompt {idx}: empty, keeping as-is")
                skipped_indices.add(idx)
            else:
                prompts_to_enhance.append((idx, original))

        # If nothing to enhance, keep originals
        if not prompts_to_enhance:
            _DEBUG_LOG.write("  → all prompts empty, skipping enhancement")
            _DEBUG_LOG.write(f"  p.all_prompts (output) = {p.all_prompts!r}")
            return

        # Start a single server instance (the helper drains the pipes,
        # streams stderr into the debug log, and waits for health)
        sp = _spawn_healthy_server(
            label=f", {len(prompts_to_enhance)} prompt(s)",
        )
        if sp is None:
            _DEBUG_LOG.write(f"  p.all_prompts (output) = {p.all_prompts!r}")
            return

        base_url = sp.base_url

        try:
            # Discover the loaded model name
            model_name = _discover_model_name(base_url)
            _DEBUG_LOG.write(f"  Discovered model: {model_name}")

            # Discover how many parallel slots the server has
            slot_count = _discover_slot_count(base_url, model_name)
            _DEBUG_LOG.write(f"  Discovered {slot_count} slot(s)")

            # Resolve the effective bucket size (max concurrent LLM requests
            # per bucket). The dropdown's Auto choice maps to the slot count.
            bucket = _parse_bucket_size(bucket_size)
            if bucket == 0:
                bucket = slot_count
                _DEBUG_LOG.write(f"  bucket size: auto → {slot_count} (server slots)")
            else:
                suffix = f" (server has {slot_count} slots - requests will queue)" if bucket > slot_count else ""
                _DEBUG_LOG.write(f"  bucket size: user choice {bucket}{suffix}")

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
            _DEBUG_LOG.write(f"  → positive batch: {len(prompts_to_enhance)} prompt(s) in {num_buckets} bucket(s) of {bucket}, cross-bucket variance enabled")
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
                    _DEBUG_LOG.write(
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
                    _DEBUG_LOG.write(f"  → prompt {idx}: positive enhancement failed ({error_detail}), keeping original")
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
                    neg_user_prompt = build_negative_user_prompt(enhanced_prompts[idx], original_negative)
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
                        cleaned_negative, reject_reason = validate_negative_output(
                            enhanced_prompts[idx], original_negative, result.content
                        )
                        if cleaned_negative is None:
                            _DEBUG_LOG.write(f"  → prompt {idx}: negative rejected ({reject_reason}), keeping original")
                            print(f"  Negative {idx + 1} rejected ({reject_reason}) - keeping original")
                        else:
                            _DEBUG_LOG.write(
                                f"  → prompt {idx}: negative enhanced ({len(cleaned_negative)} chars, "
                                f"{result.completion_tokens} tokens, {result.generation_tps:.1f} tok/s, "
                                f"{result.total_ms:.0f}ms total)"
                            )

                            # Escape accidental DP syntax in the enhanced negative prompt
                            escaped_negative = _escape_accidental_dp_syntax(original_negative, cleaned_negative)

                            logger.info("Negative %d enhanced: %s → %s", idx + 1, original_negative[:60], escaped_negative[:60])
                            print(f"  Enhanced negative {idx + 1} (full): {escaped_negative}")
                            enhanced_negatives[idx] = escaped_negative
                    # else: keep original negative (already set as baseline)

            p.all_prompts = enhanced_prompts
            p.all_negative_prompts = enhanced_negatives
            _DEBUG_LOG.write(f"  p.all_prompts (output) = {p.all_prompts!r}")
            _DEBUG_LOG.write(f"  p.all_negative_prompts (output) = {p.all_negative_prompts!r}")
        finally:
            # Always kill the server: an unexpected exception above must not
            # leak a GPU-resident llama-server (or its port) until restart.
            _kill_server(sp)
            stderr_tail = sp.stderr_tail(3000)
            if stderr_tail.strip():
                _DEBUG_LOG.write(f"  llama-server stderr (last 3000 chars):\n{stderr_tail}")

    def postprocess(self, p, processed, enable, *args):
        """Called after all processing ends. Unload SD models to free VRAM."""
        if not enable:
            return
        _DEBUG_LOG.write("postprocess() — unloading SD models to free VRAM")
        _free_gpu_memory()


# Register settings callback on import
script_callbacks.on_ui_settings(_settings_module.on_ui_settings)
