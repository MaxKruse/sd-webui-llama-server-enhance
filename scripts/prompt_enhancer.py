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
from prompt_enhancer.client import chat_completion_request
from prompt_enhancer.flow import (
    BUCKET_SIZE_AUTO,
    BUCKET_SIZE_CHOICES,
    FlowContext,
    run_once,
    run_per_image,
)
from prompt_enhancer.logfile import RotatingLog
from prompt_enhancer.negative import build_negative_instruction
from prompt_enhancer.prompts import get_effective_dimensions
from prompt_enhancer.server import _ServerProcess, _discover_model_name, _kill_server
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
        # The accordion's open state doubles as the enable flag: a closed
        # accordion silently disables the extension - tell users explicitly.
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
            self._build_test_controls()

        return [enable, enhance_mode, bucket_size, enhance_negative]

    def _build_test_controls(self) -> None:
        """Test-server button + result box, inside the accordion.

        Validates the configured server (live spawn + health + small chat)
        without generating anything; blocked while a generation runs.
        """
        test_button = gr.Button("Test server", variant="secondary")
        test_output = gr.Textbox(
            label="Test result",
            interactive=False,
            lines=6,
            elem_id=self.elem_id("test-result"),
        )
        test_button.click(fn=self._test_server, outputs=[test_output])

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
        ok, lines = self._check_server_prereqs()
        if not ok:
            return "\n".join(lines)

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

    def _check_server_prereqs(self) -> tuple[bool, list[str]]:
        """Check the configured binary and model file. Returns (ok, lines):
        a non-empty lines list is the full user-facing result on failure."""
        server_path = shared.opts.llama_enhance_server_path or "llama-server"
        model_path = shared.opts.llama_enhance_model_path or ""

        # 1. Binary check
        binary = str(Path(server_path).resolve()) if Path(server_path).is_file() else shutil.which(server_path)
        if not binary:
            msg = f"Binary: FAILED - {server_path!r} not found. Set 'llama-server path' in Settings -> LLama Server Enhance."
            _DEBUG_LOG.write(f"  → {msg}")
            return False, [msg]
        _DEBUG_LOG.write(f"  binary: {binary}")

        # 2. Model file check
        if not model_path or not Path(model_path).is_file():
            msg = f"Model: FAILED - {model_path or '<not set>'!r} not found. Set 'Model path' in Settings -> LLama Server Enhance."
            _DEBUG_LOG.write(f"  → {msg}")
            return False, [f"Binary: OK ({binary})", msg]
        _DEBUG_LOG.write(f"  model: {model_path}")
        return True, [f"Binary: OK ({binary})", f"Model: OK ({Path(model_path).name})"]

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

        # Resolve preset + model path, build the flow context, run the mode
        resolved = self._resolve_preset(enhance_negative)
        if resolved is None:
            return
        preset_content, skip_negatives = resolved

        ctx = self._make_flow_ctx(p, bucket_size, enhance_negative, skip_negatives, preset_content)
        if enhance_mode == "Once":
            new_prompts, new_negatives = run_once(ctx, p.all_prompts, p.all_negative_prompts)
        else:
            new_prompts, new_negatives = run_per_image(ctx, p.all_prompts, p.all_negative_prompts)

        if new_prompts is not None:
            p.all_prompts = new_prompts
            p.all_negative_prompts = new_negatives
        self._log_output_state(p)

    def _resolve_preset(self, enhance_negative: bool) -> tuple[str, bool] | None:
        """Resolve the LLM preset from the Forge-Neo UI preset and validate
        the model path. Returns (preset_content, skip_negatives), or None
        (with a console skip notice) when enhancement cannot run."""
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
            return None

        preset_content = load_preset(resolved)
        if not preset_content:
            print(f"  Llama Enhance: skipping - preset '{resolved}' not found (look in the presets/ directory)")
            logger.warning("Preset '%s' not found, skipping enhancement", resolved)
            _DEBUG_LOG.write(f"  → preset '{resolved}' not found, skipping")
            return None

        # Some presets (FLUX / Z-Image) have no negative-prompt conditioning;
        # enhancing a negative prompt against them would contradict the preset.
        skip_negatives = negatives_ignored(resolved)
        if enhance_negative and skip_negatives:
            print(f"  Llama Enhance: negative enhancement skipped - preset '{resolved}' ignores negative prompts (model has no negative-prompt conditioning)")
            _DEBUG_LOG.write(f"  → negative enhancement skipped: preset '{resolved}' ignores negative prompts")

        if not _opts.read_model_path():
            print("  Llama Enhance: skipping - no model path set (Settings > LLama Server Enhance > Model path)")
            logger.warning("Model path not set, skipping enhancement")
            _DEBUG_LOG.write("  → model path not set, skipping")
            return None

        return preset_content, skip_negatives

    def _build_generation_prompts(
        self, p: StableDiffusionProcessing, preset_content: str, enhance_negative: bool
    ) -> tuple[str, str | None]:
        """Build the (positive, negative) system prompts for this generation:
        preset + resolution context + DP wildcard note."""
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
        return system_prompt, negative_system_prompt

    def _make_flow_ctx(
        self,
        p: StableDiffusionProcessing,
        bucket_size: str,
        enhance_negative: bool,
        skip_negatives: bool,
        preset_content: str,
    ) -> FlowContext:
        """Wire the FlowContext: settings-backed spawn, DP separators, sinks."""
        system_prompt, negative_system_prompt = self._build_generation_prompts(
            p, preset_content, enhance_negative
        )
        v_start, v_end, wc_wrap = _opts.read_dp_separators()
        return FlowContext(
            system_prompt=system_prompt,
            negative_system_prompt=negative_system_prompt,
            request_timeout=_opts.read_request_timeout(),
            bucket_size=bucket_size,
            enhance_negative=enhance_negative,
            skip_negatives=skip_negatives,
            v_start=v_start,
            v_end=v_end,
            wc_wrap=wc_wrap,
            spawn=lambda label="": _spawn_healthy_server(label=label),
            kill=_kill_server,
            log=_DEBUG_LOG.write,
            out=print,
            info=logger.info,
        )

    def _log_output_state(self, p) -> None:
        """Log the prompt lists that reach the generation pipeline."""
        _DEBUG_LOG.write(f"  p.all_prompts (output) = {p.all_prompts!r}")
        _DEBUG_LOG.write(f"  p.all_negative_prompts (output) = {p.all_negative_prompts!r}")

    def postprocess(self, p, processed, enable, *args):
        """Called after all processing ends. Unload SD models to free VRAM."""
        if not enable:
            return
        _DEBUG_LOG.write("postprocess() — unloading SD models to free VRAM")
        _free_gpu_memory()


# Register settings callback on import
script_callbacks.on_ui_settings(_settings_module.on_ui_settings)
