#!/usr/bin/env python3
"""Standalone test for the llama-server prompt enhancement pipeline.

Usage:
    python test_llm.py [OPTIONS]

Examples:
    # Single-prompt test (default)
    python test_llm.py --preset anima --model C:/models/my-model.gguf

    # Batch test with multiple prompts
    python test_llm.py --mode batch --preset anima --model C:/models/my-model.gguf \\
        --prompts "a cat on a windowsill" "sunset over mountains" "cyberpunk city street"

    # Batch test reading prompts from a file (one per line)
    python test_llm.py --mode batch --preset anima --model C:/models/my-model.gguf \\
        --prompts-file prompts.txt

    # Inline system prompt
    python test_llm.py --system "You are a prompt enhancer." --model C:/models/my-model.gguf

    # Custom flags
    python test_llm.py --preset anima --model C:/models/my-model.gguf --flags "-ngl 99 --temp 0.5"

    # Dry run (show command only, don't execute)
    python test_llm.py --preset anima --model C:/models/my-model.gguf --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

# Resolve paths relative to this script's directory
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from prompt_enhancer.llm import (
    _SERVER_STARTUP_TIMEOUT,
    _build_command,
    _find_free_port,
    _kill_server,
    _wait_for_server,
    _batch_chat_completions,
    enhance_prompt,
)
from prompt_enhancer.presets import list_presets, load_preset, PRESETS_DIR


def _print_header(text: str):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def _print_section(title: str, value: str, *, indent: int = 2):
    pad = " " * indent
    lines = value.split("\n")
    print(f"\n{pad}{title}:")
    for line in lines:
        print(f"{pad}  {line}")


def test_preset_loading():
    _print_header("Preset Discovery")

    presets = list_presets()
    print(f"\nPresets directory: {PRESETS_DIR}")
    print(f"Found {len(presets)} preset(s): {presets or '(none)'}")

    if not presets:
        print("\n  No presets found. Create a .txt file in the presets/ directory.")
        print(f"  e.g.  {PRESETS_DIR / 'default.txt'}")
        return None

    return presets


def test_command_build(server_path: str, model_path: str, extra_flags: str):
    _print_header("Command Construction")

    port = random.randint(49152, 65535)  # dummy port for dry-run display
    cmd = _build_command(server_path, model_path, port, extra_flags)

    print(f"\n  Binary:        {cmd[0]}")
    print(f"  Model:         {cmd[2]}")
    print(f"  Extra flags:   {extra_flags or '(none)'}")

    print(f"\n  Full command (shell):")
    print(f"    {' '.join(cmd)}")

    print(f"\n  Full command (JSON):")
    print(f"    {json.dumps(cmd, indent=4)}")

    return cmd


def test_binary_exists(server_path: str):
    _print_header("Binary Check")

    # Try as absolute path first, then PATH lookup
    p = Path(server_path)
    if p.is_file():
        print(f"  Found: {p.resolve()}")
        return True

    import shutil
    found = shutil.which(server_path)
    if found:
        print(f"  Found via PATH: {found}")
        return True

    print(f"  NOT FOUND: {server_path}")
    print(f"  Check that llama-server exists at that path or is in your PATH.")
    return False


def test_model_exists(model_path: str):
    _print_header("Model Check")

    p = Path(model_path)
    if p.is_file():
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  Found: {p.resolve()} ({size_mb:.1f} MB)")
        return True

    print(f"  NOT FOUND: {model_path}")
    return False


def test_single(server_path: str, model_path: str, system_prompt: str, user_prompt: str, extra_flags: str):
    """Single-prompt mode: start server, send one prompt, kill server."""
    _print_header("Live Inference Test (Single Prompt)")

    print(f"  Running llama-server...")
    start = time.monotonic()

    result = enhance_prompt(
        server_path=server_path,
        model_path=model_path,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        extra_flags=extra_flags,
    )

    elapsed = time.monotonic() - start

    if result:
        print(f"  Success in {elapsed:.1f}s")
        _print_section("Enhanced prompt", result)
    else:
        print(f"\n  FAILED after {elapsed:.1f}s — check logs above for details.")

    return result


def _start_server(server_path: str, model_path: str, extra_flags: str):
    """Start a llama-server and return (base_url, server_proc) or (None, None) on failure."""
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    cmd = _build_command(server_path, model_path, port, extra_flags)

    print(f"\n  Running llama-server (model: {Path(model_path).name}, port: {port})...")

    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    try:
        server_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        print(f"  Started llama-server (pid={server_proc.pid}, port={port})")
    except FileNotFoundError:
        print(f"  FAILED: llama-server not found at: {server_path}")
        return None, None
    except Exception:
        print("  FAILED: could not start llama-server")
        return None, None

    print(f"  Waiting for server to start (timeout: {_SERVER_STARTUP_TIMEOUT}s)...")
    start = time.monotonic()

    if not _wait_for_server(base_url):
        elapsed = time.monotonic() - start
        print(f"  FAILED: Server did not become healthy within {_SERVER_STARTUP_TIMEOUT}s ({elapsed:.1f}s elapsed)")
        _kill_server(server_proc)
        return None, None

    ready_time = time.monotonic()
    print(f"  Server ready on port {port} in {ready_time - start:.1f}s")
    return base_url, server_proc


def test_batch(
    server_path: str,
    model_path: str,
    system_prompt: str,
    user_prompts: list[str],
    extra_flags: str,
):
    """Batch mode: start ONE server, send ALL prompts in parallel, collect all, kill."""
    _print_header(f"Live Inference Test (Batch — {len(user_prompts)} prompt(s))")

    # Start a single server
    base_url, server_proc = _start_server(server_path, model_path, extra_flags)
    if not base_url:
        return None

    # Build indexed prompt list
    prompts_to_enhance = list(enumerate(user_prompts))

    # Send all prompts concurrently
    print(f"  Sending {len(prompts_to_enhance)} prompt(s) in parallel...")
    start = time.monotonic()
    results = _batch_chat_completions(base_url, system_prompt, prompts_to_enhance)
    elapsed = time.monotonic() - start

    # Kill the server
    _kill_server(server_proc)

    # Display results
    result_map = {idx: content for idx, content in results}
    success_count = 0
    for idx, original in enumerate(user_prompts):
        content = result_map.get(idx)
        print(f"\n  --- Prompt {idx + 1} ---")
        print(f"  Input:  {original}")
        if content:
            print(f"  Output: {content}")
            print(f"  ({len(content)} chars)")
            success_count += 1
        else:
            print("  Output: (FAILED)")

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Results: {success_count}/{len(user_prompts)} succeeded")

    return result_map


def main():
    parser = argparse.ArgumentParser(
        description="Test the llama-server prompt enhancement pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["single", "batch"],
        default="single",
        help="Test mode: single (one prompt, default) or batch (multiple prompts, parallel)",
    )
    parser.add_argument("--preset", type=str, help="Preset name (without .txt)")
    parser.add_argument("--system", type=str, help="Inline system prompt (overrides --preset)")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a cat sitting on a windowsill",
        help="Test user prompt (single mode, or first prompt in batch mode)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        help="Additional prompts for batch mode (combined with --prompt)",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        help="File with one prompt per line for batch mode (appended to --prompt / --prompts)",
    )
    parser.add_argument("--model", type=str, help="Path to .gguf model file")
    parser.add_argument("--server", type=str, default="llama-server", help="Path to llama-server binary")
    parser.add_argument("--flags", type=str, default="", help="Extra inference flags for llama-server")
    parser.add_argument("--dry-run", action="store_true", help="Build command but don't execute")

    args = parser.parse_args()

    # --- Preset loading ---
    presets = test_preset_loading()

    # Resolve system prompt
    system_prompt = args.system
    if not system_prompt and args.preset:
        system_prompt = load_preset(args.preset)
        if not system_prompt:
            print(f"\nError: Preset '{args.preset}' not found.")
            if presets:
                print(f"Available presets: {presets}")
            sys.exit(1)
    elif not system_prompt:
        system_prompt = "You are a Stable Diffusion prompt enhancer. Expand the given prompt with vivid visual details. Output ONLY the enhanced prompt."
        print(f"\n  (No preset or --system given, using fallback)")

    _print_section("Active system prompt", system_prompt)

    # --- Collect prompts ---
    user_prompts = [args.prompt]
    if args.prompts:
        user_prompts.extend(args.prompts)
    if args.prompts_file:
        pf = Path(args.prompts_file)
        if pf.is_file():
            file_prompts = [line.strip() for line in pf.read_text(encoding="utf-8").splitlines() if line.strip()]
            user_prompts.extend(file_prompts)
            print(f"\n  Loaded {len(file_prompts)} prompt(s) from {args.prompts_file}")
        else:
            print(f"\nWarning: prompts file not found: {args.prompts_file}")

    if args.mode == "batch":
        _print_header(f"Batch Mode — {len(user_prompts)} prompt(s)")
    else:
        _print_header("Single Mode")

    for idx, p in enumerate(user_prompts):
        print(f"  [{idx + 1}] {p}")

    # --- Command construction ---
    cmd = test_command_build(
        server_path=args.server,
        model_path=args.model or "",
        extra_flags=args.flags,
    )

    if args.dry_run:
        _print_header("Dry Run — Stopping here")
        print("\n  Command built successfully. Use without --dry-run to execute.")
        return

    # --- Pre-flight checks ---
    if not test_binary_exists(args.server):
        print("\n  Skipping live test (binary not found).")
        return

    if not args.model:
        print("\n  Skipping live test (--model not provided).")
        return

    if not test_model_exists(args.model):
        print("\n  Skipping live test (model not found).")
        return

    # --- Live inference ---
    if args.mode == "batch":
        test_batch(
            server_path=args.server,
            model_path=args.model,
            system_prompt=system_prompt,
            user_prompts=user_prompts,
            extra_flags=args.flags,
        )
    else:
        test_single(
            server_path=args.server,
            model_path=args.model,
            system_prompt=system_prompt,
            user_prompt=user_prompts[0],
            extra_flags=args.flags,
        )

    _print_header("All tests complete")


if __name__ == "__main__":
    main()
