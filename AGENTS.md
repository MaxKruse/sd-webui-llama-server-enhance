# Repository Guidelines

## Project Overview

Gradio extension for Stable Diffusion WebUI Forge (Forge Neo) that rewrites image-generation prompts with a local LLM. For each generation it spawns a transient llama.cpp `llama-server` (OpenAI-compatible HTTP API) on `127.0.0.1` at a random ephemeral port (49152-65535), sends chat-completion requests using an architecture-specific GGUF system-prompt preset, writes results back into the generation pipeline, then kills the server. Fully local - no cloud APIs, no persistent server.

## Architecture & Data Flow

- **Entry point**: `scripts/prompt_enhancer.py` defines `LlamaServerEnhance(scripts.Script)` with `sorting_priority = 999` and `scripts.AlwaysVisible`. The priority runs it *before* sd-dynamic-prompts (priority 1000) so raw DP wildcards/variants survive enhancement.
- **Import layout**: Forge adds the extension root to `sys.path`, so the sibling package `prompt_enhancer/` is imported as top-level (`from prompt_enhancer.llm import ...`). Only `scripts/prompt_enhancer.py` and `prompt_enhancer/settings.py` import host `modules.*` - the rest of the package is host-independent (stdlib + aiohttp), which is what makes `test_llm.py` runnable standalone.
- **Generation flow** (`scripts/prompt_enhancer.py` `process()`):
  1. `_free_gpu_memory()` - lazy Forge-only import of `backend.memory_management` (try/except so non-Forge installs degrade gracefully).
  2. `_spawn_healthy_server()` - build command, `Popen`, poll `/health`, discover model name via `GET /v1/models`, discover slot count via `GET /slots`, warm up the chat endpoint per slot.
  3. Enhance prompts: "Per image" mode = one server, positive prompts batched in **strict buckets of the slot count** (`_batch_chat_completions`, aiohttp; a whole bucket finishes before the next starts). With `vary_across_buckets=True` each bucket after the first gets the previous bucket's prompts (enhanced text, originals after failure) as a "do not repeat exactly, introduce variance" reference in the system prompt - so similar prompts across a multi-bucket batch don't collapse into clones. "Once" mode = single sequential request replicated across the batch. Negative prompts batched second, only for successfully enhanced positives, without a variation reference.
  4. Write back into `p.all_prompts` / `p.all_negative_prompts`; failures keep the original prompt per index (graceful degradation). Server is killed in all paths; `postprocess()` frees VRAM again.
- **Endpoints used** (all on `http://127.0.0.1:<port>`): `GET /health`, `GET /v1/models`, `GET /slots?model=`, `POST /v1/chat/completions` (`Authorization: Bearer no-key`, no inference params - server defaults).
- **DP interop**: `_escape_accidental_dp_syntax()` replaces `|` with `,` inside variant blocks the LLM newly invented; wildcards (`__x__`) always preserved; custom separators read defensively from DP settings (`shared.opts.dp_parser_variant_start` / `variant_end` / `wildcard_wrap`).
- **Preset resolution** (`prompt_enhancer/presets.py`): `presets/*.txt` files, name = filename stem (kebab-case). `Auto` mode maps `shared.opts.forge_preset` via `FORGE_TO_PRESET_MAP` (`flux`-`flux-dev`, `zit`-`z-image-turbo`, `anima`-`anima`), with checkpoint-substring overrides (`FLUX_CHECKPOINT_OVERRIDES`, e.g. `krea`-`flux-krea-dev`).
- **Debug log**: `enhance_debug.log` in the extension root, written via `_log_to_file()` under `_LOG_LOCK`, rotated to `enhance_debug.log.1` at 5MB; llama-server stderr streamed in as `[llama-server]` lines. Gitignored.

## Key Directories

- `scripts/` - Forge extension entry point (Forge loads `.py` files here).
- `prompt_enhancer/` - logic package: `llm.py` (server lifecycle + HTTP client), `settings.py` (UI options), `presets.py` (preset discovery/resolution).
- `presets/` - user-editable system-prompt `.txt` files, consumed verbatim.
- Extension root - `metadata.ini` manifest, `README.md`, standalone test scripts, `enhance_debug.log`.

## Development Commands

No build or packaging step. The extension is run by the host:

- Launch host (Windows): `webui.bat` from `C:\Dev\sd-webui-forge-neo` (auto-creates and activates `venv/`). Linux: `./webui.sh`.
- Standalone pipeline tests (from the extension dir):
  ```
  python test_llm.py --preset anima --model C:/models/my-model.gguf --dry-run
  python test_llm.py --preset anima --model C:/models/my-model.gguf --prompt "a cat"
  python test_llm.py --mode batch --preset anima --model C:/models/my-model.gguf --prompts "p1" "p2" "p3"
  python test_llm.py --server C:/tools/llamacpp/llama-server.exe --model C:/models/my-model.gguf --flags "-ngl 99 --temp 0.5" --timeout 60
  ```
  Live modes need the `llama-server` binary + a `.gguf` model; `--dry-run` only prints the built command.
- Logic tests: `python test_escape.py` (pure stdlib, exit code 0 on pass) and `python test_batching.py` (drives `_batch_chat_completions` against a fake local llama-server: bucket splitting, the strict bucket barrier, cross-bucket variance reference; needs the venv's aiohttp).
- In-UI verification: txt2img/img2img accordion "LLama Server Enhance" - "Test server" button (live spawn + health + small chat; blocked while a generation runs).

## Code Conventions & Common Patterns

- **A1111 script pattern**: `ui()` returns a list of Gradio controls positionally matched by `process(self, p, ...)` (e.g. `return [enable, enhance_mode, enhance_negative]`). The `InputAccordion` open state doubles as the enable flag.
- **Settings**: register via `shared.opts.add_option(key="llama_enhance_*", info=shared.OptionInfo(..., section=("llama_enhance", "LLama Server Enhance")))` in `prompt_enhancer/settings.py`; hook it with `script_callbacks.on_ui_settings(...)` at module import in `scripts/prompt_enhancer.py`. All config is Forge UI settings - the extension reads/sets no environment variables.
- **Defensive cross-extension access**: `getattr(shared.opts, "dp_parser_variant_start", "{") or "{"` - never hard-require other extensions' or Forge's options.
- **Error handling**: failures return a `ChatResult` with `.error` set rather than raising; `_is_transient_error()` classifies 408/429/5xx/network errors as retryable (max 5 attempts, backoff 3/6/12/24s) while deterministic 4xx fail fast.
- **Async patterns**: single requests use stdlib `urllib`; batch mode uses `aiohttp` + `asyncio.run()` with strict buckets sized to the server's parallel slot count (a bucket barrier, not a free-running semaphore).
- **Windows subprocess care (do not "simplify" away)**: `CREATE_NO_WINDOW` creation flags, and daemon pipe-drainer threads reading stdout/stderr into bounded 256KB buffers. Without draining, the Windows 4KB pipe buffer fills and the server's own logging threads block - `/health` stays 200 while chat hangs. Rationale documented inline in `prompt_enhancer/llm.py`.
- **Logging**: `logging.getLogger(__name__)` for the WebUI console, `print()` for user-facing progress lines, `_log_to_file()` for the rotating debug log on every branch decision.
- **Module style**: `from __future__ import annotations`, `__slots__` on result objects, `_`-prefixed private helpers, module-level `threading.Lock()` guards (`_LOG_LOCK`, `_TEST_LOCK`).
- **Presets**: plain `.txt`, kebab-case filenames, content is the raw system prompt; mapping tables live at module level in `prompt_enhancer/presets.py`.
- **README is the living source of truth** - update `README.md` when features, settings, or commands change.

## Important Files

- `scripts/prompt_enhancer.py` (~940 lines) - UI, generation pipeline, DP escaping, debug-log rotation, VRAM freeing.
- `prompt_enhancer/llm.py` (~830 lines) - `spawn_server`, `_wait_for_server`, `_discover_model_name`, `_discover_slot_count`, `_warmup_chat_endpoint`, `_chat_completion(_async)`, `_plan_buckets`, `_variation_reference_block`, `_batch_chat_completions` (strict buckets; `vary_across_buckets` passes the previous bucket's prompts as a variance reference), `chat_completion_request`, `enhance_prompt` (all-in-one convenience used by `test_llm.py`), `_kill_server`.
- `prompt_enhancer/settings.py` - the 5 options: `llama_enhance_server_path` (default `llama-server` from PATH), `llama_enhance_model_path`, `llama_enhance_timeout` (120s), `llama_enhance_startup_timeout` (60s, read via `_read_startup_timeout()` with the module constant as fallback), `llama_enhance_extra_flags`.
- `prompt_enhancer/presets.py` + `presets/*.txt` (`anima`, `flux-dev`, `flux-krea-dev`, `z-image-turbo`).
- `metadata.ini` - the only manifest (`[Extension] Name = ...`); there is no `requirements.txt`/`pyproject.toml`/`install.py`.
- `README.md` - install, settings table, preset mapping, test-script usage, reliability notes.

## Runtime/Tooling Preferences

- **Python 3.13.x** in the host Forge venv (the host warns on other versions; live bytecode is `cpython-313`). Run extension code with the venv's `python`, not a system one.
- **No package manager for this extension**: it has no manifest, no `install.py`, no lockfile. All dependencies come from the host environment (`gradio==4.40.0`, Forge `modules.*` API). `aiohttp` is an undeclared hard dependency of `prompt_enhancer/llm.py` - it must already exist in the venv.
- **External binary**: a llama.cpp `llama-server` executable on PATH or in the server-path setting; `llama-cpp-python`/torch are NOT used by this extension. GPU behavior depends entirely on user-provided flags (e.g. `-ngl 99`).
- **Host-side tooling does not cover this extension**: host `pyproject.toml` configures ruff with `exclude = ["extensions"]` (and black line-length 1024) - no lint gate applies here. Keep the host's black line-length (1024) in mind when editing.
- **Git**: the extension dir is its own repository (remote `MaxKruse/sd-webui-llama-server-enhance`, branch `main`) nested inside the Forge repo - not a submodule. Commit granularly with Conventional Commits on short feature branches; keep `.gitignore` coverage for `enhance_debug.log*` and `__pycache__/`.

## Testing & QA

- **No formal test infrastructure**: no pytest suite, no CI, no coverage tooling. Do not run bare `pytest` in the extension dir - `test_llm.py`'s helper functions get collected and fail without live-model args.
- `test_escape.py` - deterministic logic tests (~40 assertions) for `_escape_accidental_dp_syntax`. It **duplicates the production function inline** (because `scripts/prompt_enhancer.py` cannot be imported outside the host) - when you change that function in production, you MUST update the copy in `test_escape.py` to match, or the tests silently drift.
- `test_batching.py` - deterministic tests (~34 assertions) for the batch pipeline, run against a fake local llama-server (stdlib HTTP, real aiohttp client, no model needed): `_plan_buckets` splitting, the strict bucket barrier (no request of bucket N+1 starts before bucket N finishes), the cross-bucket variance reference (bucket 1's enhanced prompts appear in bucket 2's system prompt; a failed slot falls back to its original prompt; later buckets reference the immediately previous bucket), and the default no-reference behavior.
- `test_llm.py` - live integration harness: `--dry-run` needs nothing; single/batch modes spawn a real `llama-server` with a real `.gguf` and kill it after. Batch mode mirrors production: slot discovery, warmup, strict buckets, `--concurrency` to override the bucket size. It imports only `prompt_enhancer.llm`/`presets` (plus transitive aiohttp).
- **Verification workflow for changes**: `python test_escape.py` for escaping logic; `python test_batching.py` for batch pipeline logic; `python test_llm.py ... --dry-run` for command construction; live `test_llm.py` run and/or the in-UI "Test server" button for end-to-end. Watch `enhance_debug.log` for pipeline decisions and streamed server stderr.
