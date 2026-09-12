# sd-webui-llama-server-enhance

LLM-powered prompt enhancement for **Stable Diffusion WebUI Forge**, using [llama.cpp](https://github.com/ggerganov/llama.cpp)'s `llama-server` HTTP API.

Drop this extension into your `extensions/` folder and get per-image prompt enhancement powered by any local GGUF model - no cloud API keys needed.

## Features

- **Batch mode** - starts a single `llama-server` instance and sends all prompts in parallel, eliminating per-image startup penalties
- **Auto preset selection** - automatically selects the correct system prompt preset based on the Forge-Neo UI preset (flux, zit, anima)
- **Hires fix aware** - provides the LLM with final output dimensions (after hires fix upscaling) so it can frame compositions appropriately
- **Dynamic Prompts integration** - automatically escapes accidental DP syntax in LLM output; supports custom DP separators (configurable `variant_start`, `variant_end`, `wildcard_wrap`). See [Dynamic Prompts Integration](#dynamic-prompts-integration) for execution order
- **Two enhancement modes**:
  - **Per image** - enhance each prompt individually (batch-optimized: one server, strict buckets of the server's slot count, cross-bucket variation so similar prompts don't clone)
  - **Once** - enhance one prompt and apply it to all images in the batch
- **Test server button** - in the UI accordion: validates the binary, model file, and inference flags with a live spawn + health check + small chat request, without generating anything
- **OpenAI-compatible API** - works with any model that `llama-server` can load

## Installation

1. Clone or copy this repository into your WebUI Forge extensions folder:

   ```
   extensions/sd-webui-llama-server-enhance/
   ```

2. Ensure `llama-server` is available in your `PATH` or set the full path in WebUI settings.

3. Set your model path in **Settings -> LLama Server Enhance**.

4. Restart WebUI Forge.

## Configuration

All settings are in **Settings -> LLama Server Enhance**:

| Setting | Description |
|---------|-------------|
| **llama-server path** | Path to the `llama-server` binary (default: `llama-server` from PATH) |
| **Model path** | Full path to your `.gguf` model file |
| **Inference flags** | Extra flags passed to `llama-server` (e.g., `-ngl 99 --temp 0.8 --top-p 0.9`) |
| **Request timeout** | Seconds to wait for a single LLM request before retrying (default: `120`) |
| **Server startup timeout** | Seconds to wait for `llama-server` to load the model and become healthy (default: `60`) |

## Presets

Create `.txt` files in the `presets/` directory. Each file becomes an auto-selectable preset. The file content is used as the system prompt sent to the LLM.

The preset is automatically selected based on the Forge-Neo UI preset:

| Forge-Neo Preset | LLM Preset |
|------------------|------------|
| `flux` | `flux-dev` |
| `flux` (with "krea" checkpoint) | `flux-krea-dev` |
| `zit` | `z-image-turbo` |
| `anima` | `anima` |

Included presets:
- `presets/anima.txt` - for the [Anima](https://github.com/CircleStone-Labs/Anima) anime model. Enforces *scene completeness*: every enhanced output must include mood, lighting, environment, movement/pose, and character-environment interaction - the LLM invents fitting details for anything the user's prompt leaves unspecified. Runs *uncensored*: prompts with sexual or sensual intent are enhanced to be fully explicit (clothing removed, `nsfw, explicit` tags, direct anatomical description in the natural-language part), and the `safe` tag is never emitted in the positive - for adult content it is pushed into the negative alongside `sensitive` instead, because Anima's safety tags steer content in both directions
- `presets/flux-dev.txt` - for FLUX.1 [dev]
- `presets/flux-krea-dev.txt` - for FLUX.1 Krea [dev]
- `presets/z-image-turbo.txt` - for Z-Image Turbo

## Enhancement Modes

### Per image (default)

Each prompt in the batch is enhanced individually. The extension starts **one** `llama-server` instance and sends the positive prompts in **strict buckets** sized to the server's parallel slot count (e.g. 4 concurrent): a whole bucket must finish before the next bucket starts. When more than one bucket runs, each bucket after the first receives the previous bucket's prompts (the enhanced text actually used, or the original after a failure) as a *variation reference* - the LLM is instructed not to repeat them exactly and to introduce fresh variance (mood, lighting, environment, pose, composition) while staying faithful to your original intent. This keeps large batches of similar prompts from collapsing into near-identical images. Negative prompts are batched the same way, without variation references.

### Once

Enhances only the first prompt and applies the result to all images in the batch. Useful for grid generation or when you want consistent enhancement across all outputs. The positive and negative requests share **one** server instance, so the model is loaded only once.

## Resolution Awareness

The extension automatically detects hires fix settings and provides the LLM with:

- **Final output resolution** - the dimensions after hires fix upscaling (or base dimensions if no hires fix)
- **Orientation and aspect ratio** - portrait, landscape, or square
- **Base resolution** - when hires fix is enabled, the LLM also knows the initial generation dimensions

This allows the LLM to frame compositions, describe layouts, and choose appropriate camera angles for the final output size.

## Dynamic Prompts Integration

When the [sd-dynamic-prompts](https://github.com/DecadenceAI/sd-dynamic-prompts) extension is installed, the two extensions interact. **Execution order**: Forge runs script `process()` callbacks in extension discovery order (directory name, alphabetical), so **DP runs first** and this extension enhances the already-expanded prompt. If you want the other way around - enhance the raw prompt with wildcards/variants intact, then let DP expand them - pin the order via **Settings -> "script_process callback priority"**: move `LlamaServerEnhance` above `Script` (DP's class name).

In the intended order (this extension first), the pipeline is:

1. **Raw prompt with wildcards** is sent to the LLM for enhancement
2. The LLM is instructed to preserve `__wildcard__` and `{variant|syntax}` tokens verbatim
3. **Auto-escaping** - any variant syntax the LLM accidentally generates (that wasn't in your original prompt) has its pipes escaped to commas, preventing DP misinterpretation
4. **Custom separator support** - reads your DP settings (`dp_parser_variant_start`, `dp_parser_variant_end`, `dp_parser_wildcard_wrap`) and handles any custom separator configuration
5. DP then expands the wildcards/variants in the enhanced prompt

In the default order (DP first) the escaping logic is harmless: it only touches variant syntax the LLM generated that was not present in the prompt it received.

### How Escaping Works

The extension compares variant blocks in the enhanced output against your original prompt:

- **Original variants preserved** - `{red|blue|green}` in your original stays as-is
- **Accidental variants escaped** - `{indoor|outdoor}` generated by the LLM becomes `{indoor,outdoor}` (pipes replaced with commas)
- **Wildcards untouched** - `__style__` tokens are always preserved

This ensures the LLM can freely describe things like "a cat or dog" without the output being misinterpreted as a DP variant.

### Single prompt (Once mode)

1. Find a free TCP port in the ephemeral range (49152-65535)
2. Start `llama-server` on that port with `--no-ui --no-warmup`
3. Poll `/health` until the model is loaded and ready (configurable startup timeout, default 60s, aborts early if the process exits)
4. Send the positive prompt - and the negative prompt too, if enabled - via `/v1/chat/completions`, both against the same server
5. Kill the server process

### Batch (Per image mode)

1. Find a free TCP port
2. Start **one** `llama-server` instance
3. Wait for `/health` (once), discover the slot count via `/slots`, warm up the chat endpoint
4. Send the positive prompts in **strict buckets** of the slot count - each bucket finishes before the next starts
5. Each bucket after the first gets the previous bucket's prompts appended to the system prompt as a "do not repeat exactly, introduce variance" reference
6. Batch the negative prompts (if enabled), same bucketing, no variation reference
7. Kill the server

This avoids the Nxstartup penalty of spawning a new server per prompt, and the bucketing keeps multi-bucket jobs varied instead of repetitive.

### Reliability details

- **Pipe draining** - the server's stdout/stderr are continuously drained by background threads into bounded in-memory buffers (last 256KB kept). Without draining, the OS pipe buffer (4KB on Windows) fills up as the server logs its startup and every task, and the server's own threads block on write: `/health` keeps answering while chat requests hang. This was the root cause of intermittent "server is healthy but times out" behavior.
- **Transient-only retries** - failed requests are retried up to 5 times (3s/6s/12s/24s backoff), but only for transient errors (timeouts, connection failures, HTTP 408/429/5xx, truncated responses). Deterministic 4xx errors (e.g., context overflow) fail immediately instead of burning ~45s of retries.
- **Per-request timeout** - each chat request times out after the configured *Request timeout* (default 120s), not after the whole retry budget.

## Test Script

A standalone test script is included for validating your setup:

```bash
# Dry run - see the command that would be built
python test_llm.py --preset anima --model C:/models/my-model.gguf --dry-run

# Single prompt test
python test_llm.py --preset anima --model C:/models/my-model.gguf --prompt "a cat"

# Batch test - multiple prompts, sent in buckets (cross-bucket variation enabled)
python test_llm.py --mode batch --preset anima --model C:/models/my-model.gguf \
    --prompts "a cat on a windowsill" "sunset over mountains" "cyberpunk city street"

# Batch test with a fixed bucket size (default: the server's parallel slot count)
python test_llm.py --mode batch --preset anima --model C:/models/my-model.gguf \
    --concurrency 2 --prompts "p1" "p2" "p3" "p4"

# Batch test from file (one prompt per line)
python test_llm.py --mode batch --preset anima --model C:/models/my-model.gguf \
    --prompts-file prompts.txt

# Custom server path + inference flags
python test_llm.py --server C:/tools/llamacpp/llama-server.exe \
    --preset anima --model C:/models/my-model.gguf --flags "-ngl 99 --temp 0.5"

# Override the per-request timeout (seconds)
python test_llm.py --preset anima --model C:/models/my-model.gguf --timeout 60
```

## Debugging

A debug log is written to `enhance_debug.log` in the extension root directory. It records every step of the enhancement pipeline with timestamps, plus the server's own stderr streamed live as `[llama-server]` lines - model load progress, per-task logs, OOM errors. On failure it also captures the last 3000 characters of the server's stderr. The log rotates to `enhance_debug.log.1` automatically once it exceeds 5MB (one backup generation is kept).

## Requirements

- **Stable Diffusion WebUI Forge** (or compatible Auto1111 fork)
- **llama.cpp** - specifically the `llama-server` binary
- A GGUF model file
