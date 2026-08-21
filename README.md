# sd-webui-llama-server-enhance

LLM-powered prompt enhancement for **Stable Diffusion WebUI Forge**, using [llama.cpp](https://github.com/ggerganov/llama.cpp)'s `llama-server` HTTP API.

Drop this extension into your `extensions/` folder and get per-image prompt enhancement powered by any local GGUF model - no cloud API keys needed.

## Features

- **Batch mode** - starts a single `llama-server` instance and sends all prompts in parallel, eliminating per-image startup penalties
- **Auto preset selection** - automatically selects the correct system prompt preset based on the Forge-Neo UI preset (flux, zit, anima)
- **Hires fix aware** - provides the LLM with final output dimensions (after hires fix upscaling) so it can frame compositions appropriately
- **Dynamic Prompts integration** - automatically escapes accidental DP syntax in LLM output; supports custom DP separators (configurable `variant_start`, `variant_end`, `wildcard_wrap`). See [Dynamic Prompts Integration](#dynamic-prompts-integration) for execution order
- **Two enhancement modes**:
  - **Per image** - enhance each prompt individually (batch-optimized: one server, parallel requests)
  - **Once** - enhance one prompt and apply it to all images in the batch
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
- `presets/anima.txt` - for the [Anima](https://github.com/CircleStone-Labs/Anima) anime model
- `presets/flux-dev.txt` - for FLUX.1 [dev]
- `presets/flux-krea-dev.txt` - for FLUX.1 Krea [dev]
- `presets/z-image-turbo.txt` - for Z-Image Turbo

## Enhancement Modes

### Per image (default)

Each prompt in the batch is enhanced individually. The extension starts **one** `llama-server` instance and sends all prompts concurrently, then collects all responses before returning. This leverages the server's ability to run parallel inference workflows.

### Once

Enhances only the first prompt and applies the result to all images in the batch. Useful for grid generation or when you want consistent enhancement across all outputs.

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
3. Poll `/health` until the model is loaded and ready (60s timeout, aborts early if the process exits)
4. Send the prompt via `/v1/chat/completions`
5. Kill the server process

### Batch (Per image mode)

1. Find a free TCP port
2. Start **one** `llama-server` instance
3. Wait for `/health` (once)
4. Send **all prompts concurrently** via thread pool
5. Collect all responses
6. Kill the server

This avoids the Nxstartup penalty of spawning a new server per prompt.

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

# Batch test - multiple prompts, parallel inference
python test_llm.py --mode batch --preset anima --model C:/models/my-model.gguf \
    --prompts "a cat on a windowsill" "sunset over mountains" "cyberpunk city street"

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

A debug log is written to `enhance_debug.log` in the extension root directory. It records every step of the enhancement pipeline with timestamps. On failure it also captures the last 3000 characters of the server's stderr (model load errors, OOM, etc.). Note: the log grows unbounded - delete it periodically if it gets large.

## Requirements

- **Stable Diffusion WebUI Forge** (or compatible Auto1111 fork)
- **llama.cpp** - specifically the `llama-server` binary
- A GGUF model file
