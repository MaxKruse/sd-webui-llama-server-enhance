# sd-webui-llama-server-enhance

LLM-powered prompt enhancement for **Stable Diffusion WebUI Forge**, using [llama.cpp](https://github.com/ggerganov/llama.cpp)'s `llama-server` HTTP API.

Drop this extension into your `extensions/` folder and get per-image prompt enhancement powered by any local GGUF model — no cloud API keys needed.

## Features

- **Batch mode** — starts a single `llama-server` instance and sends all prompts in parallel, eliminating per-image startup penalties
- **Preset system** — load system prompts from `presets/*.txt` files
- **Dynamic Prompts integration** — automatically preserves `__wildcard__` and `{variant|syntax}` tokens when the sd-dynamic-prompts extension is installed
- **Two enhancement modes**:
  - **Per image** — enhance each prompt individually (batch-optimized: one server, parallel requests)
  - **Once** — enhance one prompt and apply it to all images in the batch
- **OpenAI-compatible API** — works with any model that `llama-server` can load

## Installation

1. Clone or copy this repository into your WebUI Forge extensions folder:

   ```
   extensions/sd-webui-llama-server-enhance/
   ```

2. Ensure `llama-server` is available in your `PATH` or set the full path in WebUI settings.

3. Set your model path in **Settings → LLama Server Enhance**.

4. Restart WebUI Forge.

## Configuration

All settings are in **Settings → LLama Server Enhance**:

| Setting | Description |
|---------|-------------|
| **llama-server path** | Path to the `llama-server` binary (default: `llama-server` from PATH) |
| **Model path** | Full path to your `.gguf` model file |
| **Inference flags** | Extra flags passed to `llama-server` (e.g., `-ngl 99 --temp 0.8 --top-p 0.9`) |

## Presets

Create `.txt` files in the `presets/` directory. Each file becomes a selectable preset in the UI. The file content is used as the system prompt sent to the LLM.

A preset for the [Anima](https://github.com/CircleStone-Labs/Anima) anime model is included (`presets/anima.txt`).

## Enhancement Modes

### Per image (default)

Each prompt in the batch is enhanced individually. The extension starts **one** `llama-server` instance and sends all prompts concurrently, then collects all responses before returning. This leverages the server's ability to run parallel inference workflows.

### Once

Enhances only the first prompt and applies the result to all images in the batch. Useful for grid generation or when you want consistent enhancement across all outputs.

## How It Works

### Single prompt (Once mode)

1. Find a free TCP port in the ephemeral range (49152–65535)
2. Start `llama-server` on that port with `--no-ui --no-warmup`
3. Poll `/health` until the model is loaded and ready (60s timeout)
4. Send the prompt via `/v1/chat/completions`
5. Kill the server process

### Batch (Per image mode)

1. Find a free TCP port
2. Start **one** `llama-server` instance
3. Wait for `/health` (once)
4. Send **all prompts concurrently** via thread pool
5. Collect all responses
6. Kill the server

This avoids the N×startup penalty of spawning a new server per prompt.

## Test Script

A standalone test script is included for validating your setup:

```bash
# Dry run — see the command that would be built
python test_llm.py --preset anima --model C:/models/my-model.gguf --dry-run

# Single prompt test
python test_llm.py --preset anima --model C:/models/my-model.gguf --prompt "a cat"

# Batch test — multiple prompts, parallel inference
python test_llm.py --mode batch --preset anima --model C:/models/my-model.gguf \
    --prompts "a cat on a windowsill" "sunset over mountains" "cyberpunk city street"

# Batch test from file (one prompt per line)
python test_llm.py --mode batch --preset anima --model C:/models/my-model.gguf \
    --prompts-file prompts.txt

# Custom server path + inference flags
python test_llm.py --server C:/tools/llamacpp/llama-server.exe \
    --preset anima --model C:/models/my-model.gguf --flags "-ngl 99 --temp 0.5"
```

## Debugging

A debug log is written to `enhance_debug.log` in the extension root directory. It records every step of the enhancement pipeline with timestamps.

## Requirements

- **Stable Diffusion WebUI Forge** (or compatible Auto1111 fork)
- **llama.cpp** — specifically the `llama-server` binary
- A GGUF model file
