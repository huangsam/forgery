# Forgery

A high-performance CLI pipeline for downloading, merging sharded GGUFs from Hugging Face, and registering them directly into Ollama on macOS.

## Why Forgery?
Hugging Face splits large models (>50 GB) into multi-part shards (`-00001-of-0000X.gguf`), which Ollama's native `hf.co` importer cannot assemble. `forgery` automates the entire workflow in a single hands-off command:
- **Fast parallel downloads**: Rust-accelerated `hf-transfer` + parallel workers.
- **Auto-merge**: Detects split shards and merges via `llama-gguf-split`.
- **Template-aware**: Inherits native Jinja/ChatML templates and stop tokens from GGUF metadata.
- **Hardware sizing**: Automatically inspects host RAM and warns if model exceeds macOS limits.
- **Disk management**: `--cleanup-downloads` and `--cleanup-merged` to prevent SSD exhaustion.

## Prerequisites
- [uv](https://docs.astral.sh/uv/): `brew install uv`
- [Ollama](https://ollama.com): `brew install ollama`
- [llama.cpp](https://github.com/ggerganov/llama.cpp): `brew install llama.cpp`

## Usage

```bash
# General syntax
uv run forgery build --repo "<owner>/<repo>" --quant "<tag>" --name "<ollama-model-name>"

# Example: Llama 3.3 70B (auto-merged & cleaned up)
uv run forgery build \
  --repo "bartowski/Llama-3.3-70B-Instruct-GGUF" \
  --quant "Q4_K_M" \
  --name "llama3.3:70b-q4_k_m" \
  --cleanup-downloads

# Inspect CLI options
uv run forgery build --help
```
