# Forgery: Local LLM Model Workshop

A local model forge for downloading, merging sharded GGUFs, and registering models in Ollama.

## Active Build: Qwen3.8-Flash-Next (UD-IQ4_XS)
- **Model**: `qwen3.8-flash-next:iq4_xs` (176.9B params, `qwen4exp` architecture).
- **Quantization**: 4-bit `UD-IQ4_XS` (~93.7 GB).
- **Hardware Sizing**: Sized to sit directly under macOS's ~96 GB wired memory ceiling (on 128 GB Unified Memory) to prevent OS disk swap thrashing.
- **Runtime Target**: Served by the local `launchd` Ollama daemon with a 64K context window (`OLLAMA_CONTEXT_LENGTH=65536`, `OLLAMA_KV_CACHE_TYPE=q8_0`).

## Pipeline Toolchain (`src/forgery/`)
- `uv run forgery build`: Native Python CLI using `huggingface_hub` and `llama-gguf-split`:
  - Automatically downloads shards with Rust `hf-transfer` acceleration.
  - Dynamically detects split parts (`*-00001-of-*.gguf`) and merges them.
  - Generates `Modelfile` and registers with `ollama create`.
  - Supports `--cleanup-downloads` and `--cleanup-merged` flags.

## Quick Run
```bash
uv run forgery build --help
ollama run qwen3.8-flash-next:iq4_xs
```
