"""LLM build pipeline module.
Downloads, merges, and registers GGUFs in Ollama.
"""

import os
import shutil
import subprocess
from pathlib import Path


def check_tool(name: str, preferred_paths: list[str] | None = None) -> str:
    """Find a binary in PATH or common fallback locations."""
    found = shutil.which(name)
    if found:
        return found
    if preferred_paths:
        for path_str in preferred_paths:
            if Path(path_str).is_file() and os.access(path_str, os.X_OK):
                return path_str
    raise FileNotFoundError(f"Required binary '{name}' was not found. Please ensure it is installed and in your PATH.")


def get_system_specs(work_dir: Path) -> tuple[float, float]:
    """Return (total_ram_gb, free_disk_gb) for the host machine."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        total_ram_gb = (page_size * phys_pages) / (1024**3)
    except AttributeError, ValueError, OSError:
        total_ram_gb = 0.0

    try:
        free_disk_gb = shutil.disk_usage(work_dir).free / (1024**3)
    except OSError:
        free_disk_gb = 0.0

    return total_ram_gb, free_disk_gb


def download_shards(repo_id: str, quant_pattern: str, output_dir: Path) -> Path:
    """Download GGUF shards from Hugging Face."""
    # Enable high-performance Rust transfer if installed
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    from huggingface_hub import snapshot_download

    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = f"*{quant_pattern}*" if not quant_pattern.startswith("*") else quant_pattern
    print(f"[*] Downloading '{repo_id}' matching pattern '{pattern}' into {output_dir}...")

    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[pattern],
        local_dir=str(output_dir),
        max_workers=8,
    )
    return output_dir


def find_and_merge_shards(download_dir: Path, output_file: Path) -> Path:
    """Find GGUF files in directory; merge if split, or return single file."""
    gguf_files = sorted(download_dir.rglob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf files found in {download_dir}")

    # If only a single GGUF exists, no merge needed
    if len(gguf_files) == 1:
        single_file = gguf_files[0]
        print(f"[*] Found single GGUF: {single_file.name}. No merge required.")
        return single_file

    print(f"[*] Found {len(gguf_files)} GGUF parts. Preparing to merge...")
    # Find first shard (e.g. -00001-of-0000X.gguf)
    first_shard = next(
        (f for f in gguf_files if "-00001-of-" in f.name or f.name.endswith(".00001.gguf")),
        gguf_files[0],
    )
    print(f"[*] Identified first shard: {first_shard}")

    split_bin = check_tool(
        "llama-gguf-split",
        ["/opt/homebrew/bin/llama-gguf-split", "/usr/local/bin/llama-gguf-split"],
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [split_bin, "--merge", str(first_shard), str(output_file)]
    print(f"[*] Executing merge: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[✓] Merged GGUF created at: {output_file} ({output_file.stat().st_size / (1024**3):.2f} GB)")
    return output_file


def generate_modelfile(
    gguf_path: Path,
    modelfile_path: Path,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    num_ctx: int | None = None,
    stop_tokens: list[str] | None = None,
) -> Path:
    """Generate an Ollama Modelfile."""
    lines = [f"FROM {gguf_path.resolve()}"]

    if num_ctx:
        lines.append(f"PARAMETER num_ctx {num_ctx}")

    lines.append(f"PARAMETER temperature {temperature}")
    lines.append(f"PARAMETER top_p {top_p}")
    lines.append(f"PARAMETER top_k {top_k}")

    if stop_tokens:
        for stop in stop_tokens:
            lines.append(f'PARAMETER stop "{stop}"')

    content = "\n".join(lines) + "\n"
    modelfile_path.write_text(content)
    print(f"[✓] Generated Modelfile at: {modelfile_path}")
    return modelfile_path


def register_with_ollama(model_name: str, modelfile_path: Path) -> None:
    """Register model with running Ollama daemon."""
    ollama_bin = check_tool(
        "ollama",
        ["/opt/homebrew/bin/ollama", "/usr/local/bin/ollama"],
    )
    cmd = [ollama_bin, "create", model_name, "-f", str(modelfile_path)]
    print(f"[*] Registering '{model_name}' with Ollama...")
    subprocess.run(cmd, check=True)
    print(f"[✓] Model '{model_name}' successfully created in Ollama!")


def build_pipeline(
    repo_id: str = "unsloth/Qwen3.8-Flash-Next-GGUF",
    quant: str = "UD-IQ4_XS",
    name: str = "qwen3.8-flash-next:iq4_xs",
    work_dir: Path | None = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    num_ctx: int | None = None,
    stop_tokens: list[str] | None = None,
    cleanup_downloads: bool = False,
    cleanup_merged: bool = False,
) -> None:
    """Run full download -> merge -> modelfile -> register pipeline."""
    base_dir = work_dir or Path.cwd()
    download_dir = base_dir / "downloads" / quant
    merged_file = base_dir / f"{name.replace(':', '_')}.gguf"
    modelfile_path = base_dir / "Modelfile"

    ram_gb, free_disk_gb = get_system_specs(base_dir)
    max_b_params = ram_gb * 1.25

    print("=" * 60)
    print(f"  LLM Build Pipeline: {name}")
    print(f"  Source: {repo_id} [{quant}]")
    if ram_gb > 0:
        print(f"  Host: {ram_gb:.1f} GB RAM (est. max ~{max_b_params:.0f}B params @ 4-bit) | {free_disk_gb:.1f} GB Free Disk")
    print("=" * 60)

    # 1. Download
    download_shards(repo_id, quant, download_dir)

    # 2. Merge
    target_gguf = find_and_merge_shards(download_dir, merged_file)
    model_size_gb = target_gguf.stat().st_size / (1024**3)
    wired_limit_gb = ram_gb * 0.75
    if ram_gb > 0 and model_size_gb > wired_limit_gb:
        print(
            f"\n[!] ADVISORY: Model size ({model_size_gb:.1f} GB) exceeds estimated 75% wired memory limit ({wired_limit_gb:.1f} GB).\n"
            "    Running this model in Ollama will likely cause severe swap thrashing or high latency.\n"
        )

    # 3. Modelfile
    generate_modelfile(
        gguf_path=target_gguf,
        modelfile_path=modelfile_path,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_ctx=num_ctx,
        stop_tokens=stop_tokens,
    )

    # 4. Register with Ollama
    register_with_ollama(name, modelfile_path)

    # 5. Cleanup
    if cleanup_downloads and download_dir.exists():
        print(f"[*] Cleaning up raw download shards in {download_dir}...")
        shutil.rmtree(download_dir)
        print("[✓] Raw shards cleaned up.")

    if cleanup_merged and target_gguf.is_file() and target_gguf == merged_file:
        print(f"[*] Cleaning up merged GGUF file {target_gguf}...")
        target_gguf.unlink()
        print("[✓] Merged GGUF cleaned up.")

    print("=" * 60)
    print("  Build complete! Run your model with:")
    print(f"    ollama run {name}")
    print("=" * 60)
