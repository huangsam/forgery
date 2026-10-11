"""CLI entrypoint for LLM practice and build toolchain."""

import argparse
import sys
from pathlib import Path

from .builder import build_pipeline


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forgery",
        description="Forgery: Local LLM Model Workshop & Quantization Toolchain",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # Build subcommand
    build_parser = subparsers.add_parser(
        "build",
        help="Download, merge, and register a GGUF model in Ollama",
    )
    build_parser.add_argument(
        "--repo",
        default="unsloth/Qwen3.8-Flash-Next-GGUF",
        help="Hugging Face repo ID (default: unsloth/Qwen3.8-Flash-Next-GGUF)",
    )
    build_parser.add_argument(
        "--quant",
        default="UD-IQ4_XS",
        help="Quantization tag / subfolder to download (default: UD-IQ4_XS)",
    )
    build_parser.add_argument(
        "--name",
        default="qwen3.8-flash-next:iq4_xs",
        help="Model name to register in Ollama (default: qwen3.8-flash-next:iq4_xs)",
    )
    build_parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path.cwd(),
        help=("Working directory for downloads and intermediate files (default: current directory)"),
    )
    build_parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="Context window length for Modelfile (e.g. 65536)",
    )
    build_parser.add_argument(
        "--temp",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    build_parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter (default: 0.95)",
    )
    build_parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling parameter (default: 20)",
    )
    build_parser.add_argument(
        "--cleanup-downloads",
        action="store_true",
        help="Delete raw downloaded shards after merging",
    )
    build_parser.add_argument(
        "--cleanup-merged",
        action="store_true",
        help="Delete merged .gguf file after Ollama import completes",
    )

    return parser


def main() -> None:
    parser = create_parser()

    # Default to 'build' if flags given without subcommand,
    # or show help if completely empty
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # If first arg looks like a flag rather than a subcommand, default to 'build'
    if sys.argv[1].startswith("-") and sys.argv[1] not in ("-h", "--help"):
        args = parser.parse_args(["build"] + sys.argv[1:])
    else:
        args = parser.parse_args()

    if args.subcommand == "build":
        build_pipeline(
            repo_id=args.repo,
            quant=args.quant,
            name=args.name,
            work_dir=args.work_dir,
            temperature=args.temp,
            top_p=args.top_p,
            top_k=args.top_k,
            num_ctx=args.num_ctx,
            cleanup_downloads=args.cleanup_downloads,
            cleanup_merged=args.cleanup_merged,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
