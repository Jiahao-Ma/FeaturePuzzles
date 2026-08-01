"""Download pretrained weights from Hugging Face Hub.

Examples:
    # Pi3X (default, backward-compatible)
    python scripts/download_weights.py

    # VGGT-1B
    python scripts/download_weights.py --model vggt

    # Custom repo
    python scripts/download_weights.py --repo-id facebook/VGGT-1B --filename model.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path


PRESETS = {
    "pi3x": {"repo_id": "yyfz233/Pi3X",       "filename": "model.safetensors"},
    "vggt": {"repo_id": "facebook/VGGT-1B",   "filename": "model.pt"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument("--model", choices=list(PRESETS), default="pi3x",
                        help="Preset: pi3x or vggt")
    parser.add_argument("--repo-id", default=None,
                        help="Override Hugging Face repo ID")
    parser.add_argument("--filename", default=None,
                        help="Override weight file name")
    parser.add_argument(
        "--local-dir",
        default=str(Path(__file__).resolve().parents[1] / "checkpoints"),
        help="Local directory to save the weights",
    )
    args = parser.parse_args()

    preset = PRESETS[args.model]
    repo_id = args.repo_id or preset["repo_id"]
    filename = args.filename or preset["filename"]

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface-hub is not installed. Run: pip install huggingface-hub[torch]"
        )

    print(f"Downloading {filename} from {repo_id} -> {args.local_dir}")
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=args.local_dir)
    print(f"Saved to: {path}")


if __name__ == "__main__":
    main()
