"""Download Pi3X pretrained weights from Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Pi3X model weights")
    parser.add_argument(
        "--repo-id",
        default="yyfz233/Pi3X",
        help="Hugging Face repo ID (default: yyfz233/Pi3X)",
    )
    parser.add_argument(
        "--filename",
        default="model.safetensors",
        help="Weight file name in the repo (default: model.safetensors)",
    )
    parser.add_argument(
        "--local-dir",
        default=str(Path(__file__).resolve().parents[1] / "checkpoints"),
        help="Local directory to save the weights",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface-hub is not installed. Run: pip install huggingface-hub[torch]"
        )

    print(f"Downloading {args.filename} from {args.repo_id} -> {args.local_dir}")
    path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        local_dir=args.local_dir,
    )
    print(f"Saved to: {path}")


if __name__ == "__main__":
    main()
