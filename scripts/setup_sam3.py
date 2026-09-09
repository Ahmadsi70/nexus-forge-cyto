#!/usr/bin/env python3
"""
Download and verify SAM 3 segmentation model weights.

Usage:
  export HF_TOKEN=hf_xxx  # get one from https://huggingface.co/settings/tokens
  python scripts/setup_sam3.py

This downloads ~2.4 GB to ~/.cache/nexus-forge/sam3.pt by default.
Set NEXUS_SAM3_MODEL_PATH to override the destination.
"""

import os, sys, hashlib
from pathlib import Path

def main():
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        print("FATAL: export HF_TOKEN first")
        print("  Get a token: https://huggingface.co/settings/tokens")
        sys.exit(1)

    dest = Path(os.environ.get("NEXUS_SAM3_MODEL_PATH",
                Path.home() / ".cache" / "nexus-forge" / "sam3.pt"))
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        size_mb = dest.stat().st_size / 1e6
        print(f"Already present: {dest} ({size_mb:.0f} MB)")
        # Quick sanity: weights should be ~2.4 GB
        if size_mb > 2300:
            print("Model looks intact.")
            return
        print("Model seems too small, re-downloading...")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface-hub...")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "huggingface-hub"])
        from huggingface_hub import hf_hub_download

    print("Downloading SAM 3 from HuggingFace (facebook/sam3)...")
    print("  (~2.4 GB, may take a few minutes)")
    path = hf_hub_download(
        repo_id="facebook/sam3",
        filename="sam3.pt",
        local_dir=str(dest.parent),
        token=hf_token,
    )
    final = Path(path)
    size_mb = final.stat().st_size / 1e6
    print(f"Downloaded: {final} ({size_mb:.0f} MB)")

    # Symlink / copy to expected location if different
    if final.resolve() != dest.resolve():
        if dest.exists():
            dest.unlink()
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(final, dest)
            print(f"Symlinked: {dest} → {final}")
        except OSError:
            import shutil
            shutil.copy2(final, dest)
            print(f"Copied to: {dest}")

    print("SAM 3 ready.")
    print("Set in .env:  NEXUS_SAM3_MODEL_PATH=<path>")

if __name__ == "__main__":
    main()