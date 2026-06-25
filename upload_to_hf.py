#!/usr/bin/env python3
"""
upload_to_hf.py — Upload AgentSight weights and tokeniser to HuggingFace Hub.

Usage
-----
    # 1. Log in once (browser or token):
    huggingface-cli login

    # 2. Run this script:
    python upload_to_hf.py --repo YOUR_USERNAME/agentsight

    # That's it. The repo is created automatically if it doesn't exist.

What gets uploaded
------------------
    best_agentsight.pth          — model weights (PyTorch state dict)
    best_agentsight_meta.json    — threshold + val metrics
    tokenizer_config/            — DeBERTa-v3-base tokeniser config
    README.md                    — model card (from hf_model_card.md)
"""
import argparse
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        required=True,
        help="HuggingFace repo id, e.g.  your_username/agentsight",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository (default: public)",
    )
    args = parser.parse_args()

    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    # ── 1. Create repo (idempotent) ────────────────────────────────────────
    print(f"Creating / verifying repo: {args.repo} …")
    create_repo(
        repo_id=args.repo,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    # ── 2. Collect files to upload ─────────────────────────────────────────
    weights_pth  = HERE / "src" / "models" / "best_agentsight.pth"
    meta_json    = HERE / "src" / "models" / "best_agentsight_meta.json"
    card_src     = HERE / "hf_model_card.md"

    assert weights_pth.exists(), f"Weights not found: {weights_pth}"
    assert meta_json.exists(),   f"Meta not found: {meta_json}"
    assert card_src.exists(),    f"Model card not found: {card_src}"

    # Save tokeniser config into a temp directory
    tmpdir = Path(tempfile.mkdtemp())
    tok_dir = tmpdir / "tokenizer_config"
    tok_dir.mkdir()

    try:
        from transformers import AutoTokenizer
        print("Saving tokeniser config …")
        tok = AutoTokenizer.from_pretrained(
            "microsoft/deberta-v3-base", use_fast=True
        )
        tok.save_pretrained(str(tok_dir))

        # ── 3. Upload ──────────────────────────────────────────────────────
        print(f"\nUploading to https://huggingface.co/{args.repo} …")
        print(f"  • best_agentsight.pth  ({weights_pth.stat().st_size / 1e6:.0f} MB)")
        api.upload_file(
            path_or_fileobj=str(weights_pth),
            path_in_repo="best_agentsight.pth",
            repo_id=args.repo,
            repo_type="model",
            commit_message="Add model weights (AgentSight best checkpoint)",
        )

        print("  • best_agentsight_meta.json")
        api.upload_file(
            path_or_fileobj=str(meta_json),
            path_in_repo="best_agentsight_meta.json",
            repo_id=args.repo,
            repo_type="model",
            commit_message="Add threshold + validation metadata",
        )

        print("  • README.md (model card)")
        api.upload_file(
            path_or_fileobj=str(card_src),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="model",
            commit_message="Add model card",
        )

        print("  • tokenizer_config/")
        api.upload_folder(
            folder_path=str(tok_dir),
            path_in_repo="tokenizer_config",
            repo_id=args.repo,
            repo_type="model",
            commit_message="Add DeBERTa-v3-base tokeniser config (max_len=512)",
        )

        print(f"\n✓ Upload complete!")
        print(f"  Model page : https://huggingface.co/{args.repo}")
        print(f"  Load with  : AgentMonitor('{args.repo}')")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
