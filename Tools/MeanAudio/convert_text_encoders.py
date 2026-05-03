#!/usr/bin/env python3
"""
Convert CLAP text encoder and T5 encoder weights for MeanAudio's MLX Swift pipeline.

Subcommands:
    clap    - Extract CLAP text branch (RoBERTa + text projection) from LAION checkpoint
    t5      - Download/convert flan-t5-large encoder-only weights
    bundle  - Run both and package into a single directory structure

Usage:
    python convert_text_encoders.py clap --input /path/to/music_speech_audioset_epoch_15_esc_89.98.pt --output ./clap/
    python convert_text_encoders.py t5 --output ./t5/
    python convert_text_encoders.py bundle --clap-ckpt /path/to/clap.pt --output ./text_encoders/
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# CLAP text encoder conversion
# ---------------------------------------------------------------------------

# RoBERTa config for LAION CLAP text branch
CLAP_CONFIG = {
    "vocab_size": 50265,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "max_position_embeddings": 514,
    "type_vocab_size": 1,
    "layer_norm_eps": 1e-5,
    "hidden_act": "gelu",
    "pad_token_id": 1,
    "projection_dim": 512,
}

# Regex to match encoder layer keys and decompose them
_ENCODER_LAYER_RE = re.compile(
    r"^text_branch\.encoder\.layer\.(\d+)\.(.+)$"
)

# Mapping from the sub-key within each encoder layer to the Swift target
_LAYER_SUB_MAP = {
    "attention.self.query": "self_attention.q",
    "attention.self.key": "self_attention.k",
    "attention.self.value": "self_attention.v",
    "attention.output.dense": "self_attention.output",
    "attention.output.LayerNorm": "attention_layer_norm",
    "intermediate.dense": "intermediate",
    "output.dense": "output",
    "output.LayerNorm": "output_layer_norm",
}

# Top-level (non-layer) CLAP text branch key prefixes
_TOP_LEVEL_MAP = OrderedDict([
    ("text_branch.embeddings.LayerNorm", "roberta.embeddings.layer_norm"),
    ("text_branch.embeddings.word_embeddings", "roberta.embeddings.word_embeddings"),
    ("text_branch.embeddings.position_embeddings", "roberta.embeddings.position_embeddings"),
    ("text_branch.embeddings.token_type_embeddings", "roberta.embeddings.token_type_embeddings"),
    ("text_branch.pooler.dense", "roberta.pooler"),
])

# Text projection Sequential indices to named submodules
_PROJECTION_MAP = {
    "text_projection.0": "text_projection.linear1",
    "text_projection.2": "text_projection.linear2",
}


def _map_clap_key(key: str) -> Optional[str]:
    """Map a single CLAP checkpoint key to the Swift CLAPTextEncoder key.

    Returns None if the key should be skipped.
    """
    # Skip audio branch entirely
    if key.startswith("audio_") or key.startswith("model.audio_") or key.startswith("module.audio_"):
        return None

    # Strip optional 'model.' or 'module.' prefix from raw LAION CLAP state dict
    clean = key
    if clean.startswith("module."):
        clean = clean[len("module."):]
    elif clean.startswith("model."):
        clean = clean[len("model."):]

    # Skip cross-attention keys
    if "crossattention" in clean:
        return None

    # Encoder layer keys
    m = _ENCODER_LAYER_RE.match(clean)
    if m:
        layer_idx = m.group(1)
        sub_key = m.group(2)
        for src_prefix, dst_prefix in _LAYER_SUB_MAP.items():
            if sub_key.startswith(src_prefix):
                suffix = sub_key[len(src_prefix):]  # e.g. ".weight" or ".bias"
                return f"roberta.encoder.layers.{layer_idx}.{dst_prefix}{suffix}"
        # Unknown sub-key within an encoder layer -- skip
        return None

    # Top-level (embeddings, pooler)
    for src_prefix, dst_prefix in _TOP_LEVEL_MAP.items():
        if clean.startswith(src_prefix):
            suffix = clean[len(src_prefix):]
            return f"{dst_prefix}{suffix}"

    # Text projection
    for src_prefix, dst_prefix in _PROJECTION_MAP.items():
        if clean.startswith(src_prefix):
            suffix = clean[len(src_prefix):]
            return f"{dst_prefix}{suffix}"

    # Anything not matching a known prefix is skipped (audio branch leftovers, logit_scale, etc.)
    return None


def convert_clap(input_path: str, output_dir: str):
    """Extract CLAP text branch from a LAION CLAP checkpoint and save as safetensors."""
    import torch
    from safetensors.numpy import save_file

    input_path = Path(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"ERROR: Checkpoint not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading CLAP checkpoint: {input_path}")
    ckpt = torch.load(str(input_path), map_location="cpu", weights_only=False)

    # The LAION CLAP checkpoint stores the state dict under various possible keys
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    print(f"  Total keys in checkpoint: {len(state_dict)}")

    mapped = OrderedDict()
    skipped = 0

    for key, tensor in state_dict.items():
        new_key = _map_clap_key(key)
        if new_key is None:
            skipped += 1
            continue

        arr = tensor.detach().cpu().float().numpy()
        mapped[new_key] = arr
        print(f"  {key} -> {new_key}  shape={arr.shape}")

    print(f"\n  Mapped: {len(mapped)} tensors, skipped: {skipped}")

    if len(mapped) == 0:
        print("ERROR: No text branch keys found. Check checkpoint format.", file=sys.stderr)
        sys.exit(1)

    # Save weights
    save_file(mapped, str(out / "model.safetensors"))
    print(f"  Saved model.safetensors ({len(mapped)} tensors)")

    # Save config
    with open(out / "config.json", "w") as f:
        json.dump(CLAP_CONFIG, f, indent=2)
    print("  Saved config.json")

    # Save RoBERTa tokenizer files
    _save_roberta_tokenizer(out)

    print(f"\nCLAP text encoder conversion complete: {out}")


def _save_roberta_tokenizer(output_dir: Path):
    """Download and save RoBERTa-base tokenizer files for the CLAP text encoder."""
    try:
        from transformers import RobertaTokenizerFast
    except ImportError:
        print("  WARNING: transformers not installed; skipping tokenizer download.", file=sys.stderr)
        print("  Install with: pip install transformers", file=sys.stderr)
        return

    print("  Downloading roberta-base tokenizer...")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    tokenizer.save_pretrained(str(output_dir))

    # Clean up model-card files that sneak in
    for unwanted in ["README.md", "config.json.bak"]:
        p = output_dir / unwanted
        if p.exists():
            p.unlink()

    # List what we saved
    tok_files = sorted(output_dir.glob("tokenizer*")) + sorted(output_dir.glob("vocab*")) + sorted(output_dir.glob("merges*")) + sorted(output_dir.glob("special_tokens*"))
    for f in tok_files:
        print(f"  Saved {f.name}")


# ---------------------------------------------------------------------------
# T5 encoder conversion
# ---------------------------------------------------------------------------

def convert_t5(output_dir: str, model_name: str = "google/flan-t5-large"):
    """Download flan-t5-large, strip decoder, save encoder weights as safetensors."""
    from safetensors.numpy import save_file

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from transformers import T5EncoderModel, AutoTokenizer, AutoConfig
    except ImportError:
        print("ERROR: transformers not installed. Install with: pip install transformers", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {model_name} (encoder only)...")
    config = AutoConfig.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name)
    model.eval()

    state_dict = model.state_dict()
    print(f"  Encoder state dict: {len(state_dict)} keys")

    # The T5EncoderModel already excludes decoder weights.
    # Convert to numpy and save.
    mapped = OrderedDict()
    for key, tensor in state_dict.items():
        # Skip any decoder / lm_head keys that might leak through
        if key.startswith("decoder.") or key.startswith("lm_head."):
            continue
        arr = tensor.detach().cpu().float().numpy()
        mapped[key] = arr

    print(f"  Saving {len(mapped)} tensors...")

    # Check total size — if > 4GB, split into shards
    total_bytes = sum(arr.nbytes for arr in mapped.values())
    max_shard_bytes = 4 * 1024 * 1024 * 1024  # 4 GB

    if total_bytes > max_shard_bytes:
        _save_sharded(mapped, out, max_shard_bytes)
    else:
        save_file(mapped, str(out / "model.safetensors"))
        print(f"  Saved model.safetensors ({total_bytes / 1024 / 1024:.1f} MB)")

    # Save config.json (the HF T5 config)
    config_dict = config.to_dict()
    with open(out / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    print("  Saved config.json")

    # Save tokenizer
    print(f"  Downloading {model_name} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(str(out))

    # Clean up unwanted files
    for unwanted in ["README.md"]:
        p = out / unwanted
        if p.exists():
            p.unlink()

    tok_files = sorted(out.glob("tokenizer*")) + sorted(out.glob("spiece*")) + sorted(out.glob("special_tokens*"))
    for f in tok_files:
        print(f"  Saved {f.name}")

    print(f"\nT5 encoder conversion complete: {out}")


def _save_sharded(tensors: OrderedDict, output_dir: Path, max_shard_bytes: int):
    """Save tensors across multiple safetensors shards."""
    from safetensors.numpy import save_file

    shards = []
    current_shard = OrderedDict()
    current_bytes = 0

    for key, arr in tensors.items():
        if current_bytes + arr.nbytes > max_shard_bytes and len(current_shard) > 0:
            shards.append(current_shard)
            current_shard = OrderedDict()
            current_bytes = 0
        current_shard[key] = arr
        current_bytes += arr.nbytes

    if current_shard:
        shards.append(current_shard)

    num_shards = len(shards)
    index_map = {}

    for i, shard in enumerate(shards, 1):
        filename = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
        save_file(shard, str(output_dir / filename))
        for key in shard:
            index_map[key] = filename
        shard_bytes = sum(arr.nbytes for arr in shard.values())
        print(f"  Saved {filename} ({shard_bytes / 1024 / 1024:.1f} MB, {len(shard)} tensors)")

    # Write index file
    total_bytes = sum(arr.nbytes for arr in tensors.values())
    index = {
        "metadata": {"total_size": total_bytes},
        "weight_map": index_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"  Saved model.safetensors.index.json")


# ---------------------------------------------------------------------------
# Bundle subcommand
# ---------------------------------------------------------------------------

def bundle(clap_ckpt: str, output_dir: str):
    """Run both conversions and package into a single directory tree."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    clap_dir = out / "clap"
    t5_dir = out / "t5"

    print("=" * 60)
    print("  STEP 1: Converting CLAP text encoder")
    print("=" * 60)
    convert_clap(clap_ckpt, str(clap_dir))

    print()
    print("=" * 60)
    print("  STEP 2: Converting T5 encoder")
    print("=" * 60)
    convert_t5(str(t5_dir))

    print()
    print("=" * 60)
    print("  Bundle complete")
    print("=" * 60)
    print(f"\nOutput directory: {out}")
    print("\nStructure:")
    _print_tree(out)


def _print_tree(root: Path, prefix: str = "  "):
    """Print a simple tree listing with file sizes."""
    entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name))
    for entry in entries:
        if entry.is_dir():
            print(f"{prefix}{entry.name}/")
            _print_tree(entry, prefix + "  ")
        else:
            size_mb = entry.stat().st_size / (1024 * 1024)
            if size_mb >= 1.0:
                print(f"{prefix}{entry.name}  ({size_mb:.1f} MB)")
            else:
                size_kb = entry.stat().st_size / 1024
                print(f"{prefix}{entry.name}  ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert CLAP text encoder and T5 encoder for MeanAudio MLX Swift pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s clap --input ./weights/music_speech_audioset_epoch_15_esc_89.98.pt --output ./clap/
  %(prog)s t5 --output ./t5/
  %(prog)s bundle --clap-ckpt ./weights/music_speech_audioset_epoch_15_esc_89.98.pt --output ./text_encoders/
""",
    )
    sub = parser.add_subparsers(dest="command")

    # --- clap ---
    clap_parser = sub.add_parser(
        "clap",
        help="Extract CLAP text branch (RoBERTa + projection) from LAION checkpoint",
    )
    clap_parser.add_argument(
        "--input", required=True,
        help="Path to LAION CLAP checkpoint (.pt)",
    )
    clap_parser.add_argument(
        "--output", required=True,
        help="Output directory for CLAP text encoder",
    )

    # --- t5 ---
    t5_parser = sub.add_parser(
        "t5",
        help="Download and convert flan-t5-large encoder-only weights",
    )
    t5_parser.add_argument(
        "--output", required=True,
        help="Output directory for T5 encoder",
    )
    t5_parser.add_argument(
        "--model", default="google/flan-t5-large",
        help="HuggingFace model name (default: google/flan-t5-large)",
    )

    # --- bundle ---
    bundle_parser = sub.add_parser(
        "bundle",
        help="Convert both encoders and package into a single directory",
    )
    bundle_parser.add_argument(
        "--clap-ckpt", required=True,
        help="Path to LAION CLAP checkpoint (.pt)",
    )
    bundle_parser.add_argument(
        "--output", required=True,
        help="Output directory (will contain clap/ and t5/ subdirs)",
    )

    args = parser.parse_args()

    if args.command == "clap":
        convert_clap(args.input, args.output)
    elif args.command == "t5":
        convert_t5(args.output, model_name=args.model)
    elif args.command == "bundle":
        bundle(args.clap_ckpt, args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
