#!/usr/bin/env python3
"""
Convert MeanAudio PyTorch checkpoint (.pth) to MLX-compatible safetensors.

Usage:
    python convert_meanaudio_weights.py \
        --input path/to/ema_ckpt.pth \
        --output path/to/weights/

Handles:
- Stripping 'ema_model.' prefix (MeanAudio saves EMA weights with this prefix)
- Removing non-persistent buffers (t_embed.freqs, r_embed.freqs, latent_rot, text_rot)
- Removing _extra_state keys
- Mapping PyTorch nn.Sequential indices to named Swift submodules
- Transposing Conv1d weights from PyTorch (out, in, kernel) to MLX (out, kernel, in)
- Saving as safetensors + config.json
"""

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


def remove_prefix(state_dict: dict) -> dict:
    new = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("ema_model.", "")
        new[name] = v
    return new


# Keys to drop (non-persistent buffers and extras)
DROP_KEYS = {
    "t_embed.freqs", "r_embed.freqs",
    "latent_rot", "text_rot",
    "_extra_state",
}

# PyTorch nn.Sequential index → Swift named submodule
SEQUENTIAL_MAP = {
    # audio_input_proj: Sequential(ChannelLastConv1d, SELU, ConvMLP)
    "audio_input_proj.0.conv": "audio_input_proj.conv1.conv",
    "audio_input_proj.2": "audio_input_proj.mlp",
    # text_input_proj: Sequential(Linear, MLP)
    "text_input_proj.0": "text_input_proj.linear",
    "text_input_proj.1": "text_input_proj.mlp",
    # text_cond_proj: Sequential(Linear, MLP)
    "text_cond_proj.0": "text_cond_proj.linear",
    "text_cond_proj.1": "text_cond_proj.mlp",
    # adaLN_modulation: Sequential(SiLU, Linear) → just the Linear
    # SiLU has no params, so index 1 maps to our single linear
}

# Regex for adaLN_modulation.1 → adaLN_modulation.linear
ADALN_PATTERN = re.compile(r"(.*)adaLN_modulation\.1\.(.*)")

# Regex for t_embed.mlp.0 → t_embed.linear1, t_embed.mlp.2 → t_embed.linear2
EMBED_MLP_PATTERN = re.compile(r"(t_embed|r_embed)\.mlp\.(\d+)\.(.*)")

# Conv1d weight keys (need transpose)
CONV_WEIGHT_PATTERN = re.compile(r".*conv\.weight$")

# Frozen parameter name mapping (snake_case → camelCase for Swift)
PARAM_RENAME = {
    "latent_mean": "latentMean",
    "latent_std": "latentStd",
    "empty_string_feat": "emptyStringFeat",
    "empty_string_feat_c": "emptyStringFeatC",
}


def map_key(key: str) -> str:
    # 0. Frozen parameter renames
    if key in PARAM_RENAME:
        return PARAM_RENAME[key]
    # 1. adaLN_modulation.1.X → adaLN_modulation.linear.X
    m = ADALN_PATTERN.match(key)
    if m:
        return f"{m.group(1)}adaLN_modulation.linear.{m.group(2)}"

    # 2. t_embed.mlp.0/2 → t_embed.linear1/linear2
    m = EMBED_MLP_PATTERN.match(key)
    if m:
        prefix = m.group(1)
        idx = int(m.group(2))
        suffix = m.group(3)
        name = "linear1" if idx == 0 else "linear2"
        return f"{prefix}.{name}.{suffix}"

    # 3. Sequential index mappings
    for pytorch_prefix, swift_prefix in SEQUENTIAL_MAP.items():
        if key.startswith(pytorch_prefix):
            return key.replace(pytorch_prefix, swift_prefix, 1)

    return key


def convert(input_path: str, output_dir: str):
    print(f"Loading checkpoint: {input_path}")
    ckpt = torch.load(input_path, map_location="cpu", weights_only=True)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    state_dict = remove_prefix(state_dict)

    # Drop non-persistent buffers
    for k in list(state_dict.keys()):
        if k in DROP_KEYS or k.endswith("_extra_state"):
            del state_dict[k]

    # Map keys and transpose conv weights
    mapped = OrderedDict()
    for key, tensor in state_dict.items():
        new_key = map_key(key)
        arr = tensor.numpy()

        # Conv1d: PyTorch (out_ch, in_ch, kernel) → MLX (out_ch, kernel, in_ch)
        if CONV_WEIGHT_PATTERN.match(key) and arr.ndim == 3:
            arr = np.transpose(arr, (0, 2, 1))

        mapped[new_key] = arr
        if new_key != key:
            print(f"  {key} → {new_key}  shape={arr.shape}")

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from safetensors.numpy import save_file
        save_file(mapped, str(out / "model.safetensors"))
        print(f"Saved {len(mapped)} tensors to {out / 'model.safetensors'}")
    except ImportError:
        np.savez(str(out / "weights.npz"), **mapped)
        print(f"safetensors not installed; saved as npz. Install: pip install safetensors")

    # Write config
    config = {
        "latent_dim": 20,
        "text_dim": 1024,
        "text_c_dim": 512,
        "hidden_dim": 448,
        "depth": 12,
        "fused_depth": 8,
        "num_heads": 7,
        "mlp_ratio": 4.0,
        "latent_seq_len": 312,
        "text_seq_len": 77,
        "use_rope": False,
        "sample_rate": 16000,
        "duration_seconds": 9.975,
        "cfg_strength": 4.5,
        "steps": 1,
    }
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MeanAudio weights to MLX safetensors")
    parser.add_argument("--input", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", required=True, help="Output directory for safetensors + config")
    args = parser.parse_args()
    convert(args.input, args.output)
