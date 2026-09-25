#!/usr/bin/env python3
"""
Merge LoRA adapter weights into a base model and save merged weights locally.

Usage:
    python lora_scripts/merge_lora_weights.py \
        --lora_path /path/to/lora_output \
        --output_path /path/to/merged_model

    # Optional: manually specify base model path
    python lora_scripts/merge_lora_weights.py \
        --lora_path /path/to/lora_output \
        --base_model_path /path/to/base_model \
        --output_path /path/to/merged_model
"""

import argparse
import os
from typing import Optional

import yaml
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model and save merged model."
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to LoRA output directory (contains adapter_model.safetensors).",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help=(
            "Base model path. If not provided, script reads model_name_or_path from "
            "lora_path/training.yaml."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory to save merged full model weights.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype used when loading base model.",
    )
    return parser.parse_args()


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {dtype_name}")


def read_base_model_from_training_yaml(lora_path: str) -> Optional[str]:
    training_yaml = os.path.join(lora_path, "training.yaml")
    if not os.path.exists(training_yaml):
        return None

    with open(training_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("model_name_or_path")


def main() -> None:
    args = parse_args()

    lora_path = os.path.abspath(args.lora_path)
    output_path = os.path.abspath(args.output_path)
    base_model_path = args.base_model_path

    if base_model_path is None:
        base_model_path = read_base_model_from_training_yaml(lora_path)
        if base_model_path is None:
            raise ValueError(
                "Cannot infer base model path. Please pass --base_model_path explicitly."
            )

    base_model_path = os.path.abspath(base_model_path)
    torch_dtype = resolve_torch_dtype(args.torch_dtype)

    print(f"LoRA path: {lora_path}")
    print(f"Base model path: {base_model_path}")
    print(f"Output path: {output_path}")
    print(f"Torch dtype: {args.torch_dtype}")

    os.makedirs(output_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        lora_path,
        trust_remote_code=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        dtype=torch_dtype,
        device_map="cpu",
    )

    peft_model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        is_trainable=False,
    )
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print("Merge complete.")
    print(f"Merged model saved to: {output_path}")


if __name__ == "__main__":
    main()
