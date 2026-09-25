#!/usr/bin/env python3
"""
Generate a LLaMA-Factory ASFT training YAML configuration

Usage:
    python generate_asft_yaml.py \
        --model_path /path/to/model \
        --dataset my_dataset \
        --output_dir /path/to/output \
        --yaml_output_path /path/to/training.yaml

This script generates configuration for ASFT full fine-tuning.
"""

import argparse
import os
import yaml
from typing import Dict, Any


def parse_args():
    parser = argparse.ArgumentParser(description='Generate a LLaMA-Factory ASFT training configuration')

    # Model options
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the pretrained model'
    )

    # Dataset options
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name registered in dataset_info.json'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default=None,
        help='Directory containing dataset_info.json (uses the LLaMA-Factory default when omitted)'
    )

    # Output options
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Model output directory'
    )
    parser.add_argument(
        '--yaml_output_path',
        type=str,
        default=None,
        help='YAML output path (defaults to output_dir/training.yaml)'
    )

    # Training hyperparameters
    parser.add_argument(
        '--num_train_epochs',
        type=float,
        default=3.0,
        help='Training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--per_device_train_batch_size',
        type=int,
        default=1,
        help='Training batch size per device'
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=8,
        help='Gradient accumulation steps'
    )
    parser.add_argument(
        '--cutoff_len',
        type=int,
        default=4096,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--warmup_ratio',
        type=float,
        default=0.1,
        help='Warmup ratio'
    )
    parser.add_argument(
        '--lr_scheduler_type',
        type=str,
        default='cosine',
        help='Learning-rate scheduler type'
    )
    parser.add_argument(
        '--lr_scheduler_kwargs',
        type=str,
        default=None,
        help='Extra scheduler parameters (JSON string, for example \'{"min_lr": 2.0e-6}\')'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='Weight decay'
    )
    parser.add_argument(
        '--val_size',
        type=float,
        default=None,
        help='Validation split ratio (for example, 0.05 means 5%%)'
    )
    parser.add_argument(
        '--packing',
        action='store_true',
        default=False,
        help='Enable dataset packing'
    )
    parser.add_argument(
        '--plot_loss',
        action='store_true',
        default=True,
        help='Plot the loss curve'
    )

    # ASFT options
    parser.add_argument(
        '--asft_alpha',
        type=float,
        default=0.05,
        help='ASFT alpha coefficient (default: 0.05)'
    )
    parser.add_argument(
        '--use_asft_loss',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use ASFT loss (enabled by default)'
    )

    # Optimization options
    parser.add_argument(
        '--bf16',
        action='store_true',
        default=True,
        help='Use bfloat16 precision'
    )
    parser.add_argument(
        '--flash_attn',
        type=str,
        default='fa2',
        choices=['disabled', 'sdpa', 'fa2'],
        help='Flash Attention implementation'
    )
    parser.add_argument(
        '--gradient_checkpointing',
        action='store_true',
        default=True,
        help='Enable gradient checkpointing'
    )

    # DeepSpeed options
    parser.add_argument(
        '--deepspeed',
        type=str,
        default=None,
        help='Path to a DeepSpeed configuration'
    )

    # Logging and checkpoint options
    parser.add_argument(
        '--logging_steps',
        type=int,
        default=10,
        help='Steps between log entries'
    )
    parser.add_argument(
        '--save_steps',
        type=int,
        default=50000,
        help='Steps between checkpoints'
    )
    parser.add_argument(
        '--save_total_limit',
        type=int,
        default=1,
        help='Maximum number of checkpoints to retain'
    )

    # Miscellaneous options
    parser.add_argument(
        '--report_to',
        type=str,
        default='none',
        help='Logging destination (none, wandb, or tensorboard)'
    )
    parser.add_argument(
        '--overwrite_output_dir',
        action='store_true',
        default=True,
        help='Overwrite the output directory'
    )
    parser.add_argument(
        '--template',
        type=str,
        default='qwen3_nothink',
        help='Conversation template (qwen3 includes <think> tags; qwen3_nothink does not)'
    )

    # Dataset mixing options
    parser.add_argument(
        '--mix_strategy',
        type=str,
        default=None,
        choices=['concat', 'interleave_under', 'interleave_over'],
        help='Dataset mixing strategy: concat or interleave'
    )
    parser.add_argument(
        '--interleave_probs',
        type=str,
        default=None,
        help='Interleave probabilities (comma-separated)'
    )

    return parser.parse_args()


def generate_asft_config(args) -> Dict[str, Any]:
    """Generate an ASFT full fine-tuning configuration"""

    config = {
        # Model configuration
        "model_name_or_path": args.model_path,
        "trust_remote_code": True,

        # Training stage
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full",

        # ASFT configuration
        "use_asft_loss": args.use_asft_loss,
        "asft_alpha": args.asft_alpha,

        # Dataset configuration
        "dataset": args.dataset,
        "template": args.template,
        "cutoff_len": args.cutoff_len,
        "preprocessing_num_workers": 16,
        "overwrite_cache": True,
        "packing": args.packing,
        "neat_packing": args.packing,
        "dataloader_num_workers": 4,

        # Output configuration
        "output_dir": args.output_dir,
        "overwrite_output_dir": args.overwrite_output_dir,

        # Training hyperparameters
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": 1.0,

        # Precision and optimization
        "bf16": args.bf16,
        "flash_attn": args.flash_attn,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},

        # Logging and checkpoints
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "report_to": args.report_to,

        # Miscellaneous options
        "ddp_timeout": 180000000,
        "eval_strategy": "no",
    }

    if args.val_size is not None:
        config["val_size"] = args.val_size
        config["eval_strategy"] = "steps"
        config["eval_steps"] = 5000

    if args.lr_scheduler_kwargs:
        import json
        try:
            config["lr_scheduler_kwargs"] = json.loads(args.lr_scheduler_kwargs)
        except json.JSONDecodeError:
            config["lr_scheduler_kwargs"] = args.lr_scheduler_kwargs

    if args.plot_loss:
        config["plot_loss"] = True

    if args.dataset_dir:
        config["dataset_dir"] = args.dataset_dir

    if args.deepspeed:
        config["deepspeed"] = args.deepspeed

    if args.mix_strategy:
        config["mix_strategy"] = args.mix_strategy
        if "interleave" in args.mix_strategy and args.interleave_probs:
            config["interleave_probs"] = args.interleave_probs

    return config


def write_yaml(file_path: str, data: Dict[str, Any]) -> None:
    """Write a YAML file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"YAML configuration saved to: {file_path}")


def main():
    args = parse_args()

    config = generate_asft_config(args)

    yaml_output_path = args.yaml_output_path
    if yaml_output_path is None:
        yaml_output_path = os.path.join(args.output_dir, "training.yaml")

    write_yaml(yaml_output_path, config)

    print("\n" + "=" * 50)
    print("Training configuration:")
    print("=" * 50)
    print(f"  Model path: {args.model_path}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Template: {args.template}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Fine-tuning type: full (ASFT)")
    print(f"  Use ASFT loss: {args.use_asft_loss}")
    print(f"  ASFT Alpha: {args.asft_alpha}")
    print(f"  Training epochs: {args.num_train_epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Maximum length: {args.cutoff_len}")
    print(f"  DeepSpeed: {args.deepspeed if args.deepspeed else 'disabled'}")
    if args.mix_strategy:
        print(f"  Mixing strategy: {args.mix_strategy}")
        if args.interleave_probs:
            print(f"  Sampling probabilities: {args.interleave_probs}")
    print("=" * 50)

    print(f"\nStart training with:")
    print(f"  llamafactory-cli train {yaml_output_path}")


if __name__ == '__main__':
    main()
