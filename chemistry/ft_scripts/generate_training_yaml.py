#!/usr/bin/env python3
"""
Generate a LLaMA-Factory training YAML configuration

Usage:
    python generate_training_yaml.py \
        --model_path /path/to/model \
        --dataset my_dataset \
        --output_dir /path/to/output \
        --yaml_output_path /path/to/training.yaml

This script supports full fine-tuning and optional layer-selective tuning.
"""

import argparse
import os
import yaml
from typing import Dict, Any, Optional


def parse_args():
    parser = argparse.ArgumentParser(description='Generate a LLaMA-Factory full fine-tuning configuration')
    
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
        default=1e-5,
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
        default=False,
        help='Plot the loss curve'
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
        help='Path to a DeepSpeed configuration (for example, ds_z3_config.json)'
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
        help='Interleave probabilities (comma-separated; used only with an interleave strategy)'
    )
    
    # Layer-selective tuning options
    parser.add_argument(
        '--trainable_layers',
        type=int,
        default=None,
        help='Number of trainable bottom layers (starting at layer 0)'
    )
    parser.add_argument(
        '--top_trainable_layers',
        type=int,
        default=None,
        help='Number of trainable top layers (starting at the final layer)'
    )
    parser.add_argument(
        '--specific_layers',
        type=str,
        default=None,
        help='Specific trainable layers (comma-separated, for example, "1,2,3,4")'
    )
    parser.add_argument(
        '--freeze_extra_modules',
        type=str,
        default=None,
        help='Additional modules to freeze (comma-separated)'
    )
    
    return parser.parse_args()


def generate_full_finetune_config(args) -> Dict[str, Any]:
    """Generate a full or layer-selective fine-tuning configuration"""
    from transformers import AutoConfig
    
    # Determine whether layer-selective tuning is requested.
    do_freeze = False
    trainable_layers = None
    
    if args.trainable_layers is not None or args.top_trainable_layers is not None or args.specific_layers is not None:
        do_freeze = True
        
        # Read the model's total layer count.
        model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        total_layer_number = model_config.num_hidden_layers if hasattr(model_config, 'num_hidden_layers') else model_config.text_config.num_hidden_layers
        
        trainable_layer_list = []
        
        # Bottom layers.
        if args.trainable_layers is not None:
            trainable_layer_list.extend([str(lyr) for lyr in range(int(args.trainable_layers))])
        
        # Top layers, following the original generator's convention.
        if args.top_trainable_layers is not None:
            trainable_layer_list.extend([str(total_layer_number - lyr) for lyr in range(int(args.top_trainable_layers))])
        
        # Explicitly selected layers.
        if args.specific_layers is not None:
            specific_layers = [lyr.strip() for lyr in args.specific_layers.split(",")]
            trainable_layer_list.extend([lyr for lyr in specific_layers if lyr not in trainable_layer_list])
        
        if trainable_layer_list:
            trainable_layers = ",".join(trainable_layer_list)
    
    config = {
        # Model configuration
        "model_name_or_path": args.model_path,
        "trust_remote_code": True,
        
        # Training stage
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "freeze" if do_freeze else "full",  # layer-selective or full fine-tuning
        
        # Dataset configuration
        "dataset": args.dataset,
        "template": args.template,  
        "cutoff_len": args.cutoff_len,
        "preprocessing_num_workers": 16,  # use 1 if multiprocessing serialization fails
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
    
    # Add a validation split when requested.
    if args.val_size is not None:
        config["val_size"] = args.val_size
        config["eval_strategy"] = "steps"  # Enable evaluation when val_size is set.
        config["eval_steps"] = 5000
    
    # Add extra scheduler parameters when provided.
    if args.lr_scheduler_kwargs:
        import json
        try:
            config["lr_scheduler_kwargs"] = json.loads(args.lr_scheduler_kwargs)
        except json.JSONDecodeError:
            # Preserve the value as a string if JSON parsing fails.
            config["lr_scheduler_kwargs"] = args.lr_scheduler_kwargs
    
    # Enable loss plotting when requested.
    if args.plot_loss:
        config["plot_loss"] = True
    
    # Add the dataset directory when provided.
    if args.dataset_dir:
        config["dataset_dir"] = args.dataset_dir
    
    # config["freeze_extra_modules"] = None
    # config["freeze_language_model"] = False
    # config["freeze_multi_modal_projector"] = True
    # config["freeze_vision_tower"] = True
    
    # Add the DeepSpeed configuration when provided.
    if args.deepspeed:
        config["deepspeed"] = args.deepspeed
    
    # Add dataset mixing options when provided.
    if args.mix_strategy:
        config["mix_strategy"] = args.mix_strategy
        
        # Add sampling probabilities for an interleave strategy.
        if "interleave" in args.mix_strategy and args.interleave_probs:
            config["interleave_probs"] = args.interleave_probs
    
    # Add layer-selective tuning options when provided.
    if do_freeze:
        if trainable_layers:
            config["freeze_trainable_layers_ids"] = trainable_layers
        if args.freeze_extra_modules:
            config["freeze_extra_modules"] = args.freeze_extra_modules
    
    return config


def write_yaml(file_path: str, data: Dict[str, Any]) -> None:
    """Write a YAML file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"YAML configuration saved to: {file_path}")


def main():
    args = parse_args()
    
    # Generate the configuration.
    config = generate_full_finetune_config(args)
    
    # Resolve the YAML output path.
    yaml_output_path = args.yaml_output_path
    if yaml_output_path is None:
        yaml_output_path = os.path.join(args.output_dir, "training.yaml")
    
    # Write the YAML file.
    write_yaml(yaml_output_path, config)
    
    # Print a configuration summary.
    print("\n" + "=" * 50)
    print("Training configuration:")
    print("=" * 50)
    print(f"  Model path: {args.model_path}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Template: {args.template}")
    print(f"  Output directory: {args.output_dir}")
    
    # Report the fine-tuning type.
    if args.trainable_layers is not None or args.top_trainable_layers is not None or args.specific_layers is not None:
        print(f"  Fine-tuning type: freeze (layer-selective tuning)")
        if args.trainable_layers is not None:
            print(f"  Trainable bottom layers: {args.trainable_layers}")
        if args.top_trainable_layers is not None:
            print(f"  Trainable top layers: {args.top_trainable_layers}")
        if args.specific_layers is not None:
            print(f"  Specific trainable layers: {args.specific_layers}")
    else:
        print(f"  Fine-tuning type: full (full fine-tuning)")
    
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
