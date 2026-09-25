#!/usr/bin/env python3
"""
Register a data file in LLaMA-Factory's dataset_info.json.

Usage:
    python data_prepare.py \
        --data_file /path/to/your/data.json \
        --dataset_name my_dataset \
        --dataset_info_path /path/to/LLaMA-Factory/data/dataset_info.json

Expected ShareGPT format:
    One JSON object per line with a "messages" field:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

import argparse
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Validate and register data with LLaMA-Factory')
    parser.add_argument(
        '--data_file',
        type=str,
        required=True,
        help='Path to the input JSONL file'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default=None,
        help='Dataset name (default: input filename)'
    )
    parser.add_argument(
        '--dataset_info_path',
        type=str,
        required=True,
        help='Path to LLaMA-Factory dataset_info.json'
    )
    parser.add_argument(
        '--copy_to_dir',
        type=str,
        default=None,
        help='Optional directory to copy the data file into'
    )
    return parser.parse_args()


def validate_data_format(data_file: str) -> int:
    """Validate that a data file follows the ShareGPT format."""
    count = 0
    with open(data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if 'messages' not in data:
                    raise ValueError(f"Line {line_num} is missing the 'messages' field")
                messages = data['messages']
                if not isinstance(messages, list):
                    raise ValueError(f"Line {line_num}: 'messages' must be a list")
                for msg in messages:
                    if 'role' not in msg or 'content' not in msg:
                        raise ValueError(f"Line {line_num}: a message is missing 'role' or 'content'")
                count += 1
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}")
    return count


def register_dataset(dataset_info_path: str, dataset_name: str, data_file_path: str):
    """Register a dataset in dataset_info.json."""
    # Read the existing dataset_info.json
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}
        # Create the directory
        os.makedirs(os.path.dirname(dataset_info_path), exist_ok=True)
    
    # Add the new dataset configuration
    dataset_info[dataset_name] = {
        "file_name": data_file_path,
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages"
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system"
        }
    }
    
    # Write the updated data back to disk
    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"Registered dataset '{dataset_name}' in {dataset_info_path}")


def main():
    args = parse_args()
    
    # Verify that the input file exists
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")
    
    # Resolve the dataset name
    dataset_name = args.dataset_name
    if dataset_name is None:
        dataset_name = Path(args.data_file).stem
    
    # Validate the data format
    print(f"Validating data format: {args.data_file}")
    sample_count = validate_data_format(args.data_file)
    print(f"Validation passed: {sample_count} samples")
    
    # Resolve the final data file path
    final_data_path = args.data_file
    
    # When a copy directory is specified
    if args.copy_to_dir:
        os.makedirs(args.copy_to_dir, exist_ok=True)
        dest_file = os.path.join(args.copy_to_dir, os.path.basename(args.data_file))
        import shutil
        shutil.copy2(args.data_file, dest_file)
        final_data_path = dest_file
        print(f"Copied data file to: {dest_file}")
    
    # Register the dataset
    register_dataset(args.dataset_info_path, dataset_name, final_data_path)
    
    print(f"\nDataset name: {dataset_name}")
    print(f"Data file: {final_data_path}")
    print(f"Sample count: {sample_count}")


if __name__ == '__main__':
    main()
