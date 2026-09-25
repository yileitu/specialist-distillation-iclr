#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Randomly sample JSONL records grouped by the task field.

Read a source data file, sample a requested number of records for each task,
and save each task to a separate file.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def load_jsonl_data(data_file: str) -> List[Dict[str, Any]]:
    """
    Load records from a JSONL file.
    
    Args:
        data_file: Path to the JSONL data file.
    
    Returns:
        A list of records.
    """
    data = []
    data_path = Path(data_file)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print(f"Loading data file: {data_file}")
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue
    
    print(f"Data loaded: {len(data)} total samples")
    return data


def group_data_by_task(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group records by their task field.
    
    Args:
        data: Records to group.
    
    Returns:
        A mapping of the form {task_name: [samples]}.
    """
    task_groups = defaultdict(list)
    no_task_count = 0
    
    for sample in data:
        task = sample.get('task')
        if task:
            task_groups[task].append(sample)
        else:
            no_task_count += 1
    
    if no_task_count > 0:
        print(f"Skipped {no_task_count} records without a task field")
    
    return dict(task_groups)


def sample_and_save_by_task(
    data_file: str,
    output_dir: str,
    samples_per_task: int = 50000,
    seed: int = 42,
):
    """
    Group records by task, randomly sample them, and save separate files.
    
    Args:
        data_file: Input data path.
        output_dir: Output directory.
        samples_per_task: Number of samples to keep per task.
        seed: Random seed.
    """
    # Set the random seed
    random.seed(seed)
    
    print(f"=" * 60)
    print("Sample JSONL data by task")
    print(f"=" * 60)
    print(f"Input file: {data_file}")
    print(f"Output directory: {output_dir}")
    print(f"Samples per task: {samples_per_task}")
    print(f"Random seed: {seed}")
    print(f"=" * 60)
    
    # Create the output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {output_path}")
    
    # Load data
    all_data = load_jsonl_data(data_file)
    
    # Group by task
    print("\nGrouping records by task...")
    task_groups = group_data_by_task(all_data)
    
    print(f"\nDetected {len(task_groups)} distinct tasks:")
    for task, samples in sorted(task_groups.items()):
        print(f"  - {task}: {len(samples)} samples")
    
    # Sample and save each task
    print("\nProcessing tasks...")
    task_statistics = []  # Track statistics for each task
    
    for task, samples in sorted(task_groups.items()):
        print(f"\n{'=' * 40}")
        print(f"Processing task: {task}")
        print(f"{'=' * 40}")
        print(f"  Original samples: {len(samples)}")
        
        # Sample records
        if samples_per_task < len(samples):
            sampled_data = random.sample(samples, samples_per_task)
            print(f"  Sampled records: {len(sampled_data)}")
        else:
            sampled_data = samples
            print(f"  Fewer than {samples_per_task} records available; using all {len(sampled_data)}")
        
        # Save to a file named after the task
        # Sanitize special characters in task names for safe filenames
        safe_task_name = task.replace('/', '_').replace('\\', '_').replace(' ', '_')
        output_file = output_path / f"{safe_task_name}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in sampled_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"  Saved to: {output_file}")
        
        # Record statistics
        task_statistics.append({
            'task': task,
            'original_count': len(samples),
            'sampled_count': len(sampled_data),
            'output_file': f"{safe_task_name}.jsonl"
        })
        
        # Print the first two examples
        print("  First two sample records:")
        for i, sample in enumerate(sampled_data[:2], start=1):
            # Print selected sample fields
            sample_id = sample.get('sample_id', 'N/A')
            input_preview = str(sample.get('input', ''))[:80]
            print(f"    [{i}] sample_id={sample_id}, input={input_preview}...")
    
    # Save statistics to a text file
    statistics_file = output_path / "task_statistics.txt"
    with open(statistics_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Per-task sampling statistics\n")
        f.write("=" * 80 + "\n")
        f.write(f"Input file: {data_file}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Target samples per task: {samples_per_task}\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"Total tasks: {len(task_statistics)}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write detailed statistics
        f.write(f"{'Task':<40} {'Original':>12} {'Sampled':>12} {'Output file':<40}\n")
        f.write("-" * 80 + "\n")
        
        total_original = 0
        total_sampled = 0
        for stat in task_statistics:
            f.write(f"{stat['task']:<40} {stat['original_count']:>12,} {stat['sampled_count']:>12,} {stat['output_file']:<40}\n")
            total_original += stat['original_count']
            total_sampled += stat['sampled_count']
        
        f.write("-" * 80 + "\n")
        f.write(f"{'Total':<40} {total_original:>12,} {total_sampled:>12,}\n")
        f.write("=" * 80 + "\n")
        
        # Write supplementary information
        f.write("\nAdditional information:\n")
        f.write(f"- Tasks with fewer than {samples_per_task} records: {sum(1 for s in task_statistics if s['sampled_count'] < samples_per_task)}\n")
        f.write(f"- Tasks with at least {samples_per_task} records: {sum(1 for s in task_statistics if s['sampled_count'] >= samples_per_task)}\n")
        
        # List tasks with insufficient samples
        insufficient_tasks = [s for s in task_statistics if s['sampled_count'] < samples_per_task]
        if insufficient_tasks:
            f.write(f"\nTasks with fewer than {samples_per_task} records:\n")
            for stat in insufficient_tasks:
                f.write(f"  - {stat['task']}: {stat['original_count']} samples\n")
    
    print(f"\nStatistics saved to: {statistics_file}")
    
    print(f"\n{'=' * 60}")
    print("Processing complete!")
    print(f"All task data saved to: {output_dir}")
    print(f"Statistics saved to: {statistics_file}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description='Randomly sample JSONL records grouped by task',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sample 50,000 records per task
  python sample_by_task.py --data_file /path/to/input.jsonl

  # Set a custom sample size and output directory
  python sample_by_task.py --data_file /path/to/input.jsonl --samples_per_task 10000 --output_dir /path/to/output
        """
    )

    parser.add_argument(
        '--data_file',
        type=str,
        required=True,
        help='Path to the input JSONL data file'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(Path(__file__).resolve().parent / 'sample_50k_all_subtasks'),
        help='Output directory (default: sample_50k_all_subtasks next to this script)'
    )
    
    parser.add_argument(
        '--samples_per_task',
        type=int,
        default=50000,
        help='Number of samples per task (default: 50000)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    sample_and_save_by_task(
        data_file=args.data_file,
        output_dir=args.output_dir,
        samples_per_task=args.samples_per_task,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
