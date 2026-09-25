#!/usr/bin/env python3
"""
Align and merge fine-tuning data across multiple inference modes.

This script:
1. Reads ft_data.jsonl from multiple inference mode folders
2. For each task, downsamples to match the mode with minimum data points
3. Merges all tasks within each mode into a single shuffled JSONL file
4. Ensures all modes have the same total number of data points
5. Removes special tokens from assistant responses during data loading
"""

import json
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import sys

# Add parent directory to path to import util
sys.path.append(str(Path(__file__).parent.parent))
from util import ALL_TASKS


# Special tokens to remove (based on InternLM tokenizer)
SPECIAL_TOKENS = [
    # Chat tokens
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    
    # Object reference tokens
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    
    # Vision tokens
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    
    # Tool tokens
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    
    # Fill-in-middle tokens
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    
    # Code tokens
    "<|repo_name|>",
    "<|file_sep|>",
    
    # Domain-specific tokens
    "<SELFIES>",
    "</SELFIES>",
    "<FASTA>",
    "</FASTA>",
    
    # Image context tokens
    "<IMG_CONTEXT>",
    "<image>",
    "</image>",
    "<img>",
    "</img>",
    "<quad>",
    "</quad>",
    "<ref>",
    "</ref>",
    "<box>",
    "</box>",
    
    # Action tokens
    "<|action_start|>",
    "<|action_end|>",
    "<|interpreter|>",
    "<|plugin|>",
    
    # Video token
    "<video>",
    "</video>",
]


def remove_special_tokens(text: str, tokens: List[str] = SPECIAL_TOKENS) -> str:
    """
    Remove all special tokens from the text.
    
    Args:
        text: Input text containing special tokens
        tokens: List of special tokens to remove (default: SPECIAL_TOKENS)
        
    Returns:
        Text with all special tokens removed
    """
    cleaned_text = text
    for token in tokens:
        cleaned_text = cleaned_text.replace(token, "")
    return cleaned_text


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file and remove special tokens from assistant responses.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries with special tokens removed from assistant content
    """
    data = []
    if not file_path.exists():
        return data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                
                # Process messages to remove special tokens from assistant content
                if 'messages' in item:
                    for message in item['messages']:
                        if message.get('role') == 'assistant' and 'content' in message:
                            message['content'] = remove_special_tokens(message['content'])
                
                # Process direct format with role and content
                elif item.get('role') == 'assistant' and 'content' in item:
                    item['content'] = remove_special_tokens(item['content'])
                
                data.append(item)
    
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: Path):
    """
    Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to output JSONL file
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def collect_task_data(mode_folders: List[Path]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Collect data for all tasks from all modes.
    
    Args:
        mode_folders: List of inference mode folder paths
        
    Returns:
        Dictionary: {task_name: {mode_name: [data_points]}}
    """
    task_data = defaultdict(dict)
    
    print("Collecting data from all modes...")
    for mode_folder in tqdm(mode_folders, desc="Reading modes", unit="mode"):
        mode_name = mode_folder.name
        
        for task_name in ALL_TASKS:
            task_dir = mode_folder / task_name
            ft_data_file = task_dir / "ft_data.jsonl"
            
            if ft_data_file.exists():
                data = load_jsonl(ft_data_file)
                if data:
                    task_data[task_name][mode_name] = data
                    print(f"  {mode_name}/{task_name}: {len(data)} samples")
    
    return task_data


def align_task_data(task_data: Dict[str, Dict[str, List[Dict[str, Any]]]],
                    seed: int = 42) -> Tuple[Dict[str, Dict[str, List[Dict[str, Any]]]], 
                                             Dict[str, Dict[str, Tuple[int, int]]]]:
    """
    Align data across modes by downsampling to the minimum count for each task.
    
    Args:
        task_data: Dictionary of task data from all modes
        seed: Random seed for sampling
        
    Returns:
        Tuple of (aligned_data, sampling_stats)
        - aligned_data: Aligned task data with same counts across modes for each task
        - sampling_stats: {task_name: {mode_name: (original_count, sampled_count)}}
    """
    aligned_data = defaultdict(dict)
    sampling_stats = defaultdict(dict)
    random.seed(seed)
    
    print("\nAligning data across modes...")
    alignment_report = []
    
    for task_name, mode_dict in task_data.items():
        if not mode_dict:
            continue
        
        # Find minimum count across all modes for this task
        min_count = min(len(data) for data in mode_dict.values())
        
        alignment_report.append(f"\n{task_name}:")
        alignment_report.append(f"  Min count: {min_count}")
        
        # Downsample each mode to min_count
        for mode_name, data in mode_dict.items():
            original_count = len(data)
            
            if original_count > min_count:
                # Random sampling without replacement
                sampled_data = random.sample(data, min_count)
                aligned_data[task_name][mode_name] = sampled_data
                sampling_stats[task_name][mode_name] = (original_count, min_count)
                alignment_report.append(f"  {mode_name}: {original_count} -> {min_count} (sampled)")
            else:
                aligned_data[task_name][mode_name] = data
                sampling_stats[task_name][mode_name] = (original_count, original_count)
                alignment_report.append(f"  {mode_name}: {original_count} (unchanged)")
    
    # Print alignment report
    print("\nAlignment Report:")
    print("=" * 70)
    for line in alignment_report:
        print(line)
    print("=" * 70)
    
    return aligned_data, sampling_stats


def merge_and_shuffle_mode_data(aligned_data: Dict[str, Dict[str, List[Dict[str, Any]]]],
                                mode_name: str,
                                seed: int = 42) -> List[Dict[str, Any]]:
    """
    Merge all tasks for a single mode and shuffle.
    
    Args:
        aligned_data: Aligned task data
        mode_name: Name of the mode to merge
        seed: Random seed for shuffling
        
    Returns:
        Merged and shuffled data for the mode
    """
    merged_data = []
    
    for task_name, mode_dict in aligned_data.items():
        if mode_name in mode_dict:
            merged_data.extend(mode_dict[mode_name])
    
    # Shuffle the merged data
    random.seed(seed)
    random.shuffle(merged_data)
    
    return merged_data


def main():
    parser = argparse.ArgumentParser(
        description="Align and merge fine-tuning data across multiple inference modes"
    )
    parser.add_argument(
        'mode_folders',
        nargs='+',
        type=str,
        help='Paths to inference mode folders (e.g., ft_data/Intern-S1-mini_...)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: same as script directory)'
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='aligned_ft_data',
        help='Prefix for output files (default: aligned_ft_data)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling and shuffling (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    mode_folders = [Path(folder) for folder in args.mode_folders]
    
    # Validate folders
    valid_folders = []
    for folder in mode_folders:
        if not folder.exists():
            print(f"Warning: Folder does not exist: {folder}", file=sys.stderr)
        elif not folder.is_dir():
            print(f"Warning: Not a directory: {folder}", file=sys.stderr)
        else:
            valid_folders.append(folder)
    
    if len(valid_folders) < 2:
        print("Error: Need at least 2 valid mode folders", file=sys.stderr)
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir is None:
        # Default to ft_data/aligned_data
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "ft_data" / "aligned_data"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Align and Merge Fine-tuning Data")
    print("=" * 70)
    print(f"Number of modes: {len(valid_folders)}")
    print(f"Output directory: {output_dir}")
    print(f"Output prefix: {args.output_prefix}")
    print(f"Random seed: {args.seed}")
    print(f"Special tokens removal: Enabled ({len(SPECIAL_TOKENS)} token types)")
    print("=" * 70)
    print("\nMode folders:")
    for folder in valid_folders:
        print(f"  - {folder.name}")
    print("")
    
    # Step 1: Collect data from all modes
    task_data = collect_task_data(valid_folders)
    
    if not task_data:
        print("Error: No data found in any mode folder", file=sys.stderr)
        sys.exit(1)
    
    # Step 2: Align data across modes
    aligned_data, sampling_stats = align_task_data(task_data, seed=args.seed)
    
    # Step 3: Merge and shuffle for each mode
    print("\nMerging and shuffling data for each mode...")
    mode_stats = {}
    
    for mode_folder in tqdm(valid_folders, desc="Processing modes", unit="mode"):
        mode_name = mode_folder.name
        
        # Merge all tasks for this mode
        merged_data = merge_and_shuffle_mode_data(aligned_data, mode_name, seed=args.seed)
        
        if not merged_data:
            print(f"Warning: No data for mode {mode_name}, skipping", file=sys.stderr)
            continue
        
        # Save merged data
        output_file = output_dir / f"{args.output_prefix}_{mode_name}.jsonl"
        save_jsonl(merged_data, output_file)
        
        mode_stats[mode_name] = {
            "output_file": str(output_file),
            "total_samples": len(merged_data)
        }
        
        print(f"  {mode_name}: {len(merged_data)} samples -> {output_file.name}")
    
    # Step 4: Save statistics
    stats_file = output_dir / f"{args.output_prefix}_statistics.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Aligned and Merged Fine-tuning Data Statistics\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Random seed: {args.seed}\n")
        f.write(f"Number of modes: {len(mode_stats)}\n")
        f.write(f"Special tokens removed: {len(SPECIAL_TOKENS)} token types\n")
        f.write("  (Removed from assistant responses during data loading)\n")
        f.write("\n")
        
        f.write("Per-mode Statistics:\n")
        f.write("-" * 70 + "\n")
        for mode_name, stats in mode_stats.items():
            f.write(f"\n{mode_name}:\n")
            f.write(f"  Output file: {stats['output_file']}\n")
            f.write(f"  Total samples: {stats['total_samples']}\n")
        
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("Per-task Alignment (Before -> After Sampling):\n")
        f.write("=" * 70 + "\n")
        
        for task_name in sorted(aligned_data.keys()):
            f.write(f"\n{task_name}:\n")
            mode_dict = aligned_data[task_name]
            stats_dict = sampling_stats.get(task_name, {})
            
            for mode_name in sorted(mode_dict.keys()):
                if mode_name in stats_dict:
                    original_count, sampled_count = stats_dict[mode_name]
                    if original_count == sampled_count:
                        f.write(f"  {mode_name}: {sampled_count} samples (unchanged)\n")
                    else:
                        f.write(f"  {mode_name}: {original_count} -> {sampled_count} samples\n")
                else:
                    count = len(mode_dict[mode_name])
                    f.write(f"  {mode_name}: {count} samples\n")
        
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("Verification:\n")
        f.write("-" * 70 + "\n")
        
        # Verify all modes have the same total count
        total_counts = [stats['total_samples'] for stats in mode_stats.values()]
        if len(set(total_counts)) == 1:
            f.write(f"✓ All modes have the same total samples: {total_counts[0]}\n")
        else:
            f.write(f"✗ WARNING: Modes have different total samples: {total_counts}\n")
        
        # Calculate per-task contribution
        f.write("\nPer-task contribution to total:\n")
        for task_name, mode_dict in sorted(aligned_data.items()):
            if mode_dict:
                # All modes should have same count per task after alignment
                sample_count = len(next(iter(mode_dict.values())))
                f.write(f"  {task_name}: {sample_count} samples\n")
        
        f.write("=" * 70 + "\n")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("Final Summary")
    print("=" * 70)
    
    for mode_name, stats in mode_stats.items():
        print(f"\n{mode_name}:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Output: {stats['output_file']}")
    
    # Verify all modes have same count
    total_counts = [stats['total_samples'] for stats in mode_stats.values()]
    print("\nVerification:")
    if len(set(total_counts)) == 1:
        print(f"  ✓ All modes have the same total samples: {total_counts[0]}")
    else:
        print(f"  ✗ WARNING: Modes have different totals: {total_counts}")
    
    print(f"\nStatistics saved to: {stats_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()
