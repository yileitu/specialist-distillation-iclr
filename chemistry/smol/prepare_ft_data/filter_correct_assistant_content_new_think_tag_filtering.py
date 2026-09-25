#!/usr/bin/env python3
"""
Prepare fine-tuning data from postprocessed inference results.

This script:
1. Reads postprocessed JSONL files from multiple inference folders
2. Filters data points where extracted_core_answer_correctness is True
3. Randomly selects one generation if multiple are True
4. Converts prompt and generated text to ShareGPT format
5. Saves to ft_data directory organized by inference mode and task
"""

import json
import argparse
import random
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
from datetime import datetime
import sys

# Add parent directory to path to import util
sys.path.append(str(Path(__file__).parent.parent))
from util import ALL_TASKS


def find_postprocessed_jsonl(task_dir: Path) -> Optional[Path]:
    """
    Find the postprocessed JSONL file in a task directory.
    
    Looks for files matching patterns:
    - *_postprocessed*.jsonl
    - *_bool_postprocessed.jsonl
    - *_num_postprocessed*.jsonl
    - *_smiles_postprocessed.jsonl
    - *_postprocessed_lev*.jsonl
    
    Args:
        task_dir: Path to task directory
        
    Returns:
        Path to postprocessed JSONL file, or None if not found
    """
    # Try common patterns
    patterns = [
        "*_postprocessed*.jsonl",
        "*_bool_postprocessed.jsonl",
        "*_num_postprocessed*.jsonl",
        "*_smiles_postprocessed.jsonl",
        "*_postprocessed_lev*.jsonl",
        "*_postprocessed_METEOR*.jsonl",
    ]
    
    for pattern in patterns:
        matches = list(task_dir.glob(pattern))
        if matches:
            # Return the first match (should be only one)
            return matches[0]
    
    return None


def extract_text_from_prompt(prompt: List[Dict[str, Any]]) -> str:
    """
    Extract text content from prompt format.
    
    Args:
        prompt: List of prompt dictionaries with role and content
        
    Returns:
        Combined text content
    """
    text_parts = []
    for item in prompt:
        if item.get("role") == "user":
            content = item.get("content", [])
            if isinstance(content, list):
                for content_item in content:
                    if isinstance(content_item, dict) and content_item.get("type") == "text":
                        text_parts.append(content_item.get("text", ""))
            elif isinstance(content, str):
                text_parts.append(content)
    
    return "".join(text_parts)


def _has_substantive_content(text: str) -> bool:
    """Check if text contains substantive content (not just whitespace or punctuation)."""
    return bool(re.search(r'\w', text))


def convert_to_sharegpt_format(
    prompt: List[Dict[str, Any]], 
    generated_text: str,
    task: str,
    sample_id: str,
    ground_truth: str,
    output_core_tag_left: str,
    output_core_tag_right: str
) -> Dict[str, Any]:
    """
    Convert prompt and generated text to ShareGPT format with additional metadata.
    
    Args:
        prompt: Original prompt format
        generated_text: Generated text from model
        task: Task name
        sample_id: Sample ID
        ground_truth: Ground truth answer
        output_core_tag_left: Left tag for output
        output_core_tag_right: Right tag for output
        
    Returns:
        ShareGPT format dictionary with metadata
    """
    # Extract user message from prompt
    user_content = extract_text_from_prompt(prompt)
    
    # Create ShareGPT format
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": generated_text}
    ]
    
    return {
        "messages": messages,
        "task": task,
        "sample_id": sample_id,
        "ground_truth": ground_truth,
        "output_core_tag_left": output_core_tag_left,
        "output_core_tag_right": output_core_tag_right
    }


def process_jsonl_file(jsonl_path: Path, output_path: Path) -> Dict[str, int]:
    """
    Process a single postprocessed JSONL file.
    
    Args:
        jsonl_path: Path to input JSONL file
        output_path: Path to output JSONL file
        
    Returns:
        Dictionary with statistics (total, filtered, saved)
    """
    stats = {
        "total": 0,
        "filtered": 0,           # Has at least one True in correctness
        "saved": 0,              # Actually saved (after random selection)
        "correct_generations": 0  # Total number of correct generations (can be > total if multiple correct per sample)
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Count total lines for progress bar
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f if line.strip())
    
    with open(jsonl_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        # Create progress bar
        pbar = tqdm(total=total_lines, desc=f"  Processing {jsonl_path.name}", unit="lines", leave=False)
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            stats["total"] += 1
            pbar.update(1)
            data = json.loads(line)
            
            # Get correctness dictionary
            correctness = data.get("extracted_core_answer_correctness", {})
            if not correctness:
                continue
            
            # Find all generations that are True
            true_generations = [
                gen_key for gen_key, is_correct in correctness.items()
                if is_correct is True
            ]
            
            if not true_generations:
                continue
            
            stats["filtered"] += 1
            stats["correct_generations"] += len(true_generations)  # Count all correct generations
            
            # Get the generated texts for all correct generations
            generated = data.get("generated", {})
            
            # Filter generations that meet think-tag formatting requirements:
            #   Case 1: exactly one <think> at the start and one </think> after it,
            #           with substantive content both between the tags and after </think>.
            #   Case 2: no <think> tag, exactly one </think>, with substantive content
            #           both before and after </think>. Will prepend "<think> " later.
            valid_generations: List[Tuple[str, str]] = []
            for gen_key in true_generations:
                gen_text = generated.get(gen_key, "")
                if not gen_text:
                    continue

                think_open_count = gen_text.count("<think>")
                think_close_count = gen_text.count("</think>")

                if think_open_count == 1 and think_close_count == 1:
                    if not gen_text.lstrip().startswith("<think>"):
                        continue
                    open_idx = gen_text.index("<think>")
                    close_idx = gen_text.index("</think>")
                    if close_idx <= open_idx:
                        continue
                    between = gen_text[open_idx + len("<think>"):close_idx]
                    after = gen_text[close_idx + len("</think>"):]
                    if _has_substantive_content(between) and _has_substantive_content(after):
                        valid_generations.append((gen_key, "case1"))

                elif think_open_count == 0 and think_close_count == 1:
                    close_idx = gen_text.index("</think>")
                    before = gen_text[:close_idx]
                    after = gen_text[close_idx + len("</think>"):]
                    if _has_substantive_content(before) and _has_substantive_content(after):
                        valid_generations.append((gen_key, "case2"))

            if not valid_generations:
                continue

            selected_gen, case_type = random.choice(valid_generations)
            generated_text = generated.get(selected_gen, "")

            if not generated_text:
                continue

            if case_type == "case2":
                generated_text = "<think> " + generated_text
            
            # Get prompt
            prompt = data.get("prompt", [])
            if not prompt:
                continue
            
            # Get metadata fields
            task = data.get("task", "")
            sample_id = data.get("sample_id", "")
            ground_truth = data.get("ground_truth", "")
            output_core_tag_left = data.get("output_core_tag_left", "")
            output_core_tag_right = data.get("output_core_tag_right", "")
            
            # Convert to ShareGPT format
            sharegpt_data = convert_to_sharegpt_format(
                prompt, 
                generated_text,
                task,
                sample_id,
                ground_truth,
                output_core_tag_left,
                output_core_tag_right
            )
            
            # Write to output
            f_out.write(json.dumps(sharegpt_data, ensure_ascii=False) + '\n')
            stats["saved"] += 1
        
        pbar.close()
    
    return stats


def process_inference_folder(inference_folder: Path, output_base: Path) -> Dict[str, Any]:
    """
    Process all task folders in an inference folder.
    
    Args:
        inference_folder: Path to inference folder (e.g., Intern-S1-mini/presence0.0_...)
        output_base: Base output directory (ft_data)
        
    Returns:
        Dictionary with statistics
    """
    # Extract inference mode name from folder path
    # e.g., Intern-S1-mini/presence0.0_freq0.0_repetition1.0_loop5-200_rep3
    # -> Intern-S1-mini_presence0.0_freq0.0_repetition1.0_loop5-200_rep3
    folder_parts = inference_folder.parts
    
    # Get the last two parts (model_name and param_folder)
    # Handle both absolute and relative paths
    if len(folder_parts) >= 2:
        model_name = folder_parts[-2]
        param_folder = folder_parts[-1]
        inference_mode = f"{model_name}_{param_folder}"
    else:
        # Fallback: use the folder name
        inference_mode = inference_folder.name
    
    # Sanitize inference_mode name (remove invalid characters for file paths)
    inference_mode = inference_mode.replace('/', '_').replace('\\', '_')
    
    output_dir = output_base / inference_mode
    
    print(f"\nProcessing inference folder: {inference_folder}")
    print(f"  Output directory: {output_dir}")
    
    all_stats = {}
    total_stats = {"total": 0, "filtered": 0, "saved": 0, "correct_generations": 0}
    
    # Collect available tasks first
    available_tasks = []
    for task_name in ALL_TASKS:
        task_dir = inference_folder / task_name
        if task_dir.exists() and task_dir.is_dir():
            jsonl_file = find_postprocessed_jsonl(task_dir)
            if jsonl_file is not None:
                available_tasks.append((task_name, task_dir, jsonl_file))
    
    print(f"  Found {len(available_tasks)} tasks with postprocessed files")
    print("")
    
    # Process each task folder with progress bar
    for task_name, task_dir, jsonl_file in tqdm(available_tasks, desc=f"Processing tasks", unit="task"):
        print(f"\n  Task {task_name}: Processing {jsonl_file.name}")
        
        # Output path
        output_file = output_dir / task_name / "ft_data.jsonl"
        
        # Process the file
        stats = process_jsonl_file(jsonl_file, output_file)
        all_stats[task_name] = stats
        
        # Update totals
        total_stats["total"] += stats["total"]
        total_stats["filtered"] += stats["filtered"]
        total_stats["saved"] += stats["saved"]
        total_stats["correct_generations"] += stats["correct_generations"]
        
        # Calculate rates
        save_rate = (stats['saved'] / stats['total'] * 100) if stats['total'] > 0 else 0
        correct_gen_rate = (stats['correct_generations'] / stats['total'] * 100) if stats['total'] > 0 else 0
        
        print(f"    Total: {stats['total']}, Saved: {stats['saved']} ({save_rate:.2f}%), Correct Gens: {stats['correct_generations']} ({correct_gen_rate:.2f}%)")
    
    print(f"\n  Summary for {inference_mode}:")
    save_rate = (total_stats['saved'] / total_stats['total'] * 100) if total_stats['total'] > 0 else 0
    correct_gen_rate = (total_stats['correct_generations'] / total_stats['total'] * 100) if total_stats['total'] > 0 else 0
    print(f"    Total: {total_stats['total']}, Saved: {total_stats['saved']} ({save_rate:.2f}%), Correct Gens: {total_stats['correct_generations']} ({correct_gen_rate:.2f}%)")
    
    # Save statistics to text file
    stats_file = output_dir / "statistics.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"Statistics for: {inference_mode}\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source folder: {Path(inference_folder).resolve()}\n")
        f.write("\n")
        f.write("Per-task Statistics:\n")
        f.write("-" * 70 + "\n")
        
        for task_name in ALL_TASKS:
            if task_name in all_stats:
                stats = all_stats[task_name]
                f.write(f"\n{task_name}:\n")
                f.write(f"  Total samples: {stats['total']}\n")
                f.write(f"  Saved to ft_data: {stats['saved']}\n")
                f.write(f"  Total correct generations: {stats['correct_generations']}\n")
                if stats['total'] > 0:
                    save_rate = (stats['saved'] / stats['total']) * 100
                    correct_gen_rate = (stats['correct_generations'] / stats['total']) * 100
                    f.write(f"  Save rate: {save_rate:.2f}%\n")
                    f.write(f"  Correct generation rate: {correct_gen_rate:.2f}%\n")
        
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("Overall Summary:\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total samples: {total_stats['total']}\n")
        f.write(f"Saved to ft_data: {total_stats['saved']}\n")
        f.write(f"Total correct generations: {total_stats['correct_generations']}\n")
        
        if total_stats['total'] > 0:
            save_rate = (total_stats['saved'] / total_stats['total']) * 100
            correct_gen_rate = (total_stats['correct_generations'] / total_stats['total']) * 100
            f.write(f"Save rate: {save_rate:.2f}%\n")
            f.write(f"Correct generation rate: {correct_gen_rate:.2f}%\n")
            f.write(f"\nNote: Correct generation rate can exceed 100% when multiple\n")
            f.write(f"      generations per sample are correct. For example, with 3\n")
            f.write(f"      generations per sample, the maximum rate is 300%.\n")
        
        f.write("=" * 70 + "\n")
    
    print(f"  Statistics saved to: {stats_file}")
    
    return {
        "inference_mode": inference_mode,
        "stats": all_stats,
        "total_stats": total_stats,
        "stats_file": str(stats_file)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare fine-tuning data from postprocessed inference results"
    )
    parser.add_argument(
        'inference_folders',
        nargs='+',
        type=str,
        help='Paths to inference folders (each represents an inference mode)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: smol/ft_data)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for selecting generations (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Determine output directory
    if args.output_dir is None:
        script_dir = Path(__file__).parent
        output_base = script_dir.parent / "ft_data"
    else:
        output_base = Path(args.output_dir)
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Fine-tuning Data Preparation")
    print("=" * 70)
    print(f"Output directory: {output_base}")
    print(f"Random seed: {args.seed}")
    print(f"Number of inference folders: {len(args.inference_folders)}")
    print("=" * 70)
    
    all_results = []
    
    # Process each inference folder
    for folder_path in args.inference_folders:
        inference_folder = Path(folder_path)
        
        if not inference_folder.exists():
            print(f"Warning: Folder does not exist: {inference_folder}", file=sys.stderr)
            continue
        
        if not inference_folder.is_dir():
            print(f"Warning: Not a directory: {inference_folder}", file=sys.stderr)
            continue
        
        result = process_inference_folder(inference_folder, output_base)
        all_results.append(result)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("Final Summary")
    print("=" * 70)
    
    grand_total = {"total": 0, "saved": 0, "correct_generations": 0}
    
    for result in all_results:
        mode = result["inference_mode"]
        stats = result["total_stats"]
        stats_file = result.get("stats_file", "N/A")
        print(f"\n{mode}:")
        print(f"  Total: {stats['total']}")
        print(f"  Saved: {stats['saved']}")
        print(f"  Correct generations: {stats['correct_generations']}")
        if stats['total'] > 0:
            save_rate = (stats['saved'] / stats['total']) * 100
            correct_gen_rate = (stats['correct_generations'] / stats['total']) * 100
            print(f"  Save rate: {save_rate:.2f}%")
            print(f"  Correct generation rate: {correct_gen_rate:.2f}%")
        print(f"  Statistics file: {stats_file}")
        
        grand_total["total"] += stats["total"]
        grand_total["saved"] += stats["saved"]
        grand_total["correct_generations"] += stats.get("correct_generations", 0)
    
    print("\n" + "=" * 70)
    print("Grand Total:")
    print(f"  Total: {grand_total['total']}")
    print(f"  Saved: {grand_total['saved']}")
    print(f"  Correct generations: {grand_total['correct_generations']}")
    if grand_total['total'] > 0:
        save_rate = (grand_total['saved'] / grand_total['total']) * 100
        correct_gen_rate = (grand_total['correct_generations'] / grand_total['total']) * 100
        print(f"  Save rate: {save_rate:.2f}%")
        print(f"  Correct generation rate: {correct_gen_rate:.2f}%")
    print("=" * 70)
    
    print(f"\nAll data saved to: {output_base}")
    print(f"Statistics files saved in each inference mode folder")


if __name__ == '__main__':
    main()
