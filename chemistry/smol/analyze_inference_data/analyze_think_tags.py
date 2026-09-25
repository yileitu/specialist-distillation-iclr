#!/usr/bin/env python3
"""
Analyze <think> and </think> tags in inference results.

This script analyzes postprocessed JSONL files from multiple inference directories
and generates statistics reports about the usage of <think> tags in generated text.
"""

import argparse
import json
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def count_tag_occurrences(text: str, tag: str) -> int:
    """Count the number of occurrences of a tag in text."""
    # Escape special regex characters in the tag
    escaped_tag = re.escape(tag)
    return len(re.findall(escaped_tag, text))


def has_valid_think_content(text: str) -> bool:
    """
    Check if text has exactly one <think> and one </think> tag,
    and the content between them is not empty (not just whitespace/newlines).
    """
    # Find all <think> and </think> positions
    think_open = list(re.finditer(r'<think>', text))
    think_close = list(re.finditer(r'</think>', text))
    
    # Must have exactly one of each
    if len(think_open) != 1 or len(think_close) != 1:
        return False
    
    # Check that close comes after open
    open_pos = think_open[0].end()
    close_pos = think_close[0].start()
    
    if close_pos <= open_pos:
        return False
    
    # Extract content between tags
    content = text[open_pos:close_pos]
    
    # Check if content is not just whitespace
    return bool(content.strip())


def analyze_generation(text: str) -> Dict[str, bool]:
    """
    Analyze a single generation text and return statistics.
    
    Returns a dict with:
    - has_close_think: has at least one </think>
    - has_exactly_one_close_think: has exactly one </think>
    - has_open_think: has at least one <think>
    - has_exactly_one_open_think: has exactly one <think>
    - has_valid_pair: has exactly one of each and content between them is not empty
    """
    open_count = count_tag_occurrences(text, '<think>')
    close_count = count_tag_occurrences(text, '</think>')
    
    return {
        'has_close_think': close_count > 0,
        'has_exactly_one_close_think': close_count == 1,
        'has_open_think': open_count > 0,
        'has_exactly_one_open_think': open_count == 1,
        'has_valid_pair': has_valid_think_content(text)
    }


def analyze_jsonl_file(jsonl_path: Path, verbose: bool = False) -> Tuple[Dict[str, Dict[str, int]], int]:
    """
    Analyze a single JSONL file.
    
    Returns:
    - stats: Dict mapping generation_idx to Dict of stat_name -> count
    - total_datapoints: Total number of datapoints in the file
    """
    stats = defaultdict(lambda: defaultdict(int))
    total_datapoints = 0
    
    if verbose:
        print(f"    Processing: {jsonl_path.name}", end='', flush=True)
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            total_datapoints += 1
            data = json.loads(line)
            
            # Get all generations
            generated = data.get('generated', {})
            
            for gen_key, gen_text in generated.items():
                if not isinstance(gen_text, str):
                    continue
                
                # Analyze this generation
                result = analyze_generation(gen_text)
                
                # Update stats
                for stat_name, stat_value in result.items():
                    if stat_value:
                        stats[gen_key][stat_name] += 1
    
    if verbose:
        print(f" - {total_datapoints} datapoints")
    
    return dict(stats), total_datapoints


def analyze_inference_directory(inference_dir: Path, verbose: bool = False) -> Dict[str, Tuple[Dict[str, Dict[str, int]], int]]:
    """
    Analyze all postprocessed JSONL files in an inference directory.
    
    Returns:
    - Dict mapping subtask_name to (stats, total_datapoints)
    """
    results = {}
    
    # Find all postprocessed JSONL files (any file containing "postprocessed" in the name)
    jsonl_files = list(inference_dir.rglob('*postprocessed*.jsonl'))
    
    if verbose:
        print(f"  Found {len(jsonl_files)} postprocessed JSONL files")
    
    for jsonl_file in jsonl_files:
        # Get subtask name (parent directory name)
        subtask_name = jsonl_file.parent.name
        
        # Analyze this file
        stats, total_datapoints = analyze_jsonl_file(jsonl_file, verbose=verbose)
        
        results[subtask_name] = (stats, total_datapoints)
    
    return results


def calculate_average_stats(results: Dict[str, Tuple[Dict[str, Dict[str, int]], int]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate average statistics across all generations for each subtask.
    
    Returns:
    - Dict mapping subtask_name to Dict of stat_name -> average_percentage
    """
    averages = {}
    
    for subtask_name, (stats, total_datapoints) in results.items():
        if not stats or total_datapoints == 0:
            continue
        
        # Initialize accumulator for each stat
        stat_sums = defaultdict(float)
        stat_counts = defaultdict(int)
        
        # Get all generation keys
        gen_keys = [k for k in stats.keys() if k.startswith('generation')]
        
        if not gen_keys:
            continue
        
        # Accumulate stats across all generations
        for gen_key in gen_keys:
            gen_stats = stats[gen_key]
            
            for stat_key in ['has_close_think', 'has_exactly_one_close_think', 
                            'has_open_think', 'has_exactly_one_open_think', 'has_valid_pair']:
                count = gen_stats.get(stat_key, 0)
                percentage = (count / total_datapoints * 100) if total_datapoints > 0 else 0
                stat_sums[stat_key] += percentage
                stat_counts[stat_key] += 1
        
        # Calculate averages
        subtask_averages = {}
        for stat_key in stat_sums.keys():
            subtask_averages[stat_key] = stat_sums[stat_key] / stat_counts[stat_key] if stat_counts[stat_key] > 0 else 0
        
        averages[subtask_name] = subtask_averages
    
    return averages


def format_report(inference_dir: Path, results: Dict[str, Tuple[Dict[str, Dict[str, int]], int]], 
                 averages: Dict[str, Dict[str, float]]) -> str:
    """
    Format the analysis results into a text report.
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"Think Tag Analysis Report")
    lines.append(f"Inference Directory: {inference_dir}")
    lines.append("=" * 80)
    lines.append("")
    
    # Sort subtasks by name for consistent output
    sorted_subtasks = sorted(results.keys())
    
    for subtask_name in sorted_subtasks:
        stats, total_datapoints = results[subtask_name]
        
        lines.append("-" * 80)
        lines.append(f"Subtask: {subtask_name}")
        lines.append(f"Total Datapoints: {total_datapoints}")
        lines.append("-" * 80)
        
        if not stats:
            lines.append("  No generation data found.")
            lines.append("")
            continue
        
        # Get all generation keys and sort them
        gen_keys = sorted(stats.keys(), key=lambda x: int(x.replace('generation', '')))
        
        for gen_key in gen_keys:
            gen_stats = stats[gen_key]
            
            lines.append(f"\n  {gen_key}:")
            lines.append(f"    Total Datapoints: {total_datapoints}")
            
            # Statistics
            stat_labels = [
                ('has_close_think', '1. Contains </think> tag'),
                ('has_exactly_one_close_think', '2. Has exactly one </think>'),
                ('has_open_think', '3. Contains <think> tag'),
                ('has_exactly_one_open_think', '4. Has exactly one <think>'),
                ('has_valid_pair', '5. Has valid <think>...</think> pair with non-empty content')
            ]
            
            for stat_key, stat_label in stat_labels:
                count = gen_stats.get(stat_key, 0)
                percentage = (count / total_datapoints * 100) if total_datapoints > 0 else 0
                lines.append(f"    {stat_label}:")
                lines.append(f"      Count: {count}")
                lines.append(f"      Percentage: {percentage:.2f}%")
        
        # Add average statistics
        if subtask_name in averages:
            lines.append(f"\n  Average across all generations:")
            stat_labels = [
                ('has_close_think', '1. Contains </think> tag'),
                ('has_exactly_one_close_think', '2. Has exactly one </think>'),
                ('has_open_think', '3. Contains <think> tag'),
                ('has_exactly_one_open_think', '4. Has exactly one <think>'),
                ('has_valid_pair', '5. Has valid <think>...</think> pair with non-empty content')
            ]
            
            for stat_key, stat_label in stat_labels:
                avg_percentage = averages[subtask_name].get(stat_key, 0)
                lines.append(f"    {stat_label}: {avg_percentage:.2f}%")
        
        lines.append("")
    
    lines.append("=" * 80)
    lines.append("End of Report")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def plot_statistics(inference_dir: Path, averages: Dict[str, Dict[str, float]], output_path: Path):
    """
    Create a bar chart showing average statistics for each subtask.
    """
    if not averages:
        print("  Warning: No data to plot")
        return
    
    # Sort subtasks by name
    sorted_subtasks = sorted(averages.keys())
    
    # Define the statistics to plot
    stat_keys = [
        'has_close_think',
        'has_exactly_one_close_think',
        'has_open_think',
        'has_exactly_one_open_think',
        'has_valid_pair'
    ]
    
    stat_labels = [
        'Contains </think>',
        'Exactly one </think>',
        'Contains <think>',
        'Exactly one <think>',
        'Valid pair with content'
    ]
    
    # Prepare data
    x = np.arange(len(sorted_subtasks))
    width = 0.15  # Width of each bar
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(14, len(sorted_subtasks) * 0.8), 8))
    
    # Plot bars for each statistic
    for i, (stat_key, stat_label) in enumerate(zip(stat_keys, stat_labels)):
        values = [averages[subtask].get(stat_key, 0) for subtask in sorted_subtasks]
        offset = width * (i - 2)  # Center the bars
        ax.bar(x + offset, values, width, label=stat_label)
    
    # Customize plot
    ax.set_xlabel('Subtask', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Think Tag Statistics (Average across generations)\n{inference_dir}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_subtasks, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (stat_key, stat_label) in enumerate(zip(stat_keys, stat_labels)):
        values = [averages[subtask].get(stat_key, 0) for subtask in sorted_subtasks]
        offset = width * (i - 2)
        for j, v in enumerate(values):
            if v > 2:  # Only show label if value is significant
                ax.text(j + offset, v + 1, f'{v:.1f}', 
                       ha='center', va='bottom', fontsize=7, rotation=90)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Chart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze <think> and </think> tags in inference results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python analyze_think_tags.py \\
    /path/to/inference/dir1 \\
    /path/to/inference/dir2 \\
    /path/to/inference/dir3

This will analyze all postprocessed JSONL files in each directory and generate
a statistics report saved to the same directory as 'think_tag_analysis.txt'.
        """
    )
    
    parser.add_argument(
        'inference_dirs',
        nargs='+',
        type=str,
        help='One or more inference directory paths to analyze'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print verbose progress information'
    )
    
    args = parser.parse_args()
    
    # Process each inference directory
    for inference_dir_str in args.inference_dirs:
        inference_dir = Path(inference_dir_str).resolve()
        
        if not inference_dir.exists():
            print(f"Error: Directory does not exist: {inference_dir}")
            continue
        
        if not inference_dir.is_dir():
            print(f"Error: Not a directory: {inference_dir}")
            continue
        
        print(f"\nAnalyzing: {inference_dir}")
        
        # Analyze all subtasks in this directory
        results = analyze_inference_directory(inference_dir, verbose=args.verbose)
        
        if not results:
            print(f"  Warning: No postprocessed JSONL files found in {inference_dir}")
            continue
        
        # Calculate averages
        averages = calculate_average_stats(results)
        
        # Generate report
        report = format_report(inference_dir, results, averages)
        
        # Save report to the inference directory
        report_path = inference_dir / 'think_tag_analysis.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"  Report saved to: {report_path}")
        
        # Generate plot
        plot_path = inference_dir / 'think_tag_analysis.png'
        plot_statistics(inference_dir, averages, plot_path)
        
        print(f"  Analyzed {len(results)} subtasks")


if __name__ == '__main__':
    main()
