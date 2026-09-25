#!/usr/bin/env python3
"""
Analyze METEOR score distribution from postprocessed molecule captioning task files.

This script reads JSONL files containing METEOR scores from the extracted_core_answer field
and generates a single distribution plot combining all generations.

Usage:
    python analyze_meteor_distribution.py <path_to_jsonl_file1> [<path_to_jsonl_file2> ...]
    
Example:
    python analyze_meteor_distribution.py /path/to/inference_split1000_postprocessed_METEOR0.30.jsonl
    python analyze_meteor_distribution.py file1.jsonl file2.jsonl file3.jsonl
"""

import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Read JSONL file and return list of records."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num} due to JSON decode error: {e}")
    return data


def extract_meteor_scores(data: List[Dict[str, Any]]) -> List[float]:
    """
    Extract METEOR scores from extracted_core_answer field.
    
    Concatenates all scores from all generations into a single list.
    
    Returns a list of all METEOR scores.
    """
    all_scores = []
    
    for record in data:
        extracted_answer = record.get('extracted_core_answer', {})
        if extracted_answer:
            for gen_name, score in extracted_answer.items():
                if score is not None and isinstance(score, (int, float)):
                    all_scores.append(float(score))
    
    return all_scores


def plot_score_distribution(scores: List[float], 
                            output_dir: str,
                            base_filename: str):
    """
    Plot METEOR score distribution for all generations combined.
    
    Args:
        scores: List of all METEOR scores (all generations concatenated)
        output_dir: Directory to save the plots
        base_filename: Base filename for the output plots
    """
    if len(scores) == 0:
        print("No valid METEOR scores found in the data.")
        return
    
    # Calculate statistics
    scores_array = np.array(scores)
    mean_score = np.mean(scores_array)
    median_score = np.median(scores_array)
    std_score = np.std(scores_array)
    min_score = np.min(scores_array)
    max_score = np.max(scores_array)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot histogram
    n, bins, patches = ax.hist(scores, bins=50, alpha=0.7, color='skyblue', 
                               edgecolor='black', density=False)
    
    # Add KDE (kernel density estimation) overlay
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(scores_array)
        # METEOR score is in [0, 1], fix x-axis range
        x_range = np.linspace(0.0, 1.0, 200)
        kde_values = kde(x_range)
        # Scale KDE to match histogram
        kde_scaled = kde_values * len(scores) * (bins[1] - bins[0])
        ax2 = ax.twinx()
        ax2.plot(x_range, kde_values, 'r-', linewidth=2, label='KDE')
        ax2.set_ylabel('Density', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.legend(loc='upper right')
    except Exception as e:
        print(f"Warning: Could not compute KDE: {e}")
    
    # Add vertical lines for mean and median
    ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {mean_score:.4f}')
    ax.axvline(median_score, color='green', linestyle='--', linewidth=2, 
              label=f'Median: {median_score:.4f}')
    
    # Set labels and title
    ax.set_xlabel('METEOR Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'METEOR Score Distribution (All Generations Combined)\n'
                f'N={len(scores)}, Mean={mean_score:.4f}, Median={median_score:.4f}, '
                f'Std={std_score:.4f}\nMin={min_score:.4f}, Max={max_score:.4f}',
                fontsize=12)
    # Fix x-axis range to [0, 1]
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'{base_filename}_meteor_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved distribution plot to: {output_path}")
    plt.close()


def print_statistics(scores: List[float], output_path: str = None):
    """
    Print statistical summary of METEOR scores and save to file.
    
    Args:
        scores: List of METEOR scores
        output_path: Optional path to save statistics to file
    """
    scores_array = np.array(scores)
    
    # Prepare statistics text
    stats_lines = []
    stats_lines.append("=" * 80)
    stats_lines.append("METEOR Score Statistics Summary (All Generations Combined)")
    stats_lines.append("=" * 80)
    stats_lines.append("")
    stats_lines.append(f"Total samples: {len(scores)}")
    stats_lines.append(f"Mean:          {np.mean(scores_array):.6f}")
    stats_lines.append(f"Median:        {np.median(scores_array):.6f}")
    stats_lines.append(f"Std Dev:       {np.std(scores_array):.6f}")
    stats_lines.append(f"Min:           {np.min(scores_array):.6f}")
    stats_lines.append(f"Max:           {np.max(scores_array):.6f}")
    stats_lines.append(f"25th percentile: {np.percentile(scores_array, 25):.6f}")
    stats_lines.append(f"75th percentile: {np.percentile(scores_array, 75):.6f}")
    
    # Count zero scores
    zero_count = np.sum(scores_array == 0)
    zero_pct = (zero_count / len(scores)) * 100
    stats_lines.append(f"Zero scores:   {zero_count} ({zero_pct:.2f}%)")
    stats_lines.append("=" * 80)
    
    # Print to console
    print("\n" + "\n".join(stats_lines))
    
    # Save to file if output_path is provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(stats_lines) + "\n")
        print(f"\nStatistics saved to: {output_path}")


def process_single_file(input_file: str, output_dir: str = None) -> int:
    """
    Process a single input file and generate distribution plots.
    
    Args:
        input_file: Path to the postprocessed JSONL file
        output_dir: Optional output directory for plots
        
    Returns:
        0 on success, 1 on error
    """
    # Validate input file
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return 1
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_file))
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename without extension
    base_filename = Path(input_file).stem
    
    print(f"\n{'='*80}")
    print(f"Processing file: {input_file}")
    print(f"{'='*80}")
    
    print(f"Reading data from: {input_file}")
    data = read_jsonl(input_file)
    print(f"Total records read: {len(data)}")
    
    print("\nExtracting METEOR scores...")
    scores = extract_meteor_scores(data)
    
    if not scores:
        print("Error: No valid METEOR scores found in the data.")
        return 1
    
    # Print statistics and save to file
    stats_output_path = os.path.join(output_dir, f'{base_filename}_meteor_statistics.txt')
    print_statistics(scores, stats_output_path)
    
    # Plot distribution
    print("\nGenerating distribution plot...")
    plot_score_distribution(scores, output_dir, base_filename)
    
    print("\nFile analysis complete!")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Analyze METEOR score distribution from postprocessed molecule captioning files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single file
  python analyze_meteor_distribution.py inference_split1000_postprocessed_METEOR0.30.jsonl
  
  # Analyze multiple files
  python analyze_meteor_distribution.py file1.jsonl file2.jsonl file3.jsonl
  
  # Analyze with full path
  python analyze_meteor_distribution.py /path/to/molecule_captioning/inference_split1000_postprocessed_METEOR0.30.jsonl
        """
    )
    parser.add_argument('input_files', type=str, nargs='+',
                       help='Path(s) to the postprocessed JSONL file(s)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: same as input file directory)')
    
    args = parser.parse_args()
    
    print(f"Total files to process: {len(args.input_files)}")
    
    # Process each input file
    errors = 0
    for input_file in args.input_files:
        result = process_single_file(input_file, args.output_dir)
        if result != 0:
            errors += 1
    
    print(f"\n{'='*80}")
    print(f"All files processed!")
    print(f"Total files: {len(args.input_files)}")
    print(f"Successful: {len(args.input_files) - errors}")
    print(f"Failed: {errors}")
    print(f"{'='*80}")
    
    return 0 if errors == 0 else 1


if __name__ == '__main__':
    exit(main())
