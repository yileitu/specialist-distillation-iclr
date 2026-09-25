#!/usr/bin/env python3
"""
Re-threshold METEOR scores for postprocessed molecule captioning JSONL files.

This script reads a postprocessed JSONL file containing METEOR scores in 
extracted_core_answer, applies a new METEOR threshold, and updates the
extracted_core_answer_correctness field accordingly. It outputs a new JSONL
file with the new threshold in the filename and a statistics log file.

Usage:
    python rethreshold_meteor_scores.py <input_jsonl> <threshold>

Example:
    python rethreshold_meteor_scores.py \
        /path/to/inference_split1000_postprocessed_METEOR0.30.jsonl \
        0.25
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Re-threshold METEOR scores in postprocessed molecule captioning JSONL files"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input postprocessed JSONL file"
    )
    parser.add_argument(
        "threshold",
        type=float,
        help="New METEOR threshold (e.g., 0.25)"
    )
    return parser.parse_args()


def get_output_paths(input_path: Path, new_threshold: float) -> tuple[Path, Path]:
    """
    Generate output file paths based on input path and new threshold.
    
    Args:
        input_path: Path to input JSONL file
        new_threshold: New METEOR threshold value
        
    Returns:
        Tuple of (output_jsonl_path, output_log_path)
    """
    # Get the directory and base name
    directory = input_path.parent
    stem = input_path.stem
    
    # Replace old threshold in filename with new threshold
    # Handle different possible formats: METEOR0.30 or METEOR0.3
    if "METEOR" in stem:
        # Split at METEOR and rebuild with new threshold
        parts = stem.split("_METEOR")
        base = parts[0]
        new_stem = f"{base}_METEOR{new_threshold:.2f}"
    else:
        # If no METEOR in filename, append it
        new_stem = f"{stem}_METEOR{new_threshold:.2f}"
    
    output_jsonl = directory / f"{new_stem}.jsonl"
    output_log = directory / f"{new_stem}_log.txt"
    
    return output_jsonl, output_log


def apply_threshold(
    line_data: Dict[str, Any],
    threshold: float
) -> Dict[str, Any]:
    """
    Apply new threshold to a single JSONL line entry.
    
    Args:
        line_data: Dictionary containing the JSONL line data
        threshold: METEOR threshold value
        
    Returns:
        Updated line_data dictionary
    """
    extracted_core_answer = line_data.get("extracted_core_answer", {})
    extracted_core_answer_correctness = line_data.get("extracted_core_answer_correctness", {})
    
    # Update correctness based on new threshold
    for gen_key, meteor_score in extracted_core_answer.items():
        if meteor_score is not None and isinstance(meteor_score, (int, float)):
            # Set correctness to True if METEOR score exceeds threshold
            extracted_core_answer_correctness[gen_key] = meteor_score >= threshold
        else:
            # Keep as False if no valid METEOR score
            extracted_core_answer_correctness[gen_key] = False
    
    line_data["extracted_core_answer_correctness"] = extracted_core_answer_correctness
    return line_data


def process_jsonl_file(
    input_path: Path,
    output_path: Path,
    threshold: float
) -> Dict[str, Any]:
    """
    Process the entire JSONL file and collect statistics.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        threshold: METEOR threshold value
        
    Returns:
        Dictionary containing processing statistics
    """
    stats = {
        "total_lines": 0,
        "task_lines": 0,
        "total_generations": 0,
        "generations_with_score": 0,
        "generations_without_score": 0,
        "generations_meeting_threshold": 0,
        "meteor_scores": [],
    }
    
    print(f"Processing {input_path}...")
    print(f"New threshold: {threshold}")
    print(f"Output will be written to: {output_path}")
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            if line_num % 10000 == 0:
                print(f"  Processed {line_num} lines...")
            
            stats["total_lines"] += 1
            
            try:
                line_data = json.loads(line.strip())
                
                # Check if this is a molecule_captioning task
                if line_data.get("task") == "molecule_captioning":
                    stats["task_lines"] += 1
                    
                    # Get extracted_core_answer scores
                    extracted_core_answer = line_data.get("extracted_core_answer", {})
                    
                    # Count generations and collect scores
                    for gen_key, meteor_score in extracted_core_answer.items():
                        stats["total_generations"] += 1
                        
                        if meteor_score is not None and isinstance(meteor_score, (int, float)):
                            stats["generations_with_score"] += 1
                            stats["meteor_scores"].append(meteor_score)
                            
                            # Check if meets new threshold
                            if meteor_score >= threshold:
                                stats["generations_meeting_threshold"] += 1
                        else:
                            stats["generations_without_score"] += 1
                    
                    # Apply new threshold
                    line_data = apply_threshold(line_data, threshold)
                
                # Write updated line to output
                outfile.write(json.dumps(line_data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
    
    print(f"Processing complete! Processed {stats['total_lines']} lines.")
    return stats


def calculate_statistics(stats: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """
    Calculate summary statistics from collected data.
    
    Args:
        stats: Dictionary containing raw statistics
        threshold: METEOR threshold value
        
    Returns:
        Dictionary with calculated statistics
    """
    meteor_scores = stats["meteor_scores"]
    
    summary = {
        "total_lines": stats["total_lines"],
        "task_lines": stats["task_lines"],
        "total_generations": stats["total_generations"],
        "generations_with_score": stats["generations_with_score"],
        "generations_without_score": stats["generations_without_score"],
        "generations_meeting_threshold": stats["generations_meeting_threshold"],
        "accuracy": 0.0,
        "avg_meteor": 0.0,
        "max_meteor": 0.0,
        "min_meteor": 0.0,
        "threshold": threshold,
    }
    
    # Calculate accuracy
    if stats["total_generations"] > 0:
        summary["accuracy"] = (stats["generations_meeting_threshold"] / 
                              stats["total_generations"]) * 100
    
    # Calculate METEOR statistics
    if meteor_scores:
        summary["avg_meteor"] = sum(meteor_scores) / len(meteor_scores)
        summary["max_meteor"] = max(meteor_scores)
        summary["min_meteor"] = min(meteor_scores)
    
    return summary


def write_log_file(
    log_path: Path,
    input_path: Path,
    output_path: Path,
    summary: Dict[str, Any],
    threshold: float
):
    """
    Write statistics log file.
    
    Args:
        log_path: Path to output log file
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        summary: Dictionary containing summary statistics
        threshold: METEOR threshold value
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Molecule Captioning Re-threshold Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"Output file: {output_path}\n")
        f.write(f"METEOR threshold: {threshold}\n")
        f.write("\n")
        f.write("Processing complete:\n")
        f.write(f"  Total lines processed: {summary['total_lines']}\n")
        f.write(f"  Molecule captioning task lines: {summary['task_lines']}\n")
        f.write(f"  Total generations processed: {summary['total_generations']}\n")
        f.write("\n")
        f.write("Generation statistics:\n")
        f.write(f"  Generations with METEOR score: {summary['generations_with_score']}\n")
        f.write(f"  Generations without METEOR score: {summary['generations_without_score']}\n")
        f.write(f"  Generations meeting threshold: {summary['generations_meeting_threshold']}\n")
        f.write(f"  Accuracy (threshold={threshold}): {summary['accuracy']:.2f}%\n")
        f.write("\n")
        f.write("METEOR score statistics (for generations with scores):\n")
        f.write(f"  Average METEOR: {summary['avg_meteor']:.4f}\n")
        f.write(f"  Max METEOR: {summary['max_meteor']:.4f}\n")
        f.write(f"  Min METEOR: {summary['min_meteor']:.4f}\n")
        f.write(f"  Total scores calculated: {summary['generations_with_score']}\n")
        f.write("=" * 70 + "\n")
    
    print(f"\nLog file written to: {log_path}")


def main():
    """Main function."""
    args = parse_args()
    
    # Validate inputs
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    if args.threshold < 0 or args.threshold > 1:
        print(f"Error: Threshold must be between 0 and 1, got {args.threshold}")
        return 1
    
    # Get output paths
    output_jsonl, output_log = get_output_paths(input_path, args.threshold)
    
    # Check if output file already exists
    if output_jsonl.exists():
        response = input(f"\nWarning: Output file already exists: {output_jsonl}\n"
                        f"Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0
    
    print("\n" + "=" * 70)
    print("METEOR Re-threshold Tool")
    print("=" * 70)
    
    # Process the file
    stats = process_jsonl_file(input_path, output_jsonl, args.threshold)
    
    # Calculate summary statistics
    summary = calculate_statistics(stats, args.threshold)
    
    # Write log file
    write_log_file(output_log, input_path, output_jsonl, summary, args.threshold)
    
    # Print summary to console
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total lines processed: {summary['total_lines']}")
    print(f"Total generations: {summary['total_generations']}")
    print(f"Generations meeting new threshold ({args.threshold}): {summary['generations_meeting_threshold']}")
    print(f"Accuracy: {summary['accuracy']:.2f}%")
    print(f"Average METEOR: {summary['avg_meteor']:.4f}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
