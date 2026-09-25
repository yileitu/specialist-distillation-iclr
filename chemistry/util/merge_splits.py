#!/usr/bin/env python3
"""
Merge split files into JSONL.

Merge files in the input directory directly, or recursively process child
directories when the input directory has no matching files.
"""

import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any
import re


def merge_splits_to_json(
    input_dir: str,
    output_path: str = None,
    output_dir: str = None,
    output_filename: str = None,
    file_pattern: str = "*.jsonl"
    ) -> str:
    """
    Merge split files into JSONL with one JSON object per line.
    
    By default, match both *.jsonl and *.json files. Try reading .json files as
    JSONL first, then fall back to a single JSON array or object. If the input
    directory has no matching files, process matching child directories recursively.
    
    Args:
        input_dir: Directory containing split files.
        output_path: Complete output path; overrides output_dir and output_filename.
        output_dir: Output directory; defaults to the parent of input_dir.
        output_filename: Output filename; derived from input_dir when omitted.
        file_pattern: Filename pattern; defaults to *.jsonl and also matches *.json.
    
    Returns:
        The complete output path.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    candidate_dirs = _collect_candidate_dirs(input_path, file_pattern)
    if not candidate_dirs:
        raise ValueError(f"No files matching {file_pattern} found in {input_dir}")

    if output_path and len(candidate_dirs) > 1:
        raise ValueError("Cannot use a single --output-path when merging multiple subdirectories")
    if output_filename and len(candidate_dirs) > 1:
        raise ValueError("Cannot use one --output-filename when merging multiple subdirectories")

    output_paths: List[str] = []
    for dir_idx, merge_dir in enumerate(candidate_dirs, 1):
        if len(candidate_dirs) > 1:
            print(f"\n[{dir_idx}/{len(candidate_dirs)}] Processing directory: {merge_dir}")
        output_paths.append(_merge_single_dir(
            input_path=merge_dir,
            output_path=output_path,
            output_dir=output_dir,
            output_filename=output_filename,
            file_pattern=file_pattern
        ))

    return "\n".join(output_paths)


def _collect_candidate_dirs(input_path: Path, file_pattern: str) -> List[Path]:
    direct_files = _list_split_files(input_path, file_pattern)
    if direct_files:
        return [input_path]

    candidate_dirs: List[Path] = []
    for root, dirs, _files in os.walk(input_path):
        root_path = Path(root)
        if _list_split_files(root_path, file_pattern):
            candidate_dirs.append(root_path)
            dirs[:] = []
    return candidate_dirs


def _list_split_files(input_path: Path, file_pattern: str) -> List[Path]:
    if file_pattern == "*.jsonl":
        split_files = sorted(list(input_path.glob("*.jsonl")) + list(input_path.glob("*.json")))
    else:
        split_files = sorted(input_path.glob(file_pattern))
    return split_files


def _merge_single_dir(
    input_path: Path,
    output_path: str = None,
    output_dir: str = None,
    output_filename: str = None,
    file_pattern: str = "*.jsonl"
) -> str:
    split_files = _list_split_files(input_path, file_pattern)
    if not split_files:
        raise ValueError(f"No files matching {file_pattern} found in {input_path}")

    print(f"Found {len(split_files)} files; starting merge...")

    # Read all input files
    all_data: List[Dict[Any, Any]] = []
    for i, split_file in enumerate(split_files, 1):
        print(f"Processing [{i}/{len(split_files)}]: {split_file.name}")
        try:
            # For .json files, first try JSONL (one JSON object per line)
            # If the first non-empty line fails to parse, retry as a single JSON document
            if split_file.suffix == '.json':
                # Try reading as JSONL first
                with open(split_file, 'r', encoding='utf-8') as f:
                    line_count = 0
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        line_count += 1
                        try:
                            data = json.loads(line)
                            all_data.append(data)
                        except json.JSONDecodeError:
                            # If the first non-empty line fails to parse, try reading a single JSON document
                            if line_count == 1:
                                print(f"Note: {split_file.name} is not JSONL; trying to read it as one JSON document...")
                                f.seek(0)  # Reset the file pointer
                                try:
                                    content = f.read().strip()
                                    parsed = json.loads(content)
                                    # Expand arrays into individual objects
                                    if isinstance(parsed, list):
                                        all_data.extend(parsed)
                                        print(f"  Read {len(parsed)} records from a JSON array")
                                    else:
                                        # Single object
                                        all_data.append(parsed)
                                        print("  Read one record from a JSON object")
                                except json.JSONDecodeError as e2:
                                    print(f"Warning: {split_file.name} is neither valid JSONL nor valid JSON: {e2}")
                            else:
                                print(f"Warning: Invalid JSON on line {line_num} of {split_file.name}; skipping line")
                            break  # Exit the loop
            else:
                # Read .jsonl files directly as JSONL
                with open(split_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            all_data.append(data)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Invalid JSON on line {line_num} of {split_file.name}: {e}")
        except Exception as e:
            print(f"Error reading {split_file.name}: {e}")
            raise

    print(f"Read {len(all_data)} records in total")

    # Resolve the output path
    if output_path:
        final_output_path = Path(output_path)
    else:
        if output_dir:
            output_dir_path = Path(output_dir)
        else:
            # Use the parent of the input directory
            output_dir_path = input_path.parent

        if output_filename:
            final_output_path = output_dir_path / output_filename
        else:
            # Generate a filename from the input directory name
            # For example: full_50k_ar -> full_50k_ar.jsonl
            dir_name = input_path.name
            # Remove _split and everything after it, retaining only the preceding part
            dir_name = re.sub(r'_split.*', '', dir_name)
            filename = f'{dir_name}.jsonl'
            final_output_path = output_dir_path / filename

    # Ensure the output directory exists
    final_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as JSONL (one JSON object per line)
    print(f"Saving to: {final_output_path}")
    with open(final_output_path, 'w', encoding='utf-8') as f:
        for data in all_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"Merge complete: saved {len(all_data)} records to {final_output_path}")
    return str(final_output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Merge split files into a JSONL file with one object per line',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use the default output path (parent directory with an automatically generated filename)
  python merge_splits.py /path/to/split/directory

  # Use a parent input directory and recursively merge its subdirectories
  python merge_splits.py /path/to/parent/directory
  
  # Specify an output directory
  python merge_splits.py /path/to/split/directory --output-dir /path/to/output
  
  # Specify the output filename
  python merge_splits.py /path/to/split/directory --output-filename merged.jsonl
  
  # Specify the complete output path
  python merge_splits.py /path/to/split/directory --output-path /path/to/output/merged.jsonl
  
  # Set a filename pattern, for example to merge .json files
  python merge_splits.py /path/to/split/directory --file-pattern "*.json"
        """
    )
    
    parser.add_argument(
        'input_dir',
        type=str,
        help='Path containing split files, or a parent directory of such paths'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Complete output path; overrides --output-dir and --output-filename'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (defaults to the parent of input_dir)'
    )
    
    parser.add_argument(
        '--output-filename',
        type=str,
        default=None,
        help='Output filename (derived from input_dir when omitted)'
    )
    
    parser.add_argument(
        '--file-pattern',
        type=str,
        default='*.jsonl',
        help='File pattern (default: *.jsonl; matching *.json files are also read)'
    )
    
    args = parser.parse_args()
    
    try:
        output_path = merge_splits_to_json(
            input_dir=args.input_dir,
            output_path=args.output_path,
            output_dir=args.output_dir,
            output_filename=args.output_filename,
            file_pattern=args.file_pattern
        )
        print(f"\nSuccess! Output file: {output_path}")
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

