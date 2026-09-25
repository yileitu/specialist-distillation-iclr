#!/usr/bin/env python3
"""
Sample records from a JSON/JSONL file or every JSON/JSONL file in a directory.

A single-file input produces one sampled file. A directory input samples each
file and writes the results to a new directory.
"""

import json
import argparse
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Union


def detect_json_format(file_path: Path) -> str:
    """
    Detect whether a JSON file uses a JSON array or JSONL.
    
    Args:
        file_path: Path to the JSON file.
    
    Returns:
        "json_array" or "jsonl".
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        
        # A leading [ usually indicates a JSON array
        if first_char == '[':
            return 'json_array'
        else:
            # Otherwise parse the first line; a valid JSON object indicates JSONL
            first_line = f.readline().strip()
            if first_line:
                try:
                    json.loads(first_line)
                    return 'jsonl'
                except json.JSONDecodeError:
                    # If the first line is not valid JSON, the file may be a pretty-printed JSON array
                    # Read the entire file again
                    f.seek(0)
                    content = f.read().strip()
                    if content.startswith('['):
                        return 'json_array'
            return 'jsonl'  # Default to JSONL


def read_json_data(file_path: Path) -> List[Dict[Any, Any]]:
    """
    Read JSON data after detecting its format.
    
    Args:
        file_path: Path to the JSON file.
    
    Returns:
        A list of records.
    """
    format_type = detect_json_format(file_path)
    
    if format_type == 'json_array':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"File {file_path.name} is not a JSON array")
            return data
    else:  # jsonl
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num} of {file_path.name}: {e}")
        return data


def write_json_data(data: List[Dict[Any, Any]], output_path: Path, format_type: str):
    """
    Write JSON data while preserving the source format.
    
    Args:
        data: Records to write.
        output_path: Output path.
        format_type: Either "json_array" or "jsonl".
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format_type == 'json_array':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:  # jsonl
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


def sample_json_files(
    input_dir: str,
    sample_size: int,
    output_dir: str = None,
    sample_mode: str = 'first'
) -> str:
    """
    Sample one JSON/JSONL file or every JSON/JSONL file in a directory.
    
    Args:
        input_dir: Input directory or JSON/JSONL file.
        sample_size: Number of records to sample.
        output_dir: Output directory or file; creates a sibling path when omitted.
        sample_mode: "first" for the first N records or "random" for sampling
            N records without replacement.
    
    Returns:
        The output directory or file path.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_dir}")
    
    if sample_size <= 0:
        raise ValueError(f"Sample size must be greater than zero: {sample_size}")
    
    if sample_mode not in ['first', 'random']:
        raise ValueError(f"Sample mode must be 'first' or 'random': {sample_mode}")
    
    # Determine whether the input is a file or directory
    if input_path.is_file():
        # Process a single file
        if input_path.suffix.lower() not in ['.json', '.jsonl']:
            raise ValueError(f"Input file must use .json or .jsonl format: {input_dir}")
        
        print(f"Processing one file: {input_path.name}")
        print(f"Sample mode: {sample_mode}; sample size: {sample_size}")
        
        try:
            # Read data
            data = read_json_data(input_path)
            original_count = len(data)
            
            # Sample records
            if sample_mode == 'first':
                # Take the first N records
                sampled_data = data[:sample_size]
            else:  # random
                # Sample N records without replacement
                if sample_size >= original_count:
                    # If the requested sample size covers the dataset, return all records in shuffled order
                    sampled_data = data.copy()
                    random.shuffle(sampled_data)
                else:
                    # Select N records randomly
                    sampled_data = random.sample(data, sample_size)
            
            sampled_count = len(sampled_data)
            
            # Detect the source file format
            format_type = detect_json_format(input_path)
            
            # Resolve the output file path
            if output_dir:
                output_file = Path(output_dir)
            else:
                # Create a sibling file with the _sample_{sample_size}_{mode} suffix
                parent_dir = input_path.parent
                stem = input_path.stem
                suffix = input_path.suffix
                new_file_name = f"{stem}_sample_{sample_size}_{sample_mode}{suffix}"
                output_file = parent_dir / new_file_name
            
            # Ensure the output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the sampled data
            write_json_data(sampled_data, output_file, format_type)
            
            print(f"  Original: {original_count}; sampled: {sampled_count}")
            print("\nSampling complete!")
            print(f"Output file: {output_file}")
            return str(output_file)
            
        except Exception as e:
            print(f"  Error processing {input_path.name}: {e}")
            raise
    
    elif input_path.is_dir():
        # Process a directory
        # Find all JSON files
        json_files = sorted(input_path.glob("*.json"))
        jsonl_files = sorted(input_path.glob("*.jsonl"))
        all_files = json_files + jsonl_files
        
        if not all_files:
            raise ValueError(f"No JSON or JSONL files found in {input_dir}")
        
        print(f"Found {len(all_files)} files; starting sampling...")
        print(f"Sample mode: {sample_mode}; sample size: {sample_size}")
        
        # Resolve the output directory
        if output_dir:
            output_path = Path(output_dir)
        else:
            # Create a sibling directory with the _sample_{sample_size}_{mode} suffix
            parent_dir = input_path.parent
            dir_name = input_path.name
            new_dir_name = f"{dir_name}_sample_{sample_size}_{sample_mode}"
            output_path = parent_dir / new_dir_name
        
        # Ensure the output directory exists
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process each file
        for i, file_path in enumerate(all_files, 1):
            print(f"Processing [{i}/{len(all_files)}]: {file_path.name}")
            
            try:
                # Read data
                data = read_json_data(file_path)
                original_count = len(data)
                
                # Sample records
                if sample_mode == 'first':
                    # Take the first N records
                    sampled_data = data[:sample_size]
                else:  # random
                    # Sample N records without replacement
                    if sample_size >= original_count:
                        # If the requested sample size covers the dataset, return all records in shuffled order
                        sampled_data = data.copy()
                        random.shuffle(sampled_data)
                    else:
                        # Select N records randomly
                        sampled_data = random.sample(data, sample_size)
                
                sampled_count = len(sampled_data)
                
                # Detect the source file format
                format_type = detect_json_format(file_path)
                
                # Build the output path while preserving the original filename and extension
                output_file = output_path / file_path.name
                
                # Write the sampled data
                write_json_data(sampled_data, output_file, format_type)
                
                print(f"  Original: {original_count}; sampled: {sampled_count}")
                
            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")
                raise
        
        print(f"\nSampling complete: processed {len(all_files)} files")
        print(f"Output directory: {output_path}")
        return str(output_path)
    else:
        raise ValueError(f"Input path is neither a file nor a directory: {input_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Sample records from a JSON file or every JSON file in a directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Take the first 100 records from every JSON file in a directory
  python sample_json_files.py /path/to/data/folder --sample-size 100
  
  # Take the first 100 records from one JSON file
  python sample_json_files.py /path/to/data.json --sample-size 100
  
  # Randomly sample 100 records from one JSONL file
  python sample_json_files.py /path/to/data.jsonl --sample-size 100 --sample-mode random
  
  # Specify an output directory or file
  python sample_json_files.py /path/to/data/folder --sample-size 100 --output-dir /path/to/output
  python sample_json_files.py /path/to/data.json --sample-size 100 --output-dir /path/to/output.json
        """
    )
    
    parser.add_argument(
        'input_dir',
        type=str,
        help='Input directory or JSON/JSONL file'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        required=True,
        help='Number of records to sample'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (defaults to a sibling directory with a _sample_{sample_size}_{mode} suffix)'
    )
    
    parser.add_argument(
        '--sample-mode',
        type=str,
        choices=['first', 'random'],
        default='first',
        help='Sampling mode: first takes the first N records; random samples N without replacement (default: first)'
    )
    
    args = parser.parse_args()
    
    try:
        output_dir = sample_json_files(
            input_dir=args.input_dir,
            sample_size=args.sample_size,
            output_dir=args.output_dir,
            sample_mode=args.sample_mode
        )
        print(f"\nSuccess! Output directory: {output_dir}")
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

