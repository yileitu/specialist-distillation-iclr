#!/usr/bin/env python3
"""
Analyze token length of assistant responses in JSONL files.

This script analyzes the token count of assistant messages using either a tokenizer
or whitespace splitting, including statistics and examples of messages exceeding a threshold.
Generates histograms showing token distribution for each file.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
from datetime import datetime
import statistics
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def load_tokenizer(tokenizer_path: str):
    """
    Load tokenizer from the specified path.
    
    Args:
        tokenizer_path: Path to the tokenizer directory
        
    Returns:
        Loaded tokenizer
    """
    from transformers import AutoTokenizer
    
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True
    )
    print(f"Tokenizer loaded successfully: {tokenizer.__class__.__name__}")
    return tokenizer


def count_tokens_or_words(text: str, tokenizer=None, use_whitespace: bool = False) -> int:
    """
    Count tokens using tokenizer or words by splitting on whitespace.
    
    Args:
        text: Input text
        tokenizer: Tokenizer to use (if use_whitespace is False)
        use_whitespace: If True, use whitespace splitting; if False, use tokenizer
        
    Returns:
        Number of tokens or words
    """
    if use_whitespace:
        return len(text.split())
    else:
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided when use_whitespace is False")
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)


def analyze_jsonl_file(
    file_path: Path,
    tokenizer=None,
    use_whitespace: bool = False,
    max_length_threshold: int = 32768
) -> Dict[str, Any]:
    """
    Analyze token/word lengths in a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        tokenizer: Tokenizer to use (required if use_whitespace is False)
        use_whitespace: If True, count words by whitespace; if False, use tokenizer
        max_length_threshold: Threshold for long messages (default: 32768)
        
    Returns:
        Dictionary with analysis results
    """
    stats = {
        'file_name': file_path.name,
        'total_datapoints': 0,
        'assistant_messages': 0,
        'token_counts': [],
        'exceeding_threshold': [],
        'total_tokens': 0
    }
    
    print(f"\nAnalyzing file: {file_path.name}")
    
    with open(file_path, 'r', encoding='utf-8') as infile:
        for line_idx, line in enumerate(tqdm(infile, desc="Processing lines"), 1):
            try:
                data = json.loads(line.strip())
                stats['total_datapoints'] += 1
                
                # Process messages format
                if 'messages' in data:
                    for message in data['messages']:
                        if message.get('role') == 'assistant' and 'content' in message:
                            stats['assistant_messages'] += 1
                            content = message['content']
                            
                            # Count tokens or words
                            token_count = count_tokens_or_words(content, tokenizer, use_whitespace)
                            
                            stats['token_counts'].append(token_count)
                            stats['total_tokens'] += token_count
                            
                            # Check if exceeds threshold
                            if token_count > max_length_threshold:
                                stats['exceeding_threshold'].append({
                                    'line_number': line_idx,
                                    'token_count': token_count,
                                    'content': content,
                                    'data': data
                                })
                
                # Process direct format (role and content at top level)
                elif data.get('role') == 'assistant' and 'content' in data:
                    stats['assistant_messages'] += 1
                    content = data['content']
                    
                    # Count tokens or words
                    token_count = count_tokens_or_words(content, tokenizer, use_whitespace)
                    
                    stats['token_counts'].append(token_count)
                    stats['total_tokens'] += token_count
                    
                    # Check if exceeds threshold
                    if token_count > max_length_threshold:
                        stats['exceeding_threshold'].append({
                            'line_number': line_idx,
                            'token_count': token_count,
                            'content': content,
                            'data': data
                        })
                        
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_idx}: {e}")
            except Exception as e:
                print(f"Warning: Error processing line {line_idx}: {e}")
    
    return stats


def plot_histogram(
    token_counts: List[int],
    file_name: str,
    output_path: Path,
    use_whitespace: bool = False,
    bin_size: int = 1000
):
    """
    Plot histogram of token/word count distribution.
    
    Args:
        token_counts: List of token/word counts
        file_name: Name of the file being analyzed
        output_path: Path to save the histogram
        use_whitespace: Whether whitespace splitting was used
        bin_size: Size of each bin (default: 1000)
    """
    if not token_counts:
        print(f"Warning: No data to plot for {file_name}")
        return
    
    unit_name = "Words" if use_whitespace else "Tokens"
    
    # Calculate statistics
    avg = statistics.mean(token_counts)
    std_dev = statistics.stdev(token_counts) if len(token_counts) > 1 else 0
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    
    # Determine bins
    max_count = max(token_counts)
    bins = range(0, int(max_count) + bin_size, bin_size)
    
    plt.hist(token_counts, bins=bins, edgecolor='black', alpha=0.7)
    
    # Set title with statistics
    plt.title(f'{unit_name} Distribution: {file_name}\n'
              f'Mean: {avg:.2f}, Std Dev: {std_dev:.2f}',
              fontsize=12, pad=20)
    
    plt.xlabel(f'{unit_name} Count', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add some statistics as text
    textstr = f'Total messages: {len(token_counts):,}\n'
    textstr += f'Min: {min(token_counts):,}\n'
    textstr += f'Max: {max(token_counts):,}\n'
    textstr += f'Median: {statistics.median(token_counts):,}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.98, 0.97, textstr, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Histogram saved to: {output_path}")


def generate_report(
    all_stats: List[Dict[str, Any]],
    output_path: Path,
    max_length_threshold: int,
    use_whitespace: bool = False,
    tokenizer_path: Optional[str] = None
):
    """
    Generate a text report of the analysis.
    
    Args:
        all_stats: List of statistics from all files
        output_path: Path to save the report
        max_length_threshold: The threshold used for long messages
        use_whitespace: Whether whitespace splitting was used
        tokenizer_path: Path to tokenizer (if used)
    """
    # Determine the unit name
    unit_name = "words" if use_whitespace else "tokens"
    unit_name_singular = "word" if use_whitespace else "token"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 100 + "\n")
        if use_whitespace:
            f.write("ASSISTANT MESSAGE WORD COUNT ANALYSIS REPORT\n")
        else:
            f.write("ASSISTANT MESSAGE TOKEN LENGTH ANALYSIS REPORT\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if use_whitespace:
            f.write(f"Counting method: Whitespace splitting\n")
        else:
            f.write(f"Counting method: Tokenizer\n")
            if tokenizer_path:
                f.write(f"Tokenizer: {tokenizer_path}\n")
        f.write(f"Max length threshold: {max_length_threshold} {unit_name}\n")
        f.write(f"Number of files analyzed: {len(all_stats)}\n")
        f.write("\n" + "=" * 100 + "\n\n")
        
        # Aggregate statistics
        total_datapoints = sum(s['total_datapoints'] for s in all_stats)
        total_assistant_messages = sum(s['assistant_messages'] for s in all_stats)
        all_token_counts = []
        for s in all_stats:
            all_token_counts.extend(s['token_counts'])
        
        total_exceeding = sum(len(s['exceeding_threshold']) for s in all_stats)
        
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 100 + "\n")
        f.write(f"Total datapoints: {total_datapoints:,}\n")
        f.write(f"Total assistant messages: {total_assistant_messages:,}\n")
        
        if all_token_counts:
            avg_tokens = statistics.mean(all_token_counts)
            std_dev = statistics.stdev(all_token_counts) if len(all_token_counts) > 1 else 0
            min_tokens = min(all_token_counts)
            max_tokens = max(all_token_counts)
            median_tokens = statistics.median(all_token_counts)
            
            f.write(f"Average {unit_name_singular} count: {avg_tokens:.2f}\n")
            f.write(f"Standard deviation: {std_dev:.2f}\n")
            f.write(f"Median {unit_name_singular} count: {median_tokens:,}\n")
            f.write(f"Min {unit_name_singular} count: {min_tokens:,}\n")
            f.write(f"Max {unit_name_singular} count: {max_tokens:,}\n")
            f.write(f"Messages exceeding {max_length_threshold} {unit_name}: {total_exceeding:,} "
                   f"({100 * total_exceeding / len(all_token_counts):.2f}%)\n")
        
        f.write("\n" + "=" * 100 + "\n\n")
        
        # Per-file statistics
        f.write("PER-FILE STATISTICS\n")
        f.write("-" * 100 + "\n\n")
        
        for idx, stats in enumerate(all_stats, 1):
            f.write(f"File {idx}: {stats['file_name']}\n")
            f.write(f"  Total datapoints: {stats['total_datapoints']:,}\n")
            f.write(f"  Assistant messages: {stats['assistant_messages']:,}\n")
            
            if stats['token_counts']:
                avg = statistics.mean(stats['token_counts'])
                std_dev = statistics.stdev(stats['token_counts']) if len(stats['token_counts']) > 1 else 0
                min_val = min(stats['token_counts'])
                max_val = max(stats['token_counts'])
                median_val = statistics.median(stats['token_counts'])
                
                f.write(f"  Average {unit_name}: {avg:.2f}\n")
                f.write(f"  Standard deviation: {std_dev:.2f}\n")
                f.write(f"  Median {unit_name}: {median_val:,}\n")
                f.write(f"  Min {unit_name}: {min_val:,}\n")
                f.write(f"  Max {unit_name}: {max_val:,}\n")
                f.write(f"  Exceeding threshold: {len(stats['exceeding_threshold']):,}\n")
            
            f.write("\n")
        
        f.write("=" * 100 + "\n\n")
        
        # Examples of messages exceeding threshold
        f.write("EXAMPLES OF MESSAGES EXCEEDING THRESHOLD\n")
        f.write("-" * 100 + "\n\n")
        
        found_example = False
        for stats in all_stats:
            if stats['exceeding_threshold']:
                for example in stats['exceeding_threshold'][:1]:  # Show first example from each file
                    found_example = True
                    f.write(f"File: {stats['file_name']}\n")
                    f.write(f"Line number: {example['line_number']}\n")
                    f.write(f"{unit_name_singular.capitalize()} count: {example['token_count']:,}\n")
                    f.write(f"\nContent preview (first 2000 characters):\n")
                    f.write("-" * 100 + "\n")
                    f.write(example['content'][:2000])
                    if len(example['content']) > 2000:
                        f.write(f"\n... (truncated, total length: {len(example['content'])} characters)")
                    f.write("\n\n")
                    f.write("-" * 100 + "\n\n")
                    
                    # Also show the full datapoint structure (without the full content)
                    f.write("Full datapoint structure (content truncated):\n")
                    f.write("-" * 100 + "\n")
                    data_copy = json.loads(json.dumps(example['data']))
                    if 'messages' in data_copy:
                        for msg in data_copy['messages']:
                            if msg.get('role') == 'assistant' and len(msg.get('content', '')) > 500:
                                msg['content'] = msg['content'][:500] + "... (truncated)"
                    elif data_copy.get('role') == 'assistant' and len(data_copy.get('content', '')) > 500:
                        data_copy['content'] = data_copy['content'][:500] + "... (truncated)"
                    
                    f.write(json.dumps(data_copy, indent=2, ensure_ascii=False))
                    f.write("\n\n" + "=" * 100 + "\n\n")
                    break  # Only show one example
        
        if not found_example:
            f.write("No messages found exceeding the threshold.\n\n")
        
        f.write("=" * 100 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze token/word length of assistant responses in JSONL files"
    )
    parser.add_argument(
        '--input',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to input JSONL file(s). Can specify multiple files.'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default=None,
        help='Path or model ID for the tokenizer; required unless --use-whitespace is set'
    )
    parser.add_argument(
        '--use-whitespace',
        action='store_true',
        help='Use whitespace splitting instead of tokenizer (counts words instead of tokens)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output report file (default: token_length_analysis_report_<timestamp>.txt in current directory)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save output files (report and histograms). Default: current directory'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=None,
        help='Token/word count threshold for identifying long messages (default: 32768 for tokenizer, 20000 for whitespace)'
    )
    parser.add_argument(
        '--bin-size',
        type=int,
        default=1000,
        help='Bin size for histogram (default: 1000)'
    )
    
    args = parser.parse_args()
    if not args.use_whitespace and not args.tokenizer:
        parser.error('--tokenizer is required unless --use-whitespace is set')
    
    # Set default threshold based on mode
    if args.threshold is None:
        args.threshold = 20000 if args.use_whitespace else 32768
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path.cwd()
    
    # Load tokenizer if needed
    tokenizer = None
    if not args.use_whitespace:
        try:
            tokenizer = load_tokenizer(args.tokenizer)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            return 1
    
    # Setup output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = "word_count" if args.use_whitespace else "token_length"
        output_path = output_dir / f"{prefix}_analysis_report_{timestamp}.txt"
    
    # Process all input files
    all_stats = []
    for input_file in args.input:
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"Warning: Input file not found: {input_path}, skipping...")
            continue
        
        try:
            stats = analyze_jsonl_file(
                input_path, 
                tokenizer=tokenizer,
                use_whitespace=args.use_whitespace,
                max_length_threshold=args.threshold
            )
            all_stats.append(stats)
            
            # Generate histogram for this file
            if stats['token_counts']:
                histogram_name = f"{input_path.stem}_histogram.png"
                histogram_path = output_dir / histogram_name
                print(f"\nGenerating histogram for {input_path.name}...")
                plot_histogram(
                    stats['token_counts'],
                    input_path.name,
                    histogram_path,
                    use_whitespace=args.use_whitespace,
                    bin_size=args.bin_size
                )
            
        except Exception as e:
            print(f"Error processing file {input_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_stats:
        print("Error: No files were successfully processed")
        return 1
    
    # Generate report
    print(f"\nGenerating report...")
    generate_report(
        all_stats, 
        output_path, 
        args.threshold,
        use_whitespace=args.use_whitespace,
        tokenizer_path=args.tokenizer if not args.use_whitespace else None
    )
    
    print(f"\nAnalysis complete!")
    print(f"Report saved to: {output_path}")
    
    # Print summary to console
    unit_name = "words" if args.use_whitespace else "tokens"
    unit_name_singular = "word" if args.use_whitespace else "token"
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_assistant_messages = sum(s['assistant_messages'] for s in all_stats)
    all_token_counts = []
    for s in all_stats:
        all_token_counts.extend(s['token_counts'])
    
    if all_token_counts:
        avg_tokens = statistics.mean(all_token_counts)
        std_dev = statistics.stdev(all_token_counts) if len(all_token_counts) > 1 else 0
        total_exceeding = sum(len(s['exceeding_threshold']) for s in all_stats)
        print(f"Total assistant messages: {total_assistant_messages:,}")
        print(f"Average {unit_name_singular} count: {avg_tokens:.2f}")
        print(f"Standard deviation: {std_dev:.2f}")
        print(f"Messages exceeding {args.threshold} {unit_name}: {total_exceeding:,}")
    
    return 0


if __name__ == "__main__":
    exit(main())
