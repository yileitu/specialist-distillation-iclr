#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inference on a single SMolInstruct JSONL task with vLLM.

The script loads an Intern-S1-mini model and supports loop detection, which
stops generation when the output falls into a repeated pattern. Loop detection
uses the vLLM V1 LogitsProcessor API.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import re
import torch
from vllm import LLM, SamplingParams
from vllm.v1.sample.logits_processor import LogitsProcessor, BatchUpdate, MoveDirectionality
from vllm.config import VllmConfig

# Import utility functions
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from util.util_func import get_available_gpu_count

SMOL_SYSTEM_PROMPT = "You are an expert reasoner with extensive experience in all areas. You approach problems through systematic thinking and rigorous reasoning. Your response should reflect deep understanding and precise logical thinking, making your solution path and reasoning clear to others. Please put your thinking process within <think>...</think> tags."
SMOL_TASK_INSTRUCTION = {
    "forward_synthesis": "You are an expert chemist. Given the SMILES representation of reactants and reagents, your task is to predict the potential product using your chemical reaction knowledge.\n    The input contains both reactants and reagents, and different reactants and reagents are separated by \".\". Your reply should contain the SMILES representation of the predicted product wrapped in <SMILES> and </SMILES> tags. Your reply must be valid and chemically reasonable.\nQuestion: ",
    "retrosynthesis": "You are an expert chemist. Given the SMILES representation of the product, your task is to predict the potential reactants and reagents using your chemical reaction knowledge.\n    The input contains the SMILES representation of the product. Your reply should contain the SMILES representation of both reactants and reagents, and all reactants and reagents should be enclosed **together** within a single pair of <SMILES> and </SMILES> tags, separated by \".\". Your reply must be valid and chemically reasonable.\nQuestion: ",
    "molecule_captioning": "You are an expert chemist. Given the SMILES representation of a molecule, your task is to describe the molecule in natural language.\n    The input contains the SMILES representation of the molecule. Your reply should contain a natural language description of the molecule. Your reply must be valid and chemically reasonable.\nQuestion: ",
    "molecule_generation": "You are an expert chemist. Given the description of a molecule, your task is to generate the potential SMILES representation of the molecule.\n    The input contains the description of the molecule. Your reply should contain the potential SMILES representation of the molecule wrapped in <SMILES> and </SMILES> tags. Your reply must be valid and chemically reasonable.\nQuestion: ",
    "name_conversion-i2f": "You are an expert chemist. Given the IUPAC representation of compounds, your task is to predict the molecular formula of the compound.\n    The input contains the IUPAC representation of the compound. Your reply should contain only the molecular formula of the compound wrapped in <MOLFORMULA> and </MOLFORMULA> tags and no other text. Your reply must be valid and chemically reasonable.\nQuestion: ",
    "name_conversion-i2s": "You are an expert chemist. Given the IUPAC representation of compounds, your task is to predict the SMILES representation of the compound.\n    The input contains the IUPAC representation of the compound. Your reply should contain only the SMILES representation of the compound wrapped in <SMILES> and </SMILES> tags and no other text. Your reply must be valid and chemically reasonable.\nQuestion: ",
    "name_conversion-s2f": "You are an expert chemist. Given the SMILES representation of compounds, your task is to predict the molecular formula of the compound.\n    The input contains the SMILES representation of the compound. Your reply should contain only the molecular formula of the compound wrapped in <MOLFORMULA> and </MOLFORMULA> tags and no other text. Your reply must be valid and chemically reasonable.\nQuestion: ",
    "name_conversion-s2i": "You are an expert chemist. Given the SMILES representation of compounds, your task is to predict the IUPAC representation of the compound.\n    The input contains the SMILES representation of the compound. Your reply should contain only the IUPAC representation of the compound wrapped in <IUPAC> and </IUPAC> tags and no other text. Your reply must be valid and chemically reasonable.\nQuestion: ",
    "property_prediction-esol": "You are an expert chemist. Given the SMILES representation of compounds, your task is to predict the log solubility of the compound.\n    The input contains the SMILES representation of the compound. Your reply should contain the log solubility of the compound wrapped in \\boxed{}. Your reply must be valid and chemically reasonable.\nQuestion: ",
    "property_prediction-lipo": "You are an expert chemist. Given the SMILES representation of compounds, your task is to predict the octanol/water partition coefficient of the compound.\n    The input contains the SMILES representation of the compound. Your reply should contain the octanol/water partition coefficient of the compound wrapped in \\boxed{}. Your reply must be valid and chemically reasonable.\nQuestion: ",
    "property_prediction-bbbp": "You are an expert chemist. Given the smiles representation of the compound, your task is to predict whether blood-brain barrier permeability (BBBP) is a property of the compound.\n    The input contains the compound. Your reply should only contain Yes or No. Your reply must be valid and chemically reasonable.\nQuestion: ",
    "property_prediction-clintox": "You are an expert chemist. Given the smiles representation of the compound, your task is to predict whether the compound is toxic.\n    The input contains the compound. Your reply should contain only Yes or No. Your reply must be valid and chemically reasonable.\nQuestion: ",
    "property_prediction-hiv": "You are an expert chemist. Given the smiles representation of the compound, your task is to predict whether the compound serve as an inhibitor of HIV replication.\n    The input contains the compound. Your reply should contain only Yes or No. Your reply must be valid and chemically reasonable.\nQuestion: ",
    "property_prediction-sider": "You are an expert chemist. Given the smiles representation of the compound, your task is to predict whether the compound has any side effects.\n    The input contains the compound. Your reply should contain only Yes or No. Your reply must be valid and chemically reasonable.\nQuestion: ",
}

# Public defaults contain no machine-specific paths.
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs")


def process_dict_updates(
    req_entries: Dict[int, Any],
    batch_update: Optional[BatchUpdate],
    new_state,
) -> bool:
    """
    Update dictionary state for a sparse LogitsProcessor.

    Based on the official vLLM implementation.
    """
    if not batch_update:
        return False

    updated = False
    for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
        state = new_state(params, prompt_tok_ids, output_tok_ids)
        if state is not None:
            req_entries[index] = state
            updated = True
        elif req_entries.pop(index, None) is not None:
            updated = True

    if req_entries:
        # Handle removed requests
        for index in batch_update.removed:
            if req_entries.pop(index, None):
                updated = True

        # Handle moved requests (one-way a->b or swapped a<->b)
        for a_index, b_index, direct in batch_update.moved:
            a_entry = req_entries.pop(a_index, None)
            b_entry = req_entries.pop(b_index, None)
            if a_entry is not None:
                req_entries[b_index] = a_entry
                updated = True
            if b_entry is not None:
                updated = True
                if direct == MoveDirectionality.SWAP:
                    req_entries[a_index] = b_entry

    return updated


class RepetitionStoppingLogitsProcessor(LogitsProcessor):
    """
    Detect repeated output loops dynamically with the vLLM V1 API.

    The processor scans the end of each sequence for consecutive repeated patterns
    of any length in the configured range. It forces EOS when a pattern reaches
    the repetition threshold.

    For min_pattern_len=5, max_pattern_len=50, and max_repeat_times=3, every pattern
    length from 5 through 50 is checked. Any pattern repeated three consecutive
    times triggers stopping.

    Examples:
    - [..., A, B, C, A, B, C, A, B, C] triggers for a length-three pattern.
    - A longer block repeated three times also triggers, regardless of its length.

    Arguments are passed through SamplingParams.extra_args:
    - enable_loop_detection: Whether loop detection is enabled.
    - min_pattern_len: Minimum pattern length; defaults to 5 to avoid short phrases.
    - max_pattern_len: Maximum pattern length; defaults to 100.
    - max_repeat_times: Maximum allowed consecutive repeats; defaults to 3.
    """

    # Default arguments
    DEFAULT_MIN_PATTERN_LEN = 5
    DEFAULT_MAX_PATTERN_LEN = 100
    DEFAULT_MAX_REPEAT_TIMES = 3

    @classmethod
    def validate_params(cls, params: SamplingParams):
        """Validate custom SamplingParams arguments."""
        if params.extra_args:
            min_pattern_len = params.extra_args.get("min_pattern_len")
            if min_pattern_len is not None and (not isinstance(min_pattern_len, int) or min_pattern_len < 1):
                raise ValueError(f"min_pattern_len must be a positive integer, got {min_pattern_len}")

            max_pattern_len = params.extra_args.get("max_pattern_len")
            if max_pattern_len is not None and (not isinstance(max_pattern_len, int) or max_pattern_len < 1):
                raise ValueError(f"max_pattern_len must be a positive integer, got {max_pattern_len}")

            max_repeat_times = params.extra_args.get("max_repeat_times")
            if max_repeat_times is not None and (not isinstance(max_repeat_times, int) or max_repeat_times < 2):
                raise ValueError(f"max_repeat_times must be >= 2, got {max_repeat_times}")

    def __init__(self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool):
        """Initialize the LogitsProcessor."""
        self.device = device
        self.pin_memory = is_pin_memory

        # Read the EOS token ID from vllm_config
        try:
            self.eos_token_id = vllm_config.model_config.hf_config.eos_token_id
            if isinstance(self.eos_token_id, list):
                self.eos_token_id = self.eos_token_id[0]
        except:
            self.eos_token_id = 2  # Default value

        # Sparse mapping: batch_index -> [output_tok_ids_ref, min_pattern_len, max_pattern_len, max_repeat_times, force_stop_flag]
        self.req_info: Dict[int, list] = {}

        # Debug mode controlled by an environment variable
        self.debug = os.getenv("LOOP_DETECTION_DEBUG", "0") == "1"

        print(f"[Loop Detection] Initialized RepetitionStoppingLogitsProcessor, device={device}, eos_token_id={self.eos_token_id}, debug={self.debug}")

    def is_argmax_invariant(self) -> bool:
        """Return False because this logits processor can change the argmax."""
        return False

    def _add_request(
        self,
        params: SamplingParams,
        prompt_tok_ids: Optional[List[int]],
        output_tok_ids: List[int]
    ) -> Optional[list]:
        """
        Create state for a new request.

        Returns:
            [output_tok_ids_ref, min_pattern_len, max_pattern_len,
            max_repeat_times, force_stop_flag], or None.
        """
        if not params.extra_args:
            return None

        enable = params.extra_args.get("enable_loop_detection", False)
        if not enable:
            return None

        min_pattern_len = params.extra_args.get("min_pattern_len", self.DEFAULT_MIN_PATTERN_LEN)
        max_pattern_len = params.extra_args.get("max_pattern_len", self.DEFAULT_MAX_PATTERN_LEN)
        max_repeat_times = params.extra_args.get("max_repeat_times", self.DEFAULT_MAX_REPEAT_TIMES)

        # Return [output_tok_ids reference, minimum pattern length, maximum pattern length, maximum repeats, force-stop flag]
        return [output_tok_ids, min_pattern_len, max_pattern_len, max_repeat_times, False]

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        """Update internal request state."""
        old_count = len(self.req_info)
        process_dict_updates(self.req_info, batch_update, self._add_request)
        new_count = len(self.req_info)

        # Print only when the request count changes
        # if old_count != new_count:
        #     print(f"[Loop Detection] update_state: tracked requests {old_count} -> {new_count}")

    def _check_consecutive_repeat(self, output_tok_ids: List[int], pattern_len: int, max_repeat_times: int) -> int:
        """
        Count consecutive repetitions of a pattern at the end of a sequence.

        Args:
            output_tok_ids: Output token sequence.
            pattern_len: Pattern length.
            max_repeat_times: Repeat threshold used for early exit.

        Returns:
            The consecutive repetition count, which is at least one.
        """
        seq_len = len(output_tok_ids)

        # The sequence is too short to contain a complete pattern
        if seq_len < pattern_len:
            return 0

        # Read the trailing pattern
        pattern = tuple(output_tok_ids[-pattern_len:])
        consecutive_count = 1  # Count at least the final occurrence

        # Scan backward for consecutive repeats
        for i in range(1, max_repeat_times):
            start_idx = seq_len - (i + 1) * pattern_len
            end_idx = seq_len - i * pattern_len

            # Stop when the index moves out of bounds
            if start_idx < 0:
                break

            # Read the preceding pattern
            prev_pattern = tuple(output_tok_ids[start_idx:end_idx])

            if prev_pattern == pattern:
                consecutive_count += 1
            else:
                # Stop checking when the repeats are not consecutive
                break

        return consecutive_count

    def _check_ngram_repeat(self, state: list) -> bool:
        """
        Detect a repeated pattern of any configured length at the sequence end.

        Scan all lengths from min_pattern_len through max_pattern_len. Stop as soon as
        one pattern reaches the consecutive-repeat threshold.

        Args:
            state: [output_tok_ids, min_pattern_len, max_pattern_len,
                max_repeat_times, force_stop_flag].

        Returns:
            Whether generation must be stopped.
        """
        output_tok_ids, min_pattern_len, max_pattern_len, max_repeat_times, force_stop_flag = state

        # Return True immediately if forced stopping is already set
        if force_stop_flag:
            return True

        seq_len = len(output_tok_ids)

        # The sequence is too short to detect the minimum pattern
        if seq_len < min_pattern_len * max_repeat_times:
            return False

        # Dynamically cap the maximum pattern length at sequence_length / max_repeat_times
        effective_max_len = min(max_pattern_len, seq_len // max_repeat_times)

        # Try every pattern length from shortest to longest
        for pattern_len in range(min_pattern_len, effective_max_len + 1):
            consecutive_count = self._check_consecutive_repeat(output_tok_ids, pattern_len, max_repeat_times)

            # Stop when a pattern of this length reaches the consecutive-repeat threshold
            if consecutive_count >= max_repeat_times:
                state[4] = True  # force_stop_flag = True

                if self.debug:
                    pattern = tuple(output_tok_ids[-pattern_len:])
                    print(f"[Loop Detection] Found a repeated pattern (length={pattern_len}), repeated {consecutive_count} times >= threshold {max_repeat_times}, sequence length={seq_len}")
                    print(f"  First five tokens of the repeated pattern: {pattern[:min(5, len(pattern))]}...")

                return True

        return False

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply n-gram repetition detection to a batch of logits.

        Args:
            logits: A logits tensor with shape (num_requests, vocab_size).

        Returns:
            The modified logits tensor.
        """
        if not self.req_info:
            return logits

        # Check every tracked request for forced stopping
        for batch_idx, state in self.req_info.items():
            if batch_idx >= logits.shape[0]:
                continue

            if self._check_ngram_repeat(state):
                # Force EOS when a repeated pattern is detected
                logits[batch_idx, :] = -float("inf")
                logits[batch_idx, self.eos_token_id] = 0

        return logits


def load_test_data(data_file: str) -> List[Dict[str, Any]]:
    """
    Load test data.

    Args:
        data_file: Path to a JSONL file.

    Returns:
        A list of test records.
    """
    data = []
    data_path = Path(data_file)
    if data_path.exists():
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        raise FileNotFoundError(f"Data file not found: {data_file}")
    return data


def build_sample_id(sample: Dict[str, Any], task_name: str, sample_idx: int) -> str:
    """Build a consistent sample ID for checkpoint matching."""
    return sample.get("sample_id", f"{task_name}.{sample_idx}")


def chunk_list(items: List[Any], chunk_size: int):
    """Split a list into fixed-size chunks."""
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def load_existing_results(output_file: Path) -> Dict[str, Any]:
    """Load saved results to support resumable inference."""
    if not output_file.exists():
        return {}
    existing = {}
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                sid = record.get("sample_id")
                if sid is not None:
                    existing[sid] = record
            except json.JSONDecodeError:
                continue
    return existing


def save_results_atomic(output_file: Path, results_map: Dict[str, Any]):
    """Write results atomically to prevent corruption on interruption."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.with_suffix(output_file.suffix + ".tmp")
    sorted_results = sorted(results_map.values(), key=lambda x: x.get("sample_idx", 0))
    with open(tmp_file, "w", encoding="utf-8") as f:
        for result in sorted_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    tmp_file.replace(output_file)


def extract_core_content(generated_text: str, output_core_tag_left: str = '', output_core_tag_right: str = '') -> str:
    """
    Extract the core content from generated text.

    Strategies:
    1. Extract content between output_core_tag_left and output_core_tag_right.
    2. If that result is empty, extract the contents of \boxed{}.
    3. Strip surrounding whitespace and newlines.

    Args:
        generated_text: Generated text.
        output_core_tag_left: Opening tag.
        output_core_tag_right: Closing tag.

    Returns:
        The extracted content, or an empty string if no strategy matches.
    """
    extracted_content = ""

    # Strategy 1: extract content between output_core_tag_left and output_core_tag_right
    if output_core_tag_left and output_core_tag_right:
        left_idx = generated_text.rfind(output_core_tag_left)
        if left_idx != -1:
            right_idx = generated_text.rfind(output_core_tag_right)
            if right_idx != -1 and right_idx > left_idx:
                start_pos = left_idx + len(output_core_tag_left)
                extracted_content = generated_text[start_pos:right_idx]

    # Strategy 2: if Strategy 1 finds nothing, try extracting from \boxed{}
    if not extracted_content.strip():
        # Match \boxed{...} or boxed{...}, including escaped forms
        boxed_pattern = r'\\?boxed\{([^}]+)\}'
        match = re.search(boxed_pattern, generated_text)
        if match:
            extracted_content = match.group(1)

    return extracted_content.strip()


def build_messages(
    prompt: str,
    enable_thinking: bool = True,
    task: str = '',
    answer_before_thinking: bool = False,
    think_from_ground_truth: bool = False,
    ground_truth_answer: str = "",
) -> List[Dict[str, Any]]:
    """
    Build conversation messages.

    Args:
        prompt: User prompt.

    Returns:
        A list of messages.
    """

    if task:
        task_instruction = SMOL_TASK_INSTRUCTION[task]
        if think_from_ground_truth:
            # Oracle mode supplies the reference answer separately; avoid duplicating the "Answer:" prompt here
            prompt = task_instruction + prompt
        else:
            prompt = task_instruction + prompt + "\nAnswer: "
    else:
        prompt = prompt

    if think_from_ground_truth:
        user_prompt = (
            f"{prompt}\n\n"
            f"Reference Ground-truth Answer:\n{ground_truth_answer}\n\n"
            "Please generate a detailed reasoning process that can derive the provided reference answer. "
            "Put your full reasoning inside <think>...</think> tags. "
            "After </think>, provide a final answer to the original question. "
            "The final answer must strictly follow the output format required by the task instruction in the question."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert reasoner with extensive experience in all areas. "
                    "You are given a question and its ground-truth answer. "
                    "Your task is to generate a valid reasoning chain that leads to the given answer, then provide a normal final answer. "
                    "All reasoning must be inside <think>...</think> tags. "
                    "After </think>, output the final answer to the original question. "
                    "The final answer must strictly comply with the task-specific output format required in the question."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        return messages

    if answer_before_thinking:
        # Return the final answer before the reasoning
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert reasoner with extensive experience in all areas. You approach problems through systematic thinking and rigorous reasoning. "
                    "First provide a concise final answer, without any explanation or intermediate steps. After that, provide your detailed thinking process inside <think> </think> tags. "
                    "Your response should reflect deep understanding and precise logical thinking, making your solution path and reasoning clear to others. Do not put the final answer inside <think> ... </think> tags. Only the reasoning, intermediate steps and thinking process goes there."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    elif enable_thinking:
        messages = [
            {
                "role": "system",
                "content": SMOL_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": "You are an expert with extensive knowledge in all areas. Provide direct answers to the user's question without showing your reasoning process. Do not use <think> and </think> tags.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

    return messages


def run_inference(
    model_path: str,
    data_file: str,
    output_dir: str,
    max_new_tokens: int = 2048,
    enable_thinking: bool = True,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    seed: int = 42,
    num_generations: int = 3,
    temperature: float = 0.8,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    enable_loop_detection: bool = False,
    min_pattern_len: int = 5,
    max_pattern_len: int = 100,
    max_repeat_times: int = 3,
    batch_size: int = 32,
    answer_before_thinking: bool = False,
    max_model_len: int = 65536,
    think_from_ground_truth: bool = False,
):
    """
    Run inference.

    Args:
        model_path: Model path.
        data_file: Path to a JSONL data file.
        output_dir: Output directory.
        max_new_tokens: Maximum number of generated tokens.
        enable_thinking: Whether thinking mode is enabled.
        tensor_parallel_size: Tensor-parallel size.
        gpu_memory_utilization: Device-memory utilization target.
        seed: Random seed.
        num_generations: Independent responses generated per question.
        temperature: Sampling temperature.
        presence_penalty: Presence penalty.
        frequency_penalty: Frequency penalty.
        repetition_penalty: Repetition penalty.
        enable_loop_detection: Whether repeated-output detection is enabled.
        min_pattern_len: Minimum repeated-pattern length in tokens.
        max_pattern_len: Maximum repeated-pattern length in tokens.
        max_repeat_times: Consecutive-repeat limit before forced stopping.
        batch_size: Records processed and checkpointed per batch.
        answer_before_thinking: Whether to put the final answer before the reasoning.
        max_model_len: Maximum vLLM context length; defaults to 65,536.
        think_from_ground_truth: Whether to generate reasoning from the reference answer.
    """
    random.seed(seed)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    # Extract the task name from data_file (filename without extension)
    task_name = Path(data_file).stem
    task = task_name.split('-')[0]

    # Extract the model name from the final component of model_path
    model_name = Path(model_path).name

    # The caller prepares the output path using the model, parameters, task, and split
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"inference_{task_name}.jsonl"

    print(f"=" * 60)
    print("SMolInstruct 50k_all_subtasks inference")
    print(f"=" * 60)
    print(f"Task: {task_name}")
    print(f"Model path: {model_path}")
    print(f"Model name: {model_name}")
    print(f"Data file: {data_file}")
    print(f"Output directory: {output_path}")
    print(f"Maximum generated tokens: {max_new_tokens}")
    print(f"Number of generations: {num_generations}")
    print(f"Sampling parameters: temperature={temperature}, top_p=1.0, top_k=50")
    print(f"Thinking mode: {'enabled' if enable_thinking else 'disabled'}")
    print(f"Ground-truth-conditioned rationale generation: {'enabled' if think_from_ground_truth else 'disabled'}")
    print(f"Loop detection: {'enabled (pattern length=' + str(min_pattern_len) + '-' + str(max_pattern_len) + ', maximum repetitions=' + str(max_repeat_times) + ')' if enable_loop_detection else 'disabled'}")
    print(f"Batch size: {batch_size}")
    print(f"=" * 60)

    # Load data
    print("\nLoading data...")
    all_data = load_test_data(data_file)
    task = all_data[0]["task"]
    total_samples = len(all_data)
    print(f"Data loaded: {total_samples} total samples")

    # Load existing results to support resumption
    existing_results = load_existing_results(output_file)
    if existing_results:
        print(f"[Checkpoint] Found {len(existing_results)} saved results; skipping those samples.")

    indexed_data = list(enumerate(all_data))
    pending_data = [
        (idx, sample)
        for idx, sample in indexed_data
        if build_sample_id(sample, task_name, idx) not in existing_results
    ]
    print(f"Pending samples: {len(pending_data)} (completed: {len(existing_results)})")
    if not pending_data:
        print(f"All samples have already been processed. Results: {output_file}")
        # Print a few examples
        sorted_results = sorted(existing_results.values(), key=lambda x: x.get("sample_idx", 0))
        print("\nExample results (first three):")
        for j, result in enumerate(sorted_results[:3]):
            print(f"--- Example {j+1} ---")
            print(f"Input: {result.get('input', '')[:100]}...")
            print(f"Expected: {result.get('ground_truth', '')[:100]}...")
            first_gen = result.get('generated', {}).get('generation1', '')
            print(f"Generation 1: {first_gen[:100]}...")
            first_extract = result.get('extracted_core_answer', {}).get('generation1', '')
            if first_extract:
                print(f"Extraction 1: {first_extract[:100]}...")
        print(f"\n{'=' * 60}")
        print("Inference complete!")
        print(f"{'=' * 60}")
        return

    # Prepare the logits_processors list to pass during LLM initialization
    custom_logits_processors = None
    if enable_loop_detection:
        # Use the V1 API to register the LogitsProcessor class during LLM initialization
        custom_logits_processors = [RepetitionStoppingLogitsProcessor]
        print("[Loop Detection] RepetitionStoppingLogitsProcessor will be registered during LLM initialization")

    # Load the model
    print("\nLoading model...")
    llm_kwargs = {
        "model": model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": "bfloat16",
        "max_model_len": max_model_len,
    }

    # Pass logits_processors during initialization when loop detection is enabled
    if custom_logits_processors is not None:
        llm_kwargs["logits_processors"] = custom_logits_processors

    llm = LLM(**llm_kwargs)
    print("Model loaded!\n")

    # Set the EOS token ID when loop detection is enabled
    if enable_loop_detection:
        tokenizer = llm.get_tokenizer()
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token) if tokenizer.eos_token else 2
        print(f"[Loop Detection] EOS token ID: {eos_token_id}")
        # The EOS token ID is passed to each request through extra_args

    # Build extra_args for the V1 LogitsProcessor
    extra_args = None
    if enable_loop_detection:
        extra_args = {
            "enable_loop_detection": True,
            "min_pattern_len": min_pattern_len,
            "max_pattern_len": max_pattern_len,
            "max_repeat_times": max_repeat_times,
        }
        print(f"[Loop Detection] extra_args: {extra_args}")

    # Set sampling parameters
    sampling_params = SamplingParams(
        n=num_generations,
        temperature=temperature,
        top_p=1.0,  # Recommended value
        top_k=50,  # Recommended value
        max_tokens=max_new_tokens,
        skip_special_tokens=True,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repetition_penalty=repetition_penalty,
        extra_args=extra_args,  # V1 API: pass custom arguments through extra_args
    )

    # Run batched inference, saving each batch to support resumption
    print(f"Starting batched inference ({batch_size} samples per batch)...")
    chat_template_kwargs = {}
    if enable_thinking:
        chat_template_kwargs["enable_thinking"] = True
    else:
        chat_template_kwargs["enable_thinking"] = False

    for batch_idx, batch_items in enumerate(chunk_list(pending_data, batch_size), start=1):
        prompts = []
        for idx, sample in batch_items:
            formatted_prompt = sample["input"]
            messages = build_messages(
                formatted_prompt,
                enable_thinking,
                task=task,
                answer_before_thinking=answer_before_thinking,
                think_from_ground_truth=think_from_ground_truth,
                ground_truth_answer=sample.get("raw_output", ""),
            )
            prompts.append(messages)

        outputs = llm.chat(
            messages=prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
            chat_template_kwargs=chat_template_kwargs
        )

        for local_idx, ((idx, sample), output) in enumerate(zip(batch_items, outputs)):
            generated_dict = {}
            extracted_dict = {}

            for gen_idx, gen_output in enumerate(output.outputs, start=1):
                gen_key = f"generation{gen_idx}"
                generated_text = gen_output.text
                generated_dict[gen_key] = generated_text

                output_core_tag_left = sample.get("output_core_tag_left", "")
                output_core_tag_right = sample.get("output_core_tag_right", "")

                extracted_content = extract_core_content(
                    generated_text=generated_text,
                    output_core_tag_left=output_core_tag_left,
                    output_core_tag_right=output_core_tag_right
                )

                extracted_dict[gen_key] = extracted_content

            sample_id = build_sample_id(sample, task_name, idx)
            result = {
                "task": sample.get("task", task_name),
                "sample_idx": idx,
                "sample_id": sample_id,
                "input": sample["input"],
                "ground_truth": sample.get("raw_output", ""),
                "prompt": prompts[local_idx] if prompts else [],
                "generated": generated_dict,
                "extracted_core_answer": extracted_dict,
                "output_core_tag_left": sample.get("output_core_tag_left", ""),
                "output_core_tag_right": sample.get("output_core_tag_right", ""),
            }
            existing_results[sample_id] = result

        save_results_atomic(output_file, existing_results)
        print(f"[Checkpoint] Batch {batch_idx}: completed {len(existing_results)}/{total_samples}; results saved to {output_file}")

    # Summarize final results and print examples
    sorted_results = sorted(existing_results.values(), key=lambda x: x.get("sample_idx", 0))
    print(f"Inference complete! Results saved to: {output_file}")
    print("\nExample results (first three):")
    for j, result in enumerate(sorted_results[:3]):
        print(f"--- Example {j+1} ---")
        print(f"Input: {result['input'][:100]}...")
        print(f"Expected: {result['ground_truth'][:100]}...")
        first_gen = result['generated'].get('generation1', '')
        print(f"Generation 1: {first_gen[:100]}...")
        first_extract = result['extracted_core_answer'].get('generation1', '')
        if first_extract:
            print(f"Extraction 1: {first_extract[:100]}...")

    print(f"\n{'=' * 60}")
    print("Inference complete!")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description='Run SMolInstruct inference with vLLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference on one JSONL file
  python interns1_infer_subtasks.py --model_path /path/to/model --data_file /path/to/task.jsonl

  # Set the output directory and tensor-parallel size
  python interns1_infer_subtasks.py --model_path /path/to/model --data_file /path/to/task.jsonl --output_dir /path/to/output --tensor_parallel_size 4

  # Enable loop detection
  python interns1_infer_subtasks.py --model_path /path/to/model --data_file /path/to/task.jsonl --enable_loop_detection
        """
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Model path or model ID'
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
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )

    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=60000,
        help='Maximum number of generated tokens'
    )

    parser.add_argument(
        '--presence_penalty',
        type=float,
        default=0.0,
        help='Presence penalty (default: 0.0)'
    )

    parser.add_argument(
        '--frequency_penalty',
        type=float,
        default=0.0,
        help='Frequency penalty (default: 0.0)'
    )

    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=1.0,
        help='Repetition penalty (default: 1.0)'
    )

    parser.add_argument(
        '--disable_thinking',
        action='store_true',
        help='Disable thinking mode (enabled by default)'
    )

    parser.add_argument(
        '--answer_before_thinking',
        action='store_true',
        help='Place the final answer before the reasoning and force thinking mode'
    )

    parser.add_argument(
        '--tensor_parallel_size',
        type=int,
        default=None,
        help='Tensor-parallel size (default: detect available devices)'
    )

    parser.add_argument(
        '--gpu_memory_utilization',
        type=float,
        default=0.9,
        help='Device-memory utilization target (default: 0.9)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    parser.add_argument(
        '--num_generations',
        type=int,
        default=3,
        help='Independent responses per question (default: 3)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (default: 0.8)'
    )

    parser.add_argument(
        '--enable_loop_detection',
        action='store_true',
        help='Stop generation when a repeated pattern is detected'
    )

    parser.add_argument(
        '--min_pattern_len',
        type=int,
        default=5,
        help='Minimum repeated-pattern length in tokens (default: 5)'
    )

    parser.add_argument(
        '--max_pattern_len',
        type=int,
        default=200,
        help='Maximum repeated-pattern length in tokens (default: 200)'
    )

    parser.add_argument(
        '--max_repeat_times',
        type=int,
        default=3,
        help='Consecutive-repeat threshold before stopping (default: 3)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Inference batch size; each batch is checkpointed (default: 32)'
    )

    parser.add_argument(
        '--max_model_len',
        type=int,
        default=65536,
        help='Maximum vLLM context length (default: 65536)'
    )

    parser.add_argument(
        '--think_from_ground_truth',
        action='store_true',
        help='Generate reasoning conditioned on the reference answer'
    )

    args = parser.parse_args()

    # Detect the available device count when tensor_parallel_size is omitted
    tensor_parallel_size = args.tensor_parallel_size
    if tensor_parallel_size is None:
        tensor_parallel_size = get_available_gpu_count()

    # Determine thinking mode from the arguments:
    # - Default: enable_thinking = not disable_thinking
    # - If answer_before_thinking is explicitly set, always enable thinking mode
    enable_thinking = not args.disable_thinking
    if args.answer_before_thinking:
        enable_thinking = True
    if args.think_from_ground_truth:
        enable_thinking = True

    if args.answer_before_thinking and args.think_from_ground_truth:
        raise ValueError("--answer_before_thinking and --think_from_ground_truth are mutually exclusive")

    run_inference(
        model_path=args.model_path,
        data_file=args.data_file,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=enable_thinking,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        num_generations=args.num_generations,
        temperature=args.temperature,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        repetition_penalty=args.repetition_penalty,
        enable_loop_detection=args.enable_loop_detection,
        min_pattern_len=args.min_pattern_len,
        max_pattern_len=args.max_pattern_len,
        max_repeat_times=args.max_repeat_times,
        batch_size=args.batch_size,
        answer_before_thinking=args.answer_before_thinking,
        max_model_len=args.max_model_len,
        think_from_ground_truth=args.think_from_ground_truth,
    )


if __name__ == '__main__':
    main()
