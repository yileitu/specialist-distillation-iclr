#!/usr/bin/env bash
# Split a JSONL file and run translation inference for each split locally.

set -euo pipefail

show_usage() {
  cat <<'EOF'
Usage:
  batch_submit_tasks_with_params.sh --data_file PATH --model_path PATH [OPTIONS] [-- INFERENCE_OPTIONS]

Options:
  --data_file PATH              Source JSONL file (required).
  --model_path PATH             Model directory (required).
  --output_dir PATH             Output root (default: inference/output).
  --split_size N                Records per split (default: 500).
  --tensor_parallel_size N      Optional vLLM tensor-parallel size.
  --help                        Show this message.

All arguments after -- are forwarded to infer_translation.py. The script runs
splits sequentially. Use your own scheduler or launcher if parallel execution is
required.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/infer_translation.py"

DATA_FILE=""
MODEL_PATH=""
OUTPUT_DIR="${SCRIPT_DIR}/output"
SPLIT_SIZE=500
TENSOR_PARALLEL_SIZE=""
INFERENCE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_file)
      DATA_FILE="$2"
      shift 2
      ;;
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --split_size)
      SPLIT_SIZE="$2"
      shift 2
      ;;
    --tensor_parallel_size)
      TENSOR_PARALLEL_SIZE="$2"
      shift 2
      ;;
    --help|-h)
      show_usage
      exit 0
      ;;
    --)
      shift
      INFERENCE_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$DATA_FILE" || -z "$MODEL_PATH" ]]; then
  echo "--data_file and --model_path are required." >&2
  show_usage >&2
  exit 2
fi
if [[ ! -f "$DATA_FILE" ]]; then
  echo "Data file not found: $DATA_FILE" >&2
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model directory not found: $MODEL_PATH" >&2
  exit 1
fi

DATA_FILE="$(realpath "$DATA_FILE")"
MODEL_PATH="$(realpath "$MODEL_PATH")"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
DATA_NAME="$(basename "${DATA_FILE%.jsonl}")"
MODEL_NAME="$(basename "$MODEL_PATH")"
SPLIT_DIR="$(dirname "$DATA_FILE")/splits/${DATA_NAME}_split${SPLIT_SIZE}"
OUTPUT_TASK_DIR="${OUTPUT_DIR}/${MODEL_NAME}/${DATA_NAME}/split${SPLIT_SIZE}"

mkdir -p "$SPLIT_DIR" "$OUTPUT_TASK_DIR"
shopt -s nullglob
SPLIT_FILES=("$SPLIT_DIR"/*.jsonl)
if [[ ${#SPLIT_FILES[@]} -eq 0 ]]; then
  split -l "$SPLIT_SIZE" -d -a 4 --additional-suffix=.jsonl \
    "$DATA_FILE" "$SPLIT_DIR/${DATA_NAME}_part"
  SPLIT_FILES=("$SPLIT_DIR"/*.jsonl)
fi
shopt -u nullglob

if [[ ${#SPLIT_FILES[@]} -eq 0 ]]; then
  echo "No split files were created." >&2
  exit 1
fi

echo "Running ${#SPLIT_FILES[@]} split(s) sequentially."
for split_file in "${SPLIT_FILES[@]}"; do
  command=(
    python3 "$PYTHON_SCRIPT"
    --model_path "$MODEL_PATH"
    --data_file "$split_file"
    --output_dir "$OUTPUT_TASK_DIR"
  )
  if [[ -n "$TENSOR_PARALLEL_SIZE" ]]; then
    command+=(--tensor_parallel_size "$TENSOR_PARALLEL_SIZE")
  fi
  command+=("${INFERENCE_ARGS[@]}")
  echo "Processing $(basename "$split_file")"
  "${command[@]}"
done

echo "Inference completed. Results: $OUTPUT_TASK_DIR"
