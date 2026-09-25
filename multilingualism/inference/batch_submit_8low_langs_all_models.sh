#!/usr/bin/env bash
# Run batched inference for multiple models and language-direction files.

set -euo pipefail

show_usage() {
  cat <<'EOF'
Usage:
  batch_submit_8low_langs_all_models.sh --data-root DIR --output-dir DIR \
    --model PATH [--model PATH ...] [--direction en2xx|xx2en|both] \
    [--languages "bn cs hu sr sw te th vi"] [-- INFERENCE_OPTIONS]

The data root must contain en2xx/ and xx2en/ directories. Arguments after --
are forwarded to batch_submit_tasks_with_params.sh after its own -- marker.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INNER_SCRIPT="${SCRIPT_DIR}/batch_submit_tasks_with_params.sh"
DATA_ROOT=""
OUTPUT_DIR=""
DIRECTION="both"
LANGUAGES=""
MODELS=()
INFERENCE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --model)
      MODELS+=("$2")
      shift 2
      ;;
    --direction)
      DIRECTION="$2"
      shift 2
      ;;
    --languages)
      LANGUAGES="$2"
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

if [[ -z "$DATA_ROOT" || -z "$OUTPUT_DIR" || ${#MODELS[@]} -eq 0 ]]; then
  show_usage >&2
  exit 2
fi
if [[ "$DIRECTION" != "en2xx" && "$DIRECTION" != "xx2en" && "$DIRECTION" != "both" ]]; then
  echo "Invalid direction: $DIRECTION" >&2
  exit 2
fi

language_selected() {
  local name="$1"
  [[ -z "$LANGUAGES" ]] && return 0
  local lang
  lang="$(sed -E 's/.*en2([[:alpha:]]{2,3}).*/\1/; t; s/.*([[:alpha:]]{2,3})2en.*/\1/' <<<"$name")"
  [[ " $LANGUAGES " == *" $lang "* ]]
}

run_direction() {
  local direction="$1"
  local data_dir="${DATA_ROOT}/${direction}"
  if [[ ! -d "$data_dir" ]]; then
    echo "Skipping missing directory: $data_dir" >&2
    return
  fi

  shopt -s nullglob
  local files=("$data_dir"/*.jsonl)
  shopt -u nullglob
  for data_file in "${files[@]}"; do
    language_selected "$(basename "$data_file")" || continue
    for model_path in "${MODELS[@]}"; do
      "$INNER_SCRIPT" \
        --data_file "$data_file" \
        --model_path "$model_path" \
        --output_dir "${OUTPUT_DIR}/${direction}" \
        -- "${INFERENCE_ARGS[@]}"
    done
  done
}

if [[ "$DIRECTION" == "en2xx" || "$DIRECTION" == "both" ]]; then
  run_direction en2xx
fi
if [[ "$DIRECTION" == "xx2en" || "$DIRECTION" == "both" ]]; then
  run_direction xx2en
fi

echo "All inference runs completed."
