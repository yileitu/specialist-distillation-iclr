#!/usr/bin/env bash
# Postprocess every directory beneath a root that directly contains JSONL files.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 ROOT [POSTPROCESS_OPTIONS]" >&2
  exit 2
fi

ROOT="$1"
shift
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "$ROOT" ]]; then
  echo "Directory not found: $ROOT" >&2
  exit 1
fi

mapfile -t INPUT_DIRS < <(find "$ROOT" -type f -name '*.jsonl' -printf '%h\n' | sort -u)
if [[ ${#INPUT_DIRS[@]} -eq 0 ]]; then
  echo "No JSONL files found under: $ROOT" >&2
  exit 1
fi

for input_path in "${INPUT_DIRS[@]}"; do
  echo "Postprocessing: $input_path"
  python3 "${SCRIPT_DIR}/postprocess_translation.py" \
    --input_path "$input_path" \
    "$@"
done

echo "Postprocessing completed."
