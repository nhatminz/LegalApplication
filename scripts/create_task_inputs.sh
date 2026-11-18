#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/workspace/storage-shared/nlp/users/minhpn19/LegalApplication}"
export TASK_SOURCE_JSONL="${TASK_SOURCE_JSONL:-${PROJECT_ROOT}/input/input_application_clean_v2.jsonl}"
export TASK_INPUT_DIR="${TASK_INPUT_DIR:-${PROJECT_ROOT}/input/input_temp_18_11}"

cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
# Dùng python của môi trường đã activate, không phụ thuộc vị trí .venv.
exec python "$PROJECT_ROOT/scripts/create_task_inputs.py" "$@"
