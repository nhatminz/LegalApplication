#!/usr/bin/env bash
set -euo pipefail

# Runtime paths are based on PROJECT_ROOT. Relative path exports also resolve from it.
export PROJECT_ROOT="${PROJECT_ROOT:-/workspace/storage-shared/nlp/minhpn19/LegalApplication}"
export INPUT_JSONL="${INPUT_JSONL:-${PROJECT_ROOT}/input/judgment_with_issues_sample.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs}"
export RUN_ID="${RUN_ID:-}"  # Empty -> Vietnam local timestamp + random suffix.
export MODE="${MODE:-full}"  # full | judge-only | validate-only
export GENERATION_JSONL="${GENERATION_JSONL:-}"
export RESUME="${RESUME:-0}"

# Required for full mode: the exact served model name and the endpoint you deploy.
export GENERATOR_BASE_URL="${GENERATOR_BASE_URL:-}"
export GENERATOR_MODEL="${GENERATOR_MODEL:-}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://172.29.62.173:8000/v1}"
export JUDGE_MODEL="${JUDGE_MODEL:-DeepSeek-V4.1-Flash-1017}"
export GENERATOR_API_KEY="${GENERATOR_API_KEY:-EMPTY}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
export GENERATOR_THINKING_MODE="${GENERATOR_THINKING_MODE:-none}"
export JUDGE_RESPONSE_FORMAT="${JUDGE_RESPONSE_FORMAT:-json_schema}"

export LIMIT="${LIMIT:-}"  # Empty/all = ALL input; 0 = select zero; no hidden cap.
export BATCH_SIZE="${BATCH_SIZE:-8}"
export GENERATOR_CONCURRENCY="${GENERATOR_CONCURRENCY:-2}"
export JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-2}"
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-32768}"
export API_TIMEOUT_SECONDS="${API_TIMEOUT_SECONDS:-1800}"
export API_MAX_RETRIES="${API_MAX_RETRIES:-3}"
export JUDGE_JSON_RETRIES="${JUDGE_JSON_RETRIES:-2}"
export RETRY_BACKOFF_SECONDS="${RETRY_BACKOFF_SECONDS:-2}"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment missing: $PYTHON_BIN. Install with Python >=3.10; see README.md." >&2
  exit 2
fi
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m legal_application
