#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/workspace/storage-shared/nlp/users/minhpn19/LegalApplication}"
export RAW_INPUT_JSONL="${RAW_INPUT_JSONL:-${PROJECT_ROOT}/input/final_raw_600_input_application.jsonl}"
# Each successful sample is written to both final files. Use separate paths for test runs.
export FINAL_APPLICATION_JSONL="${FINAL_APPLICATION_JSONL:-${PROJECT_ROOT}/input/final/final_600_rule_application.jsonl}"
export FINAL_ISSUE_SPOTTING_JSONL="${FINAL_ISSUE_SPOTTING_JSONL:-${PROJECT_ROOT}/input/final/final_600_issue_spotting.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs}"
export EXPECTED_COUNT="${EXPECTED_COUNT:-600}"
export RUBRIC_BASE_URL="${RUBRIC_BASE_URL:-http://172.29.116.48:8000/v1}"
export RUBRIC_MODEL="${RUBRIC_MODEL:-DeepSeek-V4.1-Flash-1017}"
export RUBRIC_API_KEY="${RUBRIC_API_KEY:-EMPTY}"
export RUBRIC_THINKING_MODE="${RUBRIC_THINKING_MODE:-deepseek_vllm_xhigh}"
export RUBRIC_CONCURRENCY="${RUBRIC_CONCURRENCY:-20}"
export RUBRIC_MAX_OUTPUT_TOKENS="${RUBRIC_MAX_OUTPUT_TOKENS:-65536}"
export RUBRIC_RESPONSE_FORMAT="${RUBRIC_RESPONSE_FORMAT:-json_schema}"
export RUBRIC_JSON_RETRIES="${RUBRIC_JSON_RETRIES:-2}"
export API_TIMEOUT_SECONDS="${API_TIMEOUT_SECONDS:-1800}"
export API_MAX_RETRIES="${API_MAX_RETRIES:-3}"
export RETRY_BACKOFF_SECONDS="${RETRY_BACKOFF_SECONDS:-2}"
export BATCH_SIZE="${BATCH_SIZE:-20}"
export LIMIT="${LIMIT:-all}"
export RUN_ID="${RUN_ID:-}"
export RESUME="${RESUME:-0}"
export MODE=build-rubrics
unset STATISTIC_SEED_JSONL

cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m legal_application
