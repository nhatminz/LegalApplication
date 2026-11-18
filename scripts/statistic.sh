#!/usr/bin/env bash
set -euo pipefail

# Data Completeness Classification: chỉ đọc input, không xóa/sửa file nguồn.
# Record JSON/schema lỗi được ghi invalid_input_records.jsonl và bỏ qua; tiếp tục với records hợp lệ.
export PROJECT_ROOT="${PROJECT_ROOT:-/workspace/storage-shared/nlp/users/minhpn19/LegalApplication}"
export INPUT_JSONL="${INPUT_JSONL:-${PROJECT_ROOT}/input/judgment_with_issues_sample.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs}"
export RUN_ID="${RUN_ID:-}"
export LIMIT="${LIMIT:-}"  # Rỗng/all = toàn bộ; không có hidden cap.
export BATCH_SIZE="${BATCH_SIZE:-20}"
export JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-20}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://172.29.62.173:8000/v1}"
export JUDGE_MODEL="${JUDGE_MODEL:-DeepSeek-V4.1-Flash-1017}"
export JUDGE_THINKING_MODE="${JUDGE_THINKING_MODE:-deepseek_vllm_xhigh}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
export JUDGE_RESPONSE_FORMAT="${JUDGE_RESPONSE_FORMAT:-json_schema}"
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-32768}"
export API_TIMEOUT_SECONDS="${API_TIMEOUT_SECONDS:-1800}"
export API_MAX_RETRIES="${API_MAX_RETRIES:-3}"
export JUDGE_JSON_RETRIES="${JUDGE_JSON_RETRIES:-2}"
export RETRY_BACKOFF_SECONDS="${RETRY_BACKOFF_SECONDS:-2}"
export RESUME="${RESUME:-0}"
export STATISTIC_SEED_JSONL="${STATISTIC_SEED_JSONL:-}"
export MODE=statistic  # Không kế thừa MODE=full/validate-only của application.

cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m legal_application
