#!/usr/bin/env bash
set -euo pipefail

# MODEL ĐƯỢC ĐÁNH GIÁ: sửa hai giá trị mặc định dưới đây khi đổi model/server.
# Ví dụ Qwen3-4B được host trên cùng máy B200 ở port 8001:
#   vllm serve Qwen/Qwen3-4B --served-model-name Qwen/Qwen3-4B \
#     --host 0.0.0.0 --port 8001 --reasoning-parser qwen3
# Lệnh serve chạy riêng trên B200; script này chỉ gọi endpoint đã được host.
# Nếu host máy khác, đổi 127.0.0.1 thành IP máy đó; giữ /v1 ở cuối URL.
# GENERATOR_MODEL phải khớp --served-model-name, không phải URL tải checkpoint.
export GENERATOR_BASE_URL="${GENERATOR_BASE_URL:-http://127.0.0.1:8001/v1}"
export GENERATOR_MODEL="${GENERATOR_MODEL:-Qwen/Qwen3-4B}"
export GENERATOR_API_KEY="${GENERATOR_API_KEY:-EMPTY}"
# DeepSeek V4.1 vLLM: low=25, high=50, xhigh=75, max=100 hoặc số nguyên 1–100.
# on/off = bật/tắt DeepSeek; default/none = dùng mặc định server (none là alias cũ).
# Qwen3: qwen3_on/qwen3_off. Không dùng các mức effort DeepSeek cho Qwen3.
# Các tên cũ như deepseek_vllm_xhigh vẫn được hỗ trợ.
export GENERATOR_THINKING_MODE="${GENERATOR_THINKING_MODE:-none}"

# Runtime paths are based on PROJECT_ROOT. Relative path exports also resolve from it.
export PROJECT_ROOT="${PROJECT_ROOT:-/workspace/storage-shared/nlp/minhpn19/LegalApplication}"
# Dòng JSON lỗi sẽ được ghi log rồi xóa khỏi INPUT_JSONL trước khi gọi model.
export INPUT_JSONL="${INPUT_JSONL:-${PROJECT_ROOT}/input/judgment_with_issues_sample.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs}"
export RUN_ID="${RUN_ID:-}"  # Empty -> Vietnam local timestamp + random suffix.
export MODE="${MODE:-full}"  # full | judge-only | validate-only
export GENERATION_JSONL="${GENERATION_JSONL:-}"
export RESUME="${RESUME:-0}"

# MODEL JUDGE: chấm bài làm của generator, mặc định DeepSeek với effort xhigh.
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://172.29.62.173:8000/v1}"
export JUDGE_MODEL="${JUDGE_MODEL:-DeepSeek-V4.1-Flash-1017}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
# Cùng các lựa chọn như GENERATOR_THINKING_MODE ở trên; mặc định Judge xhigh=75.
export JUDGE_THINKING_MODE="${JUDGE_THINKING_MODE:-deepseek_vllm_xhigh}"
export JUDGE_RESPONSE_FORMAT="${JUDGE_RESPONSE_FORMAT:-json_schema}"

export LIMIT="${LIMIT:-}"  # Empty/all = ALL input; 0 = select zero; no hidden cap.
# Mỗi batch tối đa 20 samples; mỗi giai đoạn gọi tối đa 20 requests đồng thời.
# LIMIT vẫn là tổng số samples của run, độc lập với mức chạy đồng thời này.
export BATCH_SIZE="${BATCH_SIZE:-20}"
export GENERATOR_CONCURRENCY="${GENERATOR_CONCURRENCY:-20}"
export JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-20}"
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-32768}"
export API_TIMEOUT_SECONDS="${API_TIMEOUT_SECONDS:-1800}"
export API_MAX_RETRIES="${API_MAX_RETRIES:-3}"
export JUDGE_JSON_RETRIES="${JUDGE_JSON_RETRIES:-2}"
export RETRY_BACKOFF_SECONDS="${RETRY_BACKOFF_SECONDS:-2}"

cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
# Dùng python trên PATH của môi trường đã activate, dù .venv nằm ở đâu.
exec python -m legal_application
