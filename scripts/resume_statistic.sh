#!/usr/bin/env bash
set -euo pipefail

# Chỉ thay BATCH_SIZE/JUDGE_CONCURRENCY; cấu hình Judge/input được khôi phục từ run cũ.
# Phải Ctrl-C và đợi process cũ thoát trước khi gọi helper này.
export PROJECT_ROOT="${PROJECT_ROOT:-/workspace/storage-shared/nlp/minhpn19/LegalApplication}"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
exec python "$PROJECT_ROOT/scripts/resume_statistic.py" "$@"
