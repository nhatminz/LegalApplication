#!/usr/bin/env bash
set -euo pipefail

# Thay batch/concurrency; dùng --relocate khi chuyển project/output sang thư mục mới.
# Phải Ctrl-C và đợi process cũ thoát trước khi gọi helper này.
export PROJECT_ROOT="${PROJECT_ROOT:-/workspace/storage-shared/nlp/users/minhpn19/LegalApplication}"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
exec python "$PROJECT_ROOT/scripts/resume_statistic.py" "$@"
