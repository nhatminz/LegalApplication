#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/workspace/storage-shared/nlp/users/minhpn19/LegalApplication}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs}"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
exec python "$PROJECT_ROOT/scripts/resume_rubric_run.py" "$@"
