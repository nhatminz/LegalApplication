#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/workspace/storage-shared/nlp/users/minhpn19/LegalApplication}"
export INPUT_JSONL="${INPUT_JSONL:-/workspace/storage-shared/nlp/users/minhpn19/LegalDataset/output/2k_clean_hoi_final/judgment_with_issues.jsonl}"
export CLASSIFICATION_DETAILS="${CLASSIFICATION_DETAILS:-${PROJECT_ROOT}/outputs/20261117T202530_5e87eebe/classification_details.jsonl}"
export FILTERED_JSONL="${FILTERED_JSONL:-${PROJECT_ROOT}/input/input_application_clean_v1.jsonl}"

cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m legal_application.filter_full "$@"
