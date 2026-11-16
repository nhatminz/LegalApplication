from __future__ import annotations

import json
from importlib.resources import files

from jsonschema import Draft202012Validator

from .data import Sample

LABEL_SCHEMA = json.loads(files("legal_application").joinpath("schemas", "judge_labels.schema.json").read_text())
VALIDATOR = Draft202012Validator(LABEL_SCHEMA)


class LabelError(ValueError):
    pass


def validate_labels(labels: dict, sample: Sample) -> None:
    errors = sorted(VALIDATOR.iter_errors(labels), key=lambda e: str(list(e.path)))
    if errors:
        # No raw invalid output in logs or repair instructions.
        error = errors[0]
        raise LabelError(f"schema violation at {'.'.join(map(str, error.path)) or '$'} ({error.validator})")
    if labels["doc_name"] != sample.doc_name:
        raise LabelError("doc_name does not match input")
    rows = labels["issues_evaluation"]
    if len(rows) != len(sample.issues):
        raise LabelError("issues_evaluation must contain every original issue exactly once")
    for i, (row, original) in enumerate(zip(rows, sample.issues), 1):
        if row["issue_index"] != i or row["original_issue"] != original:
            raise LabelError(f"issue {i}: index/order/original_issue mismatch")
        if not row["explanation"].strip():
            raise LabelError(f"issue {i}: explanation is empty")
        if row["needs_review"]:
            if row["correct"] is not None or row["inferred_reference_verdict"] is not None:
                raise LabelError(f"issue {i}: uncertain gold requires null correct/reference verdict")
            if not labels["needs_review"]:
                raise LabelError(f"issue {i}: issue review must propagate to whole record")
        else:
            if type(row["correct"]) is not bool or not (row["inferred_reference_verdict"] or "").strip():
                raise LabelError(f"issue {i}: scorable issue requires correct and reference verdict")
            if not row["covered"] and row["correct"]:
                raise LabelError(f"issue {i}: uncovered issue cannot have correct=true")
    review = labels["needs_review"]
    if labels["evaluation_status"] != ("needs_review" if review else "completed"):
        raise LabelError("evaluation_status inconsistent with needs_review")
    if (review and labels["final_order_correct"] is not None) or (not review and type(labels["final_order_correct"]) is not bool):
        raise LabelError("final_order_correct must be null for review, boolean otherwise")
    if not labels["final_order_explanation"].strip():
        raise LabelError("final_order_explanation is empty")


def score_labels(labels: dict, sample: Sample) -> dict:
    validate_labels(labels, sample)
    if labels["needs_review"]:
        return dict.fromkeys(["final_order", "coverage", "correctness", "total"], None)
    rows = labels["issues_evaluation"]
    n = len(sample.issues)
    order = 5 * int(labels["final_order_correct"])
    coverage = 2 * sum(row["covered"] for row in rows) / n
    correctness = 3 * sum(row["covered"] and row["correct"] for row in rows) / n
    return {"final_order": order, "coverage": coverage, "correctness": correctness,
            "total": order + coverage + correctness}

