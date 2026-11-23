from __future__ import annotations

import json
from importlib.resources import files
from statistics import mean

from jsonschema import Draft202012Validator

from .data import Sample
from .rubrics import criteria_for, issue_blocked, load_schema, require_schema

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



# LEGIT helpers above are used only by the explicit legacy evaluation profile.
RUBRIC_JUDGE_SCHEMA = load_schema("rubric_judge")


def evaluation_plan(sample: Sample) -> dict:
    rubric = sample.raw["rubrics"]
    simple = rubric["version"] >= 3
    issues = []
    for issue in rubric["issues"]:
        criteria = criteria_for(issue)
        blocked = not simple and issue_blocked(issue)
        eligible = [c["criterion_id"] for c in criteria if simple or c["scorable"]]
        status = "needs_review" if blocked or any(c.get("needs_review") for c in criteria) or not eligible else "scored"
        if not criteria:
            status = "unscorable"
        outcome = issue.get("reference_verdict") is not None and not blocked
        issues.append({"issue_index": issue["issue_index"], "evaluation_status": status,
                       "criterion_ids": eligible, "application_scorable": bool(eligible) and not blocked,
                       "outcome_scorable": outcome})
    review = not simple and any(row["evaluation_status"] != "scored" or not row["outcome_scorable"] for row in issues)
    return {"issues": issues, "needs_review": review}


def score_rubric_labels(labels: dict, sample: Sample, answer: str) -> dict:
    try:
        require_schema(RUBRIC_JUDGE_SCHEMA, labels, "rubric Judge")
    except ValueError as exc:
        raise LabelError(str(exc)) from None
    if labels["doc_name"] != sample.doc_name or labels["category"] != sample.raw["category"]:
        raise LabelError("Judge doc_name/category mismatch")
    plan = evaluation_plan(sample)
    planned_issues = plan["issues"]
    rows = labels["issues_evaluation"]
    if len(rows) != len(planned_issues):
        raise LabelError("Judge must return exactly one result per issue")
    if labels["needs_review"] != plan["needs_review"] or labels["evaluation_status"] != (
            "needs_review" if plan["needs_review"] else "completed"):
        raise LabelError("Judge sample review status does not match the evaluation plan")
    issue_results = []
    simple = sample.raw["rubrics"]["version"] >= 3
    for row, planned in zip(rows, planned_issues):
        issue = planned["issue_index"]
        if row["issue_index"] != issue or row["evaluation_status"] != planned["evaluation_status"]:
            raise LabelError(f"Issue {issue}: index or review status mismatch")
        scores = row["criterion_scores"]
        ids = [item["criterion_id"] for item in scores]
        if len(ids) != len(set(ids)) or set(ids) != set(planned["criterion_ids"]):
            raise LabelError(f"Issue {issue}: must score every eligible criterion exactly once")
        for item in scores:
            if type(item["score"]) is not int:
                raise LabelError(f"{item['criterion_id']}: score must be integer 0, 1 or 2")
            evidence = item["evidence_from_model_answer"]
            if not simple and ((item["score"] > 0 and not evidence)
                               or any(not quote.strip() or quote not in answer for quote in evidence)):
                raise LabelError(f"{item['criterion_id']}: cite a short verbatim excerpt of the model answer")
            if not item["explanation"].strip():
                raise LabelError(f"{item['criterion_id']}: missing explanation")
        outcome = row["outcome_correct"]
        if (outcome is not None) != planned["outcome_scorable"] or not row["outcome_explanation"].strip():
            raise LabelError(f"Issue {issue}: outcome eligibility or explanation invalid")
        issue_results.append({"issue_index": issue,
                              "application_score": sum(item["score"] for item in scores) / (2 * len(scores))
                              if planned["application_scorable"] else None,
                              "scored_criteria": len(scores),
                              "outcome_accuracy": int(outcome) if outcome is not None else None,
                              "evaluation_status": planned["evaluation_status"]})

    def average(key):
        values = [row[key] for row in issue_results if row[key] is not None]
        return mean(values) if values else None

    application = average("application_score")
    return {"application_score": application, "application_score_10": 10 * application if application is not None else None,
            "outcome_accuracy": average("outcome_accuracy"), "issues": issue_results}
