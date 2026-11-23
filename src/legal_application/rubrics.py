"""Dataset validation for instance-specific Rule Application rubrics."""
from __future__ import annotations

import json
import re
from importlib.resources import files

from jsonschema import Draft202012Validator

from .data import Sample, parse_sample, rule_inputs, text_field
from .rubric_taxonomy import TAXONOMY, canonical_category


def load_schema(name: str) -> dict:
    return json.loads(files("legal_application").joinpath("schemas", name + ".schema.json").read_text(encoding="utf-8"))


DATASET_SCHEMA = load_schema("rule_application_dataset")
RUBRIC_PROTOCOL = 4


def require_schema(schema: dict, value, context: str) -> None:
    error = next(Draft202012Validator(schema).iter_errors(value), None)
    if error:
        path = ".".join(map(str, error.absolute_path)) or "$"
        raise ValueError(f"{context} schema at {path}: {error.validator}")


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def unique(values, description: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {description}")


def raw_rubric_sample(raw: dict, index: int) -> Sample:
    sample = parse_sample(raw, index)
    canonical_category(raw.get("category"))
    answer = text_field(raw["benchmark_seed"]["application"], "benchmark_seed.application")
    projected = {"doc_name": sample.doc_name, "category": raw["category"], "issues": sample.issues,
                 "input": {"facts_clean": raw["benchmark_seed"]["facts_clean"], "issues": sample.issues,
                           "rules": rule_inputs(sample)}, "expected_answer": answer}
    return Sample(index, projected, [])


def criteria_for(issue: dict) -> list[dict]:
    if "criteria" in issue:
        return issue["criteria"]
    # Read historical v1–v3 datasets through the same scoring interface.
    return [criterion for group in issue["groups"] for criterion in group["criteria"]]


def issue_blocked(issue: dict) -> bool:
    return issue["issue_review_reason"] is not None or any(c["needs_review"] and c["decisive"] for c in criteria_for(issue))


def validate_rubric_record(row: dict) -> None:
    require_schema(DATASET_SCHEMA, row, "Rule Application dataset")
    version = row["rubrics"]["version"]
    simple = version >= 3
    flat = version == RUBRIC_PROTOCOL
    taxonomy = TAXONOMY[canonical_category(row["category"])]
    payload = row["input"]
    for field, value in [("doc_name", row["doc_name"]), ("facts_clean", payload["facts_clean"]), ("expected_answer", row["expected_answer"])]:
        text_field(value, field)
    if not simple:
        unique([normalized(issue) for issue in payload["issues"]], "issue texts")
    for rule in payload["rules"]:
        for key, value in rule.items():
            if key != "citation_str" or value is not None:
                text_field(value, f"rule.{key}")
    issues = row["rubrics"]["issues"]
    current = version == 2
    if current and row["rubrics"].get("status") not in {"draft", "verified"}:
        raise ValueError("Version 2 rubrics require draft or verified status")
    if [issue["issue_text"] for issue in issues] != payload["issues"]:
        raise ValueError("Rubric issues must match input issues exactly and in order")
    all_criteria = []
    for i, issue in enumerate(issues, 1):
        text_field(issue["issue_text"], "issue_text")
        if issue["issue_index"] != i:
            raise ValueError(f"Expected rubric issue_index={i}")
        if not flat:
            unique([group["group_id"] for group in issue["groups"]], f"group IDs in issue {i}")
            for group in issue["groups"]:
                group_id = group["group_id"]
                if not simple and taxonomy.get(group_id) != group["group_name"] and not re.fullmatch(r"other_[1-9][0-9]*", group_id):
                    raise ValueError(f"Invalid group {group_id} for {row['category']}")
                text_field(group_id, "group_id")
                text_field(group["group_name"], "group_name")
        criteria = criteria_for(issue)
        for index, criterion in enumerate(criteria, 1):
            if criterion["criterion_id"] != f"I{i}_C{index}":
                raise ValueError("Criterion IDs must be unique and in issue order")
            if simple:
                text_field(criterion["description"], "description")
                text_field(criterion["reference_reasoning"], "reference_reasoning")
                continue
            for field in ("criterion", "unit_id", "reference_reasoning"):
                text_field(criterion[field], field)
            for field, original in [("source_evidence", row["expected_answer"]), ("relevant_facts", payload["facts_clean"])]:
                if any(not quote.strip() or quote not in original for quote in criterion[field]):
                    raise ValueError(f"Criterion {criterion['criterion_id']} has ungrounded {field}")
            refs = criterion["rule_refs"]
            unique(refs, "rule references")
            if any(index > len(payload["rules"]) for index in refs):
                raise ValueError("Rule reference is outside the input rules")
            missing = criterion["missing_information"]
            review = criterion["needs_review"]
            if current:
                if not {"reasoning_type", "legal_condition", "application_link"} <= criterion.keys():
                    raise ValueError("Criterion lacks reasoning_type or grounding fields")
                for field in ["legal_condition", "application_link"]:
                    if criterion[field] is not None:
                        text_field(criterion[field], field)
                if criterion["reasoning_type"] == "fact_to_rule_application":
                    if not criterion["application_link"] or not criterion["legal_condition"]:
                        raise ValueError(f"{criterion['criterion_id']}: explain the legal condition and how it applies")
                    if not review and (not criterion["relevant_facts"] or not refs):
                        raise ValueError(f"{criterion['criterion_id']}: Fact-to-Rule needs input facts and rules or explicit review")
            if review == criterion["scorable"] or review != bool(missing) or review != bool(criterion["review_reason"]):
                raise ValueError("Criterion review requires missing_information, review_reason and scorable=false")
            for explanation in missing + ([criterion["review_reason"]] if review else []):
                text_field(explanation, "criterion review information")
        if simple:
            if issue.get("reference_verdict") is not None:
                text_field(issue["reference_verdict"], "reference_verdict")
            continue
        reason = issue["issue_review_reason"]
        if not criteria or issue_blocked(issue):
            text_field(reason, "issue_review_reason")
        verdict = issue["reference_verdict"]
        if verdict is None:
            text_field(issue["outcome_review_reason"], "outcome_review_reason")
            if issue["verdict_evidence"]:
                raise ValueError("Unresolved verdict must have empty evidence")
        else:
            text_field(verdict, "reference_verdict")
            if (issue["outcome_review_reason"] is not None or not issue["verdict_evidence"] or
                    any(not quote.strip() or quote not in row["expected_answer"] for quote in issue["verdict_evidence"])):
                raise ValueError("Reference verdict requires verbatim expected-answer evidence")
        all_criteria.extend(criteria)
    unique([c["unit_id"] for c in all_criteria], "scored atomic units")
    unique([normalized(c["criterion"]) for c in all_criteria], "criteria across issues")


def rubric_sample(raw: dict, index: int) -> Sample:
    validate_rubric_record(raw)
    if raw["rubrics"].get("status") == "draft":
        raise ValueError("Draft rubric dataset must be verified before evaluation")
    return Sample(index, raw, [])
