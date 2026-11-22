"""Validation and materialization of instance-specific Rule Application rubrics."""
from __future__ import annotations

import json
import re
from importlib.resources import files

from jsonschema import Draft202012Validator

from .data import Sample, canonical_hash, parse_sample, rule_inputs, text_field
from .rubric_taxonomy import TAXONOMY, canonical_category


def load_schema(name: str) -> dict:
    return json.loads(files("legal_application").joinpath("schemas", name + ".schema.json").read_text(encoding="utf-8"))


DECOMPOSITION_SCHEMA = load_schema("rubric_decomposition")
PLAN_SCHEMA = load_schema("rubric_plan")
AUDIT_SCHEMA = load_schema("rubric_audit")
DATASET_SCHEMA = load_schema("rule_application_dataset")
RUBRIC_PROTOCOL = 2


def strict_wire_schema(schema: dict) -> dict:
    """Require current fields on the wire while retaining historical rubric readers."""
    result = {}
    for key, value in schema.items():
        if isinstance(value, dict):
            result[key] = {n: strict_wire_schema(field) for n, field in value.items()} if key == "properties" else strict_wire_schema(value)
        elif isinstance(value, list):
            result[key] = [strict_wire_schema(item) if isinstance(item, dict) else item for item in value]
        else:
            result[key] = value
    if "properties" in result:
        result["required"] = list(result["properties"])
    return result


DECOMPOSITION_V2 = strict_wire_schema(DECOMPOSITION_SCHEMA)
PLAN_V2 = strict_wire_schema(PLAN_SCHEMA)


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


def source_segments(text: str, prefix: str) -> dict[str, str]:
    """Index contiguous source wording, retaining clause boundaries where possible."""
    segments = {}
    start = 0
    while start < len(text):
        end = min(start + 600, len(text))
        if end < len(text):
            window = text[start:end]
            boundaries = list(re.finditer(r"[.!?;](?=\s)|\n+", window))
            if not boundaries:
                boundaries = list(re.finditer(r"\s+", window))
            if boundaries:
                end = start + boundaries[-1].end()
        excerpt = text[start:end]
        if excerpt.strip():
            segments[f"{prefix}{len(segments) + 1}"] = excerpt
        start = end
    return segments


def raw_rubric_sample(raw: dict, index: int) -> Sample:
    sample = parse_sample(raw, index)
    canonical_category(raw.get("category"))
    answer = text_field(raw["benchmark_seed"]["application"], "benchmark_seed.application")
    projected = {"doc_name": sample.doc_name, "category": raw["category"], "issues": sample.issues,
                 "input": {"facts_clean": raw["benchmark_seed"]["facts_clean"], "issues": sample.issues,
                           "rules": rule_inputs(sample)}, "expected_answer": answer}
    for key in ["code", "instance_level"]:
        if raw.get(key) is not None:
            projected[key] = raw[key]
    return Sample(index, projected, [])


def unit_spans(unit: dict, sources: dict[str, str]) -> list[tuple[str, int, int]]:
    spans = []
    for evidence in unit["evidence"]:
        quote = text_field(evidence["quote"], "evidence.quote").strip()
        sid = evidence["source_id"]
        source = sources.get(sid)
        if source is None:
            raise ValueError(f"{unit['unit_id']}: unknown evidence source ID {evidence['source_id']!r}")
        start = source.find(quote)
        if start < 0:
            # Restore exact source spacing when the service flattened a line break.
            match = re.search(r"\s+".join(re.escape(word) for word in quote.split()), source)
            if match is None:
                raise ValueError(f"{unit['unit_id']}: evidence must quote {evidence['source_id']} verbatim")
            start, end = match.span()
        else:
            end = start + len(quote)
        spans.append((sid, start, end))
    return list(dict.fromkeys(spans))


def unit_quotes(unit: dict, sources: dict[str, str]) -> list[str]:
    return list(dict.fromkeys(sources[sid][start:end] for sid, start, end in unit_spans(unit, sources)))


def validate_unit(unit: dict, sources: dict[str, str], taxonomy: dict) -> list[str]:
    text_field(unit["reasoning"], "unit reasoning")
    if not re.fullmatch(r"[UP][1-9][0-9]*", unit["unit_id"]):
        raise ValueError("Use a positive numbered unit or proposition ID")
    if unit["group_id"] == "other":
        text_field(unit["group_rationale"], "other group rationale")
    elif unit["group_id"] not in taxonomy or unit["group_rationale"] is not None:
        raise ValueError(f"{unit['unit_id']}: invalid decomposition group or group_rationale")
    if {e["source_id"] for e in unit["evidence"]} != set(unit["source_ids"]):
        raise ValueError(f"{unit['unit_id']}: cite verbatim evidence in each source_id")
    return unit_quotes(unit, sources)


def validate_decomposition(value: dict, sources: dict[str, str], *, taxonomy: dict | None = None,
                           prefix: str = "U") -> dict[str, dict]:
    require_schema(DECOMPOSITION_V2 if taxonomy is not None else DECOMPOSITION_SCHEMA, value, "decomposition")
    units = value["units"]
    unique([row["unit_id"] for row in units], "unit IDs")
    unique([normalized(row["reasoning"]) for row in units], "atomic reasoning")
    mapped = set()
    for row in units:
        text_field(row["unit_id"], "unit_id")
        text_field(row["reasoning"], "unit reasoning")
        unique(row["source_ids"], f"source IDs for {row['unit_id']}")
        if taxonomy is not None:
            if not row["unit_id"].startswith(prefix):
                raise ValueError(f"Use {prefix} IDs for this stage")
            validate_unit(row, sources, taxonomy)
        mapped.update(row["source_ids"])
    excluded = [row["source_id"] for row in value["non_reasoning"]]
    unique(excluded, "non-reasoning source IDs")
    for row in value["non_reasoning"]:
        text_field(row["reason"], "non_reasoning reason")
    if mapped & set(excluded) or mapped | set(excluded) != set(sources):
        raise ValueError("Every expected-answer segment must be accounted for by a unit or a non_reasoning reason")
    return {row["unit_id"]: row for row in units}


def materialize(sample: Sample, decomposition: dict, plan: dict, *, version: int = 1) -> tuple[dict, dict]:
    """Bind source IDs, validate accounting and build the fixed public rubric."""
    raw = sample.raw
    source = source_segments(raw["expected_answer"], "S")
    facts = source_segments(raw["input"]["facts_clean"], "F")
    groups = TAXONOMY[canonical_category(raw["category"])]
    current = version == RUBRIC_PROTOCOL
    units = validate_decomposition(decomposition, source, taxonomy=groups if current else None)
    require_schema(PLAN_V2 if current else PLAN_SCHEMA, plan, "rubric plan")
    quotes = {uid: unit_quotes(unit, source) if current else [source[sid] for sid in unit["source_ids"]]
              for uid, unit in units.items()}
    issues = sample.issues + [row["issue_text"] for row in plan["added_issues"]]
    assignments = plan["assignments"]
    exclusions = plan["excluded_units"]
    classified = [row["unit_id"] for row in assignments + exclusions]
    unique(classified, "unit assignments/exclusions")
    if set(classified) != set(units):
        raise ValueError("Every atomic unit must be mapped exactly once or excluded with a reason")
    by_issue = {i: {} for i in range(1, len(issues) + 1)}
    mapped_ids = {i: set() for i in by_issue}

    def excerpts(ids, catalog, label):
        unique(ids, label)
        if not set(ids) <= catalog.keys():
            raise ValueError(f"Unknown {label}: {sorted(set(ids) - catalog.keys())}")
        return [catalog[key] for key in ids]

    for row in assignments:
        i, group, unit = row["issue_index"], row["group_id"], units[row["unit_id"]]
        if i not in by_issue:
            raise ValueError(f"Invalid assignment issue_index={i}")
        if group == "other":
            title = text_field(row["other_group_name"], "other_group_name")
        elif group in groups and row["other_group_name"] is None:
            title = groups[group]
        else:
            raise ValueError(f"Unknown or inconsistent group {group!r} for {raw['category']}")
        if current:
            adjustment = row["group_adjustment_reason"]
            if group != unit["group_id"]:
                text_field(adjustment, f"{row['unit_id']}: group_adjustment_reason")
            elif adjustment is not None:
                raise ValueError(f"{row['unit_id']}: group_adjustment_reason must be null when the group is unchanged")
        group_key = (group, title)
        bucket = by_issue[i].setdefault(group_key, [])
        mapped_ids[i].add(row["unit_id"])
        criterion = {
            "criterion_id": "pending", "unit_id": unit["unit_id"], "criterion": row["criterion"],
            "reference_reasoning": unit["reasoning"], "source_evidence": quotes[unit["unit_id"]],
            "relevant_facts": excerpts(row["relevant_fact_ids"], facts, "facts_clean IDs"),
            "rule_refs": row["rule_indices"], "scorable": not row["missing_information"], "needs_review": bool(row["missing_information"]),
            "review_reason": row["review_reason"], "missing_information": row["missing_information"], "decisive": row["decisive"],
        }
        if current:
            criterion.update(reasoning_type=unit["reasoning_type"], legal_condition=row["legal_condition"],
                             application_link=row["application_link"])
        bucket.append(criterion)

    added = []
    for i, row in enumerate(plan["added_issues"], len(sample.issues) + 1):
        if not row["unit_ids"] or set(row["unit_ids"]) != mapped_ids[i]:
            raise ValueError(f"Added issue {i} must be backed by exactly its assigned units")
        unique(row["unit_ids"], "added issue unit IDs")
        text_field(row["reason"], "added issue reason")
        proposal = {"issue_index": i, "original_issue_count": len(sample.issues), "new_issue": row["issue_text"],
                    "reason": row["reason"], "source_evidence": list(dict.fromkeys(
                        quote for uid in row["unit_ids"] for quote in quotes[uid]))}
        if current:
            proposal["why_existing_issues_do_not_cover"] = text_field(
                row["why_existing_issues_do_not_cover"], "why_existing_issues_do_not_cover")
            proposal["proposal_id"] = canonical_hash({"sample_sha256": sample.fingerprint, **proposal})
            proposal["status"] = "proposed_added_issue"
        added.append(proposal)

    rubric_issues = []
    if [row["issue_index"] for row in plan["issues"]] != list(by_issue):
        raise ValueError("Rubric plan must cover all issues in exact order")
    for i, outcome in enumerate(plan["issues"], 1):
        rubric_groups, number = [], 0
        other_number = 0
        for (group, title), criteria in by_issue[i].items():
            for criterion in criteria:
                number += 1
                criterion["criterion_id"] = f"I{i}_C{number}"
            if group == "other":
                other_number += 1
                group = f"other_{other_number}"
            rubric_groups.append({"group_id": group, "group_name": title, "criteria": criteria})
        review_reason = outcome["issue_review_reason"]
        if review_reason is None:
            decisive_reasons = [c["review_reason"] for group in rubric_groups for c in group["criteria"]
                                if c["needs_review"] and c["decisive"] and c["review_reason"]]
            if decisive_reasons:
                review_reason = " | ".join(dict.fromkeys(decisive_reasons))
            elif not rubric_groups:
                review_reason = "Không có lập luận từ nguồn được ánh xạ vào issue này."
        rubric_issues.append({
            "issue_index": i, "issue_text": issues[i-1], "groups": rubric_groups,
            "reference_verdict": outcome["reference_verdict"],
            "verdict_evidence": excerpts(resolve_verdict_ids(outcome["verdict_source_ids"], units, source),
                                         source, "verdict source IDs"),
            "outcome_review_reason": outcome["outcome_review_reason"], "issue_review_reason": review_reason,
        })
    dataset = {"input": {**raw["input"], "issues": issues}, "expected_answer": raw["expected_answer"],
               "doc_name": sample.doc_name, "category": raw["category"], "rubrics": {"version": version, "issues": rubric_issues}}
    if current:
        dataset["rubrics"]["status"] = "draft"
    validate_rubric_record(dataset)
    excluded = []
    for row in exclusions:
        text_field(row["exclusion_reason"], "exclusion_reason")
        excluded.append({**row, "source_evidence": quotes[row["unit_id"]]})
    audit = {"added_issues": added, "excluded_units": excluded,
             "non_reasoning": [{**row, "source_evidence": source[row["source_id"]]} for row in decomposition["non_reasoning"]],
             "source_units": [{**unit, "source_evidence": quotes[uid]} for uid, unit in units.items()],
             "mapping": [{"unit_id": c["unit_id"], "issue_index": issue["issue_index"], "criterion_id": c["criterion_id"]}
                         for issue in rubric_issues for c in criteria_for(issue)]}
    return dataset, audit


def criteria_for(issue: dict) -> list[dict]:
    return [criterion for group in issue["groups"] for criterion in group["criteria"]]


def issue_blocked(issue: dict) -> bool:
    return issue["issue_review_reason"] is not None or any(c["needs_review"] and c["decisive"] for c in criteria_for(issue))


def resolve_verdict_ids(ids: list[str], units: dict[str, dict], source: dict[str, str]) -> list[str]:
    """Resolve a known unit to the exact source spans already validated for that unit."""
    unique(ids, "verdict source IDs")
    resolved = []
    for identifier in ids:
        if identifier in source:
            resolved.append(identifier)
        elif identifier in units:
            resolved.extend(units[identifier]["source_ids"])
        else:
            raise ValueError(f"Unknown verdict source ID {identifier!r}; use a key from expected_answer_segments")
    return list(dict.fromkeys(resolved))


def validate_rubric_record(row: dict) -> None:
    require_schema(DATASET_SCHEMA, row, "Rule Application dataset")
    taxonomy = TAXONOMY[canonical_category(row["category"])]
    payload = row["input"]
    for field, value in [("doc_name", row["doc_name"]), ("facts_clean", payload["facts_clean"]), ("expected_answer", row["expected_answer"])]:
        text_field(value, field)
    unique([normalized(issue) for issue in payload["issues"]], "issue texts")
    for rule in payload["rules"]:
        for key, value in rule.items():
            if key != "citation_str" or value is not None:
                text_field(value, f"rule.{key}")
    issues = row["rubrics"]["issues"]
    current = row["rubrics"]["version"] == RUBRIC_PROTOCOL
    if current and row["rubrics"].get("status") not in {"draft", "verified"}:
        raise ValueError("Version 2 rubrics require draft or verified status")
    if [issue["issue_text"] for issue in issues] != payload["issues"]:
        raise ValueError("Rubric issues must match input issues exactly and in order")
    all_criteria = []
    for i, issue in enumerate(issues, 1):
        text_field(issue["issue_text"], "issue_text")
        if issue["issue_index"] != i:
            raise ValueError(f"Expected rubric issue_index={i}")
        unique([group["group_id"] for group in issue["groups"]], f"group IDs in issue {i}")
        for group in issue["groups"]:
            group_id = group["group_id"]
            if taxonomy.get(group_id) != group["group_name"] and not re.fullmatch(r"other_[1-9][0-9]*", group_id):
                raise ValueError(f"Invalid group {group_id} for {row['category']}")
            text_field(group["group_name"], "group_name")
        criteria = criteria_for(issue)
        for index, criterion in enumerate(criteria, 1):
            if criterion["criterion_id"] != f"I{i}_C{index}":
                raise ValueError("Criterion IDs must be unique and in issue order")
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
