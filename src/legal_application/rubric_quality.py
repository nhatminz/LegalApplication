"""Select rubric candidates and keep QA feedback out of the published rubric."""
from __future__ import annotations

import re

from .rubrics import load_schema

QUALITY_SCHEMA = load_schema("rubric_quality")
MAX_REVISIONS = 2

# Catch explicit Vietnamese citations even if a quality response misses them.
CITATION = re.compile(
    r"\b(?:điều\s*(?:số\s*)?\d+|(?:theo|tại)\s+(?:khoản\s*\d+\b|điểm\s+[a-zđ]\b))",
    re.IGNORECASE,
)


def check_review(review, candidate):
    gold, added = candidate["issues"], candidate["added_issues"]
    added_ids = {item["issue_index"] for item in added}
    reviewed = [item["issue_index"] for item in review["added_issue_reviews"]]
    if len(reviewed) != len(added_ids) or set(reviewed) != added_ids:
        raise ValueError("Quality Judge must review each added issue exactly once")
    criteria_counts = {issue["issue_index"]: len(issue["criteria"]) for issue in gold + added}
    for problem in review["rubric_problems"]:
        index = problem["issue_index"]
        if index not in criteria_counts or (problem.get("criterion_index") or 0) > criteria_counts[index]:
            raise ValueError("Quality Judge returned an invalid issue or criterion index")
    return review


def citation_findings(candidate):
    return [{"issue_index": issue["issue_index"], "severity": "major",
             "problem": "Issue bổ sung chứa viện dẫn điều luật cụ thể.",
             "suggestion": "Viết câu hỏi trung lập và chuyển viện dẫn sang phần lập luận."}
            for issue in candidate["added_issues"] if CITATION.search(issue["issue_text"])]


def blocked_additions(review, findings):
    blocked = {item["issue_index"] for item in findings}
    if review is None:
        return blocked
    blocked.update(item["issue_index"] for item in review["added_issue_reviews"] if item["decision"] != "keep")
    return blocked


def needs_revision(review, findings):
    return bool(blocked_additions(review, findings)) or any(p["severity"] == "major" for p in review["rubric_problems"])


def candidate_rank(round_info):
    review = round_info["assessment"]
    if review is None:
        return (False, 0, False, 0, round_info["round"])
    major_count = sum(p["severity"] == "major" for p in review["rubric_problems"])
    major_count += len(blocked_additions(review, round_info["citation_findings"]))
    return (True, -major_count, review["approved"], sum(review["rubric_scores"].values()), round_info["round"])
