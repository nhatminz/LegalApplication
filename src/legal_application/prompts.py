from __future__ import annotations

import json
from importlib.resources import files

from .data import Sample


def prompt_text(name: str) -> str:
    return files("legal_application").joinpath("prompts", name + ".txt").read_text(encoding="utf-8")


def generator_messages(sample: Sample) -> list[dict]:
    # Explicit allowlist. Do not pass raw record/law metadata, explains, or benchmark_seed.
    parts = ["FACTS", sample.raw["benchmark_seed"]["facts_clean"], "LEGAL ISSUES"]
    parts.extend(f"{i}. {issue}" for i, issue in enumerate(sample.issues, 1))
    parts.append("APPLICABLE LAWS")
    for block in sample.law_blocks:
        parts.append(f"Tên luật: {block['law_str']}")
        if block["citation_str"] is not None:
            parts.append(f"Điều luật: {block['citation_str']}")
        parts.append(f"Nguồn nội dung: {block['source']}" + (" (fallback cấp luật)" if block["fallback"] else ""))
        parts.append(block["content"])
    return [{"role": "system", "content": prompt_text("generator_system")},
            {"role": "user", "content": "\n\n".join(parts)}]


def judge_messages(sample: Sample, response: str, schema: dict, repair: str | None = None) -> list[dict]:
    payload = {"doc_name": sample.doc_name, "benchmark_seed": sample.raw["benchmark_seed"],
               "issues": sample.issues, "laws": sample.raw["laws"], "evaluated_model_response": response}
    system = prompt_text("legacy_judge_system") + "\nJSON SCHEMA:\n" + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    if repair:
        # Regenerate with the same full evidence. Do not grow context by appending invalid drafts.
        system += "\nLần trả trước không hợp lệ: " + repair + ". Hãy trả lại đủ JSON hợp lệ theo schema."
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)}]



def rubric_generator_messages(sample: Sample) -> list[dict]:
    payload = sample.raw["input"]
    parts = ["FACTS", payload["facts_clean"], "LEGAL ISSUES"]
    parts.extend(f"{i}. {issue}" for i, issue in enumerate(payload["issues"], 1))
    parts.append("APPLICABLE LAWS")
    for rule in payload["rules"]:
        parts.append(f"Tên luật: {rule['law_str']}")
        if rule["citation_str"] is not None:
            parts.append(f"Điều luật: {rule['citation_str']}")
        parts.append(rule["context_text"])
    return [{"role": "system", "content": prompt_text("generator_system")},
            {"role": "user", "content": "\n\n".join(parts)}]


def rubric_judge_messages(sample: Sample, response: str, schema: dict, repair: str | None = None) -> list[dict]:
    from .scoring import evaluation_plan

    raw = sample.raw
    payload = {"doc_name": sample.doc_name, "category": raw["category"], **raw["input"],
               "rubrics": raw["rubrics"], "evaluated_model_response": response,
               "evaluation_plan": evaluation_plan(sample)}
    system = prompt_text("judge_system") + "\nJSON SCHEMA:\n" + json.dumps(schema, ensure_ascii=False)
    if repair:
        system += "\nSửa lỗi validation sau và trả lại JSON đầy đủ: " + repair
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)}]
