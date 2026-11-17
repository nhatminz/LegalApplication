from __future__ import annotations

import asyncio
import json
import logging
from importlib.resources import files

from jsonschema import Draft202012Validator

from .api import APIClient, APIError, IncompleteResponse
from .data import Sample, strict_json
from .prompts import prompt_text
from .scoring import LabelError
from .storage import now

log = logging.getLogger(__name__)
COMPLETENESS_SCHEMA = json.loads(files("legal_application").joinpath("schemas", "completeness.schema.json").read_text())
VALIDATOR = Draft202012Validator(COMPLETENESS_SCHEMA)


def completeness_messages(sample: Sample, repair: str | None = None) -> list[dict]:
    system = prompt_text("completeness_system") + "\nJSON SCHEMA:\n" + json.dumps(COMPLETENESS_SCHEMA, ensure_ascii=False)
    if repair:
        system += "\nLần trả trước không hợp lệ: " + repair + ". Trả lại đầy đủ JSON và giữ đúng trích dẫn nguồn."
    payload = {"doc_name": sample.doc_name, "issues": sample.issues,
               "benchmark_seed": {key: sample.raw["benchmark_seed"][key] for key in ["facts_clean", "facts", "application", "conclusion"]}}
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def validate_classification(labels: dict, sample: Sample) -> None:
    errors = sorted(VALIDATOR.iter_errors(labels), key=lambda e: str(list(e.path)))
    if errors:
        error = errors[0]
        raise LabelError(f"classification schema violation at {list(error.path)} ({error.validator})")
    if labels["doc_name"] != sample.doc_name:
        raise LabelError("classification doc_name mismatch")
    if len(labels["issues_analysis"]) != len(sample.issues):
        raise LabelError("classification must analyze every original issue")
    if not labels["reason"].strip():
        raise LabelError("classification reason is empty")
    missing = False
    seed = sample.raw["benchmark_seed"]

    def quote(text, source):
        if not text.strip() or text not in seed[source]:
            raise LabelError(f"evidence quote is not a verbatim substring of benchmark_seed.{source}")

    for index, (row, original) in enumerate(zip(labels["issues_analysis"], sample.issues), 1):
        if row["issue_index"] != index or row["original_issue"] != original:
            raise LabelError(f"classification issue {index}: index/order/original_issue mismatch")
        if not row["explanation"].strip():
            raise LabelError(f"classification issue {index}: empty explanation")
        for fact in row["decisive_facts"]:
            if not fact["fact"].strip() or not fact["explanation"].strip():
                raise LabelError("fact/explanation must be nonempty")
            for evidence in fact["reference_evidence"]:
                quote(evidence["quote"], evidence["source"])
            for evidence in fact["facts_clean_evidence"]:
                quote(evidence, "facts_clean")
            for evidence in fact["raw_facts_evidence"]:
                quote(evidence, "facts")
            if fact["availability"] == "missing":
                if not fact["decisive"] or fact["facts_clean_evidence"]:
                    raise LabelError("missing facts must be decisive with no claimed facts_clean evidence")
                missing = True
            elif not fact["facts_clean_evidence"]:
                raise LabelError("present/inferred facts require facts_clean supporting quotes")
    if labels["classification_status"] == "unclassifiable":
        if labels["data_status"] is not None:
            raise LabelError("unclassifiable must have null data_status")
    elif labels["data_status"] != ("partial" if missing else "full"):
        raise LabelError("data_status inconsistent with per-issue decisive missing facts")


async def classify_sample(config, sample: Sample, client: APIClient) -> dict:
    repair, calls, errors = None, [], []
    for attempt in range(config.judge_json_retries + 1):
        try:
            completion = await client.complete(completeness_messages(sample, repair), doc_name=sample.doc_name, schema=COMPLETENESS_SCHEMA)
            calls.append(completion.metadata)
            try:
                labels = strict_json(completion.content)
            except ValueError:
                raise LabelError("classification response is not strict JSON") from None
            validate_classification(labels, sample)
            if labels["classification_status"] == "unclassifiable":
                raise APIError("Judge could not classify reliably: " + labels["reason"], {"judge_calls": calls})
            missing_issues, missing_facts = [], []
            for row in labels["issues_analysis"]:
                facts = [fact for fact in row["decisive_facts"] if fact["availability"] == "missing"]
                if facts:
                    missing_issues.append({"issue_index": row["issue_index"], "original_issue": row["original_issue"]})
                    missing_facts.extend({"issue_index": row["issue_index"], **fact} for fact in facts)
            return {**labels, **sample.identity(), "issues_missing_facts": missing_issues,
                    "decisive_missing_facts": missing_facts, "judge_model": config.judge_model,
                    "raw_judge_response": completion.content,
                    "metadata": {"completed_at": now(), "judge_calls": calls, "validation_errors": errors, "label_attempts": attempt + 1}}
        except (LabelError, IncompleteResponse) as exc:
            if isinstance(exc, IncompleteResponse):
                calls.append(exc.metadata)
            repair = str(exc)
            errors.append(repair)
            log.warning("Classification invalid doc_name=%s attempt=%d reason=%s", sample.doc_name, attempt + 1, repair)
            if attempt == config.judge_json_retries:
                raise APIError("classification exhausted structured-output retries", {"judge_calls": calls, "validation_errors": errors}) from None
            if config.retry_backoff:
                await asyncio.sleep(min(60.0, config.retry_backoff * 2 ** attempt))
    raise AssertionError("unreachable")
