from __future__ import annotations

import asyncio
import json
import logging
import re
from copy import deepcopy
from importlib.resources import files

from jsonschema import Draft202012Validator

from .api import APIClient, APIError, IncompleteResponse
from .data import Sample, strict_json
from .evidence import EVIDENCE_FORMAT, evidence_catalog, evidence_schema, materialize_evidence
from .prompts import prompt_text
from .scoring import LabelError
from .storage import now

log = logging.getLogger(__name__)
COMPLETENESS_SCHEMA = json.loads(files("legal_application").joinpath("schemas", "completeness.schema.json").read_text())
VALIDATOR = Draft202012Validator(COMPLETENESS_SCHEMA)


def completeness_messages(sample: Sample, repair: str | None = None, previous: dict | None = None, *, catalog: dict | None = None) -> list[dict]:
    catalog = evidence_catalog(sample) if catalog is None else catalog
    system = prompt_text("completeness_system") + "\nJSON SCHEMA:\n" + json.dumps(evidence_schema(COMPLETENESS_SCHEMA, catalog), ensure_ascii=False)
    if repair:
        system += "\nLần trả trước không hợp lệ: " + repair + ". Trả lại đầy đủ JSON, dùng đúng evidence ID trong catalog."
    payload = {"doc_name": sample.doc_name, "issues": sample.issues,
               "benchmark_seed": {key: sample.raw["benchmark_seed"][key] for key in ["facts_clean", "facts", "application", "conclusion"]},
               "evidence_catalog": {key: {"source": span["source"], "text": span["text"]} for key, span in catalog.items()}}
    messages = [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    if repair and previous is not None:
        messages.extend([{"role": "assistant", "content": json.dumps(previous, ensure_ascii=False)},
                         {"role": "user", "content": "Sửa lỗi kiểm tra: " + repair +
                          ". Trả lại TOÀN BỘ JSON, đủ mọi issue, chỉ dùng evidence ID từ đúng nguồn. Giữ giải thích ngắn; không sửa dữ kiện để vượt validation."}])
    return messages


def decode_classification(content: str):
    normalizations = []
    fenced = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", content, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        content = fenced[1]
        normalizations.append({"path": "$", "kind": "json_code_fence"})
    labels = strict_json(content)
    return labels, normalizations


def parse_classification(content: str, sample: Sample, *, catalog: dict | None = None) -> tuple[dict, list[dict]]:
    """Resolve source IDs or align formatting in legacy quotes. Never infer labels."""
    labels, normalizations = decode_classification(content)
    if not isinstance(labels, dict):
        return labels, normalizations
    labels = deepcopy(labels)
    if "evidence_format" in labels:
        if labels["evidence_format"] != EVIDENCE_FORMAT:
            raise LabelError("Unsupported classification evidence_format")
        catalog = evidence_catalog(sample) if catalog is None else catalog
        validator = Draft202012Validator(evidence_schema(COMPLETENESS_SCHEMA, catalog))
        error = next(validator.iter_errors(labels), None)
        if error is not None:
            raise LabelError(f"classification evidence schema violation at {list(error.path)} ({error.validator}): {error.message[:400]}")
        labels, references = materialize_evidence(labels, catalog)
        normalizations.extend(references)
    seed = sample.raw["benchmark_seed"]
    # Keep offsets so accepted quotes can be replaced by the exact original source bytes/text.
    sources = {}

    def align(text, source, path):
        if not isinstance(text, str) or not text.strip() or text in source:
            return text
        if source not in sources:
            chars, spans = [], []
            for match in re.finditer(r"\s+|\S+", source):
                token = match[0]
                if token.isspace():
                    chars.append(" ")
                    spans.append((match.start(), match.end()))
                else:
                    chars.extend(token)
                    spans.extend((offset, offset + 1) for offset in range(match.start(), match.end()))
            sources[source] = ("".join(chars), spans)
        normalized, spans = sources[source]
        query = re.sub(r"\s+", " ", text).strip()
        start = normalized.find(query)
        if start < 0:
            return text  # Numbers, words, punctuation and meaning must still match exactly.
        exact = source[spans[start][0]:spans[start + len(query) - 1][1]]
        normalizations.append({"path": path, "kind": "source_whitespace", "original": text, "normalized": exact})
        return exact

    rows = labels.get("issues_analysis")
    if isinstance(rows, list):
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            prefix = f"issues_analysis[{i}]"
            if i < len(sample.issues) and isinstance(row.get("original_issue"), str):
                original = sample.issues[i]
                if re.sub(r"\s+", " ", row["original_issue"]).strip() == re.sub(r"\s+", " ", original).strip():
                    if row["original_issue"] != original:
                        normalizations.append({"path": prefix + ".original_issue", "kind": "issue_whitespace"})
                        row["original_issue"] = original
            facts = row.get("decisive_facts")
            for j, fact in enumerate(facts if isinstance(facts, list) else []):
                if not isinstance(fact, dict):
                    continue
                path = f"{prefix}.decisive_facts[{j}]"
                refs = fact.get("reference_evidence")
                for k, evidence in enumerate(refs if isinstance(refs, list) else []):
                    if isinstance(evidence, dict) and isinstance(evidence.get("source"), str) and evidence["source"] in {"application", "conclusion"}:
                        evidence["quote"] = align(evidence.get("quote"), seed[evidence["source"]], f"{path}.reference_evidence[{k}].quote")
                for field, source in [("facts_clean_evidence", "facts_clean"), ("raw_facts_evidence", "facts")]:
                    values = fact.get(field)
                    if isinstance(values, list):
                        fact[field] = [align(value, seed[source], f"{path}.{field}[{k}]") for k, value in enumerate(values)]
    return labels, normalizations


def validate_classification(labels: dict, sample: Sample) -> None:
    errors = sorted(VALIDATOR.iter_errors(labels), key=lambda e: str(list(e.path)))
    if errors:
        error = errors[0]
        raise LabelError(f"classification schema violation at {list(error.path)} ({error.validator}): {error.message[:400]}")
    if labels["doc_name"] != sample.doc_name:
        raise LabelError("classification doc_name mismatch")
    if len(labels["issues_analysis"]) != len(sample.issues):
        raise LabelError(f"classification must analyze every original issue: expected {len(sample.issues)}, received {len(labels['issues_analysis'])}")
    if not labels["reason"].strip():
        raise LabelError("classification reason is empty")
    missing = False
    seed = sample.raw["benchmark_seed"]

    def quote(text, source, path):
        if not text.strip() or text not in seed[source]:
            raise LabelError(f"{path}: evidence quote is not a verbatim substring of benchmark_seed.{source}; copy a short continuous source excerpt, without ellipses or paraphrasing")

    for index, (row, original) in enumerate(zip(labels["issues_analysis"], sample.issues), 1):
        if row["issue_index"] != index or row["original_issue"] != original:
            raise LabelError(f"classification issue {index}: index/order/original_issue mismatch")
        if not row["explanation"].strip():
            raise LabelError(f"classification issue {index}: empty explanation")
        for fact_index, fact in enumerate(row["decisive_facts"]):
            prefix = f"issues_analysis[{index - 1}].decisive_facts[{fact_index}]"
            if not fact["fact"].strip() or not fact["explanation"].strip():
                raise LabelError("fact/explanation must be nonempty")
            for i, evidence in enumerate(fact["reference_evidence"]):
                quote(evidence["quote"], evidence["source"], f"{prefix}.reference_evidence[{i}].quote")
            for i, evidence in enumerate(fact["facts_clean_evidence"]):
                quote(evidence, "facts_clean", f"{prefix}.facts_clean_evidence[{i}]")
            for i, evidence in enumerate(fact["raw_facts_evidence"]):
                quote(evidence, "facts", f"{prefix}.raw_facts_evidence[{i}]")
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
    repair, previous, calls, errors, invalid_responses = None, None, [], [], []
    catalog = evidence_catalog(sample)
    schema = evidence_schema(COMPLETENESS_SCHEMA, catalog)
    for attempt in range(config.judge_json_retries + 1):
        completion = None
        try:
            completion = await client.complete(completeness_messages(sample, repair, previous, catalog=catalog), doc_name=sample.doc_name, schema=schema)
            calls.append(completion.metadata)
            try:
                value, _ = decode_classification(completion.content)
                previous = value if isinstance(value, dict) else None
                labels, normalizations = parse_classification(completion.content, sample, catalog=catalog)
            except LabelError:
                raise
            except ValueError as exc:
                previous = None
                raise LabelError(f"classification response is not strict JSON: {str(exc)[:300]}") from None
            validate_classification(labels, sample)
            if labels["classification_status"] == "unclassifiable":
                raise APIError("Judge could not classify reliably: " + labels["reason"], {"judge_calls": calls})
            missing_issues, missing_facts = [], []
            for row in labels["issues_analysis"]:
                facts = [fact for fact in row["decisive_facts"] if fact["availability"] == "missing"]
                if facts:
                    missing_issues.append({"issue_index": row["issue_index"], "original_issue": row["original_issue"]})
                    missing_facts.extend({"issue_index": row["issue_index"], **fact} for fact in facts)
            return {**labels, **sample.identity(), **({"document_id": sample.raw["document_id"]} if "document_id" in sample.raw else {}), "issues_missing_facts": missing_issues,
                    "decisive_missing_facts": missing_facts, "judge_model": config.judge_model,
                    "raw_judge_response": completion.content,
                    "metadata": {"completed_at": now(), "judge_calls": calls, "validation_errors": errors,
                                 "normalizations": normalizations,
                                 "evidence_format": value.get("evidence_format", "legacy_quotes"), "label_attempts": attempt + 1}}
        except (LabelError, IncompleteResponse) as exc:
            if isinstance(exc, IncompleteResponse):
                calls.append(exc.metadata)
            repair = str(exc)
            errors.append(repair)
            if completion is not None:
                invalid_responses.append({"attempt": attempt + 1, "content_preview": completion.content[:4000],
                                          "content_characters": len(completion.content), "error": repair})
            log.warning("Classification invalid doc_name=%s attempt=%d reason=%s", sample.doc_name, attempt + 1, repair)
            if attempt == config.judge_json_retries:
                raise APIError("classification exhausted structured-output retries; last_error=" + repair,
                               {"judge_calls": calls, "validation_errors": errors, "last_validation_error": repair,
                                "invalid_responses": invalid_responses}) from None
            if config.retry_backoff:
                await asyncio.sleep(min(60.0, config.retry_backoff * 2 ** attempt))
    raise AssertionError("unreachable")
