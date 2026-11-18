from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import httpx
import pytest
from jsonschema import Draft202012Validator

from conftest import envelope
from legal_application.completeness import COMPLETENESS_SCHEMA, completeness_messages, parse_classification, validate_classification
from legal_application.data import parse_sample
from legal_application.evidence import EVIDENCE_FORMAT, SOURCES, evidence_catalog, evidence_schema
from legal_application.scoring import LabelError
from legal_application.storage import read_jsonl
from test_statistic import labels, reference_labels, run, source


@pytest.mark.parametrize("text", ["", " \n\t ", "Tòa tuyên: A trả B.\r\nGiữ nguyên dấu và khoảng trắng.\n",
                                  "A xác nhận nợ 123.456 đồng; B đồng ý.\n" * 120, "x" * 2300,
                                  "Không tự chuẩn hóa a\u0301 hoặc dấu “ngoặc”.\t" * 60])
def test_catalog_preserves_every_source_character_without_a_total_cap(text):
    raw = source()
    raw["benchmark_seed"].update(facts=text, application=text, conclusion=text)
    catalog = evidence_catalog(parse_sample(raw, 0))
    assert len(catalog) == len(set(catalog))
    for field in SOURCES:
        spans = [span for span in catalog.values() if span["source"] == field]
        assert "".join(span["text"] for span in spans) == raw["benchmark_seed"][field]
        offset = 0
        for span in spans:
            assert span["start"] == offset and span["start"] < span["end"]
            assert span["text"] == raw["benchmark_seed"][field][span["start"]:span["end"]]
            offset = span["end"]
        assert offset == len(raw["benchmark_seed"][field])
    Draft202012Validator.check_schema(evidence_schema(COMPLETENESS_SCHEMA, catalog))


def test_request_schema_requires_known_evidence_ids_and_keeps_full_sources():
    raw = source()
    sample = parse_sample(raw, 0)
    catalog = evidence_catalog(sample)
    wire_schema = evidence_schema(COMPLETENESS_SCHEMA, catalog)
    payload = json.loads(completeness_messages(sample)[1]["content"])
    assert payload["benchmark_seed"] == raw["benchmark_seed"]
    assert payload["evidence_catalog"]["facts_clean:000001"]["text"] == raw["benchmark_seed"]["facts_clean"]
    assert wire_schema["properties"]["evidence_format"]["enum"] == [EVIDENCE_FORMAT]
    fact = wire_schema["properties"]["issues_analysis"]["items"]["properties"]["decisive_facts"]["items"]["properties"]
    assert fact["facts_clean_evidence"]["items"]["enum"] == ["facts_clean:000001"]
    assert "quote" not in fact["reference_evidence"]["items"]["properties"]
    assert "quote_id" in fact["reference_evidence"]["items"]["properties"]


def test_evidence_ids_remove_copying_failure_and_preserve_raw_model_response(config_factory):
    raw = source()
    sample = parse_sample(raw, 0)
    invalid_legacy = labels(raw)
    invalid_legacy["issues_analysis"][0]["decisive_facts"][0]["facts_clean_evidence"] = ["B xác nhận đã nợ A số tiền trên."]
    with pytest.raises(LabelError, match="not a verbatim substring"):
        validate_classification(invalid_legacy, sample)
    wire = reference_labels(raw)
    content = json.dumps(wire, ensure_ascii=False)
    config = config_factory([raw], mode="statistic", judge_json_retries=0)
    calls = []
    result = run(config, lambda request: calls.append(request) or httpx.Response(200, json=envelope(content)))
    assert len(calls) == 1 and result["counts"]["processed"] == 1 and result["counts"]["failed"] == 0
    record = read_jsonl(config.run_dir / "classification_details.jsonl")[0]
    assert record["raw_judge_response"] == content
    assert record["metadata"]["evidence_format"] == EVIDENCE_FORMAT
    fact = record["issues_analysis"][0]["decisive_facts"][0]
    assert fact["facts_clean_evidence"] == [raw["benchmark_seed"]["facts_clean"]]
    assert fact["reference_evidence"] == [{"source": "application", "quote": raw["benchmark_seed"]["application"]}]
    assert all(item["kind"] == "source_reference" for item in record["metadata"]["normalizations"])
    assert run(replace(config, resume=True), lambda request: pytest.fail("Saved source IDs must replay exactly"))["counts"]["skipped"] == 1


@pytest.mark.parametrize("kind", ["unknown", "wrong_source", "wrong_reference_source", "unknown_protocol"])
def test_fabricated_ids_or_wrong_sources_remain_sample_failures(config_factory, kind):
    raw = source()
    wire = reference_labels(raw)
    fact = wire["issues_analysis"][0]["decisive_facts"][0]
    if kind == "unknown":
        fact["facts_clean_evidence"] = ["facts_clean:999999"]
    elif kind == "wrong_source":
        fact["facts_clean_evidence"] = ["application:000001"]
    elif kind == "wrong_reference_source":
        fact["reference_evidence"][0]["source"] = "conclusion"
    else:
        wire["evidence_format"] = "guess_sources"
    config = config_factory([raw], mode="statistic", judge_json_retries=0)
    result = run(config, lambda request: httpx.Response(200, json=envelope(json.dumps(wire))))
    assert result["counts"]["failed"] == 1 and result["counts"]["processed"] == 0
    assert read_jsonl(config.run_dir / "classification_details.jsonl") == []
    error = read_jsonl(config.run_dir / "errors.jsonl")[0]
    assert "last_error=" in error["error"]
    assert "not strict JSON" not in error["error"]


def test_evidence_ids_preserve_partial_decision_and_missing_fact_details(config_factory):
    raw = source(partial=True)
    wire = reference_labels(raw, labels(raw, missing_issue=2))
    config = config_factory([raw], mode="statistic")
    result = run(config, lambda request: httpx.Response(200, json=envelope(json.dumps(wire))))
    assert result["counts"]["partial"] == 1 and result["counts"]["full"] == 0
    record = read_jsonl(config.run_dir / "classification_details.jsonl")[0]
    assert record["issues_missing_facts"] == [{"issue_index": 2, "original_issue": raw["issues"][1]}]
    assert record["decisive_missing_facts"][0]["facts_clean_evidence"] == []


def test_repair_keeps_id_response_with_specific_source_error(config_factory):
    raw = source()
    correct = reference_labels(raw)
    invalid = deepcopy(correct)
    invalid["issues_analysis"][0]["decisive_facts"][0]["reference_evidence"][0]["source"] = "conclusion"
    config = config_factory([raw], mode="statistic")
    calls = []

    def judge(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json=envelope(json.dumps(invalid if len(calls) == 1 else correct)))

    assert run(config, judge)["counts"]["processed"] == 1
    previous = json.loads(calls[1]["messages"][2]["content"])
    assert previous == invalid
    assert "belongs to application, not conclusion" in calls[1]["messages"][-1]["content"]


def test_old_quote_records_still_import_into_new_id_pipeline(config_factory):
    raw = source()
    config = config_factory([raw], mode="statistic")
    canonical = json.dumps(labels(raw), ensure_ascii=False)
    run(config, lambda request: httpx.Response(200, json=envelope(canonical)))
    old_details = config.run_dir / "classification_details.jsonl"
    retry = replace(config, run_id="import-legacy-quotes", statistic_seed_jsonl=old_details)
    result = run(retry, lambda request: pytest.fail("Valid old quote labels must be imported"))
    assert result["counts"]["skipped"] == 1
    assert read_jsonl(retry.run_dir / "classification_details.jsonl")[0]["raw_judge_response"] == canonical


def test_source_id_decoding_never_changes_the_classification_fields():
    raw = source()
    wire = reference_labels(raw, labels(raw, missing_issue=2))
    canonical, _ = parse_classification(json.dumps(wire), parse_sample(raw, 0))
    assert canonical == labels(raw, missing_issue=2)
