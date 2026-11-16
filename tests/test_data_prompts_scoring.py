from __future__ import annotations

import json
from copy import deepcopy

import pytest

from legal_application.data import DatasetError, parse_sample, read_dataset, strict_json
from legal_application.prompts import generator_messages, judge_messages
from legal_application.scoring import LABEL_SCHEMA, LabelError, score_labels
from conftest import make_labels, make_record


def test_schema_and_law_sources_preserve_full_content():
    raw = make_record()
    original = deepcopy(raw)
    sample = parse_sample(raw, 4)
    assert sample.identity()["input_line"] == 5
    assert len(sample.law_blocks) == 3
    assert sample.law_blocks[0]["content"] == raw["laws"][0]["provisions"][0]["law_content"]["context_text"]
    assert sample.law_blocks[-1]["source"] == "laws[1].law_content.exact_text"
    assert sample.law_blocks[-1]["fallback"] is True
    assert raw == original


@pytest.mark.parametrize("provisions", [None, [], "absent"])
def test_law_fallback_prefers_context(provisions):
    raw = make_record()
    law = raw["laws"][1]
    if provisions == "absent":
        law.pop("provisions")
    else:
        law["provisions"] = provisions
    law["law_content"]["context_text"] = "CONTEXT_WINS"
    assert parse_sample(raw, 0).law_blocks[-1]["content"] == "CONTEXT_WINS"


def test_generator_allowlist_no_reference_leak_and_full_order():
    sample = parse_sample(make_record(), 0)
    messages = generator_messages(sample)
    prompt = "\n".join(x["content"] for x in messages)
    for secret in ["PRIVATE_APPLICATION_SENTINEL", "PRIVATE_CONCLUSION_SENTINEL", "RAW_FACTS_SENTINEL", "PRIVATE_EXPLAIN_MUST_NOT_LEAK", "PRIVATE_MARKDOWN_MUST_NOT_LEAK", "LAW_METADATA_MUST_NOT_LEAK", "EXACT_NOT_USED"]:
        assert secret not in prompt
    user = messages[1]["content"]
    assert user.index("FACTS") < user.index("LEGAL ISSUES") < user.index("APPLICABLE LAWS")
    positions = [user.index(s) for s in sample.issues]
    assert positions == sorted(positions)
    positions = [user.index(b["content"]) for b in sample.law_blocks]
    assert positions == sorted(positions)
    assert all(b["content"] in user for b in sample.law_blocks)
    system = messages[0]["content"]
    assert "TẤT CẢ issues, đúng thứ tự gốc" in system
    assert "Phán quyết đối với Issue 1" in system
    assert "Không thêm phần mở đầu hoặc mục Kết luận chung" in system
    assert "Không trả JSON" in system


def test_judge_gets_entire_reference_and_raw_answer():
    sample = parse_sample(make_record(), 0)
    raw_response = " \nISSUE 1: nguyên văn\n "
    messages = judge_messages(sample, raw_response, LABEL_SCHEMA)
    payload = json.loads(messages[1]["content"])
    assert payload["benchmark_seed"] == sample.raw["benchmark_seed"]
    assert payload["issues"] == sample.issues
    assert payload["laws"] == sample.raw["laws"]
    assert payload["evaluated_model_response"] == raw_response
    assert "tổng hợp tập hợp phán quyết riêng theo issues" in messages[0]["content"]
    assert "Tuyệt đối không trừ điểm vì thiếu conclusion chung" in messages[0]["content"]
    assert "không đòi model giải quyết nội dung ngoài issues" in messages[0]["content"]


@pytest.mark.parametrize("covered,correct,order,expected", [
    ([True, True], [True, True], True, (5, 2, 3, 10)),
    ([True, True], [False, False], False, (0, 2, 0, 2)),
    ([True, False], [True, False], False, (0, 1, 1.5, 2.5)),
    ([False, False], [False, False], False, (0, 0, 0, 0)),
    ([True, True], [True, False], False, (0, 2, 1.5, 3.5)),
])
def test_exact_523_formula(covered, correct, order, expected):
    sample = parse_sample(make_record(), 0)
    labels = make_labels(sample.raw, covered=covered, correct=correct, order=order)
    assert tuple(score_labels(labels, sample).values()) == expected


def test_review_null_not_zero():
    sample = parse_sample(make_record(), 0)
    assert all(x is None for x in score_labels(make_labels(sample.raw, review=True), sample).values())


@pytest.mark.parametrize("change", [
    lambda x: x.update(doc_name="wrong"),
    lambda x: x["issues_evaluation"].pop(),
    lambda x: x["issues_evaluation"].reverse(),
    lambda x: x["issues_evaluation"][0].update(original_issue="changed"),
    lambda x: x["issues_evaluation"][0].update(covered=False, correct=True),
    lambda x: x["issues_evaluation"][0].update(correct=None),
    lambda x: x["issues_evaluation"][0].update(needs_review=True),
    lambda x: x.update(evaluation_status="needs_review"),
    lambda x: x.update(final_order_correct=1),
    lambda x: x.update(score_components={"total": 10}),
])
def test_labels_reject_inconsistent_or_unmapped_output(change):
    sample = parse_sample(make_record(), 0)
    labels = make_labels(sample.raw)
    change(labels)
    with pytest.raises(LabelError):
        score_labels(labels, sample)


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("doc_name"),
    lambda r: r["benchmark_seed"].pop("facts_clean"),
    lambda r: r["benchmark_seed"].pop("application"),
    lambda r: r.update(issues=[]),
    lambda r: r.update(issues=[{"issue": "unexpected schema"}]),
    lambda r: r.update(laws=[]),
    lambda r: r["laws"][0]["provisions"][0]["law_content"].pop("context_text"),
    lambda r: r["laws"][1].pop("law_content"),
    lambda r: r["laws"][1]["law_content"].update(context_text=""),
])
def test_missing_or_wrong_fields_are_errors(mutate):
    raw = make_record()
    mutate(raw)
    with pytest.raises(ValueError):
        parse_sample(raw, 0)


def test_malformed_beyond_limit_is_reported(config_factory):
    config = config_factory()
    with config.input_jsonl.open("a") as output:
        output.write('{"doc_name": "truncated')
    with pytest.raises(DatasetError) as caught:
        read_dataset(config.input_jsonl, 1)
    assert caught.value.input_count == 2
    assert caught.value.valid_count == 1
    assert caught.value.errors[0]["input_line"] == 2


@pytest.mark.parametrize("bad", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_strict_json(bad):
    with pytest.raises(ValueError):
        strict_json(bad)


def test_duplicate_doc_names_rejected(config_factory):
    config = config_factory([make_record(), make_record()])
    with pytest.raises(DatasetError, match="Input validation failed"):
        read_dataset(config.input_jsonl, None)


@pytest.mark.parametrize("limit,count", [(None, 1003), (1001, 1001), (2, 2), (0, 0), (2000, 1003)])
def test_limit_has_no_hidden_cap(config_factory, limit, count):
    config = config_factory([make_record(f"case-{i}", n=1) for i in range(1003)])
    dataset = read_dataset(config.input_jsonl, limit)
    assert dataset.input_count == 1003
    assert len(dataset.samples) == count
