from __future__ import annotations

import asyncio
import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from conftest import envelope, make_record
from legal_application.config import Config
from legal_application.data import DatasetError
from legal_application.pipeline import Pipeline
from legal_application.rubric_builder import RubricPipeline, stage_messages
from legal_application.rubric_taxonomy import TAXONOMY, canonical_category
from legal_application.rubrics import DECOMPOSITION_SCHEMA, materialize, raw_rubric_sample, rubric_sample, source_segments
from legal_application.scoring import LabelError, evaluation_plan, score_rubric_labels
from legal_application.storage import read_jsonl

ANSWER = "B ký nhận nợ. Do đó B phải trả 100 đồng."
WORDS = "Việc B xác nhận khoản nợ làm phát sinh nghĩa vụ hoàn trả đủ một trăm đồng."


def raw_record(doc="case-1", category="Civil"):
    raw = make_record(doc, n=1)
    raw.update(category=category, instance_level="First-instance", code="DS-ST")
    raw["benchmark_seed"]["application"] = ANSWER
    return raw


def artifacts(raw, *, add=False, missing=False):
    sample = raw_rubric_sample(raw, 0)
    group = list(TAXONOMY[canonical_category(raw["category"])])[0]
    source = source_segments(sample.raw["expected_answer"], "S")
    unit_rows = [
        {"unit_id": "U1", "reasoning": "B chịu nghĩa vụ trả nợ", "source_ids": list(source)},
        {"unit_id": "U2", "reasoning": "Thông báo về án phí thường lệ", "source_ids": list(source)},
    ]
    assignment = {"unit_id": "U1", "issue_index": 1, "group_id": group, "other_group_name": None,
                  "criterion": "Nối tình tiết nhận nợ với nghĩa vụ trả nợ",
                  "relevant_fact_ids": ["F1"] if not missing else [], "rule_indices": [1],
                  "missing_information": ["Giấy xác nhận nợ"] if missing else [],
                  "review_reason": "facts_clean chưa thể hiện giấy xác nhận nợ" if missing else None,
                  "decisive": missing}
    outcome = {"issue_index": 1, "reference_verdict": "B phải trả 100 đồng", "verdict_source_ids": ["S1"],
               "outcome_review_reason": None, "issue_review_reason": "Thiếu tình tiết quyết định" if missing else None}
    plan = {"added_issues": [], "assignments": [assignment], "issues": [outcome],
            "excluded_units": [{"unit_id": "U2", "exclusion_reason": "Chỉ là án phí thường lệ, không làm đổi kết quả"}]}
    if add:
        unit_rows.append({"unit_id": "U3", "reasoning": "Hệ quả của một yêu cầu độc lập", "source_ids": ["S1"]})
        plan["added_issues"].append({"issue_text": "Có chấp nhận yêu cầu độc lập?", "unit_ids": ["U3"],
                                     "reason": "Gold issue chỉ nói về nghĩa vụ trả nợ"})
        plan["assignments"].append({**assignment, "unit_id": "U3", "issue_index": 2, "criterion": "Giải quyết yêu cầu độc lập"})
        plan["issues"].append({**outcome, "issue_index": 2})
    return sample, {"units": unit_rows, "non_reasoning": []}, plan


def run_build(config, handler):
    return asyncio.run(RubricPipeline(config, rubric_transport=httpx.MockTransport(handler)).run())


def run_eval(config, generator, judge):
    return asyncio.run(Pipeline(config, generator_transport=httpx.MockTransport(generator),
                                judge_transport=httpx.MockTransport(judge)).run())


def judge_result(row, score=2, outcome=True, evidence=WORDS):
    sample = rubric_sample(row, 0)
    plan = evaluation_plan(sample)
    return {"doc_name": row["doc_name"], "category": row["category"], "needs_review": plan["needs_review"],
            "evaluation_status": "needs_review" if plan["needs_review"] else "completed",
            "issues_evaluation": [{"issue_index": issue["issue_index"], "evaluation_status": issue["evaluation_status"],
                                   "criterion_scores": [{"criterion_id": cid, "score": score,
                                                         "evidence_from_model_answer": [evidence] if score else [],
                                                         "explanation": "Đánh giá theo ý nghĩa pháp lý."}
                                                        for cid in issue["criterion_ids"]],
                                   "outcome_correct": outcome if issue["outcome_scorable"] else None,
                                   "outcome_explanation": "Kết quả và lý do đủ điều kiện theo rubric."}
                                  for issue in plan["issues"]]}


@pytest.mark.parametrize("category", list(TAXONOMY))
def test_taxonomy_and_exact_raw_projection(category):
    raw = raw_record(category=category)
    raw["benchmark_seed"]["facts_clean"] += "\n giữ khoảng trắng "
    sample, decomposition, plan = artifacts(raw)
    record, audit = materialize(sample, decomposition, plan)
    assert record["category"] == raw["category"]
    assert record["input"]["issues"] == raw["issues"]
    assert record["input"]["facts_clean"] == raw["benchmark_seed"]["facts_clean"]
    assert record["expected_answer"] == ANSWER
    assert record["input"]["rules"][-1] == {"law_str": "Văn bản toàn văn", "citation_str": None,
                                           "context_text": "Toàn bộ văn bản fallback."}
    assert record["rubrics"]["issues"][0]["groups"][0]["group_name"] in TAXONOMY[category].values()
    assert len(audit["mapping"]) == 1 and audit["excluded_units"][0]["source_evidence"] == [ANSWER]


@pytest.mark.parametrize("change", ["category", "unaccounted", "unmapped", "twice", "wrong_group", "invented_evidence", "unnecessary_added", "wrong_order", "unsupported_rule"])
def test_rubric_validation_rejects_inconsistent_source_and_mapping(change):
    raw = raw_record()
    sample, decomposition, plan = artifacts(raw, add=change == "unnecessary_added")
    if change == "category":
        raw["category"] = "Unsupported"
        with pytest.raises(ValueError, match="Unsupported category"):
            raw_rubric_sample(raw, 0)
        return
    if change == "unaccounted":
        decomposition["units"].clear()
    if change == "unmapped":
        plan["excluded_units"].clear()
    if change == "twice":
        plan["excluded_units"][0]["unit_id"] = "U1"
    if change == "wrong_group":
        plan["assignments"][0]["group_id"] = "offense"
    if change == "invented_evidence":
        decomposition["units"][0]["source_ids"] = ["S999"]
    if change == "unnecessary_added":
        plan["added_issues"][0]["issue_text"] = raw["issues"][0]
    if change == "wrong_order":
        plan["issues"][0]["issue_index"] = 2
    if change == "unsupported_rule":
        plan["assignments"][0]["rule_indices"] = [999]
    with pytest.raises(ValueError):
        materialize(sample, decomposition, plan)


def test_build_retry_resume_paired_publish_and_evaluate(config_factory, tmp_path):
    originals = [raw_record("first"), raw_record("second", "Commercial")]
    originals[1]["code"] = "KDTM-ST"
    application = tmp_path / "final/application.jsonl"
    spotting = tmp_path / "final/spotting.jsonl"
    config = config_factory(originals, mode="build-rubrics", expected_count=2, rubric_json_retries=0,
                            final_application_jsonl=application, final_issue_spotting_jsonl=spotting)
    raw_bytes = config.input_jsonl.read_bytes()
    calls = []
    reject_second = True

    def rubric(request):
        body = json.loads(request.content)
        assert body["model"] == config.rubric_model
        assert body["chat_template_kwargs"]["reasoning_effort"] == 75
        assert body["max_tokens"] == 65536
        payload = json.loads(body["messages"][1]["content"])
        raw = next(r for r in originals if r["category"] == payload["category"])
        assert payload["instance_level"] == raw["instance_level"]
        assert "".join(payload["expected_answer_segments"].values()) == ANSWER
        schema = body["response_format"]["json_schema"]["schema"]["properties"]
        _, dec, plan = artifacts(raw, add=raw["doc_name"] == "first", missing=raw["doc_name"] == "second")
        stage = "decompose" if "units" in schema else "map" if "assignments" in schema else "audit"
        calls.append((raw["doc_name"], stage))
        approved = not (reject_second and raw["doc_name"] == "second")
        if stage == "audit":
            assert payload["plan"] == plan
        value = {"decompose": dec, "map": plan,
                 "audit": {"approved": approved, "problems": [] if approved else ["Unit U2 excluded without adequate reasoning"],
                           "repair_stage": None if approved else "map"}}[stage]
        return httpx.Response(200, json=envelope(json.dumps(value, ensure_ascii=False)))

    first = run_build(config, rubric)
    assert first["counts"]["failed"] == 1 and first["missing_doc_names"] == ["second"]
    assert not first["published"] and not application.exists() and not spotting.exists()
    assert "audit rejected" in read_jsonl(config.run_dir / "rubric_review.jsonl")[-1]["error"]
    calls.clear()
    reject_second = False
    complete = run_build(replace(config, resume=True), rubric)
    assert calls == [("second", stage) for stage in ("decompose", "map", "audit")]
    assert complete["published"] and complete["counts"]["skipped"] == 1
    assert complete["counts"]["issues_needing_review"] == 1
    assert complete["counts"]["samples_with_added_issues"] == 1
    assert complete["counts"]["rubric_coverage"] == 1
    assert config.input_jsonl.read_bytes() == raw_bytes
    rows = read_jsonl(application)
    spots = read_jsonl(spotting)
    assert [row["doc_name"] for row in rows] == ["first", "second"]
    assert spots == [{"input": row["input"]["facts_clean"], "expected_answer": row["input"]["issues"]} for row in rows]
    added = read_jsonl(config.run_dir / "added_issues.jsonl")
    assert added[0]["new_issue"] == spots[0]["expected_answer"][-1] and added[0]["doc_name"] == "first"
    assert len(read_jsonl(config.run_dir / "excluded_units.jsonl")) == 2
    assert len(read_jsonl(config.run_dir / "source_units.jsonl")) == 2
    assert json.loads(application.with_name(application.name + ".manifest.json").read_text())["sample_count"] == 2

    accessed = []
    def generate(request):
        body = json.loads(request.content)
        assert "Fact-to-Rule Application" in body["messages"][0]["content"]
        assert "tự chọn bố cục" in body["messages"][0]["content"]
        prompt = body["messages"][1]["content"]
        assert {"response_format", "rubrics", "category", "doc_name", "expected_answer"}.isdisjoint(body)
        assert all(secret not in prompt for secret in [ANSWER, "Commercial", "Civil", "first", "second", "I1_C1"])
        accessed.append("generator")
        return httpx.Response(200, json=envelope(WORDS))

    def judge(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        assert "expected_answer" not in payload and "benchmark_seed" not in payload
        assert "theo ý nghĩa pháp lý" in body["messages"][0]["content"]
        assert payload["evaluated_model_response"] == WORDS
        row = next(row for row in rows if row["doc_name"] == payload["doc_name"])
        assert payload["rubrics"] == row["rubrics"]
        accessed.append("judge")
        return httpx.Response(200, json=envelope(json.dumps(judge_result(row, outcome=False), ensure_ascii=False)))

    def forbidden(_request):
        pytest.fail("Unexpected model call")

    evaluation = replace(config, mode="full", evaluation_profile="rubric", input_jsonl=application, run_id="evaluate")
    summary = run_eval(evaluation, generate, judge)
    assert accessed.count("generator") == accessed.count("judge") == 2
    assert summary["application_score"] == 1 and summary["outcome_accuracy"] == 0
    assert summary["by_category"]["Commercial"]["unscorable_samples"] == 1
    assert summary["by_category"]["Civil"]["application_scored_samples"] == 1
    assert "final_order" not in json.dumps(summary)
    generations = read_jsonl(evaluation.run_dir / "generations.jsonl")
    assert all(r["raw_response"] == WORDS and r["run_id"] == evaluation.run_id for r in generations)
    resumed = run_eval(replace(evaluation, resume=True), forbidden, forbidden)
    assert resumed["counts"]["skipped"] == 2 and resumed["application_score"] == 1
    judge_only = replace(evaluation, mode="judge-only", run_id="judge-only",
                         generation_jsonl=evaluation.run_dir / "generations.jsonl")
    assert run_eval(judge_only, forbidden, judge)["outcome_accuracy"] == 0


def test_application_score_macro_averages_issues_and_keeps_outcome_separate():
    sample, dec, plan = artifacts(raw_record(), add=True)
    for i in range(2):
        # Two more separate criteria with source evidence supporting the second issue.
        uid = f"EX{i}"
        dec["units"].append({"unit_id": uid, "reasoning": f"Lập luận {i}", "source_ids": ["S1"]})
        plan["assignments"].append({**plan["assignments"][-1], "unit_id": uid, "criterion": f"Tiêu chí {i}"})
        plan["added_issues"][0]["unit_ids"].append(uid)
    row, _ = materialize(sample, dec, plan)
    sample = rubric_sample(row, 0)
    labels = judge_result(row)
    second = labels["issues_evaluation"][1]
    for rating in second["criterion_scores"]:
        rating.update(score=0, evidence_from_model_answer=[])
    second["outcome_correct"] = False
    result = score_rubric_labels(labels, sample, WORDS)
    assert result["application_score"] == 0.5 and result["application_score_10"] == 5
    assert result["outcome_accuracy"] == 0.5
    assert [row["scored_criteria"] for row in result["issues"]] == [1, 3]
    for mutate in [lambda value: value["issues_evaluation"][1]["criterion_scores"].pop(),
                   lambda value: value["issues_evaluation"][0]["criterion_scores"][0].update(score=3),
                   lambda value: value["issues_evaluation"][0]["criterion_scores"][0].update(evidence_from_model_answer=["invented"])]:
        bad = deepcopy(labels)
        mutate(bad)
        with pytest.raises(LabelError):
            score_rubric_labels(bad, sample, WORDS)


def test_decisively_missing_input_and_unclear_verdict_never_receive_zero():
    sample, dec, plan = artifacts(raw_record(), missing=True)
    plan["issues"][0].update(reference_verdict=None, verdict_source_ids=[], outcome_review_reason="Nguồn không có kết luận.")
    row, _ = materialize(sample, dec, plan)
    labels = judge_result(row)
    result = score_rubric_labels(labels, rubric_sample(row, 0), WORDS)
    assert result["application_score"] is None and result["outcome_accuracy"] is None and labels["needs_review"]
    assert labels["issues_evaluation"][0]["criterion_scores"] == []


def test_decisively_blocked_issue_keeps_scores_of_other_valid_criteria():
    sample, dec, plan = artifacts(raw_record(), missing=True)
    row, _ = materialize(sample, dec, plan)
    issue = row["rubrics"]["issues"][0]
    criterion = deepcopy(issue["groups"][0]["criteria"][0])
    criterion.update(criterion_id="I1_C2", unit_id="U3", criterion="Xác định phạm vi nghĩa vụ",
                     scorable=True, needs_review=False, decisive=False, missing_information=[], review_reason=None)
    issue["groups"][0]["criteria"].append(criterion)
    sample = rubric_sample(row, 0)
    labels = judge_result(row, score=1)
    assert labels["issues_evaluation"][0]["criterion_scores"][0]["criterion_id"] == "I1_C2"
    scores = score_rubric_labels(labels, sample, WORDS)
    assert scores["application_score"] is None
    assert scores["issues"][0]["scored_criteria"] == 1


def test_review_reason_kept_when_grounded_criteria_cover_only_part_of_issue():
    sample, dec, plan = artifacts(raw_record())
    plan["issues"][0]["issue_review_reason"] = "Nguồn giải quyết một phần và chưa đủ căn cứ cho toàn issue."
    row, _ = materialize(sample, dec, plan)
    labels = judge_result(row)
    assert labels["issues_evaluation"][0]["criterion_scores"][0]["criterion_id"] == "I1_C1"
    result = score_rubric_labels(labels, rubric_sample(row, 0), WORDS)
    assert result["application_score"] is None and result["outcome_accuracy"] is None


@pytest.mark.parametrize("repair_stage", ["map", "decompose"])
def test_rubric_repair_uses_verified_source_ids_and_retries_only_required_stage(config_factory, tmp_path, repair_stage):
    raw = raw_record("recovery")
    raw["benchmark_seed"]["application"] = ANSWER + "\n" + "Nội dung khác để tạo đoạn nguồn. " * 30
    _, dec, plan = artifacts(raw, missing=True)
    plan["issues"][0].update(issue_review_reason=None, verdict_source_ids=["U1", "S1"])
    config = config_factory([raw], mode="build-rubrics", expected_count=1, rubric_json_retries=2,
                            final_application_jsonl=tmp_path / "application.jsonl",
                            final_issue_spotting_jsonl=tmp_path / "spotting.jsonl")
    calls = []

    def rubric(request):
        body = json.loads(request.content)
        schema = body["response_format"]["json_schema"]["schema"]["properties"]
        stage = "decompose" if "units" in schema else "map" if "assignments" in schema else "audit"
        system = body["messages"][0]["content"]
        payload = json.loads(body["messages"][1]["content"])
        calls.append(stage)
        if stage in {"map", "audit"}:
            assert "Án phí, quyền kháng cáo" in system and "input cố định" in system
        if stage == "decompose":
            value = dec
            if calls.count(stage) > 1:
                assert payload["previous_attempt"] == dec and "Cần sửa U1" in system
        elif stage == "map":
            assert payload["decomposition"] == dec
            assert schema["issues"]["items"]["properties"]["verdict_source_ids"]["items"]["enum"] == list(payload["expected_answer_segments"])
            value = deepcopy(plan)
            if calls.count(stage) == 1:
                value["issues"][0]["verdict_source_ids"] = ["U404"]
            if calls.count(stage) == 2:
                assert payload["previous_attempt"]["issues"][0]["verdict_source_ids"] == ["U404"]
            if calls.count(stage) == 3:
                assert "Cần sửa U1" in system
                assert ("previous_attempt" in payload) == (repair_stage == "map")
        else:
            outcome = payload["plan"]["issues"][0]
            assert outcome["verdict_source_ids"] == list(payload["expected_answer_segments"])
            assert outcome["issue_review_reason"] == plan["assignments"][0]["review_reason"]
            approved = calls.count(stage) > 1
            value = {"approved": approved, "problems": [] if approved else ["Cần sửa U1"],
                     "repair_stage": None if approved else repair_stage}
        return httpx.Response(200, json=envelope(json.dumps(value, ensure_ascii=False)))

    assert run_build(config, rubric)["published"]
    assert calls == ["decompose", "map", "map", "audit", *(["decompose"] if repair_stage == "decompose" else []), "map", "audit"]
    saved = read_jsonl(config.run_dir / "rubric_records.jsonl")[0]
    assert saved["metadata"]["retry_stages"] == [
        {"failed_stage": "map", "retry_stage": "map"}, {"failed_stage": "audit", "retry_stage": repair_stage}]
    assert saved["dataset"]["rubrics"]["issues"][0]["verdict_evidence"] == list(source_segments(raw["benchmark_seed"]["application"], "S").values())


def test_retry_into_new_run_replays_completed_rubric_and_only_calls_failed_sample(config_factory, tmp_path):
    rows = [raw_record("old-approved"), raw_record("old-failed", "Commercial")]
    cfg = config_factory(rows, mode="build-rubrics", expected_count=2, rubric_json_retries=0, api_max_retries=0,
                         final_application_jsonl=tmp_path / "application.jsonl",
                         final_issue_spotting_jsonl=tmp_path / "spotting.jsonl")
    seen = []
    first_run = True

    def judge(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        seen.append(payload["category"])
        if first_run and payload["category"] == "Commercial":
            return httpx.Response(503)
        _, dec, plan = artifacts(next(row for row in rows if row["category"] == payload["category"]))
        fields = body["response_format"]["json_schema"]["schema"]["properties"]
        value = (dec if "units" in fields else plan if "assignments" in fields
                 else {"approved": True, "problems": [], "repair_stage": None})
        return httpx.Response(200, json=envelope(json.dumps(value, ensure_ascii=False)))

    assert run_build(cfg, judge)["counts"]["completed"] == 1
    source = cfg.run_dir / "rubric_records.jsonl"
    saved = read_jsonl(source)[0]
    saved["semantic_audit"].pop("repair_stage")  # Older approved audit records omit this field.
    source.write_text(json.dumps(saved, ensure_ascii=False) + "\n", encoding="utf-8")
    original = source.read_bytes()
    old_snapshot_path = cfg.run_dir / "config.json"
    snapshot = json.loads(old_snapshot_path.read_text())
    snapshot["implementation_sha256"] = "prior-code"
    old_snapshot_path.write_text(json.dumps(snapshot))
    first_run = False
    seen.clear()
    recovery = replace(cfg, run_id="rubric-retry", rubric_seed_run=cfg.run_dir)
    summary = run_build(recovery, judge)
    assert seen == ["Commercial"] * 3
    assert summary["published"] and summary["counts"]["skipped"] == 1
    assert source.read_bytes() == original
    assert read_jsonl(recovery.run_dir / "rubric_records.jsonl")[0] == read_jsonl(source)[0]


def test_invalid_row_after_limit_fails_read_only(config_factory):
    sample, dec, plan = artifacts(raw_record())
    row, _ = materialize(sample, dec, plan)
    invalid = deepcopy(row)
    invalid.update(doc_name="invalid", rubrics={"issues": []})
    cfg = config_factory([row, invalid], evaluation_profile="rubric", mode="validate-only", limit=1)
    before = cfg.input_jsonl.read_bytes()

    def no_api(_request):
        pytest.fail("Validation must finish before model requests")

    with pytest.raises(DatasetError):
        run_eval(cfg, no_api, no_api)
    assert cfg.input_jsonl.read_bytes() == before
    report = json.loads((cfg.run_dir / "summary.json").read_text())
    assert report["invalid_by_category"] == {"Civil": 1}


def test_decomposition_prompt_preserves_appeal_context():
    messages = stage_messages("decompose", DECOMPOSITION_SCHEMA, {"instance_level": "Appellate"}, None)
    assert "phúc thẩm, giám đốc thẩm hoặc tái thẩm" in messages[0]["content"]
    assert json.loads(messages[1]["content"])["instance_level"] == "Appellate"


def test_full_run_count_mismatch_stops_before_api_or_final_publication(config_factory, tmp_path):
    cfg = config_factory([raw_record()], mode="build-rubrics", expected_count=600,
                         final_application_jsonl=tmp_path / "application.jsonl",
                         final_issue_spotting_jsonl=tmp_path / "spotting.jsonl")
    with pytest.raises(ValueError, match="EXPECTED_COUNT=600"):
        run_build(cfg, lambda _: pytest.fail("Wrong-sized input must fail before generation"))
    summary = json.loads((cfg.run_dir / "rubric_generation_summary.json").read_text())
    assert not cfg.final_application_jsonl.exists() and not summary["published"]
    assert summary["counts"]["input"] == 1


@pytest.mark.parametrize("retry_into", [False, True])
def test_rubric_resume_helper_restores_raw_input_and_settings(config_factory, monkeypatch, tmp_path, retry_into):
    cfg = config_factory([raw_record()], mode="build-rubrics", expected_count=1, rubric_concurrency=11,
                         final_application_jsonl=tmp_path / "application.jsonl",
                         final_issue_spotting_jsonl=tmp_path / "spotting.jsonl")
    run_build(cfg, lambda _: httpx.Response(503))
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    import resume_rubric_run

    monkeypatch.setenv("PROJECT_ROOT", str(cfg.project_root))
    monkeypatch.setenv("OUTPUT_DIR", str(cfg.output_dir))
    monkeypatch.setenv("RAW_INPUT_JSONL", "wrong.jsonl")
    monkeypatch.setenv("RUBRIC_CONCURRENCY", "99")
    args = ["--retry-into", "new-run"] if retry_into else []
    monkeypatch.setattr(sys, "argv", [str(scripts / "resume_rubric_run.py"), cfg.run_id, *args])
    monkeypatch.setattr(resume_rubric_run.os, "chdir", lambda _: None)
    captured = {}
    monkeypatch.setattr(resume_rubric_run.os, "execvpe", lambda _, argv, env: captured.update(env))
    assert resume_rubric_run.main() == 0
    for key, value in captured.items():
        monkeypatch.setenv(key, value)
    restored = Config.from_env()
    assert restored.resume == (not retry_into) and restored.mode == "build-rubrics"
    assert restored.rubric_seed_run == (cfg.run_dir if retry_into else None)
    assert restored.run_id == ("new-run" if retry_into else cfg.run_id)
    assert restored.input_jsonl == cfg.input_jsonl
    assert restored.rubric_concurrency == 11 and restored.expected_count == 1
