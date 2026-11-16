from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from legal_application.data import DatasetError, canonical_hash
from legal_application.pipeline import Pipeline, bounded_map
from legal_application.storage import read_jsonl
from conftest import envelope, generator_response, make_labels, make_record


def run(config, transports):
    return asyncio.run(Pipeline(config, generator_transport=transports[0], judge_transport=transports[1]).run())


def test_full_pipeline_artifacts_and_resume(config_factory, mock_transports):
    config = config_factory([make_record(f"case-{i}") for i in range(5)], batch_size=2)
    summary = run(config, mock_transports)
    assert summary["counts"] == {"input": 5, "input_valid": 5, "input_removed": 0, "selected": 5, "attempted": 5, "completed": 5, "failed": 0,
                                 "pending": 0, "skipped": 0, "scored": 5, "needs_review": 0}
    assert summary["mean_score"] == 10
    gen = read_jsonl(config.run_dir / "generations.jsonl")
    evals = read_jsonl(config.run_dir / "evaluations.jsonl")
    assert len(gen) == len(evals) == 5
    assert len({r["doc_name"] for r in gen}) == 5
    for i, record in enumerate(gen):
        assert record["input_index"] == i
        assert record["doc_name"] == f"case-{i}"
        assert all(f"ISSUE {j}:" in record["raw_response"] for j in [1, 2])
        assert "Kết luận chung" not in record["raw_response"]
    for path in config.run_dir.glob("*.json*"):
        assert "DO_NOT_SAVE_REASONING" not in path.read_text()
    assert (config.run_dir / "config.json").exists()
    assert (config.run_dir / "run.log").exists()
    before = {p.name: p.read_bytes() for p in config.run_dir.glob("*.jsonl")}
    resumed = run(replace(config, resume=True), mock_transports)
    assert resumed["counts"]["skipped"] == 5
    assert resumed["generation"]["this_invocation"]["attempted"] == 0
    assert resumed["evaluation"]["this_invocation"]["attempted"] == 0
    assert len(mock_transports[2]["generator"]) == len(mock_transports[2]["judge"]) == 5
    assert before == {p.name: p.read_bytes() for p in config.run_dir.glob("*.jsonl")}


def test_existing_run_requires_explicit_resume(config_factory, mock_transports):
    config = config_factory()
    run(config, mock_transports)
    with pytest.raises(ValueError, match="Run already exists"):
        run(config, mock_transports)
    with pytest.raises(ValueError, match="configuration/input/code differs"):
        run(replace(config, resume=True, max_output_tokens=65536), mock_transports)


def test_input_change_blocks_resume(config_factory, mock_transports):
    config = config_factory()
    run(config, mock_transports)
    with config.input_jsonl.open("a") as handle:
        handle.write(json.dumps(make_record("new")) + "\n")
    with pytest.raises(ValueError, match="configuration/input/code differs"):
        run(replace(config, resume=True), mock_transports)


def test_judge_only_uses_saved_generation_without_generator(config_factory, mock_transports):
    original = config_factory([make_record(f"case-{i}") for i in range(5)])
    run(original, mock_transports)
    config = replace(original, run_id="judge-run", mode="judge-only", limit=3, generator_model="", generator_base_url="",
                     generation_jsonl=original.run_dir / "generations.jsonl")
    summary = run(config, mock_transports)
    assert summary["counts"]["selected"] == summary["counts"]["completed"] == 3
    assert len(mock_transports[2]["generator"]) == 5
    assert len(mock_transports[2]["judge"]) == 8
    assert len(read_jsonl(config.run_dir / "generations.jsonl")) == 3
    assert summary["generation"]["this_invocation"]["attempted"] == 0
    assert summary["generation"]["this_invocation"]["skipped"] == 3


@pytest.mark.parametrize("limit,expected", [(0, 0), (1, 1), (3, 3), (10, 5), (None, 5)])
def test_limit_and_exact_batch_sizes(config_factory, mock_transports, limit, expected):
    config = config_factory([make_record(f"case-{i}") for i in range(5)], limit=limit, batch_size=2)
    summary = run(config, mock_transports)
    assert summary["counts"]["input"] == 5
    assert summary["counts"]["selected"] == summary["counts"]["completed"] == expected
    assert len(mock_transports[2]["generator"]) == len(mock_transports[2]["judge"]) == expected
    if expected:
        log = (config.run_dir / "run.log").read_text()
        assert log.count("Batch start") == (expected + 1) // 2


def test_out_of_order_completion_preserves_mapping_and_bounded_concurrency(config_factory):
    records = [make_record(f"case-{i}", n=1) for i in range(7)]
    config = config_factory(records, batch_size=7, generator_concurrency=3, judge_concurrency=2)
    active = {"generator": 0, "judge": 0}
    maximum = active.copy()

    async def gen(request):
        active["generator"] += 1
        maximum["generator"] = max(maximum["generator"], active["generator"])
        body = json.loads(request.content)
        # Unique issue text is supplied below; delay to force out-of-order completion.
        issue = body["messages"][1]["content"].split("LEGAL ISSUES\n\n1. ")[1].split("\n\nAPPLICABLE LAWS")[0]
        i = int(issue.rsplit(" ", 1)[1])
        await asyncio.sleep((7 - i) * 0.001)
        active["generator"] -= 1
        return httpx.Response(200, json=envelope(generator_response([issue])))

    async def judge(request):
        active["judge"] += 1
        maximum["judge"] = max(maximum["judge"], active["judge"])
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        assert payload["issues"][0] in payload["evaluated_model_response"]
        await asyncio.sleep(0.002)
        active["judge"] -= 1
        return httpx.Response(200, json=envelope(json.dumps(make_labels(payload))))

    for i, row in enumerate(records):
        row["issues"] = [f"Nghĩa vụ số {i}"]
    config.input_jsonl.write_text("".join(json.dumps(r) + "\n" for r in records))
    run(config, (httpx.MockTransport(gen), httpx.MockTransport(judge)))
    assert maximum == {"generator": 3, "judge": 2}
    gen_rows = read_jsonl(config.run_dir / "generations.jsonl")
    assert [r["input_index"] for r in gen_rows] != list(range(7))
    assert len({r["input_index"] for r in gen_rows}) == 7
    for row in read_jsonl(config.run_dir / "evaluations.jsonl"):
        assert row["issues_evaluation"][0]["original_issue"] == f"Nghĩa vụ số {row['input_index']}"


def test_generation_failure_does_not_lose_other_samples_and_resume_only_failed(config_factory, mock_transports):
    config = config_factory([make_record(f"case-{i}") for i in range(3)], generator_concurrency=1)
    calls = 0

    def broken_gen(request):
        nonlocal calls
        calls += 1
        if calls <= 2:  # 503 initial + retry for first sample
            return httpx.Response(503)
        return mock_transports[0].handler(request)

    summary = run(config, (httpx.MockTransport(broken_gen), mock_transports[1]))
    assert summary["counts"]["completed"] == 2
    assert summary["counts"]["failed"] == 1
    assert summary["generation"]["this_invocation"]["attempted"] == 3
    assert len(read_jsonl(config.run_dir / "generations.jsonl")) == 2
    summary = run(replace(config, resume=True), mock_transports)
    assert summary["counts"]["completed"] == 3
    assert summary["counts"]["failed"] == 0
    assert summary["generation"]["this_invocation"]["attempted"] == 1
    assert len(read_jsonl(config.run_dir / "generations.jsonl")) == 3


def test_failed_judge_does_not_regenerate_successful_generation(config_factory, mock_transports):
    config = config_factory()
    bad_judge = httpx.MockTransport(lambda request: httpx.Response(401))
    summary = run(config, (mock_transports[0], bad_judge))
    assert summary["counts"]["failed"] == 1
    assert summary["generation"]["completed"] == 1
    assert summary["evaluation"]["completed"] == 0
    summary = run(replace(config, resume=True), mock_transports)
    assert summary["counts"]["completed"] == 1
    assert len(mock_transports[2]["generator"]) == 1


def test_judge_repairs_schema_and_mapping_errors_with_full_evidence(config_factory, mock_transports):
    config = config_factory(judge_json_retries=2)
    calls = []

    def judge(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        calls.append(body)
        assert payload["benchmark_seed"] == make_record()["benchmark_seed"]
        assert payload["evaluated_model_response"] == generator_response(payload["issues"])
        if len(calls) == 1:
            content = "INVALID_JSON_DRAFT"
        elif len(calls) == 2:
            labels = make_labels(payload)
            labels["issues_evaluation"].reverse()
            content = json.dumps(labels)
        else:
            content = json.dumps(make_labels(payload))
        return httpx.Response(200, json=envelope(content))

    summary = run(config, (mock_transports[0], httpx.MockTransport(judge)))
    assert summary["counts"]["completed"] == 1
    record = read_jsonl(config.run_dir / "evaluations.jsonl")[0]
    assert record["metadata"]["label_attempts"] == 3
    assert len(record["metadata"]["validation_errors"]) == 2
    assert "INVALID_JSON_DRAFT" not in json.dumps(calls[1:])
    assert len(mock_transports[2]["generator"]) == 1


def test_exhausted_json_repairs_fail_only_that_sample(config_factory, mock_transports):
    config = config_factory([make_record("bad-json"), make_record("good-json")])
    count = {"bad-json": 0, "good-json": 0}

    def judge(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        count[payload["doc_name"]] += 1
        content = "```not json```" if payload["doc_name"] == "bad-json" else json.dumps(make_labels(payload))
        return httpx.Response(200, json=envelope(content))

    summary = run(config, (mock_transports[0], httpx.MockTransport(judge)))
    assert count == {"bad-json": 2, "good-json": 1}
    assert summary["counts"]["completed"] == summary["counts"]["failed"] == 1
    assert len(read_jsonl(config.run_dir / "generations.jsonl")) == 2
    errors = read_jsonl(config.run_dir / "errors.jsonl")
    assert errors[0]["doc_name"] == "bad-json"
    assert errors[0]["stage"] == "evaluation"
    resumed = run(replace(config, resume=True), mock_transports)
    assert resumed["counts"]["completed"] == 2
    assert len(mock_transports[2]["generator"]) == 2


def test_judge_retries_length_response_even_if_json_parses(config_factory, mock_transports):
    config = config_factory()
    calls = 0

    def judge(request):
        nonlocal calls
        calls += 1
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        return httpx.Response(200, json=envelope(json.dumps(make_labels(payload)), finish="length" if calls == 1 else "stop"))

    assert run(config, (mock_transports[0], httpx.MockTransport(judge)))["counts"]["completed"] == 1
    evaluation = read_jsonl(config.run_dir / "evaluations.jsonl")[0]
    assert calls == evaluation["metadata"]["label_attempts"] == 2
    assert evaluation["metadata"]["judge_calls"][0]["finish_reason"] == "length"


def test_omitted_issue_is_scored_without_regeneration(config_factory, mock_transports):
    config = config_factory()
    content = generator_response(make_record()["issues"][:1])
    gen_calls = []

    def generator(request):
        gen_calls.append(request)
        return httpx.Response(200, json=envelope(content))

    def judge(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        assert payload["evaluated_model_response"] == content
        labels = make_labels(payload, covered=[True, False], correct=[True, False], order=False)
        return httpx.Response(200, json=envelope(json.dumps(labels)))

    summary = run(config, (httpx.MockTransport(generator), httpx.MockTransport(judge)))
    assert len(gen_calls) == 1
    assert summary["mean_score"] == 2.5


def test_judge_only_missing_generation_keeps_available_samples(config_factory, mock_transports):
    original = config_factory([make_record("case-0"), make_record("case-1")])
    run(original, mock_transports)
    rows = read_jsonl(original.run_dir / "generations.jsonl")
    source = original.project_root / "partial_generations.jsonl"
    source.write_text(json.dumps(rows[1]) + "\n")
    config = replace(original, run_id="partial-judge", mode="judge-only", generation_jsonl=source)
    summary = run(config, mock_transports)
    assert summary["counts"]["selected"] == 2
    assert summary["counts"]["completed"] == summary["counts"]["failed"] == 1
    assert summary["failed_doc_names"] == ["case-0"]
    assert len(mock_transports[2]["generator"]) == 2


def test_judge_only_rejects_mapping_mismatch_before_any_call(config_factory, mock_transports):
    original = config_factory()
    run(original, mock_transports)
    row = read_jsonl(original.run_dir / "generations.jsonl")[0]
    row["sample_sha256"] = "changed"
    source = original.project_root / "bad_generations.jsonl"
    source.write_text(json.dumps(row) + "\n")
    config = replace(original, run_id="bad-judge", mode="judge-only", generation_jsonl=source)
    with pytest.raises(ValueError, match="Generation mapping mismatch"):
        run(config, mock_transports)
    assert len(mock_transports[2]["generator"]) == len(mock_transports[2]["judge"]) == 1
    summary = json.loads((config.run_dir / "summary.json").read_text())
    assert summary["status"] == "preflight_failed"
    assert summary["counts"]["input"] == summary["counts"]["selected"] == 1
    assert "Generation mapping mismatch" in (config.run_dir / "run.log").read_text()


def test_all_review_has_no_mean_score(config_factory, mock_transports):
    config = config_factory()

    def judge(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        return httpx.Response(200, json=envelope(json.dumps(make_labels(payload, review=True))))

    transports = (mock_transports[0], httpx.MockTransport(judge))
    summary = run(config, transports)
    assert summary["mean_score"] is None
    assert all(v is None for v in summary["mean_score_components"].values())
    assert summary["counts"]["needs_review"] == 1
    assert summary["counts"]["scored"] == 0
    summary = run(replace(config, resume=True), transports)
    assert summary["evaluation"]["this_invocation"]["attempted"] == 0


def test_uncertain_gold_excluded_from_all_means(config_factory, mock_transports):
    config = config_factory([make_record("valid"), make_record("review")])

    def judge(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        return httpx.Response(200, json=envelope(json.dumps(make_labels(payload, review=payload["doc_name"] == "review"))))

    summary = run(config, (mock_transports[0], httpx.MockTransport(judge)))
    assert summary["status"] == "completed_with_review"
    assert summary["counts"]["completed"] == 2
    assert summary["counts"]["scored"] == summary["counts"]["needs_review"] == 1
    assert summary["mean_score"] == 10
    assert sum(summary["score_distribution"]["bins"].values()) == 1
    review = next(r for r in read_jsonl(config.run_dir / "evaluations.jsonl") if r["doc_name"] == "review")
    assert review["score_components"]["total"] is None


def test_malformed_input_beyond_limit_is_deleted_and_run_continues(config_factory, mock_transports):
    config = config_factory(limit=1)
    original = config.input_jsonl.read_bytes()
    with config.input_jsonl.open("a") as handle:
        handle.write('{"broken"')
    summary = run(config, mock_transports)
    assert summary["status"] == "completed"
    assert summary["counts"]["input"] == 2
    assert summary["counts"]["input_valid"] == summary["counts"]["input_removed"] == 1
    assert len(mock_transports[2]["generator"]) == len(mock_transports[2]["judge"]) == 1
    assert config.input_jsonl.read_bytes() == original
    removed = read_jsonl(config.run_dir / "removed_input_lines.jsonl")
    assert removed[0]["original_input_line"] == 2
    assert removed[0]["raw_line"] == '{"broken"'
    summary = run(replace(config, resume=True), mock_transports)
    assert summary["counts"]["input_removed"] == 1
    assert summary["counts"]["skipped"] == 1
    assert len(read_jsonl(config.run_dir / "removed_input_lines.jsonl")) == 1


def test_validate_only_never_calls_api(config_factory, mock_transports):
    config = config_factory(mode="validate-only", generator_model="", generator_base_url="")
    assert run(config, mock_transports)["status"] == "validated"
    assert not mock_transports[2]["generator"] and not mock_transports[2]["judge"]


def test_resume_recovers_torn_tail_with_backup(config_factory, mock_transports):
    config = config_factory()
    run(config, mock_transports)
    gen_path = config.run_dir / "generations.jsonl"
    with gen_path.open("ab") as handle:
        handle.write(b'{"doc_name": "crash')
    summary = run(replace(config, resume=True), mock_transports)
    assert summary["counts"]["completed"] == 1
    assert len(read_jsonl(gen_path)) == 1
    assert len(list(config.run_dir.glob("generations.jsonl.torn-tail.*.bak"))) == 1
    assert len(mock_transports[2]["generator"]) == 1


def test_resume_rejects_duplicate_generation(config_factory, mock_transports):
    config = config_factory()
    run(config, mock_transports)
    path = config.run_dir / "generations.jsonl"
    with path.open("ab") as handle:
        handle.write(path.read_bytes())
    with pytest.raises(ValueError, match="Duplicate artifact"):
        run(replace(config, resume=True), mock_transports)


def test_final_order_comes_from_issue_verdicts_without_overall_conclusion(config_factory, mock_transports):
    config = config_factory()
    expected = generator_response(make_record()["issues"])

    def judge(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        response = payload["evaluated_model_response"]
        assert response == expected
        assert "Kết luận chung" not in response and "Final Judgment" not in response
        labels = make_labels(payload, order=all(f"Phán quyết đối với Issue {i}: B trả A 100 đồng." in response for i in (1, 2)))
        return httpx.Response(200, json=envelope(json.dumps(labels)))

    summary = run(config, (mock_transports[0], httpx.MockTransport(judge)))
    assert summary["mean_score_components"]["final_order"] == 5
    assert summary["mean_score"] == 10


def test_bounded_workers_cancel_siblings_on_fatal_persistence_error():
    state = []

    async def case():
        async def process(i):
            if i == 0:
                await asyncio.sleep(0)
                raise OSError("disk full")
            try:
                await asyncio.sleep(10)
                state.append("unexpected completion")
            except asyncio.CancelledError:
                state.append("cancelled")
                raise
        with pytest.raises(OSError):
            await bounded_map(list(range(5)), 2, process)

    asyncio.run(case())
    assert state == ["cancelled"]
