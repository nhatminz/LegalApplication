from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import make_record
from legal_application.storage import file_hash, implementation_hash, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
HELPER = runpy.run_path(str(ROOT / "scripts/create_task_inputs.py"))
export = HELPER["export_task_inputs"]
FILENAMES = HELPER["FILENAMES"]


def write_source(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_three_task_inputs_preserve_fields_order_and_control_reference_access(tmp_path):
    rows = [make_record("first"), make_record("second")]
    for i, row in enumerate(rows):
        row.pop("data_status")  # filter_full exports keep the original unlabelled records.
        row["document_id"] = 700 + i
        row["benchmark_seed"]["facts_clean"] += "\n  Giữ nguyên khoảng trắng cuối.  "
    source, output = tmp_path / "full-v2.jsonl", tmp_path / "input_temp_18_11"
    write_source(source, rows)
    original = source.read_bytes()
    manifest = export(source, output, expected_count=2)
    assert source.read_bytes() == original
    files = {task: read_jsonl(output / filename) for task, filename in FILENAMES.items()}
    assert manifest["sample_count"] == 2
    assert manifest["samples"] == [{"output_line": i + 1, "input_line": i + 1, "doc_name": row["doc_name"],
                                     "document_id": row["document_id"]} for i, row in enumerate(rows)]
    for i, row in enumerate(rows):
        seed = row["benchmark_seed"]
        for records in files.values():
            assert set(records[i]) == {"input", "expected_answer"}
        issue = files["issue_spotting"][i]
        app = files["application"][i]
        conclusion = files["conclusion"][i]
        assert issue == {"input": seed["facts_clean"], "expected_answer": row["issues"]}
        assert app["input"]["facts_clean"] == seed["facts_clean"]
        assert set(app["input"]) == {"facts_clean", "issues", "rules"}
        assert app["input"]["issues"] == row["issues"]
        assert app["expected_answer"] == seed["application"]
        assert app["input"]["rules"] == [
            {"law_str": row["laws"][0]["law_str"], "citation_str": "Điều 1", "context_text": "Toàn bộ nội dung điều 1.\nKhông cắt ngắn."},
            {"law_str": row["laws"][0]["law_str"], "citation_str": "Điều 2", "context_text": "Toàn bộ nội dung điều 2."},
            {"law_str": row["laws"][1]["law_str"], "citation_str": None, "context_text": "Toàn bộ văn bản fallback."},
        ]
        assert conclusion == {"input": {**app["input"], "application": seed["application"]}, "expected_answer": seed["conclusion"]}
        app_input = json.dumps(app["input"])
        assert all(private not in app_input for private in [seed["application"], seed["conclusion"],
                                                          "RAW_FACTS_SENTINEL", "LAW_METADATA_MUST_NOT_LEAK",
                                                          "EXACT_NOT_USED", "PRIVATE_EXPLAIN_MUST_NOT_LEAK"])
    for task, filename in FILENAMES.items():
        assert manifest["files"][task]["sha256"] == file_hash(output / filename)
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_all_1207_samples_export_without_a_hidden_cap(tmp_path):
    source, output = tmp_path / "full-v2.jsonl", tmp_path / "tasks"
    write_source(source, [make_record(f"case-{i}") for i in range(1207)])
    manifest = export(source, output, expected_count=1207)
    assert manifest["sample_count"] == 1207
    for filename in FILENAMES.values():
        assert len(read_jsonl(output / filename)) == 1207
    assert manifest["samples"][-1]["doc_name"] == "case-1206"


@pytest.mark.parametrize("problem", ["json", "schema", "duplicate", "partial", "count", "empty"])
def test_invalid_input_preserves_source_and_previous_outputs(tmp_path, problem):
    source, output = tmp_path / "full-v2.jsonl", tmp_path / "tasks"
    output.mkdir()
    targets = [output / filename for filename in [*FILENAMES.values(), "manifest.json"]]
    for target in targets:
        target.write_bytes(b"previous output")
    row = make_record()
    if problem == "schema":
        row["issues"] = []
    if problem == "partial":
        row["data_status"] = "partial"
    write_source(source, [] if problem == "empty" else [row, row] if problem == "duplicate" else [row])
    if problem == "json":
        with source.open("ab") as handle:
            handle.write(b"{broken json}\n")
    before = source.read_bytes()
    with pytest.raises(ValueError):
        export(source, output, expected_count=1207 if problem == "count" else None)
    assert source.read_bytes() == before
    assert all(target.read_bytes() == b"previous output" for target in targets)
    assert not list(output.glob(".task-inputs-*"))


def test_law_fallback_uses_context_text_before_exact_text(tmp_path):
    row = make_record()
    row["laws"][1]["law_content"]["context_text"] = "Nội dung context được ưu tiên.\n"
    source, output = tmp_path / "source.jsonl", tmp_path / "tasks"
    write_source(source, [row])
    export(source, output)
    app = read_jsonl(output / FILENAMES["application"])[0]
    assert app["input"]["rules"][-1]["context_text"] == "Nội dung context được ưu tiên.\n"


def test_shell_command_uses_external_python_and_does_not_change_pipeline_hash(tmp_path):
    source, output = tmp_path / "source.jsonl", tmp_path / "tasks"
    write_source(source, [make_record()])
    before = implementation_hash()
    environment = {**os.environ, "PROJECT_ROOT": str(ROOT), "TASK_SOURCE_JSONL": str(source), "TASK_INPUT_DIR": str(output),
                   "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]}
    result = subprocess.run(["bash", str(ROOT / "scripts/create_task_inputs.sh"), "--expected-count", "1"],
                            env=environment, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["samples"] == 1
    assert all((output / filename).is_file() for filename in FILENAMES.values())
    assert implementation_hash() == before
