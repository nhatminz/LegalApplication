"""Build a paired Application and Issue Spotting dataset using the shared run machinery."""
from __future__ import annotations

import asyncio
import fcntl
import json
import logging
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from tqdm import tqdm

from .api import APIClient, APIError, IncompleteResponse
from .data import DatasetError, canonical_hash, read_dataset, strict_json, text_field
from .pipeline import Pipeline, bounded_map, index_records
from .prompts import prompt_text
from .rubric_taxonomy import CIVIL_ANCHOR, PROCEDURAL_ANCHORS, TAXONOMY, canonical_category
from .rubrics import (AUDIT_SCHEMA, DECOMPOSITION_SCHEMA, PLAN_SCHEMA, criteria_for, issue_blocked, materialize,
                      raw_rubric_sample, require_schema, source_segments, validate_decomposition)
from .storage import atomic_json, atomic_jsonl, file_hash, implementation_hash, now
from .thinking import resolve_thinking

log = logging.getLogger(__name__)


def stage_messages(stage, schema, payload, repair):
    system = prompt_text(f"rubric_{stage}") + "\nJSON SCHEMA:\n" + json.dumps(schema, ensure_ascii=False)
    if repair:
        system += "\nKết quả trước bị từ chối. Đọc lại nguồn và sửa phát hiện sau: " + repair
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)}]


async def build_sample(config, sample, client) -> dict:
    raw = sample.raw
    category = canonical_category(raw["category"])
    source = source_segments(raw["expected_answer"], "S")
    common = {"category": category, "expected_answer_segments": source, "taxonomy": TAXONOMY[category],
              "analysis_reference": PROCEDURAL_ANCHORS.get(category, CIVIL_ANCHOR),
              "instance_level": raw.get("instance_level"), "code": raw.get("code")}
    context = {**common, **raw["input"], "facts_clean_segments": source_segments(raw["input"]["facts_clean"], "F")}
    calls, errors = [], []
    repair = None

    async def request(stage, schema, payload):
        try:
            response = await client.complete(stage_messages(stage, schema, payload, repair), doc_name=sample.doc_name, schema=schema)
        except APIError as exc:
            calls.append({"stage": stage, **exc.metadata})
            raise
        calls.append({"stage": stage, **response.metadata})
        value = strict_json(response.content)
        require_schema(schema, value, stage)
        return value

    for attempt in range(config.rubric_json_retries + 1):
        try:
            decomposition = await request("decompose", DECOMPOSITION_SCHEMA, common)
            validate_decomposition(decomposition, source)
            plan = await request("map", PLAN_SCHEMA, {**context, "decomposition": decomposition})
            record, audit = materialize(sample, decomposition, plan)
            decision = await request("audit", AUDIT_SCHEMA, {**context, "decomposition": decomposition, "plan": plan})
            validate_audit(decision)
            return {**sample.identity(), "category": raw["category"], "dataset": record, "audit": audit,
                    "decomposition": decomposition, "plan": plan, "semantic_audit": decision,
                    "model": config.rubric_model, "dataset_sha256": canonical_hash(record),
                    "metadata": {"completed_at": now(), "calls": calls, "validation_errors": errors,
                                 "attempts": attempt + 1}}
        except (ValueError, IncompleteResponse) as exc:
            repair = str(exc)[:2500]
            errors.append(repair)
            log.warning("Rubric validation failed doc_name=%s attempt=%d error=%s", sample.doc_name, attempt + 1, repair)
            if attempt == config.rubric_json_retries:
                raise APIError("Rubric validation exhausted: " + repair,
                               {"calls": calls, "validation_errors": errors, "attempts": attempt + 1}) from None
            if config.retry_backoff:
                await asyncio.sleep(min(60, config.retry_backoff * 2 ** attempt))
        except APIError as exc:
            raise APIError(str(exc), {"calls": calls, "validation_errors": errors, "attempts": attempt + 1}) from None
    raise AssertionError("unreachable")


def validate_audit(decision):
    require_schema(AUDIT_SCHEMA, decision, "semantic audit")
    for problem in decision["problems"]:
        text_field(problem, "audit problem")
    if decision["approved"] != (not decision["problems"]):
        raise ValueError("Semantic audit approval and problems are inconsistent")
    if not decision["approved"]:
        raise ValueError("Semantic audit rejected: " + " | ".join(decision["problems"]))


def rubric_counts(samples, rows, errors):
    names = {s.doc_name for s in samples}
    selected_rows = [row for row in rows if row["doc_name"] in names]
    selected_errors = [row for row in errors if row["doc_name"] in names]
    issues = [issue for row in selected_rows for issue in row["dataset"]["rubrics"]["issues"]]
    criteria = [c for issue in issues for c in criteria_for(issue)]
    added = sum(len(row["audit"]["added_issues"]) for row in selected_rows)
    mapped = sum(len(row["audit"]["mapping"]) for row in selected_rows)
    excluded = sum(len(row["audit"]["excluded_units"]) for row in selected_rows)
    units = sum(len(row["audit"]["source_units"]) for row in selected_rows)
    unmapped = units - mapped - excluded
    failed = {row["doc_name"] for row in selected_errors} - {row["doc_name"] for row in selected_rows}
    return {"selected": len(samples), "completed": len(selected_rows), "failed": len(failed),
            "pending": len(samples) - len(selected_rows) - len(failed),
            "original_issues": sum(len(sample.issues) for sample in samples), "added_issues": added,
            "samples_with_added_issues": sum(bool(row["audit"]["added_issues"]) for row in selected_rows),
            "atomic_units": units, "mapped_units": mapped, "excluded_units": excluded, "unmapped_units": unmapped,
            "rubric_coverage": mapped / (mapped + unmapped) if mapped + unmapped else None,
            "criteria": len(criteria), "criteria_needing_review": sum(c["needs_review"] for c in criteria),
            "issues_needing_review": sum(i["issue_review_reason"] is not None or i["outcome_review_reason"] is not None or
                                        any(c["needs_review"] for c in criteria_for(i)) for i in issues),
            "retry_attempts": sum(max(0, row["metadata"].get("attempts", 1) - 1) for row in selected_rows + selected_errors),
            "retried_samples": len({row["doc_name"] for row in selected_rows + selected_errors
                                    if row["metadata"].get("validation_errors")}),
            "api_calls": sum(len(row["metadata"].get("calls", [])) for row in selected_rows + selected_errors)}


class RubricPipeline(Pipeline):
    allowed_modes = {"build-rubrics"}

    def __init__(self, config, *, rubric_transport=None):
        super().__init__(config)
        self.rubric_transport = rubric_transport
        self.completed = {}

    def report(self, store, dataset, status, skipped):
        rows = [self.completed[s.doc_name] for s in dataset.samples if s.doc_name in self.completed]
        counts = rubric_counts(dataset.samples, rows, self.errors)
        by_category = {}
        for sample in dataset.samples:
            by_category.setdefault(canonical_category(sample.raw["category"]), []).append(sample)
        coverage = {"scope": "semantically_approved_samples", "assessed_samples": len(rows),
                    "unassessed_samples": len(dataset.samples) - len(rows), "by_category": {
                        cat: rubric_counts(samples, rows, self.errors) for cat, samples in by_category.items()}}
        def event_rows(key):
            for row in rows:
                for item in row["audit"][key]:
                    yield {"doc_name": row["doc_name"], "category": row["category"], **item}

        def reviews():
            for row in rows:
                for issue in row["dataset"]["rubrics"]["issues"]:
                    reasons = [issue[k] for k in ["issue_review_reason", "outcome_review_reason"] if issue[k]]
                    criteria = [c for c in criteria_for(issue) if c["needs_review"]]
                    if reasons or criteria:
                        yield {"doc_name": row["doc_name"], "category": row["category"], "issue_index": issue["issue_index"],
                               "issue_text": issue["issue_text"], "reasons": reasons,
                               "excluded_from_application_score": issue_blocked(issue) or not any(c["scorable"] for c in criteria_for(issue)),
                               "criteria": criteria}
            failed_docs = set()
            for error in reversed(self.errors):
                doc = error["doc_name"]
                if doc not in self.completed and doc not in failed_docs:
                    failed_docs.add(doc)
                    yield {**error, "category": next(s.raw["category"] for s in dataset.samples if s.doc_name == doc),
                           "review_status": "rubric_generation_failed"}

        atomic_jsonl(store.path / "added_issues.jsonl", event_rows("added_issues"))
        atomic_jsonl(store.path / "excluded_units.jsonl", event_rows("excluded_units"))
        atomic_jsonl(store.path / "rubric_review.jsonl", reviews())
        atomic_jsonl(store.path / "source_units.jsonl",
                     ({"doc_name": row["doc_name"], "category": row["category"], **row["audit"]} for row in rows))
        atomic_json(store.path / "rubric_coverage_report.json", {**coverage, **counts})
        missing = [sample.doc_name for sample in dataset.samples if sample.doc_name not in self.completed]
        summary = {"run_id": self.config.run_id, "status": status, "updated_at": now(),
                   "counts": {"input": dataset.input_count, "skipped": skipped, **counts},
                   "by_category": coverage["by_category"], "missing_doc_names": missing,
                   "published": False}
        self.write_summary(store, summary)
        return summary, rows

    @staticmethod
    def write_summary(store, summary):
        atomic_json(store.path / "rubric_generation_summary.json", summary)
        store.summary(summary)

    async def _run(self, store):
        config = self.config
        application = config.final_application_jsonl.resolve()
        spotting = config.final_issue_spotting_jsonl.resolve()
        manifest_path = application.with_name(application.name + ".manifest.json")
        targets = [application, spotting, manifest_path]
        raw_path = config.input_jsonl.resolve()
        if (len({path.resolve() for path in targets}) != len(targets) or
                any(path.exists() and raw_path.samefile(path) for path in targets) or
                raw_path in [path.resolve() for path in targets] or any(path.is_relative_to(store.path.resolve()) for path in targets)):
            raise ValueError("Dataset outputs must be distinct from raw input and the run journal")
        try:
            dataset = read_dataset(config.input_jsonl, config.limit, parser=raw_rubric_sample)
        except DatasetError as exc:
            summary = {"run_id": config.run_id, "status": "input_validation_failed", "published": False,
                       "counts": {"input": exc.input_count, "valid": exc.valid_count, "invalid": len(exc.errors), "failed": len(exc.errors)},
                       "invalid_by_category": dict(Counter(e.get("category", "unknown") for e in exc.errors)),
                       "input_validation_errors": exc.errors}
            self.write_summary(store, summary)
            raise
        if config.expected_count is None and len(dataset.samples) == dataset.input_count:
            raise ValueError("Set EXPECTED_COUNT to verify the raw dataset size before publishing")
        if config.expected_count is not None and dataset.input_count != config.expected_count:
            summary = {"run_id": config.run_id, "status": "input_count_mismatch", "published": False,
                       "expected_count": config.expected_count, "counts": {"input": dataset.input_count, "failed": 0}}
            self.write_summary(store, summary)
            raise ValueError(f"EXPECTED_COUNT={config.expected_count}, found {dataset.input_count} raw samples")
        snapshot = {**config.snapshot(), "pipeline_kind": "rubric_generation", "artifact_version": 1,
                    "input_sha256": dataset.sha256, "implementation_sha256": implementation_hash(),
                    "rubric_thinking": resolve_thinking(config.rubric_thinking_mode)}
        store.configure(snapshot, artifact_names=["rubric_records.jsonl", "errors.jsonl"])
        samples = {sample.doc_name: sample for sample in dataset.samples}
        self.completed = index_records(store.load("rubric_records.jsonl"), samples)
        self.errors = store.load("errors.jsonl")
        for row in self.errors + list(self.completed.values()):
            sample = samples.get(row.get("doc_name"))
            if sample is None or any(row.get(k) != v for k, v in sample.identity().items()):
                raise ValueError("Rubric journal identity mismatch")
        for doc, row in self.completed.items():
            expected, audit = materialize(samples[doc], row["decomposition"], row["plan"])
            validate_audit(row["semantic_audit"])
            if row["dataset"] != expected or row["audit"] != audit or row["model"] != config.rubric_model or (
                    row["dataset_sha256"] != canonical_hash(expected) or row["category"] != expected["category"]):
                raise ValueError(f"Stored rubric integrity check failed: {doc}")
        skipped = len(self.completed)
        pending = [sample for sample in dataset.samples if sample.doc_name not in self.completed]
        client = APIClient(config, "rubric", self.rubric_transport)

        async def process(sample):
            try:
                result = await build_sample(config, sample, client)
            except APIError as exc:
                self.failure(store, sample, "rubric_generation", exc)
                return
            store.append("rubric_records.jsonl", result)
            self.completed[sample.doc_name] = result

        try:
            with tqdm(total=len(dataset.samples), initial=skipped, desc="Build rubrics", unit="sample",
                      dynamic_ncols=True, disable=None) as progress:
                def finished(_sample):
                    progress.set_postfix(completed=len(self.completed), error_events=len(self.errors), refresh=False)
                    progress.update(1)

                for start in range(0, len(pending), config.batch_size):
                    await bounded_map(pending[start:start + config.batch_size], config.rubric_concurrency, process, finished)
            if len(self.completed) != len(dataset.samples):
                summary, _ = self.report(store, dataset, "completed_with_errors", skipped)
                return summary
            if len(dataset.samples) != dataset.input_count:
                summary, rows = self.report(store, dataset, "subset_completed", skipped)
                atomic_jsonl(store.path / "rule_application_subset.jsonl", (row["dataset"] for row in rows))
                atomic_jsonl(store.path / "issue_spotting_subset.jsonl", (spotting_row(row) for row in rows))
                return summary
            if file_hash(config.input_jsonl) != dataset.sha256:
                raise ValueError("Raw input changed during rubric generation")
            summary, rows = self.report(store, dataset, "ready_to_publish", skipped)
            publish(targets, rows, snapshot, config.run_id)
            summary.update(status="completed_with_review" if summary["counts"]["issues_needing_review"] else "completed",
                           published=True, outputs=[str(path) for path in targets])
            self.write_summary(store, summary)
            return summary
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "failed"
            summary, _ = self.report(store, dataset, status, skipped)
            summary["fatal_error"] = {"type": type(exc).__name__, "error": str(exc)}
            self.write_summary(store, summary)
            raise
        finally:
            await client.close()


def spotting_row(row):
    payload = row["dataset"]["input"]
    return {"input": payload["facts_clean"], "expected_answer": payload["issues"]}


def publish(targets: list[Path], rows, snapshot, run_id):
    """Stage all files and publish the checksum manifest last. Resume an identical publication."""
    application, spotting, manifest_path = targets
    with ExitStack() as stack:
        for path in sorted(targets):
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context(path.with_name(path.name + ".publish.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        staged = [path.with_name(f".{path.name}.{run_id}.staged") for path in targets]
        for path in staged:
            stack.callback(path.unlink, missing_ok=True)
        atomic_jsonl(staged[0], (row["dataset"] for row in rows))
        atomic_jsonl(staged[1], (spotting_row(row) for row in rows))
        manifest = {"version": 1, "kind": "rule_application_rubrics", "complete": True,
                    "run_id": run_id, "sample_count": len(rows), "raw_input": snapshot["input_jsonl"],
                    "raw_sha256": snapshot["input_sha256"], "implementation_sha256": snapshot["implementation_sha256"],
                    "rubric_model": snapshot["rubric_model"], "rubric_base_url": snapshot["rubric_base_url"],
                    "rubric_thinking": snapshot["rubric_thinking"],
                    "application": {"path": str(application), "sha256": file_hash(staged[0])},
                    "issue_spotting": {"path": str(spotting), "sha256": file_hash(staged[1])},
                    "samples": [{"line": i, "doc_name": row["doc_name"], "category": row["category"],
                                 "projected_sample_sha256": row["sample_sha256"], "raw_input_line": row["input_line"],
                                 "dataset_sha256": row["dataset_sha256"], "issue_count": len(row["dataset"]["input"]["issues"])}
                                for i, row in enumerate(rows, 1)]}
        atomic_json(staged[2], manifest)
        for path, temp in zip(targets, staged):
            if path.exists() and file_hash(path) != file_hash(temp):
                raise ValueError(f"Refusing to replace existing dataset: {path}")
        for path, temp in zip(targets, staged):
            if not path.exists():
                temp.rename(path)
