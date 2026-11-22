"""Generate issue rubrics and checkpoint the paired final datasets."""
from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import re
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from tqdm import tqdm

from .api import APIClient, APIError, IncompleteResponse
from .data import DatasetError, canonical_hash, read_dataset, strict_json
from .pipeline import Pipeline, bounded_map, index_records
from .prompts import prompt_text
from .rubric_taxonomy import TAXONOMY, canonical_category
from .rubrics import (RUBRIC_PROTOCOL, criteria_for, load_schema, normalized, raw_rubric_sample, require_schema,
                      validate_rubric_record)
from .storage import append_jsonl, atomic_json, file_hash, implementation_hash, now, read_jsonl
from .thinking import resolve_thinking

log = logging.getLogger(__name__)
GENERATION_SCHEMA = load_schema("rubric_generation")


def generation_messages(sample, taxonomy, repair):
    payload = {"expected_answer": sample.raw["expected_answer"], "category": canonical_category(sample.raw["category"]),
               "facts_clean": sample.raw["input"]["facts_clean"], "rules": sample.raw["input"]["rules"],
               "gold_issues": [{"issue_index": i, "issue_text": text} for i, text in enumerate(sample.issues, 1)],
               "taxonomy": taxonomy}
    system = prompt_text("rubric_generate") + "\nJSON SCHEMA:\n" + json.dumps(GENERATION_SCHEMA, ensure_ascii=False)
    if repair:
        system += "\nKết quả trước chưa đúng cấu trúc: " + repair + ". Trả lại đầy đủ JSON và mọi gold issue."
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)}]


def parse_generation(content):
    content = content.strip().removeprefix("\ufeff").strip()
    fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", content, flags=re.IGNORECASE | re.DOTALL)
    result = strict_json(fenced[1] if fenced else content)
    require_schema(GENERATION_SCHEMA, result, "Rubric response")
    return result


def build_issue(source, index, text, taxonomy):
    groups, other_ids, seen, notes = {}, {}, set(), list(source.get("warnings", []))
    for raw_group in source["groups"]:
        model_id = raw_group["group_id"].strip()
        group_id = model_id
        if model_id not in taxonomy:
            if model_id not in other_ids:
                other_ids[model_id] = f"other_{len(other_ids) + 1}"
                notes.append(f"Nhóm ngoài taxonomy: {model_id}")
            group_id = other_ids[model_id]
        for raw_criterion in raw_group["criteria"]:
            description = raw_criterion["description"].strip()
            reasoning = raw_criterion["reference_reasoning"].strip()
            if not description or not reasoning:
                notes.append("Tiêu chí có nội dung trống.")
                continue
            key = normalized(description), normalized(reasoning)
            if key in seen:
                notes.append("Tiêu chí trùng lặp.")
                continue
            seen.add(key)
            groups.setdefault(group_id, []).append({"description": description, "reference_reasoning": reasoning})
    if not groups:
        raise ValueError(f"Issue {index} requires at least one nonempty criterion")
    issue = {"issue_index": index, "issue_text": text, "groups": []}
    number = 0
    for group_id, criteria in groups.items():
        for criterion in criteria:
            number += 1
            criterion["criterion_id"] = f"I{index}_C{number}"
        issue["groups"].append({"group_id": group_id, "group_name": taxonomy.get(group_id, "Nội dung khác"),
                                "criteria": criteria})
    if verdict := (source.get("reference_verdict") or "").strip():
        issue["reference_verdict"] = verdict
    return issue, notes


def materialize(sample, value, taxonomy):
    indexed = {row["issue_index"]: row for row in value["issues"]}
    if len(indexed) != len(value["issues"]) or indexed.keys() != set(range(1, len(sample.issues) + 1)):
        raise ValueError("Rubric response must contain each gold issue_index exactly once")
    issues, added = [], []
    warnings = [{"reason": note} for note in value.get("warnings", []) if note.strip()]
    texts = list(sample.issues)
    known = {normalized(text) for text in texts}
    for index, text in enumerate(texts, 1):
        issue, notes = build_issue(indexed[index], index, text, taxonomy)
        issues.append(issue)
        warnings.extend({"issue_index": index, "reason": note} for note in notes if note.strip())
    for extra in value.get("added_issues", []):
        text, reason = extra["issue_text"].strip(), extra["reason"].strip()
        if not text or not reason or normalized(text) in known:
            warnings.append({"issue_text": text, "reason": "Bỏ issue bổ sung trùng hoặc thiếu nội dung."})
            continue
        index = len(issues) + 1
        try:
            issue, notes = build_issue(extra, index, text, taxonomy)
        except ValueError as exc:
            warnings.append({"issue_text": text, "reason": str(exc)})
            continue
        known.add(normalized(text))
        texts.append(text)
        issues.append(issue)
        added.append({"issue_index": index, "issue_text": text, "reason": reason,
                      "reference_reasoning": [c["reference_reasoning"] for c in criteria_for(issue)]})
        warnings.extend({"issue_index": index, "reason": note} for note in notes if note.strip())
    record = {key: value for key, value in sample.raw.items() if key != "issues"}
    record["input"] = {**record["input"], "issues": texts}
    record["rubrics"] = {"version": RUBRIC_PROTOCOL, "issues": issues}
    validate_rubric_record(record)
    return record, added, warnings


async def build_sample(config, sample, client):
    taxonomy = TAXONOMY[canonical_category(sample.raw["category"])]
    calls, errors, repair = [], [], None
    for attempt in range(config.rubric_json_retries + 1):
        try:
            response = await client.complete(generation_messages(sample, taxonomy, repair),
                                             doc_name=sample.doc_name, schema=GENERATION_SCHEMA)
        except IncompleteResponse as exc:
            calls.append(exc.metadata)
            repair = str(exc)
        except APIError as exc:
            calls.append(exc.metadata)
            raise APIError(str(exc), {"calls": calls, "validation_errors": errors}) from None
        else:
            calls.append(response.metadata)
            try:
                record, added, warnings = materialize(sample, parse_generation(response.content), taxonomy)
            except ValueError as exc:
                repair = str(exc)
            else:
                return {**sample.identity(), "category": sample.raw["category"], "dataset": record,
                        "dataset_sha256": canonical_hash(record), "added_issues": added, "warnings": warnings,
                        "model": config.rubric_model,
                        "metadata": {"completed_at": now(), "calls": calls, "validation_errors": errors}}
        errors.append(repair)
        log.warning("Rubric output invalid doc_name=%s attempt=%d error=%s", sample.doc_name, attempt + 1, repair)
        if attempt < config.rubric_json_retries and config.retry_backoff:
            await asyncio.sleep(min(60.0, config.retry_backoff * 2 ** attempt))
    raise APIError(f"Rubric output exhausted retries: {repair}", {"calls": calls, "validation_errors": errors})


def read_seed_run(config, snapshot):
    seed = config.rubric_seed_run
    if seed.resolve() == config.run_dir.resolve():
        raise ValueError("The seed run and the new run must be different")
    with (seed / ".run.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        previous = strict_json((seed / "config.json").read_text(encoding="utf-8"))
        if not isinstance(previous, dict) or previous.get("rubric_protocol") != RUBRIC_PROTOCOL:
            raise ValueError("The seed run uses an older rubric format. Start a fresh run with RESUME=0.")
        keys = ("pipeline_kind", "artifact_version", "input_sha256", "limit", "rubric_model",
                "rubric_base_url", "rubric_thinking", "rubric_response_format")
        if any(previous.get(key) != snapshot[key] for key in keys):
            raise ValueError("Seed run must match the raw input checksum, selection and rubric model settings")
        path = seed / "rubric_records.jsonl"
        return read_jsonl(path), file_hash(path)


def spotting_row(record):
    payload = record["input"]
    return {"input": payload["facts_clean"], "expected_answer": payload["issues"]}


def output_rows(row):
    return [[row["dataset"]], [spotting_row(row["dataset"])],
            *([{"doc_name": row["doc_name"], "category": row["category"], **item} for item in row[key]]
              for key in ("added_issues", "warnings"))]


def reconcile_jsonl(path, records):
    """Restore a missing suffix from the journal and discard only a torn append."""
    path.touch(exist_ok=True)
    with path.open("r+b") as handle:
        restored = False
        for record in records:
            offset = handle.tell()
            line = handle.readline()
            expected = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode()
            if line == expected:
                continue
            if line and not expected.startswith(line):
                raise ValueError(f"Output differs from saved rubric journal: {path}")
            handle.seek(offset)
            handle.truncate()
            handle.write(expected)
            restored = True
        if handle.read(1):
            raise ValueError(f"Output has extra rows outside the saved rubric journal: {path}")
        if restored:
            handle.flush()
            os.fsync(handle.fileno())


def reconcile_outputs(paths, rows):
    for index, path in enumerate(paths):
        reconcile_jsonl(path, (item for row in rows for item in output_rows(row)[index]))


def rubric_counts(samples, rows, errors):
    names = {sample.doc_name for sample in samples}
    selected = [row for row in rows if row["doc_name"] in names]
    failures = [row for row in errors if row["doc_name"] in names]
    failed = {row["doc_name"] for row in failures} - {row["doc_name"] for row in selected}
    issues = [issue for row in selected for issue in row["dataset"]["rubrics"]["issues"]]
    return {"selected": len(samples), "generated": len(selected), "completed": len(selected), "failed": len(failed),
            "pending": len(samples) - len(selected) - len(failed),
            "issues": len(issues), "criteria": sum(len(criteria_for(issue)) for issue in issues),
            "added_issues": sum(len(row["added_issues"]) for row in selected),
            "samples_with_warnings": sum(bool(row["warnings"]) for row in selected),
            "api_calls": sum(len(row["metadata"].get("calls", [])) for row in selected + failures)}


class RubricPipeline(Pipeline):
    allowed_modes = {"build-rubrics"}

    def __init__(self, config, *, rubric_transport=None):
        super().__init__(config)
        self.rubric_transport = rubric_transport
        self.completed = {}

    def write_summary(self, store, summary):
        atomic_json(store.path / "rubric_generation_summary.json", summary)
        store.summary(summary)

    def report(self, store, dataset, status, skipped, paths, *, fatal_error=None):
        rows = list(self.completed.values())
        by_category = {}
        for sample in dataset.samples:
            by_category.setdefault(canonical_category(sample.raw["category"]), []).append(sample)
        counts = rubric_counts(dataset.samples, rows, self.errors)
        summary = {"run_id": self.config.run_id, "status": status, "updated_at": now(),
                   "counts": {"input": dataset.input_count, "skipped": skipped, **counts},
                   "by_category": {cat: rubric_counts(samples, rows, self.errors) for cat, samples in by_category.items()},
                   "missing_doc_names": [s.doc_name for s in dataset.samples if s.doc_name not in self.completed],
                   "outputs": {"application": str(paths[0]), "issue_spotting": str(paths[1]),
                               "added_issues": str(paths[2]), "warnings": str(paths[3])}}
        if fatal_error is not None:
            summary["fatal_error"] = {"type": type(fatal_error).__name__, "error": str(fatal_error)}
        self.write_summary(store, summary)
        return summary

    async def _run(self, store):
        config = self.config
        try:
            dataset = read_dataset(config.input_jsonl, config.limit, parser=raw_rubric_sample)
        except DatasetError as exc:
            reasons = Counter(e["error"] for e in exc.errors)
            for reason, count in reasons.most_common(10):
                log.error("Raw rubric validation: %d row(s): %s", count, reason)
            self.write_summary(store, {"run_id": config.run_id, "status": "input_validation_failed",
                                      "counts": {"input": exc.input_count, "valid": exc.valid_count,
                                                 "invalid": len(exc.errors), "failed": len(exc.errors)},
                                      "invalid_by_reason": dict(reasons),
                                      "invalid_by_category": dict(Counter(e.get("category", "unknown") for e in exc.errors)),
                                      "input_validation_errors": exc.errors})
            raise
        if config.expected_count is not None and dataset.input_count != config.expected_count:
            self.write_summary(store, {"run_id": config.run_id, "status": "input_count_mismatch",
                                      "expected_count": config.expected_count,
                                      "counts": {"input": dataset.input_count, "failed": 0}})
            raise ValueError(f"EXPECTED_COUNT={config.expected_count}, found {dataset.input_count} raw samples")
        if config.limit == 0:
            summary = {"run_id": config.run_id, "status": "input_validated",
                       "counts": {"input": dataset.input_count, "valid": dataset.valid_count,
                                  "invalid": 0, "selected": 0, "failed": 0}, "input_sha256": dataset.sha256}
            self.write_summary(store, summary)
            return summary
        targets = [config.final_application_jsonl.resolve(), config.final_issue_spotting_jsonl.resolve()]
        raw_path = config.input_jsonl.resolve()
        if (len(set(targets)) != len(targets) or raw_path in targets
                or any(path.is_relative_to(store.path.resolve()) for path in targets)
                or any(path.exists() and raw_path.samefile(path) for path in targets)
                or (all(path.exists() for path in targets) and targets[0].samefile(targets[1]))):
            raise ValueError("Dataset outputs must be distinct from each other, raw input and the run journal")
        if targets[0].with_name(targets[0].name + ".manifest.json").exists():
            raise ValueError("Application output belongs to an existing rubric publication; choose a new file")
        paths = [*targets, store.path / "added_issues.jsonl", store.path / "rubric_warnings.jsonl"]
        with ExitStack() as stack:
            for target in sorted(targets):
                target.parent.mkdir(parents=True, exist_ok=True)
                lock = stack.enter_context(target.with_name(target.name + ".publish.lock").open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not config.resume and any(path.exists() for path in targets):
                raise ValueError("Final dataset already exists; resume its run or choose new output paths")
            return await self._generate(store, dataset, paths)

    async def _generate(self, store, dataset, paths):
        config = self.config
        snapshot = {**config.snapshot(), "pipeline_kind": "rubric_generation", "artifact_version": 1,
                    "rubric_protocol": RUBRIC_PROTOCOL, "input_sha256": dataset.sha256,
                    "implementation_sha256": implementation_hash(),
                    "rubric_thinking": resolve_thinking(config.rubric_thinking_mode)}
        log.info("Rubric start run_id=%s model=%s thinking=%s count=%d batch_size=%d concurrency=%d",
                 config.run_id, config.rubric_model, snapshot["rubric_thinking"], len(dataset.samples),
                 config.batch_size, config.rubric_concurrency)
        seed_rows = []
        if config.rubric_seed_run is not None:
            seed_rows, snapshot["rubric_seed_sha256"] = read_seed_run(config, snapshot)
        store.configure(snapshot, artifact_names=["rubric_records.jsonl", "errors.jsonl"])
        samples = {sample.doc_name: sample for sample in dataset.samples}
        self.completed = index_records(store.load("rubric_records.jsonl"), samples)
        seed = index_records(seed_rows, samples)
        self.errors = store.load("errors.jsonl")
        for row in [*self.errors, *self.completed.values(), *seed.values()]:
            sample = samples.get(row.get("doc_name"))
            if sample is None or any(row.get(key) != value for key, value in sample.identity().items()):
                raise ValueError("Rubric journal identity mismatch")
        for row in [*self.completed.values(), *seed.values()]:
            record = row["dataset"]
            validate_rubric_record(record)
            if (record["rubrics"]["version"] != RUBRIC_PROTOCOL or row["model"] != config.rubric_model
                    or record["doc_name"] != row["doc_name"] or record["category"] != row["category"]
                    or canonical_hash(record) != row["dataset_sha256"]):
                raise ValueError(f"Stored rubric integrity check failed: {row['doc_name']}")
        reconcile_outputs(paths, self.completed.values())

        def save(row):
            store.append("rubric_records.jsonl", row)
            self.completed[row["doc_name"]] = row
            for path, records in zip(paths, output_rows(row)):
                for record in records:
                    append_jsonl(path, record)

        for doc, row in seed.items():
            if doc not in self.completed:
                save(row)
            elif self.completed[doc] != row:
                raise ValueError(f"Seed rubric differs from the saved checkpoint: {doc}")
        skipped = len(self.completed)
        pending = [sample for sample in dataset.samples if sample.doc_name not in self.completed]
        client = APIClient(config, "rubric", self.rubric_transport)

        async def process(sample):
            try:
                result = await build_sample(config, sample, client)
            except APIError as exc:
                self.failure(store, sample, "rubric_generation", exc)
                return
            save(result)

        try:
            with tqdm(total=len(dataset.samples), initial=skipped, desc="Build rubrics", unit="sample",
                      dynamic_ncols=True, disable=None) as progress:
                def finished(_sample):
                    progress.set_postfix(completed=len(self.completed), error_events=len(self.errors), refresh=False)
                    progress.update(1)

                for start in range(0, len(pending), config.batch_size):
                    await bounded_map(pending[start:start + config.batch_size], config.rubric_concurrency, process, finished)
            reconcile_outputs(paths, self.completed.values())
            if file_hash(config.input_jsonl) != dataset.sha256:
                raise ValueError("Raw input changed during rubric generation")
            if len(self.completed) != len(dataset.samples):
                status = "completed_with_errors"
            else:
                status = "completed" if len(dataset.samples) == dataset.input_count else "subset_completed"
            return self.report(store, dataset, status, skipped, paths)
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "failed"
            self.report(store, dataset, status, skipped, paths, fatal_error=exc)
            raise
        finally:
            await client.close()
