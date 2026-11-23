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

from tqdm import tqdm

from .api import APIClient, APIError, IncompleteResponse
from .data import DatasetError, canonical_hash, read_dataset, strict_json
from .pipeline import Pipeline, bounded_map, index_records
from .prompts import prompt_text
from .rubric_quality import (MAX_REVISIONS, QUALITY_SCHEMA, blocked_additions, candidate_rank, check_review,
                             citation_findings, needs_revision)
from .rubric_taxonomy import TAXONOMY, canonical_category
from .rubrics import (RUBRIC_PROTOCOL, criteria_for, load_schema, normalized, raw_rubric_sample, require_schema,
                      validate_rubric_record)
from .storage import append_jsonl, atomic_json, file_hash, implementation_hash, now, read_jsonl
from .thinking import resolve_thinking

log = logging.getLogger(__name__)
GENERATION_SCHEMA = load_schema("rubric_generation")
RUBRIC_RUN_ARTIFACT_VERSION = 2
REVISION_SCHEMA = {
    **GENERATION_SCHEMA,
    "properties": {**GENERATION_SCHEMA["properties"], "revision_notes": {
        "type": "array", "items": {"type": "object", "additionalProperties": False,
                                  "properties": {"issue_index": {"type": "integer", "minimum": 1},
                                                 "action": {"enum": ["revised", "disagreed"]},
                                                 "explanation": {"type": "string", "minLength": 1}},
                                  "required": ["action", "explanation"]}}},
    "required": [*GENERATION_SCHEMA["required"], "revision_notes"],
}


CATEGORY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"category": {"enum": list(TAXONOMY)}, "confidence": {"enum": ["clear", "uncertain"]},
                   "reason": {"type": "string", "minLength": 1}},
    "required": ["category", "confidence", "reason"],
}


def structured_messages(prompt, payload, schema, repair):
    system = prompt_text(prompt)
    if prompt == "rubric_revise":
        system = prompt_text("rubric_generate") + "\n" + system
    system += "\nJSON SCHEMA:\n" + json.dumps(schema, ensure_ascii=False)
    if repair:
        system += "\nKết quả trước chưa đúng cấu trúc: " + repair + ". Trả lại JSON đầy đủ theo schema."
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)}]


def parse_response(content, schema, stage):
    content = content.strip().removeprefix("\ufeff").strip()
    fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", content, flags=re.IGNORECASE | re.DOTALL)
    result = strict_json(fenced[1] if fenced else content)
    require_schema(schema, result, stage)
    return result


def build_issue(source, index, text):
    criteria, seen, notes = [], set(), list(source.get("warnings", []))
    for raw_criterion in source["criteria"]:
        description = raw_criterion["description"].strip()
        reasoning = raw_criterion["reference_reasoning"].strip()
        if not description or not reasoning:
            raise ValueError(f"Issue {index} contains a blank criterion")
        key = normalized(description), normalized(reasoning)
        if key in seen:
            notes.append("Tiêu chí trùng lặp.")
            continue
        seen.add(key)
        criteria.append({"criterion_id": f"I{index}_C{len(criteria) + 1}",
                         "description": description, "reference_reasoning": reasoning})
    if not criteria:
        reason = (source.get("missing_reference_reason") or "").strip()
        notes.append(f"Thiếu tiêu chí có đủ căn cứ tham chiếu. {reason}".strip())
    verdict = (source.get("reference_verdict") or "").strip() or None
    return {"issue_index": index, "issue_text": text, "criteria": criteria, "reference_verdict": verdict}, notes


def normalize_candidate(sample, value):
    indexed = {row["issue_index"]: row for row in value["issues"]}
    if len(indexed) != len(value["issues"]) or indexed.keys() != set(range(1, len(sample.issues) + 1)):
        raise ValueError("Rubric response must contain each gold issue_index exactly once")
    issues, added = [], []
    warnings = [{"reason": note} for note in value.get("warnings", []) if note.strip()]
    texts = list(sample.issues)
    known = {normalized(text) for text in texts}
    for index, text in enumerate(texts, 1):
        issue, notes = build_issue(indexed[index], index, text)
        issues.append(issue)
        warnings.extend({"issue_index": index, "reason": note} for note in notes if note.strip())
    for extra in value.get("added_issues", []):
        text, reason = extra["issue_text"].strip(), extra["reason"].strip()
        if not text or not reason or normalized(text) in known:
            warnings.append({"issue_text": text, "reason": "Bỏ issue bổ sung trùng hoặc thiếu nội dung."})
            continue
        index = len(issues) + len(added) + 1
        try:
            issue, notes = build_issue(extra, index, text)
            if not issue["criteria"]:
                raise ValueError("Bỏ issue bổ sung thiếu tiêu chí có căn cứ.")
        except ValueError as exc:
            warnings.append({"issue_text": text, "reason": str(exc)})
            continue
        known.add(normalized(text))
        added.append({**issue, "reason": reason})
        warnings.extend({"issue_index": index, "reason": note} for note in notes if note.strip())
    return {"issues": issues, "added_issues": added, "warnings": warnings,
            "revision_notes": value.get("revision_notes", [])}


def materialize_final(sample, category, candidate, assessment, findings):
    issues, published, omitted = list(candidate["issues"]), [], []
    blocked = blocked_additions(assessment, findings)
    indices = {issue["issue_index"]: issue["issue_index"] for issue in issues}
    for extra in candidate["added_issues"]:
        original_index = extra["issue_index"]
        if assessment is None or original_index in blocked:
            omitted.append({"issue_index": original_index, "issue_text": extra["issue_text"]})
            continue
        index = len(issues) + 1
        indices[original_index] = index
        issue = {key: value for key, value in extra.items() if key != "reason"}
        issue["issue_index"] = index
        issue["criteria"] = [{**c, "criterion_id": f"I{index}_C{i}"} for i, c in enumerate(extra["criteria"], 1)]
        issues.append(issue)
        published.append({"issue_index": index, "issue_text": issue["issue_text"], "reason": extra["reason"],
                          "reference_reasoning": [c["reference_reasoning"] for c in issue["criteria"]]})
    warnings = []
    for item in candidate["warnings"]:
        index = item.get("issue_index")
        if index is None:
            warnings.append(item)
        elif index in indices:
            warnings.append({**item, "issue_index": indices[index]})
    record = {key: value for key, value in sample.raw.items() if key != "issues"}
    record["category"] = category
    record["input"] = {**record["input"], "issues": [issue["issue_text"] for issue in issues]}
    record["rubrics"] = {"version": RUBRIC_PROTOCOL, "issues": issues}
    validate_rubric_record(record)
    return record, published, omitted, warnings


async def request_structured(config, sample, client, stage, schema, payload, convert, calls, errors):
    repair = None
    retries = config.rubric_quality_json_retries if client.role == "rubric_quality_judge" else config.rubric_json_retries
    for attempt in range(retries + 1):
        try:
            messages = structured_messages(stage, payload, schema, repair)
            response = await client.complete(messages, doc_name=sample.doc_name, schema=schema)
        except IncompleteResponse as exc:
            calls.append({"role": client.role, "stage": stage, **exc.metadata})
            repair = str(exc)
        except APIError as exc:
            calls.append({"role": client.role, "stage": stage, **exc.metadata})
            raise APIError(str(exc), {"calls": calls, "validation_errors": errors}) from None
        else:
            calls.append({"role": client.role, "stage": stage, **response.metadata})
            try:
                return convert(parse_response(response.content, schema, stage))
            except ValueError as exc:
                repair = str(exc)
        errors.append(f"{stage}: {repair}")
        log.warning("Rubric output invalid role=%s doc_name=%s stage=%s attempt=%d error=%s",
                    client.role, sample.doc_name, stage, attempt + 1, repair)
        if attempt < retries and config.retry_backoff:
            await asyncio.sleep(min(60.0, config.retry_backoff * 2 ** attempt))
    raise APIError(f"{stage} exhausted structured output retries: {repair}",
                   {"calls": calls, "validation_errors": errors})


async def build_sample(config, sample, generator, quality_judge, progress_path):
    identity = sample.identity()
    progress = {"identity": identity, "steps": {}}
    if progress_path.exists():
        progress = strict_json(progress_path.read_text(encoding="utf-8"))
        if progress["identity"] != identity:
            raise ValueError("Rubric progress identity mismatch")
    steps = progress["steps"]

    def metadata():
        return {"calls": [call for step in steps.values() for call in step["calls"]],
                "validation_errors": [error for step in steps.values() for error in step["validation_errors"]]}

    async def step(key, client, prompt, schema, payload, convert, *, allow_failure=False):
        if key in steps:
            return steps[key]
        calls, errors = [], []
        result, error = None, None
        try:
            result = await request_structured(config, sample, client, prompt, schema, payload, convert, calls, errors)
        except APIError as exc:
            if not allow_failure:
                previous = metadata()
                raise APIError(str(exc), {"calls": previous["calls"] + calls,
                                          "validation_errors": previous["validation_errors"] + errors}) from None
            error = str(exc)
            log.warning("Rubric stage exhausted role=%s stage=%s doc_name=%s error=%s",
                        client.role, key, sample.doc_name, error)
        steps[key] = {"result": result, "error": error, "calls": calls, "validation_errors": errors}
        atomic_json(progress_path, progress)
        return steps[key]

    original = sample.raw["category"]
    category, correction, warnings = original, None, []
    payload = {"expected_answer": sample.raw["expected_answer"], "facts_clean": sample.raw["input"]["facts_clean"],
               "rules": sample.raw["input"]["rules"],
               "gold_issues": [{"issue_index": i, "issue_text": text} for i, text in enumerate(sample.issues, 1)]}
    category_step = await step(
        "category", generator, "rubric_category", CATEGORY_SCHEMA,
        {**payload, "original_category": original, "allowed_categories": list(TAXONOMY)},
        lambda value: value, allow_failure=True)
    decision = category_step["result"]
    if decision is None:
        warnings.append({"reason": f"Giữ category gốc vì bước kiểm tra gặp lỗi: {category_step['error']}"})
    else:
        reason = decision["reason"].strip()
        if decision["confidence"] == "uncertain" or not reason:
            warnings.append({"reason": "Giữ category gốc vì chưa đủ căn cứ. " + reason})
        elif decision["category"] != canonical_category(original):
            category = decision["category"]
            correction = {"original_category": original, "corrected_category": category, "reason": reason}

    sources = {**payload, "category": canonical_category(category)}
    gen_payload = {**sources, "taxonomy": TAXONOMY[canonical_category(category)]}
    rounds, candidates, technical_errors = [], [], []
    feedback, previous_candidate = None, None
    for index in range(MAX_REVISIONS + 1):
        revision = index > 0
        gen_step = await step(
            f"generate_{index}", generator, "rubric_generate" if not revision else "rubric_revise",
            GENERATION_SCHEMA if not revision else REVISION_SCHEMA,
            gen_payload if not revision else {**gen_payload, "candidate": previous_candidate, "feedback": feedback},
            lambda value: normalize_candidate(sample, value), allow_failure=revision)
        candidate = gen_step["result"]
        round_info = {"round": index, "revision_notes": candidate["revision_notes"] if candidate else [],
                      "assessment": None, "citation_findings": [], "added_issues": [],
                      "generator": {k: v for k, v in gen_step.items() if k != "result"}}
        rounds.append(round_info)
        if candidate is None:
            technical_errors.append({"round": index, "role": generator.role, "error": gen_step["error"]})
            break
        judge_candidate = {key: candidate[key] for key in ("issues", "added_issues")}
        findings = citation_findings(judge_candidate)
        qa_step = await step(
            f"review_{index}", quality_judge, "rubric_quality", QUALITY_SCHEMA,
            {**sources, "candidate": judge_candidate}, lambda value: check_review(value, judge_candidate),
            allow_failure=True)
        review = qa_step["result"]
        round_info.update(assessment=review, citation_findings=findings,
                          added_issues=[{"issue_index": item["issue_index"], "issue_text": item["issue_text"]}
                                        for item in candidate["added_issues"]],
                          quality_judge={k: v for k, v in qa_step.items() if k != "result"})
        candidates.append((round_info, candidate))
        if review is None:
            technical_errors.append({"round": index, "role": quality_judge.role, "error": qa_step["error"]})
            break
        if not needs_revision(review, findings):
            break
        previous_candidate = judge_candidate
        feedback = {"assessment": review, "citation_findings": findings}

    best, candidate = max(candidates, key=lambda pair: candidate_rank(pair[0]))
    assessment, findings = best["assessment"], best["citation_findings"]
    record, added, omitted, rubric_warnings = materialize_final(sample, category, candidate, assessment, findings)
    unresolved = {"rubric_problems": assessment["rubric_problems"] if assessment else [],
                  "added_issue_reviews": [item for item in assessment["added_issue_reviews"] if item["decision"] != "keep"]
                  if assessment else [], "citation_findings": findings, "omitted_added_issues": omitted,
                  "technical_errors": technical_errors}
    quality_warning = bool(assessment is None or not assessment["approved"] or
                           needs_revision(assessment, findings) or technical_errors)
    quality = {"feedback_rounds": len(rounds) - 1, "selected_round": best["round"], "rounds": rounds,
               "final_quality_scores": assessment["rubric_scores"] if assessment else None,
               "unresolved": unresolved, "quality_warning": quality_warning,
               "category_check": {key: value for key, value in category_step.items() if key != "result"}}
    if quality_warning:
        log.warning("Rubric finalized with QA warning doc_name=%s feedback_rounds=%d selected_round=%d omitted_additions=%d",
                    sample.doc_name, quality["feedback_rounds"], best["round"], len(omitted))
    return {**identity, "category": category, "category_correction": correction, "dataset": record,
            "dataset_sha256": canonical_hash(record), "added_issues": added, "warnings": warnings + rubric_warnings,
            "quality": quality, "model": config.rubric_model, "metadata": {"completed_at": now(), **metadata()}}


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
        if (previous.get("rubric_quality_base_url") or previous["rubric_base_url"]) != snapshot["rubric_quality_base_url"]:
            raise ValueError("Seed run must match the rubric Quality Judge URL")
        path = seed / "rubric_records.jsonl"
        return read_jsonl(path), file_hash(path)


def spotting_row(record):
    payload = record["input"]
    return {"input": payload["facts_clean"], "expected_answer": payload["issues"]}


def output_rows(row):
    identity = {"doc_name": row["doc_name"], "category": row["category"]}
    correction = row["category_correction"]
    return [[row["dataset"]], [spotting_row(row["dataset"])],
            *([{**identity, **item} for item in row[key]] for key in ("added_issues", "warnings")),
            [{**identity, **correction}] if correction else [],
            [{**identity, **row["quality"]}]]


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
    requests = {call["request_id"]: call["role"] for row in selected + failures for call in row["metadata"].get("calls", [])}
    return {"selected": len(samples), "generated": len(selected), "completed": len(selected), "failed": len(failed),
            "pending": len(samples) - len(selected) - len(failed),
            "issues": len(issues), "criteria": sum(len(criteria_for(issue)) for issue in issues),
            "added_issues": sum(len(row["added_issues"]) for row in selected),
            "issues_without_criteria": sum(not criteria_for(issue) for issue in issues),
            "category_corrections": sum(row["category_correction"] is not None for row in selected),
            "samples_with_warnings": sum(bool(row["warnings"]) for row in selected),
            "quality_warnings": sum(row["quality"]["quality_warning"] for row in selected),
            "feedback_rounds": sum(row["quality"]["feedback_rounds"] for row in selected),
            "api_calls": len(requests), "api_calls_by_role": dict(Counter(requests.values()))}


class RubricPipeline(Pipeline):
    allowed_modes = {"build-rubrics"}

    def __init__(self, config, *, rubric_transport=None, quality_transport=None):
        super().__init__(config)
        self.rubric_transport = rubric_transport
        self.quality_transport = quality_transport
        self.completed = {}

    def write_summary(self, store, summary):
        atomic_json(store.path / "rubric_generation_summary.json", summary)
        store.summary(summary)

    def report(self, store, dataset, status, skipped, paths, *, fatal_error=None):
        rows = list(self.completed.values())
        by_category = {}
        for sample in dataset.samples:
            category = self.completed.get(sample.doc_name, sample.raw)["category"]
            by_category.setdefault(canonical_category(category), []).append(sample)
        counts = rubric_counts(dataset.samples, rows, self.errors)
        summary = {"run_id": self.config.run_id, "status": status, "updated_at": now(),
                   "counts": {"input": dataset.input_count, "skipped": skipped, **counts},
                   "by_category": {cat: rubric_counts(samples, rows, self.errors) for cat, samples in by_category.items()},
                   "missing_doc_names": [s.doc_name for s in dataset.samples if s.doc_name not in self.completed],
                   "outputs": {"application": str(paths[0]), "issue_spotting": str(paths[1]),
                               "added_issues": str(paths[2]), "warnings": str(paths[3]),
                               "category_corrections": str(paths[4]), "quality_reviews": str(paths[5])}}
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
        paths = [*targets, store.path / "added_issues.jsonl", store.path / "rubric_warnings.jsonl",
                 store.path / "category_corrections.jsonl", store.path / "rubric_quality_reviews.jsonl"]
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
        snapshot = {**config.snapshot(), "pipeline_kind": "rubric_generation", "artifact_version": RUBRIC_RUN_ARTIFACT_VERSION,
                    "rubric_protocol": RUBRIC_PROTOCOL, "input_sha256": dataset.sha256,
                    "implementation_sha256": implementation_hash(),
                    "rubric_thinking": resolve_thinking(config.rubric_thinking_mode)}
        log.info("Rubric start run_id=%s model=%s thinking=%s count=%d batch_size=%d generator_concurrency=%d quality_concurrency=%d",
                 config.run_id, config.rubric_model, snapshot["rubric_thinking"], len(dataset.samples),
                 config.batch_size, config.rubric_concurrency, config.rubric_quality_concurrency)
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
        progress_dir = store.path / ".rubric_progress"
        progress_dir.mkdir(exist_ok=True)
        generator = APIClient(config, "rubric_generator", self.rubric_transport)
        quality_judge = APIClient(config, "rubric_quality_judge", self.quality_transport)

        async def process(sample):
            try:
                result = await build_sample(config, sample, generator, quality_judge,
                                            progress_dir / f"{sample.input_index}.json")
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

                workers = config.rubric_concurrency + config.rubric_quality_concurrency
                for start in range(0, len(pending), config.batch_size):
                    await bounded_map(pending[start:start + config.batch_size], workers, process, finished)
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
            await asyncio.gather(generator.close(), quality_judge.close())
