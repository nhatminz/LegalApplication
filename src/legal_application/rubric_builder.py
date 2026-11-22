"""Build a paired Application and Issue Spotting dataset using the shared run machinery."""
from __future__ import annotations

import asyncio
import fcntl
import json
import logging
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path

from tqdm import tqdm

from .api import APIClient, APIError, IncompleteResponse
from .data import DatasetError, canonical_hash, read_dataset, strict_json, text_field
from .pipeline import Pipeline, bounded_map, index_records
from .prompts import prompt_text
from .rubric_quality import AUDIT_V2, INVENTORY_SCHEMA, audit_coverage, extend_inventory, validate_inventory, verified_rows
from .rubric_taxonomy import CIVIL_ANCHOR, PROCEDURAL_ANCHORS, TAXONOMY, canonical_category
from .rubrics import (AUDIT_SCHEMA, DECOMPOSITION_V2, PLAN_V2, RUBRIC_PROTOCOL, criteria_for, issue_blocked, materialize,
                      raw_rubric_sample, require_schema, resolve_verdict_ids, source_segments, validate_decomposition)
from .storage import atomic_json, atomic_jsonl, file_hash, implementation_hash, now, read_jsonl
from .thinking import resolve_thinking

log = logging.getLogger(__name__)


def stage_messages(stage, schema, payload, repair):
    system = prompt_text(f"rubric_{stage}")
    if stage in {"map", "audit"}:
        system += "\n" + prompt_text("rubric_scope")
    if repair:
        system += "\nKết quả trước bị từ chối. Đọc lại nguồn và sửa phát hiện sau: " + repair
    system += "\nJSON SCHEMA:\n" + json.dumps(schema, ensure_ascii=False)
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)}]


class AuditRejected(ValueError):
    def __init__(self, decision):
        super().__init__("Semantic audit rejected: " + " | ".join(decision["problems"]))
        self.repair_stage = decision["repair_stage"]


async def build_sample(config, sample, client) -> dict:
    raw = sample.raw
    category = canonical_category(raw["category"])
    taxonomy = TAXONOMY[category]
    source = source_segments(raw["expected_answer"], "S")
    common = {"category": category, "expected_answer_segments": source, "taxonomy": taxonomy,
              "analysis_reference": PROCEDURAL_ANCHORS.get(category, CIVIL_ANCHOR),
              "instance_level": raw.get("instance_level"), "code": raw.get("code")}
    context = {**common, **raw["input"], "facts_clean_segments": source_segments(raw["input"]["facts_clean"], "F")}

    # Constrain verdict IDs at generation time. The materializer can still resolve a
    # known U ID safely when an endpoint returns a JSON object without enforcing this enum.
    map_schema = deepcopy(PLAN_V2)
    verdict_ids = map_schema["properties"]["issues"]["items"]["properties"]["verdict_source_ids"]
    verdict_ids["items"]["enum"] = list(source)
    calls, errors, retries, drafts, proposal_attempts = [], [], [], {}, []
    stage, repair = "decompose", None
    mapping_feedback = None
    inventory, coverage, coverage_details = None, None, None
    attempt = 0

    async def request(schema, payload, *, validator_schema=None):
        if repair and stage in drafts and stage != "audit":
            payload = {**payload, "previous_attempt": drafts[stage]}
        try:
            response = await client.complete(stage_messages(stage, schema, payload, repair), doc_name=sample.doc_name, schema=schema)
        except APIError as exc:
            calls.append({"stage": stage, **exc.metadata})
            raise
        calls.append({"stage": stage, **response.metadata})
        value = strict_json(response.content)
        drafts[stage] = value
        require_schema(validator_schema or schema, value, stage)
        return value

    while True:
        try:
            if stage == "decompose":
                decomposition = await request(DECOMPOSITION_V2, common)
                units = validate_decomposition(decomposition, source, taxonomy=taxonomy)
                stage, repair = "map", mapping_feedback
                mapping_feedback = None
                # Old mapping IDs may have changed after re-decomposition.
                drafts.pop("map", None)
            if stage == "map":
                map_input = {**context, "decomposition": decomposition}
                plan = await request(map_schema, map_input, validator_schema=PLAN_V2)
                record, audit = materialize(sample, decomposition, plan, version=RUBRIC_PROTOCOL)
                pending_proposals = None
                if audit["added_issues"]:
                    pending_proposals = {"attempt": attempt + 1, "proposals": audit["added_issues"],
                                         "audit_status": "unverified"}
                    proposal_attempts.append(pending_proposals)
                stage, repair = ("inventory" if inventory is None else "audit"), None
            if stage == "inventory":
                inventory = await request(INVENTORY_SCHEMA, common)
                validate_inventory(inventory, source, taxonomy)
                stage, repair = "audit", None
            # Audit the source IDs and issue review reasons actually used in the public rubric.
            audit_plan = {**plan, "issues": [
                {**outcome, "verdict_source_ids": resolve_verdict_ids(outcome["verdict_source_ids"], units, source),
                 "issue_review_reason": issue["issue_review_reason"]}
                for outcome, issue in zip(plan["issues"], record["rubrics"]["issues"])
            ]}
            decision = await request(AUDIT_V2, {**context, "decomposition": decomposition, "plan": audit_plan,
                                               "independent_inventory": inventory})
            coverage = audit_coverage(inventory, decision, decomposition, audit, source, taxonomy)
            coverage_details = {"propositions": inventory["propositions"] + decision["additional_propositions"],
                                "dispositions": decision["coverage"]}
            if not decision["approved"]:
                inventory = extend_inventory(inventory, decision["additional_propositions"])
                if pending_proposals is not None:
                    pending_proposals.update(audit_status="requires_repair", audit_problems=decision["problems"])
            validate_audit(decision)
            if pending_proposals is not None:
                pending_proposals["audit_status"] = "audit_accepted"
            return {**sample.identity(), "category": raw["category"], "dataset": record, "audit": audit,
                    "decomposition": decomposition, "plan": plan, "semantic_audit": decision, "inventory": inventory,
                    "proposition_counts": coverage,
                    "model": config.rubric_model, "dataset_sha256": canonical_hash(record),
                    "metadata": {"completed_at": now(), "calls": calls, "validation_errors": errors,
                                 "retry_stages": retries, "attempts": attempt + 1, "proposal_attempts": proposal_attempts}}
        except (ValueError, IncompleteResponse) as exc:
            target = exc.repair_stage if isinstance(exc, AuditRejected) else stage
            repair = str(exc)
            errors.append(repair)
            log.warning("Rubric validation failed doc_name=%s stage=%s retry_stage=%s attempt=%d error=%s",
                        sample.doc_name, stage, target, attempt + 1, repair)
            if attempt == config.rubric_json_retries:
                raise APIError("Rubric validation exhausted: " + repair,
                               {"calls": calls, "validation_errors": errors, "retry_stages": retries,
                                "failed_stage": stage, "attempts": attempt + 1, "proposition_counts": coverage,
                                "proposition_audit": coverage_details, "proposal_attempts": proposal_attempts}) from None
            retries.append({"failed_stage": stage, "retry_stage": target})
            if isinstance(exc, AuditRejected) and target == "decompose":
                mapping_feedback = repair
            stage = target
            if config.retry_backoff:
                await asyncio.sleep(min(60, config.retry_backoff * 2 ** attempt))
            attempt += 1
        except APIError as exc:
            raise APIError(str(exc), {"calls": calls, "validation_errors": errors, "retry_stages": retries,
                                      "failed_stage": stage, "attempts": attempt + 1, "proposition_counts": coverage,
                                      "proposition_audit": coverage_details, "proposal_attempts": proposal_attempts}) from None


def validate_audit(decision):
    require_schema(AUDIT_SCHEMA, decision, "semantic audit")
    for problem in decision["problems"]:
        text_field(problem, "audit problem")
    if decision["approved"] != (not decision["problems"]):
        raise ValueError("Semantic audit approval and problems are inconsistent")
    if decision["approved"] and decision.get("repair_stage") is not None:
        raise ValueError("Approved semantic audit must have repair_stage=null")
    if not decision["approved"]:
        if decision.get("repair_stage") is None:
            raise ValueError("Rejected semantic audit must specify repair_stage")
        raise AuditRejected(decision)


def read_seed_run(config, snapshot):
    seed = config.rubric_seed_run
    if seed.resolve() == config.run_dir.resolve():
        raise ValueError("The seed run and the new run must be different")
    with (seed / ".run.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        previous = strict_json((seed / "config.json").read_text(encoding="utf-8"))
        if isinstance(previous, dict) and previous.get("rubric_protocol") != RUBRIC_PROTOCOL:
            raise ValueError("Seed run predates independent proposition auditing. Start a fresh generation with RESUME=0.")
        keys = ("pipeline_kind", "artifact_version", "input_sha256", "limit", "rubric_model",
                "rubric_base_url", "rubric_thinking", "rubric_response_format")
        if not isinstance(previous, dict) or any(previous.get(key) != snapshot[key] for key in keys):
            raise ValueError("Seed run must match the raw input checksum, selection and rubric model settings")
        path = seed / "rubric_records.jsonl"
        return read_jsonl(path), file_hash(path)


def rubric_counts(samples, rows, errors):
    names = {s.doc_name for s in samples}
    selected_rows = [row for row in rows if row["doc_name"] in names]
    selected_errors = [row for row in errors if row["doc_name"] in names]
    issues = [issue for row in selected_rows for issue in row["dataset"]["rubrics"]["issues"]]
    criteria = [c for issue in issues for c in criteria_for(issue)]
    additions = sum(len(row["audit"]["added_issues"]) for row in selected_rows)
    propositions = {row["doc_name"]: row["metadata"].get("proposition_counts") for row in selected_errors}
    propositions.update({row["doc_name"]: row["proposition_counts"] for row in selected_rows})
    known = [counts for counts in propositions.values() if counts is not None]
    totals = {status: sum(counts[status] for counts in known) for status in ("mapped", "excluded", "missing", "unverified")}
    denominator = totals["mapped"] + totals["missing"] + totals["unverified"]
    unassessed = len(samples) - len(known)
    failed = {row["doc_name"] for row in selected_errors} - {row["doc_name"] for row in selected_rows}
    return {"selected": len(samples), "completed": len(selected_rows), "failed": len(failed),
            "pending": len(samples) - len(selected_rows) - len(failed),
            "original_issues": sum(len(sample.issues) for sample in samples), "added_issues": additions,
            "samples_with_added_issues": sum(bool(row["audit"]["added_issues"]) for row in selected_rows),
            "atomic_units": sum(len(row["audit"]["source_units"]) for row in selected_rows),
            "atomic_propositions": sum(totals.values()), **{f"{status}_propositions": value for status, value in totals.items()},
            "unassessed_samples": unassessed, "unaudited_or_rejected_samples": len(samples) - len(selected_rows),
            "rubric_coverage": (totals["mapped"] / denominator if denominator and len(selected_rows) == len(samples)
                                and not unassessed and not totals["unverified"] else None),
            "criteria": len(criteria), "criteria_needing_review": sum(c["needs_review"] for c in criteria),
            "issues_needing_review": sum(i["issue_review_reason"] is not None or i["outcome_review_reason"] is not None or
                                        any(c["needs_review"] for c in criteria_for(i)) for i in issues),
            "issues_excluded_from_application_score": sum(issue_blocked(i) or not any(c["scorable"] for c in criteria_for(i))
                                                          for i in issues),
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
        coverage = {"scope": "independent_expected_answer_propositions",
                    "assessed_samples": len(dataset.samples) - counts["unassessed_samples"], "by_category": {
                        cat: rubric_counts(samples, rows, self.errors) for cat, samples in by_category.items()}}
        def event_rows(key):
            for row in rows:
                for item in row["audit"][key]:
                    yield {"doc_name": row["doc_name"], "category": row["category"], **item}

        def added_issues():
            for item in event_rows("added_issues"):
                yield {**item, "status": "audit_accepted"}

        def reviews():
            for item in added_issues():
                yield {**item, "review_status": "added_issue"}
            for row in rows:
                for issue in row["dataset"]["rubrics"]["issues"]:
                    reasons = [issue[k] for k in ["issue_review_reason", "outcome_review_reason"] if issue[k]]
                    criteria = [c for c in criteria_for(issue) if c["needs_review"]]
                    if reasons or criteria:
                        yield {"doc_name": row["doc_name"], "category": row["category"], "issue_index": issue["issue_index"],
                               "issue_text": issue["issue_text"], "reasons": reasons, "review_status": "needs_review",
                               "excluded_from_application_score": issue_blocked(issue) or not any(c["scorable"] for c in criteria_for(issue)),
                               "criteria": criteria}
            failed_docs = set()
            for error in reversed(self.errors):
                doc = error["doc_name"]
                if doc not in self.completed and doc not in failed_docs:
                    failed_docs.add(doc)
                    yield {**error, "category": next(s.raw["category"] for s in dataset.samples if s.doc_name == doc),
                           "review_status": "rubric_generation_failed"}

        atomic_jsonl(store.path / "added_issues.jsonl", added_issues())
        atomic_jsonl(store.path / "excluded_units.jsonl", event_rows("excluded_units"))
        atomic_jsonl(store.path / "rubric_review.jsonl", reviews())
        atomic_jsonl(store.path / "source_units.jsonl",
                     ({"doc_name": row["doc_name"], "category": row["category"], **row["audit"],
                       "independent_inventory": row["inventory"],
                       "additional_propositions": row["semantic_audit"]["additional_propositions"],
                       "coverage": row["semantic_audit"]["coverage"]} for row in rows))
        atomic_jsonl(store.path / "rule_application_draft.jsonl", (row["dataset"] for row in rows))
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
            reasons = Counter(e["error"] for e in exc.errors)
            report_path = store.path / "rubric_generation_summary.json"
            for reason, count in reasons.most_common(10):
                log.error("Raw rubric validation: %d row(s): %s", count, reason)
            log.error("Full validation report: %s", report_path)
            summary = {"run_id": config.run_id, "status": "input_validation_failed", "published": False,
                       "counts": {"input": exc.input_count, "valid": exc.valid_count, "invalid": len(exc.errors), "failed": len(exc.errors)},
                       "invalid_by_reason": dict(reasons),
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
        if config.limit == 0:
            summary = {"run_id": config.run_id, "status": "input_validated", "published": False,
                       "counts": {"input": dataset.input_count, "valid": dataset.valid_count,
                                  "invalid": 0, "selected": 0, "failed": 0},
                       "input_sha256": dataset.sha256}
            self.write_summary(store, summary)
            return summary
        if not config.resume and len(dataset.samples) == dataset.input_count and any(path.exists() for path in targets):
            raise ValueError("Verified output already exists; choose new final output paths for a fresh run")
        snapshot = {**config.snapshot(), "pipeline_kind": "rubric_generation", "artifact_version": 1,
                    "rubric_protocol": RUBRIC_PROTOCOL,
                    "input_sha256": dataset.sha256, "implementation_sha256": implementation_hash(),
                    "rubric_thinking": resolve_thinking(config.rubric_thinking_mode)}
        seed_rows = []
        if config.rubric_seed_run is not None:
            seed_rows, snapshot["rubric_seed_sha256"] = read_seed_run(config, snapshot)
        store.configure(snapshot, artifact_names=["rubric_records.jsonl", "errors.jsonl"])
        samples = {sample.doc_name: sample for sample in dataset.samples}
        self.completed = index_records(store.load("rubric_records.jsonl"), samples)
        imported = []
        for doc, row in index_records(seed_rows, samples).items():
            if doc not in self.completed:
                self.completed[doc] = row
                imported.append(row)
            elif self.completed[doc] != row:
                raise ValueError(f"Seed rubric differs from the saved checkpoint: {doc}")
        self.errors = store.load("errors.jsonl")
        for row in self.errors + list(self.completed.values()):
            sample = samples.get(row.get("doc_name"))
            if sample is None or any(row.get(k) != v for k, v in sample.identity().items()):
                raise ValueError("Rubric journal identity mismatch")
        for doc, row in self.completed.items():
            sample = samples[doc]
            expected, audit = materialize(sample, row["decomposition"], row["plan"], version=RUBRIC_PROTOCOL)
            validate_audit(row["semantic_audit"])
            coverage = audit_coverage(row["inventory"], row["semantic_audit"], row["decomposition"], audit,
                                      source_segments(sample.raw["expected_answer"], "S"),
                                      TAXONOMY[canonical_category(row["category"])])
            if row["dataset"] != expected or row["audit"] != audit or row["model"] != config.rubric_model or (
                    row["dataset_sha256"] != canonical_hash(expected) or row["category"] != expected["category"]
                    or row["proposition_counts"] != coverage):
                raise ValueError(f"Stored rubric integrity check failed: {doc}")
        for row in imported:
            store.append("rubric_records.jsonl", row)
        if imported:
            log.info("Imported %d validated rubrics from %s", len(imported), config.rubric_seed_run)
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
                summary, _ = self.report(store, dataset, "subset_completed", skipped)
                return summary
            if file_hash(config.input_jsonl) != dataset.sha256:
                raise ValueError("Raw input changed during rubric generation")
            summary, rows = self.report(store, dataset, "ready_to_publish", skipped)
            publish(targets, verified_rows(rows), snapshot, config.run_id)
            has_review_notes = summary["counts"]["issues_needing_review"] or summary["counts"]["added_issues"]
            summary.update(status="completed_with_review" if has_review_notes else "completed",
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
                    "rubric_protocol": RUBRIC_PROTOCOL, "verification": "semantic_audit",
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
        if snapshot.get("rubric_seed_run"):
            manifest["seed_run"] = {"path": snapshot["rubric_seed_run"],
                                    "rubric_records_sha256": snapshot["rubric_seed_sha256"]}
        atomic_json(staged[2], manifest)
        for path, temp in zip(targets, staged):
            if path.exists() and file_hash(path) != file_hash(temp):
                raise ValueError(f"Refusing to replace existing dataset: {path}")
        for path, temp in zip(targets, staged):
            if not path.exists():
                temp.rename(path)
