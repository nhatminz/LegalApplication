from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import tempfile
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from .api import APIClient, APIError
from .completeness import COMPLETENESS_SCHEMA, classify_sample, parse_classification, validate_classification
from .data import DatasetError, read_dataset
from .evidence import EVIDENCE_FORMAT
from .pipeline import Pipeline, bounded_map, index_records
from .storage import atomic_jsonl, file_hash, implementation_hash, now, read_jsonl
from .thinking import resolve_thinking

log = logging.getLogger(__name__)


def statistic_snapshot(config) -> dict:
    snapshot = config.snapshot()
    for key in list(snapshot):
        if key.startswith("generator_") or key in {"generation_jsonl", "evaluation_profile", "expected_count"}:
            snapshot.pop(key)
    snapshot.update(pipeline_kind="data_completeness", artifact_version=1,
                    input_sha256=file_hash(config.input_jsonl), implementation_sha256=implementation_hash(),
                    statistic_seed_sha256=file_hash(config.statistic_seed_jsonl) if config.statistic_seed_jsonl else None,
                    statistic_evidence_format=EVIDENCE_FORMAT,
                    judge_thinking=resolve_thinking(config.judge_thinking_mode))
    return snapshot


class StatisticPipeline(Pipeline):
    """Reuse the run lock/logging, API client and bounded worker pool; input is read-only."""
    allowed_modes = {"statistic"}

    def __init__(self, config, *, judge_transport=None):
        super().__init__(config, judge_transport=judge_transport)
        self.classifications = {}
        self.dataset = None
        self.configured = False
        self.input_count = 0
        self.attempted_docs = set()
        self.saved_count = 0
        self.input_errors = []
        self.input_valid_count = 0

    def _record_input_errors(self, store) -> None:
        known = {(e.get("input_index"), e.get("error")) for e in self.errors if e.get("stage") == "input_validation"}
        for error in self.input_errors:
            key = (error.get("input_index"), error["error"])
            if key not in known:
                record = {"stage": "input_validation", **error, "timestamp": now()}
                store.append("errors.jsonl", record)
                self.errors.append(record)
                known.add(key)

    def _read_input(self, store):
        """Validate every source line without editing it; stream rejected bytes to an audit artifact."""
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".invalid-input-",
                                             dir=store.path, delete=False) as output:
                temporary = Path(output.name)

                def rejected(error, raw_line):
                    try:
                        text = raw_line.decode("utf-8")
                    except UnicodeError:
                        text = None
                    record = {**error, "stage": "input_validation", "timestamp": now(),
                              "input_path": str(self.config.input_jsonl), "raw_line": text,
                              "raw_line_base64": base64.b64encode(raw_line).decode("ascii"),
                              "raw_line_sha256": hashlib.sha256(raw_line).hexdigest(), "byte_length": len(raw_line)}
                    output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    log.warning("Statistic skipping invalid input line=%s doc_name=%s error=%s (source unchanged)",
                                error["input_line"], error["doc_name"], error["error"])

                try:
                    dataset = read_dataset(self.config.input_jsonl, self.config.limit, skip_invalid=True, on_invalid=rejected)
                except DatasetError as exc:
                    self.input_count, self.input_valid_count, self.input_errors = exc.input_count, exc.valid_count, exc.errors
                    raise
                finally:
                    output.flush()
                    os.fsync(output.fileno())
            return dataset
        finally:
            # Atomic replacement keeps this file one entry per rejected source line on resume.
            if temporary is not None:
                try:
                    os.replace(temporary, store.path / "invalid_input_records.jsonl")
                finally:
                    temporary.unlink(missing_ok=True)

    def _summary(self, status: str) -> dict:
        samples = self.dataset.samples if self.dataset else []
        full = [s.doc_name for s in samples if self.classifications.get(s.doc_name, {}).get("data_status") == "full"]
        partial = [s.doc_name for s in samples if self.classifications.get(s.doc_name, {}).get("data_status") == "partial"]
        failed = {e["doc_name"] for e in self.errors if e.get("stage") == "classification"} - self.classifications.keys()
        latest_failures = {e["doc_name"]: e for e in self.errors if e.get("stage") == "classification" and e["doc_name"] in failed}
        failure_reasons = Counter()
        for error in latest_failures.values():
            metadata = error.get("metadata") or {}
            failure_reasons[metadata.get("last_validation_error") or (metadata.get("validation_errors") or [error["error"]])[-1]] += 1
        processed = len(full) + len(partial)
        complete = self.dataset is not None and processed == len(samples)
        return {"run_id": self.config.run_id, "status": status, "updated_at": now(),
                "counts": {"input": self.input_count, "input_valid": self.input_valid_count,
                           "input_invalid": len([e for e in self.input_errors if e.get("input_line") is not None]),
                           "selected": len(samples), "attempted": len(self.attempted_docs),
                           "processed": processed, "completed": processed, "full": len(full), "partial": len(partial),
                           "failed": len(failed), "pending": len(samples) - processed - len(failed), "skipped": self.saved_count},
                "ratios": {"full": len(full) / processed if processed else None,
                           "partial": len(partial) / processed if processed else None,
                           "failed": len(failed) / len(samples) if samples else None,
                           "input_invalid": len([e for e in self.input_errors if e.get("input_line") is not None]) / self.input_count if self.input_count else None},
                "input_validation_errors": self.input_errors,
                "input_error_log": "invalid_input_records.jsonl",
                "classified_output_scope": "selected_valid_records",
                "source_fully_classified": complete and status == "completed" and not self.input_errors and processed == self.input_count,
                "full_doc_names": full, "partial_doc_names": partial, "failed_doc_names": sorted(failed),
                "classified_output_complete": complete and status in {"completed", "completed_with_input_errors"},
                "classified_output": "classified_input.jsonl" if complete and status in {"completed", "completed_with_input_errors"} else None,
                "partial_output": "classified_input.partial.jsonl" if self.dataset and not complete else None,
                "failure_reasons": dict(failure_reasons), "judge_model": self.config.judge_model, "mean_score": None}

    def _export(self, store, *, allow_complete: bool = True) -> None:
        if self.dataset is None:
            return
        complete = allow_complete and len(self.classifications) == len(self.dataset.samples)
        name = "classified_input.jsonl" if complete else "classified_input.partial.jsonl"
        target = store.path / name
        if target.resolve() == self.config.input_jsonl.resolve():
            raise ValueError("Classification output cannot overwrite input")
        atomic_jsonl(target, ({**s.raw, "data_status": self.classifications[s.doc_name]["data_status"]}
                             for s in self.dataset.samples if s.doc_name in self.classifications))
        if complete:
            (store.path / "classified_input.partial.jsonl").unlink(missing_ok=True)
        # Persisted details can arrive out of order; final exports follow the input order.
        atomic_jsonl(store.path / "classification_details.jsonl",
                     (self.classifications[s.doc_name] for s in self.dataset.samples if s.doc_name in self.classifications))

    async def _run(self, store) -> dict:
        try:
            return await self._classify(store)
        except BaseException as exc:
            if not self.config.resume or self.configured:
                status = "interrupted" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "preflight_failed"
                summary = self._summary(status)
                summary["classified_output_complete"] = False
                summary["classified_output"] = None
                summary["fatal_error"] = {"error_type": type(exc).__name__, "error": str(exc)}
                if isinstance(exc, DatasetError):
                    summary["input_validation_errors"] = exc.errors
                    summary["counts"]["input"] = exc.input_count
                    self.errors = store.load("errors.jsonl")
                    self._record_input_errors(store)
                if self.dataset is not None:
                    self._export(store, allow_complete=False)
                    summary["partial_output"] = "classified_input.partial.jsonl"
                store.summary(summary)
            raise

    async def _classify(self, store) -> dict:
        config = self.config
        snapshot = statistic_snapshot(config)
        store.configure(snapshot, artifact_names=["classification_details.jsonl", "errors.jsonl"])
        self.configured = True
        log.info("Statistic start run_id=%s limit=%s batch_size=%d judge=%s thinking=%s", config.run_id,
                 config.limit, config.batch_size, config.judge_model, snapshot["judge_thinking"])
        dataset = self._read_input(store)  # Deliberately NO clean_input() and NO input rewrite.
        self.input_count = dataset.input_count
        self.input_valid_count, self.input_errors = dataset.valid_count, dataset.validation_errors
        if dataset.sha256 != snapshot["input_sha256"]:
            raise ValueError("Input changed during statistic validation; no API requests made")
        for sample in dataset.samples:
            if "data_status" in sample.raw:
                raise ValueError(f"Statistic expects original unlabelled input; data_status already exists: {sample.doc_name}")
        samples = {s.doc_name: s for s in dataset.samples}
        self.classifications = index_records(store.load("classification_details.jsonl"), samples)
        seeds = {}
        if config.statistic_seed_jsonl:
            for doc, record in index_records(read_jsonl(config.statistic_seed_jsonl), samples).items():
                if doc in self.classifications and self.classifications[doc] != record:
                    raise ValueError("Imported classification conflicts with stored result")
                if doc not in self.classifications:
                    seeds[doc] = record
            self.classifications.update(seeds)
        self.errors = store.load("errors.jsonl")
        for record in self.classifications.values():
            sample = samples[record["doc_name"]]
            if any(record.get(k) != v for k, v in sample.identity().items()) or record.get("judge_model") != config.judge_model:
                raise ValueError("Stored classification identity/model mismatch")
            labels, _ = parse_classification(record["raw_judge_response"], sample)
            validate_classification(labels, sample)
            if labels["classification_status"] != "completed" or any(record.get(k) != v for k, v in labels.items()):
                raise ValueError("Stored classification differs from raw Judge labels")
        for record in self.errors:
            if record.get("stage") == "input_validation":
                if not any(record.get("input_index") == e.get("input_index") and record.get("error") == e["error"] for e in self.input_errors):
                    raise ValueError("Stored input-validation error no longer matches the source")
                continue
            if record.get("stage") != "classification":
                raise ValueError("Stored statistic error has unsupported stage")
            sample = samples.get(record.get("doc_name"))
            if sample is None or any(record.get(k) != v for k, v in sample.identity().items()):
                raise ValueError("Stored classification error identity mismatch")
        self.dataset = dataset
        if seeds:
            if file_hash(config.statistic_seed_jsonl) != snapshot["statistic_seed_sha256"]:
                raise ValueError("Seed classifications changed during import")
            for record in seeds.values():
                store.append("classification_details.jsonl", record)
            log.info("Imported %d validated classifications; only remaining samples will call Judge", len(seeds))
        self._record_input_errors(store)
        self.saved_count = len(self.classifications)
        self.attempted_docs = set(self.classifications) | {e["doc_name"] for e in self.errors if e.get("stage") == "classification"}
        # Rebuild only after validating persisted state. Never modify an input source.
        store.summary(self._summary("running"))
        client = APIClient(config, "judge", self.judge_transport)
        try:
            full_count = sum(r["data_status"] == "full" for r in self.classifications.values())
            partial_count = len(self.classifications) - full_count
            with tqdm(total=len(dataset.samples), initial=self.saved_count, desc="Classify completeness", unit="sample",
                      miniters=1, dynamic_ncols=True, disable=None) as progress:
                progress.set_postfix(full=full_count, partial=partial_count,
                                     failed=len(self.attempted_docs - self.classifications.keys()),
                                     skipped=self.saved_count, invalid=len(self.input_errors), refresh=False)

                async def process(sample):
                    nonlocal full_count, partial_count
                    if sample.doc_name in self.classifications:
                        return
                    self.attempted_docs.add(sample.doc_name)
                    try:
                        record = await classify_sample(config, sample, client)
                    except APIError as exc:
                        self.failure(store, sample, "classification", exc)
                    else:
                        store.append("classification_details.jsonl", record)
                        self.classifications[sample.doc_name] = record
                        full_count += int(record["data_status"] == "full")
                        partial_count += int(record["data_status"] == "partial")
                        log.info("Classified doc_name=%s input_index=%d data_status=%s", sample.doc_name, sample.input_index, record["data_status"])
                    progress.set_postfix(full=full_count, partial=partial_count,
                                         failed=len(self.attempted_docs - self.classifications.keys()), skipped=self.saved_count,
                                         invalid=len(self.input_errors), refresh=False)
                    progress.update(1)

                for start in range(0, len(dataset.samples), config.batch_size):
                    batch = dataset.samples[start:start + config.batch_size]
                    log.info("Classification batch start number=%d size=%d", start // config.batch_size + 1, len(batch))
                    await bounded_map(batch, config.judge_concurrency, process)
                    store.summary(self._summary("running"))
                    log.info("Classification batch done full=%d partial=%d processed=%d", full_count, partial_count, len(self.classifications))
            if file_hash(config.input_jsonl) != dataset.sha256:
                raise ValueError("Input changed during classification; refusing to publish complete output")
            self._export(store)
            status = "completed_with_errors"
            if len(self.classifications) == len(dataset.samples):
                status = "completed_with_input_errors" if self.input_errors else "completed"
            summary = self._summary(status)
            store.summary(summary)
            return summary
        finally:
            await client.close()
