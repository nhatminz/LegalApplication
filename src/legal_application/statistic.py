from __future__ import annotations

import asyncio
import logging

from tqdm import tqdm

from .api import APIClient, APIError
from .completeness import COMPLETENESS_SCHEMA, classify_sample, validate_classification
from .data import DatasetError, read_dataset, strict_json
from .pipeline import Pipeline, bounded_map, index_records
from .storage import atomic_jsonl, file_hash, implementation_hash, now
from .thinking import resolve_thinking

log = logging.getLogger(__name__)


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

    def _summary(self, status: str) -> dict:
        samples = self.dataset.samples if self.dataset else []
        full = [s.doc_name for s in samples if self.classifications.get(s.doc_name, {}).get("data_status") == "full"]
        partial = [s.doc_name for s in samples if self.classifications.get(s.doc_name, {}).get("data_status") == "partial"]
        failed = {e["doc_name"] for e in self.errors if e.get("stage") == "classification"} - self.classifications.keys()
        processed = len(full) + len(partial)
        complete = self.dataset is not None and processed == len(samples)
        return {"run_id": self.config.run_id, "status": status, "updated_at": now(),
                "counts": {"input": self.input_count, "selected": len(samples), "attempted": len(self.attempted_docs),
                           "processed": processed, "completed": processed, "full": len(full), "partial": len(partial),
                           "failed": len(failed), "pending": len(samples) - processed - len(failed), "skipped": self.saved_count},
                "ratios": {"full": len(full) / processed if processed else None,
                           "partial": len(partial) / processed if processed else None,
                           "failed": len(failed) / len(samples) if samples else None},
                "full_doc_names": full, "partial_doc_names": partial, "failed_doc_names": sorted(failed),
                "classified_output_complete": complete and status == "completed",
                "classified_output": "classified_input.jsonl" if complete and status == "completed" else None,
                "partial_output": "classified_input.partial.jsonl" if self.dataset and not complete else None,
                "judge_model": self.config.judge_model, "mean_score": None}

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
                    for error in exc.errors:
                        log.error("Statistic input invalid line=%s error=%s (source unchanged)", error["input_line"], error["error"])
                        store.append("errors.jsonl", {"stage": "input_validation", **error, "timestamp": now()})
                if self.dataset is not None:
                    self._export(store, allow_complete=False)
                    summary["partial_output"] = "classified_input.partial.jsonl"
                store.summary(summary)
            raise

    async def _classify(self, store) -> dict:
        config = self.config
        snapshot = config.snapshot()
        for key in list(snapshot):
            if key.startswith("generator_") or key in {"generation_jsonl", "data_status_used"}:
                snapshot.pop(key)
        snapshot.update(pipeline_kind="data_completeness", artifact_version=1,
                        input_sha256=file_hash(config.input_jsonl), implementation_sha256=implementation_hash(),
                        judge_thinking=resolve_thinking(config.judge_thinking_mode))
        store.configure(snapshot, artifact_names=["classification_details.jsonl", "errors.jsonl"])
        self.configured = True
        log.info("Statistic start run_id=%s limit=%s batch_size=%d judge=%s thinking=%s", config.run_id,
                 config.limit, config.batch_size, config.judge_model, snapshot["judge_thinking"])
        dataset = read_dataset(config.input_jsonl, config.limit)  # Deliberately NO clean_input().
        self.input_count = dataset.input_count
        if dataset.sha256 != snapshot["input_sha256"]:
            raise ValueError("Input changed during statistic validation; no API requests made")
        for sample in dataset.samples:
            if "data_status" in sample.raw:
                raise ValueError(f"Statistic expects original unlabelled input; data_status already exists: {sample.doc_name}")
        samples = {s.doc_name: s for s in dataset.samples}
        self.classifications = index_records(store.load("classification_details.jsonl"), samples)
        self.errors = store.load("errors.jsonl")
        for record in self.classifications.values():
            sample = samples[record["doc_name"]]
            if any(record.get(k) != v for k, v in sample.identity().items()) or record.get("judge_model") != config.judge_model:
                raise ValueError("Stored classification identity/model mismatch")
            labels = strict_json(record["raw_judge_response"])
            validate_classification(labels, sample)
            if labels["classification_status"] != "completed" or any(record.get(k) != v for k, v in labels.items()):
                raise ValueError("Stored classification differs from raw Judge labels")
        for record in self.errors:
            if record.get("stage") != "classification":
                raise ValueError("Stored statistic error has unsupported stage")
            sample = samples.get(record.get("doc_name"))
            if sample is None or any(record.get(k) != v for k, v in sample.identity().items()):
                raise ValueError("Stored classification error identity mismatch")
        self.dataset = dataset
        self.saved_count = len(self.classifications)
        self.attempted_docs = set(self.classifications) | {e["doc_name"] for e in self.errors}
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
                                     skipped=self.saved_count, refresh=False)

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
                                         failed=len(self.attempted_docs - self.classifications.keys()), skipped=self.saved_count, refresh=False)
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
            summary = self._summary("completed" if len(self.classifications) == len(dataset.samples) else "completed_with_errors")
            store.summary(summary)
            return summary
        finally:
            await client.close()
