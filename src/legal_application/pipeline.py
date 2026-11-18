from __future__ import annotations

import asyncio
import logging
from collections import Counter
from statistics import mean
from uuid import uuid4

from tqdm import tqdm

from .api import APIClient, APIError, IncompleteResponse
from .config import Config
from .data import DatasetError, Sample, canonical_hash, read_dataset, strict_json
from .input_cleanup import clean_input
from .prompts import generator_messages, judge_messages
from .scoring import LABEL_SCHEMA, LabelError, score_labels
from .storage import RunStore, file_hash, implementation_hash, now, read_jsonl
from .filter_full import verify_full_manifest
from .thinking import resolve_thinking

log = logging.getLogger(__name__)


def check_generation(record: dict, sample: Sample) -> None:
    for key, value in sample.identity().items():
        if record.get(key) != value:
            raise ValueError(f"Generation mapping mismatch doc_name={sample.doc_name} field={key}")
    if record.get("status") != "completed":
        raise ValueError(f"Generation is not completed: {sample.doc_name}")
    if not isinstance(record.get("raw_response"), str) or not record["raw_response"].strip():
        raise ValueError(f"Generation raw_response missing: {sample.doc_name}")
    if record.get("generation_sha256") != canonical_hash(record["raw_response"]):
        raise ValueError(f"Generation checksum mismatch: {sample.doc_name}")
    if not isinstance(record.get("model"), str) or not record["model"].strip() or not isinstance(record.get("metadata"), dict):
        raise ValueError(f"Generation model/metadata missing: {sample.doc_name}")


def index_records(records: list[dict], samples: dict[str, Sample], *, allow_unselected: bool = False) -> dict[str, dict]:
    indexed, seen_names, seen_indices = {}, set(), set()
    for record in records:
        doc, index = record.get("doc_name"), record.get("input_index")
        if not isinstance(doc, str) or type(index) is not int or index < 0:
            raise ValueError("Artifact record requires string doc_name and nonnegative integer input_index")
        if doc in seen_names or index in seen_indices:
            raise ValueError(f"Duplicate artifact identity: doc_name={doc} input_index={index}")
        seen_names.add(doc)
        seen_indices.add(index)
        if doc not in samples:
            if allow_unselected:
                continue
            raise ValueError(f"Artifact contains an unselected doc_name: {doc}")
        if index != samples[doc].input_index:
            raise ValueError(f"Artifact input_index does not match doc_name={doc}")
        indexed[doc] = record
    return indexed


async def bounded_map(items: list, concurrency: int, process, on_done=None) -> None:
    """At most concurrency live tasks, regardless of batch size."""
    iterator = iter(items)

    async def worker():
        for item in iterator:
            await process(item)
            if on_done is not None:
                on_done(item)

    tasks = [asyncio.create_task(worker()) for _ in range(min(concurrency, len(items)))]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def judge_sample(config: Config, sample: Sample, generation: dict, client: APIClient) -> dict:
    repair, calls, validation_errors = None, [], []
    for attempt in range(config.judge_json_retries + 1):
        try:
            completion = await client.complete(judge_messages(sample, generation["raw_response"], LABEL_SCHEMA, repair), doc_name=sample.doc_name)
            calls.append(completion.metadata)
            try:
                labels = strict_json(completion.content)
            except ValueError:
                raise LabelError("response is not a strict JSON object") from None
            components = score_labels(labels, sample)
            return {**labels, **sample.identity(), "score_components": components,
                    "generator_model": generation["model"], "judge_model": config.judge_model,
                    "generation_sha256": generation["generation_sha256"],
                    "raw_judge_response": completion.content,
                    "metadata": {"completed_at": now(), "judge_calls": calls,
                                 "validation_errors": validation_errors, "label_attempts": attempt + 1}}
        except (LabelError, IncompleteResponse) as exc:
            if isinstance(exc, IncompleteResponse):
                calls.append(exc.metadata)
            repair = str(exc)
            validation_errors.append(repair)
            log.warning("Judge invalid doc_name=%s label_attempt=%d reason=%s", sample.doc_name, attempt + 1, repair)
            if attempt == config.judge_json_retries:
                raise APIError("judge exhausted structured-output retries", {"judge_calls": calls, "validation_errors": validation_errors}) from None
            if config.retry_backoff:
                await asyncio.sleep(min(60.0, config.retry_backoff * 2 ** attempt))
    raise AssertionError("unreachable")


class Pipeline:
    allowed_modes = {"full", "judge-only", "validate-only"}

    def __init__(self, config: Config, *, generator_transport=None, judge_transport=None):
        self.config = config
        self.generator_transport, self.judge_transport = generator_transport, judge_transport
        self.generations, self.evaluations = {}, {}
        self.errors = []
        self.invocation_id = uuid4().hex
        self.counters = {role: {"attempted": 0, "completed": 0, "failed": 0, "skipped": 0}
                         for role in ["generation", "evaluation"]}
        self.preexisting_evaluations = 0
        self.input_cleanup = {}

    def failure(self, store: RunStore, sample: Sample, stage: str, exc: Exception) -> None:
        record = {**sample.identity(), "stage": stage, "error_type": type(exc).__name__, "error": str(exc),
                  "metadata": getattr(exc, "metadata", {}), "timestamp": now(), "invocation_id": self.invocation_id}
        store.append("errors.jsonl", record)
        self.errors.append(record)
        log.error("Sample failed stage=%s doc_name=%s input_index=%d error=%s", stage, sample.doc_name, sample.input_index, exc)

    def summary(self, dataset, status: str) -> dict:
        selected = len(dataset.samples)
        scored = [v for v in self.evaluations.values() if v["evaluation_status"] == "completed"]
        review = sum(v["needs_review"] for v in self.evaluations.values())
        generation_failures = {e["doc_name"] for e in self.errors if e["stage"] in {"generation", "generation_source"}}
        evaluation_failures = {e["doc_name"] for e in self.errors if e["stage"] == "evaluation"}
        generation_failures -= self.generations.keys()
        evaluation_failures -= self.evaluations.keys()
        failed_docs = (generation_failures | evaluation_failures) - self.evaluations.keys()
        attempted = set(self.generations) | set(self.evaluations) | {e["doc_name"] for e in self.errors}
        histogram = {f"{i}–{i+1}" if i < 9 else "9–10 inclusive": 0 for i in range(10)}
        exact = Counter()
        for row in scored:
            total = row["score_components"]["total"]
            index = min(int(total), 9)
            histogram[list(histogram)[index]] += 1
            exact[format(total, ".12g")] += 1

        def stage(name, rows, failures):
            return {"completed": len(rows), "failed": len(failures), "pending": selected - len(rows) - len(failures),
                    "this_invocation": self.counters[name]}

        return {"run_id": self.config.run_id, "status": status, "updated_at": now(),
                "counts": {"input": self.input_cleanup.get("original_line_count", dataset.input_count),
                           "input_valid": dataset.input_count, "input_removed": self.input_cleanup.get("removed_count", 0), "selected": selected,
                           "attempted": len(attempted), "completed": len(self.evaluations), "failed": len(failed_docs),
                           "pending": selected - len(self.evaluations) - len(failed_docs),
                           "skipped": self.preexisting_evaluations, "scored": len(scored), "needs_review": review},
                "input_cleanup": self.input_cleanup,
                "data_status_used": "full_only",
                "data_status_counts": {"input": dataset.status_counts,
                                       "filtered_out": dataset.filtered_out,
                                       "eligible": dataset.eligible_count,
                                       "selected": {group: sum(s.data_status == group for s in dataset.samples)
                                                    for group in ["full", "partial"]},
                                       "completed": {group: sum(s.data_status == group and s.doc_name in self.evaluations
                                                                 for s in dataset.samples) for group in ["full", "partial"]}},
                "generation": stage("generation", self.generations, generation_failures),
                "evaluation": stage("evaluation", self.evaluations, evaluation_failures),
                "mean_score": mean(r["score_components"]["total"] for r in scored) if scored else None,
                "mean_score_components": {key: mean(r["score_components"][key] for r in scored) if scored else None
                                          for key in ["final_order", "coverage", "correctness", "total"]},
                "score_distribution": {"bins": histogram, "exact": dict(sorted(exact.items(), key=lambda x: float(x[0])))},
                "needs_review_doc_names": [r["doc_name"] for r in self.evaluations.values() if r["needs_review"]],
                "failed_doc_names": sorted(failed_docs), "error_events": len(self.errors),
                "invocation_id": self.invocation_id, "models": {"generator": sorted({r["model"] for r in self.generations.values()}),
                                                               "judge": self.config.judge_model}}

    async def run(self) -> dict:
        config = self.config
        config.validate()
        if config.mode not in self.allowed_modes:
            raise ValueError(f"Unsupported mode {config.mode} for {type(self).__name__}")
        with RunStore(config) as store:
            handler = logging.FileHandler(store.path / "run.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
            logging.getLogger().addHandler(handler)
            package_logger = logging.getLogger("legal_application")
            previous_level = package_logger.level
            package_logger.setLevel(logging.INFO)
            try:
                return await self._run(store)
            except Exception as exc:
                log.error("Run aborted type=%s error=%s", type(exc).__name__, exc)
                # Preserve summaries of existing runs when a resume is rejected.
                if not config.resume and not (store.path / "summary.json").exists():
                    store.summary({"run_id": config.run_id, "status": "preflight_failed", "updated_at": now(),
                                   "counts": {"input": getattr(self, "input_count", None), "selected": getattr(self, "selected_count", 0),
                                              "attempted": 0, "completed": 0, "failed": 0, "skipped": 0,
                                              "scored": 0, "needs_review": 0},
                                   "mean_score": None,
                                   "mean_score_components": dict.fromkeys(["final_order", "coverage", "correctness", "total"]),
                                   "fatal_error": {"error_type": type(exc).__name__, "error": str(exc)}})
                raise
            finally:
                package_logger.setLevel(previous_level)
                logging.getLogger().removeHandler(handler)
                handler.close()

    async def _run(self, store: RunStore) -> dict:
        config = self.config
        generator_thinking = resolve_thinking(config.generator_thinking_mode)
        judge_thinking = resolve_thinking(config.judge_thinking_mode)
        full_manifest = verify_full_manifest(config.input_jsonl)
        snapshot = {**config.snapshot(), "artifact_version": 1, "implementation_sha256": implementation_hash(),
                    "full_input_manifest_sha256": file_hash(config.input_jsonl.with_name(config.input_jsonl.name + ".manifest.json")) if full_manifest else None,
                    "generation_source_sha256": file_hash(config.generation_jsonl) if config.generation_jsonl else None,
                    "generator_thinking": generator_thinking, "judge_thinking": judge_thinking,
                    "judge_reasoning_effort": judge_thinking["effort"],
                    "judge_reasoning_parameters": judge_thinking["parameters"].get("chat_template_kwargs", {})}
        log.info("Run start run_id=%s mode=%s limit=%s batch_size=%d invocation=%s", config.run_id, config.mode, config.limit, config.batch_size, self.invocation_id)
        # Validate labels across ALL records before the existing cleanup can mutate input.
        # This also protects filtered records and records beyond LIMIT from guessed labels.
        with config.input_jsonl.open("rb") as source:
            for line_number, line in enumerate(source, 1):
                try:
                    raw = strict_json(line.decode("utf-8"))
                except (ValueError, UnicodeError):
                    continue  # Existing cleanup journals malformed JSON separately.
                if isinstance(raw, dict) and raw.get("data_status") not in ("full", "partial"):
                    if not full_manifest or "data_status" in raw:
                        raise ValueError(f"{config.input_jsonl}:{line_number}: data_status must be full or partial; use scripts/filter_full.sh for raw input")
                if full_manifest and isinstance(raw, dict) and raw.get("data_status", "full") != "full":
                    raise ValueError("Full-input manifest contradicts the record's data_status")
        self.input_cleanup = clean_input(config.input_jsonl, store.path,
                                        lambda digest: store.configure({**snapshot, "input_sha256": digest}))
        snapshot["input_sha256"] = self.input_cleanup["input_sha256_after"]
        for role, settings in [("generator", generator_thinking), ("judge", judge_thinking)]:
            log.info("Thinking role=%s mode=%s backend=%s enabled=%s budget=%s parameters=%s", role,
                     settings["requested_mode"], settings["backend"], settings["thinking_enabled"],
                     settings["effort_budget"], settings["parameters"])
        try:
            dataset = read_dataset(config.input_jsonl, config.limit, application=True, verified_full=bool(full_manifest))
        except DatasetError as exc:
            for error in exc.errors:
                log.error("Input invalid line=%s error=%s", error["input_line"], error["error"])
            summary = {"run_id": config.run_id, "status": "input_validation_failed",
                       "counts": {"input": self.input_cleanup["original_line_count"], "input_valid": exc.valid_count,
                                  "input_removed": self.input_cleanup["removed_count"], "selected": 0,
                                  "attempted": 0, "completed": 0, "failed": 0, "skipped": 0, "scored": 0, "needs_review": 0},
                       "input_validation_error_count": len(exc.errors), "input_validation_errors": exc.errors,
                       "input_cleanup": self.input_cleanup,
                       "mean_score": None, "mean_score_components": dict.fromkeys(["final_order", "coverage", "correctness", "total"]),
                       "updated_at": now()}
            store.summary(summary)
            raise
        if dataset.sha256 != snapshot["input_sha256"]:
            raise ValueError("Input changed during validation; no API requests made")
        if full_manifest and (dataset.sha256 != full_manifest["output_sha256"] or dataset.valid_count != full_manifest["sample_count"]):
            raise ValueError("Full-input manifest no longer matches the validated input")
        self.input_count = dataset.input_count
        self.selected_count = len(dataset.samples)
        samples = {s.doc_name: s for s in dataset.samples}
        self.generations = index_records(store.load("generations.jsonl"), samples)
        self.evaluations = index_records(store.load("evaluations.jsonl"), samples)
        self.errors = store.load("errors.jsonl")
        for record in self.errors:
            doc = record.get("doc_name")
            if doc not in samples or any(record.get(key) != value for key, value in samples[doc].identity().items()):
                raise ValueError("Error journal identity mismatch")
            if record.get("stage") not in {"generation", "generation_source", "evaluation"}:
                raise ValueError("Invalid error journal stage")
        for record in self.generations.values():
            sample = samples[record["doc_name"]]
            check_generation(record, sample)
            if config.mode == "full" and record["model"] != config.generator_model:
                raise ValueError("Stored generator model differs from configured model")
        for record in self.evaluations.values():
            doc = record["doc_name"]
            if doc not in self.generations:
                raise ValueError(f"Evaluation has no corresponding generation: {doc}")
            for key, value in samples[doc].identity().items():
                if record.get(key) != value:
                    raise ValueError(f"Evaluation identity mismatch: {doc}, {key}")
            labels = {key: record.get(key) for key in LABEL_SCHEMA["properties"]}
            if score_labels(labels, samples[doc]) != record.get("score_components"):
                raise ValueError(f"Stored evaluation scores inconsistent with labels: {doc}")
            if record.get("generation_sha256") != self.generations[doc]["generation_sha256"] or record.get("judge_model") != config.judge_model:
                raise ValueError(f"Evaluation generation/judge mapping mismatch: {doc}")
        self.preexisting_evaluations = len(self.evaluations)
        log.info("Input validated input=%d selected=%d stored_generations=%d stored_evaluations=%d", dataset.input_count, len(samples), len(self.generations), len(self.evaluations))
        for sample in dataset.samples:
            for block in sample.law_blocks:
                if block["fallback"]:
                    log.info("Law fallback doc_name=%s source=%s", sample.doc_name, block["source"])
        if config.mode == "validate-only":
            summary = self.summary(dataset, "validated")
            store.summary(summary)
            return summary
        source = {}
        if config.mode == "judge-only":
            source = index_records(read_jsonl(config.generation_jsonl), samples, allow_unselected=True)
            for doc, row in source.items():
                check_generation(row, samples[doc])
                if doc in self.generations and row != self.generations[doc]:
                    raise ValueError(f"Generation source differs from stored record: {doc}")
        clients = []
        try:
            generator = APIClient(config, "generator", self.generator_transport) if config.mode == "full" else None
            judge = APIClient(config, "judge", self.judge_transport)
            clients = [c for c in [generator, judge] if c is not None]

            async def generate(sample: Sample):
                counters = self.counters["generation"]
                if sample.doc_name in self.generations:
                    counters["skipped"] += 1
                    return
                if config.mode == "judge-only":
                    if sample.doc_name not in source:
                        self.failure(store, sample, "generation_source", ValueError("Selected sample has no saved generation"))
                        counters["failed"] += 1
                        return
                    record = source[sample.doc_name]
                    counters["skipped"] += 1  # imported; no generator API attempt
                else:
                    counters["attempted"] += 1
                    try:
                        completion = await generator.complete(generator_messages(sample), doc_name=sample.doc_name)
                    except APIError as exc:
                        counters["failed"] += 1
                        self.failure(store, sample, "generation", exc)
                        return
                    record = {**sample.identity(), "model": config.generator_model, "status": "completed",
                              "raw_response": completion.content, "generation_sha256": canonical_hash(completion.content),
                              "metadata": {**completion.metadata, "completed_at": now(), "law_sources": [
                                  {key: block[key] for key in ["law_index", "provision_index", "source", "fallback"]}
                                  for block in sample.law_blocks]}}
                    counters["completed"] += 1
                # Persistence errors are fatal; do not issue further calls after a lost commit.
                store.append("generations.jsonl", record)
                self.generations[sample.doc_name] = record
                log.info("Generation saved doc_name=%s input_index=%d", sample.doc_name, sample.input_index)

            async def evaluate(sample: Sample):
                counters = self.counters["evaluation"]
                if sample.doc_name in self.evaluations:
                    counters["skipped"] += 1
                    return
                if sample.doc_name not in self.generations:
                    return
                counters["attempted"] += 1
                try:
                    record = await judge_sample(config, sample, self.generations[sample.doc_name], judge)
                except APIError as exc:
                    counters["failed"] += 1
                    self.failure(store, sample, "evaluation", exc)
                    return
                store.append("evaluations.jsonl", record)
                self.evaluations[sample.doc_name] = record
                counters["completed"] += 1
                log.info("Evaluation saved doc_name=%s input_index=%d status=%s score=%s", sample.doc_name, sample.input_index, record["evaluation_status"], record["score_components"]["total"])

            store.summary(self.summary(dataset, "running"))
            saved_generations, saved_evaluations = set(self.generations), set(self.evaluations)
            blocked = 0
            review_count = sum(row["needs_review"] for row in self.evaluations.values())
            with tqdm(total=len(samples), initial=len(saved_generations), unit="sample", position=0,
                      desc="Generation" if config.mode == "full" else "Import generations",
                      miniters=1, dynamic_ncols=True, disable=None) as gen_progress, \
                 tqdm(total=len(samples), initial=len(saved_evaluations), unit="sample", position=1,
                      desc="Judge", miniters=1, dynamic_ncols=True, disable=None) as judge_progress:

                def generation_finished(sample):
                    gen_progress.set_postfix(ok=len(self.generations), failed=self.counters["generation"]["failed"],
                                             skipped=len(saved_generations), refresh=False)
                    if sample.doc_name not in saved_generations:
                        gen_progress.update(1)

                def evaluation_finished(sample):
                    nonlocal blocked, review_count
                    if sample.doc_name not in saved_evaluations:
                        if sample.doc_name not in self.generations:
                            blocked += 1
                        elif sample.doc_name in self.evaluations:
                            review_count += int(self.evaluations[sample.doc_name]["needs_review"])
                    judge_progress.set_postfix(ok=len(self.evaluations), failed=self.counters["evaluation"]["failed"],
                                               skipped=len(saved_evaluations), blocked=blocked, review=review_count, refresh=False)
                    if sample.doc_name not in saved_evaluations:
                        judge_progress.update(1)

                gen_progress.set_postfix(ok=len(saved_generations), failed=0, skipped=len(saved_generations), refresh=False)
                judge_progress.set_postfix(ok=len(saved_evaluations), failed=0, skipped=len(saved_evaluations),
                                           blocked=0, review=review_count, refresh=False)
                for start in range(0, len(dataset.samples), config.batch_size):
                    batch = dataset.samples[start:start + config.batch_size]
                    gen_progress.set_description(f"{'Generation' if config.mode == 'full' else 'Import generations'} [batch {start // config.batch_size + 1}]")
                    judge_progress.set_description(f"Judge [batch {start // config.batch_size + 1}]")
                    log.info("Batch start number=%d size=%d input_indices=%s", start // config.batch_size + 1, len(batch), [s.input_index for s in batch])
                    await bounded_map(batch, config.generator_concurrency, generate, generation_finished)
                    await bounded_map(batch, config.judge_concurrency, evaluate, evaluation_finished)
                    interim = self.summary(dataset, "running")
                    store.summary(interim)
                    log.info("Batch done selected=%d completed=%d failed=%d review=%d", len(samples), interim["counts"]["completed"], interim["counts"]["failed"], interim["counts"]["needs_review"])
            summary = self.summary(dataset, "completed")
            if summary["counts"]["failed"]:
                summary["status"] = "completed_with_errors"
            elif summary["counts"]["needs_review"]:
                summary["status"] = "completed_with_review"
            store.summary(summary)
            log.info("Run finished status=%s counts=%s", summary["status"], summary["counts"])
            return summary
        except BaseException as exc:
            summary = self.summary(dataset, "interrupted" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "failed")
            summary["fatal_error"] = {"error_type": type(exc).__name__, "error": str(exc)}
            store.summary(summary)
            raise
        finally:
            for client in clients:
                await client.close()

