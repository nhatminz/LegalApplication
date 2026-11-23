"""Restore a rubric generation or evaluation run from its saved settings."""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from legal_application.config import rooted
from legal_application.data import strict_json
from legal_application.rubric_builder import RUBRIC_RUN_ARTIFACT_VERSION
from legal_application.rubrics import RUBRIC_PROTOCOL
from resume_statistic import DEFAULT_PROJECT_ROOT, snapshot_environment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="Existing run ID in OUTPUT_DIR")
    parser.add_argument("--retry-into", metavar="NEW_RUN_ID", help="Import completed rubrics into a new run using updated code")
    args = parser.parse_args()
    try:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
            raise ValueError("RUN_ID must be a simple filename")
        root = Path(os.environ.get("PROJECT_ROOT", DEFAULT_PROJECT_ROOT)).resolve()
        output = rooted(root, os.environ.get("OUTPUT_DIR", "outputs"))
        snapshot = strict_json((output / args.run_id / "config.json").read_text(encoding="utf-8"))
        if not isinstance(snapshot, dict):
            raise ValueError("Run config must be a JSON object")
        if (snapshot.get("run_id") != args.run_id or snapshot.get("project_root") != str(root)
                or snapshot.get("output_dir") != str(output)):
            raise ValueError("PROJECT_ROOT, OUTPUT_DIR and RUN_ID must match the saved run")
        build = snapshot.get("pipeline_kind") == "rubric_generation" and snapshot.get("mode") == "build-rubrics"
        evaluation = snapshot.get("evaluation_profile") == "rubric" and snapshot.get("mode") in {"full", "judge-only", "validate-only"}
        if not (build or evaluation):
            raise ValueError("Expected an existing rubric generation or rubric evaluation run")
        if build and (snapshot.get("rubric_protocol") != RUBRIC_PROTOCOL
                      or snapshot.get("artifact_version") != RUBRIC_RUN_ARTIFACT_VERSION):
            raise ValueError("This generation run uses an older format or workflow. Start generate_rubrics.sh with a new RUN_ID.")
        environment = snapshot_environment(snapshot)
        if build:
            environment["RAW_INPUT_JSONL"] = snapshot["input_jsonl"]
            environment["RUBRIC_GENERATOR_BASE_URL"] = snapshot["rubric_base_url"]
            environment["RUBRIC_QUALITY_BASE_URL"] = snapshot.get("rubric_quality_base_url") or snapshot["rubric_base_url"]
        environment["RESUME"] = "1"
        if args.retry_into:
            if not build or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.retry_into) or args.retry_into == args.run_id:
                raise ValueError("--retry-into requires a different run ID and a rubric generation source run")
            environment.update(RUN_ID=args.retry_into, RESUME="0", RUBRIC_SEED_RUN=str(output / args.run_id))
            destination = output / f"{args.retry_into}-final"
            for key, name in [("FINAL_APPLICATION_JSONL", "application.jsonl"), ("FINAL_ISSUE_SPOTTING_JSONL", "issue_spotting.jsonl")]:
                environment[key] = os.environ.get(key) or str(destination / name)
        os.chdir(root)
        os.execvpe(sys.executable, [sys.executable, "-m", "legal_application"], environment)
    except (ValueError, OSError) as exc:
        print(f"Resume aborted: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
