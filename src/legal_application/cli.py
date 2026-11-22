from __future__ import annotations

import asyncio
import json
import logging

from tqdm.contrib.logging import logging_redirect_tqdm

from .config import Config
from .pipeline import Pipeline
from .statistic import StatisticPipeline
from .rubric_builder import RubricPipeline


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # httpx can log URLs with credentials; keep library logs quiet.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    try:
        config = Config.from_env()
        with logging_redirect_tqdm():
            runner = {"statistic": StatisticPipeline, "build-rubrics": RubricPipeline}.get(config.mode, Pipeline)(config)
            summary = asyncio.run(runner.run())
        metrics = {key: summary[key] for key in ["application_score", "application_score_10", "outcome_accuracy",
                                                  "mean_score", "published"] if key in summary}
        print(json.dumps({"run_dir": str(config.run_dir), "status": summary["status"], "counts": summary["counts"],
                          **metrics}, ensure_ascii=False))
        return 1 if summary["counts"]["failed"] else 0
    except (ValueError, OSError) as exc:
        logging.error("Run aborted: %s", exc)
        return 2
    except KeyboardInterrupt:
        logging.warning("Run interrupted; use the same configuration and RESUME=1")
        return 130

