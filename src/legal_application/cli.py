from __future__ import annotations

import asyncio
import json
import logging

from .config import Config
from .pipeline import Pipeline


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # httpx can log URLs with credentials; keep library logs quiet.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    try:
        config = Config.from_env()
        summary = asyncio.run(Pipeline(config).run())
        print(json.dumps({"run_dir": str(config.run_dir), "status": summary["status"], "counts": summary["counts"], "mean_score": summary["mean_score"]}, ensure_ascii=False))
        return 1 if summary["counts"]["failed"] else 0
    except (ValueError, OSError) as exc:
        logging.error("Run aborted: %s", exc)
        return 2
    except KeyboardInterrupt:
        logging.warning("Run interrupted; use the same configuration and RESUME=1")
        return 130

