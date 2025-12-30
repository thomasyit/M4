import json
import os
from datetime import datetime
from typing import Dict, Any


def log_run_summary(summary: Dict[str, Any]) -> str:
    out_dir = os.getenv("RUN_SUMMARY_DIR", "state/run_summaries")
    os.makedirs(out_dir, exist_ok=True)
    ts = summary.get("ts") or int(datetime.utcnow().timestamp())
    run_id = summary.get("run_id", f"run_{ts}")
    path = os.path.join(out_dir, f"{run_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path
