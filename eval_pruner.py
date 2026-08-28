"""Eval script for module 3 — LLM plausibility pruner.

All orchestration logic lives in hellmm.pruner.run_pruner(); this script is
I/O glue: load the latest enumerator JSON, call the stage, save.
"""

import json
import os
import warnings
from datetime import datetime

from hellmm.io import load_latest_json
from hellmm.pruner import run_pruner
from config import LLM_MODEL, PRUNER_THRESHOLD, PRUNER_N_RUNS, MAX_DENTICITY, RUN_DIR

warnings.simplefilter("always")

# ---------------------------------------------------------------------------
# Load most recent enumerator output
# ---------------------------------------------------------------------------

enum_path, enum_data = load_latest_json(os.path.join(RUN_DIR, "eval_enumerator_*.json"))
print(f"Loading enumerator results from: {enum_path}")

# ---------------------------------------------------------------------------
# Run pruner on each case
# ---------------------------------------------------------------------------

records = run_pruner(enum_data, LLM_MODEL, PRUNER_THRESHOLD, PRUNER_N_RUNS, MAX_DENTICITY)
for rec in records:
    rec["enumerator_file"] = enum_path

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = os.path.join(RUN_DIR, f"eval_pruner_{LLM_MODEL}_{timestamp}.json")
with open(out_path, "w") as f:
    json.dump({
        "model": LLM_MODEL,
        "timestamp": timestamp,
        "enumerator_file": enum_path,
        "results": records,
    }, f, indent=2)

print(f"\nResults saved to {out_path}")
