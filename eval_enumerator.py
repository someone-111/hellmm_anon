"""Eval script for module 2 — rule enumeration.

All orchestration logic lives in hellmm.enumerator.run_enumerator(); this
script is I/O glue: load the latest meta_reasoner JSON, build the config
inputs (catalyst cases + operating points), call the stage, save.
"""

import json
import os
import warnings
from datetime import datetime

from hellmm.enumerator import run_enumerator
from hellmm.io import load_latest_json
from config import LLM_MODEL, MAX_DEPTH, ENUMERATOR_N_RUNS, RUN_DIR, unique_catalyst_cases, operating_points_for

warnings.simplefilter("always")

# ---------------------------------------------------------------------------
# Load most recent meta_reasoner output
# ---------------------------------------------------------------------------

meta_path, meta_data = load_latest_json(os.path.join(RUN_DIR, "eval_results_*.json"))
print(f"Loading meta_reasoner results from: {meta_path}")

# ---------------------------------------------------------------------------
# Cases to evaluate — from config.py; depth taken from meta_reasoner's suggested_depth
# ---------------------------------------------------------------------------

catalyst_cases = unique_catalyst_cases()
operating_points_map = {
    (ctx.composition, ctx.facet, ctx.reaction, start): operating_points_for(
        ctx.composition, ctx.facet, ctx.reaction, start
    )
    for start, ctx in catalyst_cases
}

records = run_enumerator(
    catalyst_cases, operating_points_map, meta_data, LLM_MODEL, MAX_DEPTH, ENUMERATOR_N_RUNS,
)
for rec in records:
    rec["meta_reasoner_file"] = meta_path

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = os.path.join(RUN_DIR, f"eval_enumerator_{LLM_MODEL}_{timestamp}.json")
with open(out_path, "w") as f:
    json.dump(
        {"model": LLM_MODEL, "timestamp": timestamp, "meta_reasoner_file": meta_path, "results": records},
        f, indent=2,
    )

print(f"\nResults saved to {out_path}")
