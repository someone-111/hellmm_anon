"""Eval script for module 1 — meta-reasoner rule selection.

All orchestration logic lives in hellmm.meta_reasoner.run_meta_reasoner();
this script is I/O glue: build the catalyst cases from config.py, call the
stage, save.
"""

import json
import os
import warnings
from datetime import datetime

from hellmm.meta_reasoner import run_meta_reasoner
from config import LLM_MODEL, RUN_DIR, unique_catalyst_cases

warnings.simplefilter("always")

records = run_meta_reasoner(unique_catalyst_cases(), LLM_MODEL)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = os.path.join(RUN_DIR, f"eval_results_{LLM_MODEL}_{timestamp}.json")
with open(out_path, "w") as f:
    json.dump({"model": LLM_MODEL, "timestamp": timestamp, "results": records}, f, indent=2)

print(f"\nResults saved to {out_path}")
