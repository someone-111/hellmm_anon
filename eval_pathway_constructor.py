"""Eval script for module 5 — pathway construction (topology only, no MLIP).

Runs immediately after eval_pruner.py and before eval_adsorption_energy.py.
Catches 0-pathway cases before any GPU time is spent — pathway construction
is pure graph traversal and takes seconds regardless of catalyst size.

Saves the union of intermediate labels across all valid pathways per catalyst.
eval_adsorption_energy.py reads this file and restricts its MLIP relaxations
to only those intermediates, skipping any pruner-kept node that never appears
in a valid pathway.

All orchestration logic lives in
hellmm.pathway_constructor.run_pathway_constructor(); this script is I/O
glue: load the latest upstream JSON, call the stage, save.

Output: eval_pathway_constructor_<timestamp>.json

Run with:
    python eval_pathway_constructor.py 2>&1 | tee eval_pathway_constructor_log.txt
"""

import json
import os
import sys
import warnings
from datetime import datetime

from hellmm.io import load_latest_json
from hellmm.pathway_constructor import run_pathway_constructor
from config import MAX_DENTICITY, RUN_DIR

warnings.simplefilter("always")

# ---------------------------------------------------------------------------
# Load upstream outputs
# ---------------------------------------------------------------------------

enum_path,   enum_data   = load_latest_json(os.path.join(RUN_DIR, "eval_enumerator_*.json"))
pruner_path, pruner_data = load_latest_json(os.path.join(RUN_DIR, "eval_pruner_*.json"))
print(f"Enumerator : {enum_path}")
print(f"Pruner     : {pruner_path}")

# ---------------------------------------------------------------------------
# Run pathway construction for each catalyst case
# ---------------------------------------------------------------------------

records = run_pathway_constructor(pruner_data, enum_data, MAX_DENTICITY, max_steps=10)
any_pathways = any(r["n_pathways"] > 0 for r in records)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path  = os.path.join(RUN_DIR, f"eval_pathway_constructor_{timestamp}.json")

with open(out_path, "w") as f:
    json.dump({
        "timestamp":       timestamp,
        "enumerator_file": enum_path,
        "pruner_file":     pruner_path,
        "results":         records,
    }, f, indent=2)

print(f"\nResults saved to {out_path}")

if not any_pathways:
    print("\nERROR: 0 pathways found for all cases. "
          "Fix the reaction graph before spending GPU time on adsorption energies.")
    sys.exit(1)

print("Done.")
