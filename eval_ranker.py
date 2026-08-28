"""Eval script for module 8 — catalyst ranker.

Loads the adsorption energy state, enumerator, and pruner JSONs for all
catalyst cases, constructs pathways, and ranks by CHE overpotential.
Optionally runs the Pourbaix stability gate (requires MY_MP_API_KEY in .env).

All orchestration logic lives in hellmm.ranker.run_ranker(); this script is
I/O glue: load the latest upstream JSON, call the stage, print/save.

Prerequisites:
  eval_adsorption_energy.py → eval_adsorption_energy_state.json
  eval_enumerator.py        → eval_enumerator_*.json
  eval_pruner.py            → eval_pruner_*.json

Run with:
    python eval_ranker.py 2>&1 | tee eval_ranker_log.txt
"""

import json
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*spglib.*")
warnings.filterwarnings("ignore", message=".*inc_structure.*")

from hellmm.io import load_latest_json
from hellmm.ranker import print_leaderboard, run_ranker
from config import MAX_DENTICITY, CHE_CHEM_STEP_MAX_DG, TEMPERATURE, RUN_DIR
from hellmm.tools import _best_device, compute_gas_thermo_corrections, load_mlip

# ---------------------------------------------------------------------------
# Load state files
# ---------------------------------------------------------------------------

_ads_state_path = os.path.join(RUN_DIR, "eval_adsorption_energy_state.json")
if not os.path.exists(_ads_state_path):
    raise FileNotFoundError(
        f"{_ads_state_path} not found. "
        "Run eval_adsorption_energy.py first."
    )

with open(_ads_state_path) as f:
    state = json.load(f)

mlip_model = state["mlip_model"]

if "results" in state:
    all_catalyst_states = state["results"]
else:
    all_catalyst_states = {state["catalyst_key"]: state}

enum_path,   enum_data   = load_latest_json(os.path.join(RUN_DIR, "eval_enumerator_*.json"))
pruner_path, pruner_data = load_latest_json(os.path.join(RUN_DIR, "eval_pruner_*.json"))
pc_path,     pc_data     = load_latest_json(os.path.join(RUN_DIR, "eval_pathway_constructor_*.json"))

# ---------------------------------------------------------------------------
# Load MLIP for vibrational corrections
# ---------------------------------------------------------------------------

device     = _best_device()
calculator = load_mlip(mlip_model, device=device)
print(f"Calculator loaded ({mlip_model} on {device})")

gas_thermo = compute_gas_thermo_corrections(calculator)

# ---------------------------------------------------------------------------
# Run the ranker stage
# ---------------------------------------------------------------------------

result = run_ranker(
    enum_data, all_catalyst_states, pruner_data, pc_data,
    calculator, gas_thermo, MAX_DENTICITY, TEMPERATURE, CHE_CHEM_STEP_MAX_DG,
)

if not result.ranked:
    print("\nNo catalyst results to rank. Run eval_adsorption_energy.py first.")
else:
    print_leaderboard(result)

    # Save timestamped JSON (canonical state for downstream use / reproducibility)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = os.path.join(RUN_DIR, f"eval_ranker_{mlip_model}_{timestamp}.json")
    rows = [
        {
            "catalyst":          f"{r.context.composition}({r.context.facet})",
            "reaction":          r.context.reaction,
            "U_op":              r.context.U,
            "pH":                r.context.pH,
            "overpotential":     r.best_overpotential,
            "U_onset":           r.best_U_onset,
            "pourbaix_stable":   r.pourbaix_stable,
            "best_pathway":      r.best_pathway.intermediates,
        }
        for r in result.ranked
    ]
    output = {
        "mlip_model":  mlip_model,
        "timestamp":   timestamp,
        "temperature": result.temperature,
        "results":     rows,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    with open(os.path.join(RUN_DIR, "eval_ranker_state.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")
    print("Canonical state written to eval_ranker_state.json")

print("Done.")
