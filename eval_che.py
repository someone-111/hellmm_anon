"""Eval script for module 6 — CHE free energy analysis.

Loops over ALL catalyst cases present in the adsorption energy state file.
For each case:
  Step 1 — ZPE + TΔS vibrational corrections (ASE Vibrations + HarmonicThermo)
  Step 2 — Pathway construction (module 5) from enumerator/pruner JSON
  Step 3 — Per-step ΔG(U, T) via compute_pathway_free_energy
  Step 4 — Overpotential and limiting step summary

All orchestration logic lives in hellmm.che.run_che(); this script is I/O
glue: load the latest upstream JSON, call the stage, print the leaderboard,
save.

Prerequisites:
  Run eval_adsorption_energy.py first — it saves eval_adsorption_energy_state.json

Run with:
    python eval_che.py 2>&1 | tee eval_che_log.txt
"""

import json
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*spglib.*")
warnings.filterwarnings("ignore", message=".*inc_structure.*")

from hellmm.che import run_che
from hellmm.io import load_latest_json
from config import MAX_DENTICITY, CHE_CHEM_STEP_MAX_DG, TEMPERATURE, RUN_DIR
from hellmm.tools import _best_device, compute_gas_thermo_corrections, load_mlip

# ---------------------------------------------------------------------------
# Load state from eval_adsorption_energy.py
# ---------------------------------------------------------------------------

print("=" * 60)
print("LOADING — adsorption energy state")
print("=" * 60)

_ads_state_path = os.path.join(RUN_DIR, "eval_adsorption_energy_state.json")
if not os.path.exists(_ads_state_path):
    raise FileNotFoundError(
        f"{_ads_state_path} not found. "
        "Run eval_adsorption_energy.py first."
    )

with open(_ads_state_path) as f:
    state = json.load(f)

mlip_model = state["mlip_model"]

# Support both old format (single catalyst) and new format (multi-catalyst dict)
if "results" in state:
    all_catalyst_states = state["results"]   # new: {catalyst_key: {...}}
else:
    # Legacy single-catalyst format — wrap it
    catalyst_key = state["catalyst_key"]
    all_catalyst_states = {catalyst_key: state}

print(f"  MLIP model  : {mlip_model}")
print(f"  Catalysts   : {list(all_catalyst_states.keys())}")

# ---------------------------------------------------------------------------
# Load most recent enumerator + pruner JSONs
# ---------------------------------------------------------------------------

enum_path,   enum_data   = load_latest_json(os.path.join(RUN_DIR, "eval_enumerator_*.json"))
pruner_path, pruner_data = load_latest_json(os.path.join(RUN_DIR, "eval_pruner_*.json"))
pc_path,     pc_data     = load_latest_json(os.path.join(RUN_DIR, "eval_pathway_constructor_*.json"))
print(f"\n  Enumerator          : {enum_path}")
print(f"  Pruner              : {pruner_path}")
print(f"  Pathway constructor : {pc_path}")


# ---------------------------------------------------------------------------
# Load MLIP calculator (shared across all cases)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"LOADING — MLIP calculator ({mlip_model})")
print("=" * 60)

device = _best_device()
print(f"  Device: {device}")
calculator = load_mlip(mlip_model, device=device)
print("  Calculator loaded.")

# ---------------------------------------------------------------------------
# Gas thermo corrections (computed once)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 1a — gas phase vibrational corrections (shared)")
print("=" * 60)
gas_thermo = compute_gas_thermo_corrections(calculator)
for mol, corr in gas_thermo.items():
    print(f"  {mol:4s}: ZPE={corr.zpe:.4f} eV  TΔS={corr.ts:.4f} eV")

# ---------------------------------------------------------------------------
# Run the CHE stage
# ---------------------------------------------------------------------------

summary_rows = run_che(
    enum_data, all_catalyst_states, pruner_data, pc_data,
    calculator, gas_thermo, MAX_DENTICITY, TEMPERATURE, CHE_CHEM_STEP_MAX_DG,
)

# ---------------------------------------------------------------------------
# Final leaderboard
# ---------------------------------------------------------------------------

if summary_rows:
    print(f"\n{'='*60}")
    print("LEADERBOARD")
    print(f"{'='*60}")
    for i, row in enumerate(summary_rows, 1):
        print(f"  {i}. {row['catalyst']:20s}  {row['reaction']:6s}  "
              f"pH={row['pH']}  U={row['U']:.2f} V  "
              f"η={row['overpotential']:.3f} eV  "
              f"U_onset={row['U_onset']:.3f} V  "
              f"limiting: {row['limiting_step']}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RUN_DIR, f"eval_che_{mlip_model}_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump({"mlip_model": mlip_model, "results": summary_rows}, f, indent=2)
    with open(os.path.join(RUN_DIR, "eval_che_state.json"), "w") as f:
        json.dump({"mlip_model": mlip_model, "results": summary_rows}, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Canonical state written to {os.path.join(RUN_DIR, 'eval_che_state.json')}")

print("\nDone.")
