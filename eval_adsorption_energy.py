"""Eval script for module 4 — adsorption energy computation.

Loops over ALL kept intermediates from the pruner for ALL catalyst cases.
Surface-defect labels are handled as follows:
  *V_O        — vacancy energy via compute_vacancy_energy (defect_vacancy)
  *O_lattice  — ΔE = 0.0, healed vacancy = clean slab in CHE frame (lattice_reference)
  other       — skipped (defect_other)

All physics/selection logic lives in hellmm.adsorption_energy; this script is
I/O glue: load the latest upstream JSON, call the stage functions per
catalyst case, write trajectories and results.

Output: eval_adsorption_energy_<timestamp>.json  (multi-catalyst)
        eval_adsorption_energy_state.json         (symlink/latest — read by eval_che.py)

Run with:
    python eval_adsorption_energy.py 2>&1 | tee eval_adsorption_energy_log.txt
"""

import glob
import json
import os
import warnings
from datetime import datetime

import ase.io

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*spglib.*")
warnings.filterwarnings("ignore", message=".*inc_structure.*")

from hellmm.adsorption_energy import (
    classify_intermediates,
    run_adsorption_energy,
    select_intermediates,
)
from hellmm.meta_reasoner import CatalystContext
from hellmm.tools import _best_device, load_mlip
from config import LLM_MODEL, MLIP_MODEL, MAX_DENTICITY, RUN_DIR

# ---------------------------------------------------------------------------
# Load most recent pruner and enumerator outputs
# ---------------------------------------------------------------------------

pruner_files = glob.glob(os.path.join(RUN_DIR, "eval_pruner_*.json"))
if not pruner_files:
    raise FileNotFoundError("No eval_pruner_*.json found. Run eval_pruner.py first.")
pruner_path = max(pruner_files, key=os.path.getmtime)
print(f"Loading pruner results from: {pruner_path}")

with open(pruner_path) as f:
    pruner_data = json.load(f)

# Enumerator JSON stores the starting adsorbate (e.g. "*CO2" for CO2RR).
# We need to compute its adsorption energy as the first step of the catalytic
# cycle — it is the reactant state in the Peterson CHE convention.
enum_files = glob.glob(os.path.join(RUN_DIR, "eval_enumerator_*.json"))
enum_start_ads: dict[str, str] = {}      # catalyst_key → starting_adsorbate
enum_smiles: dict[str, dict] = {}         # catalyst_key → {label → smiles}
enum_anchor_indices: dict[str, dict] = {} # catalyst_key → {label → [i0, i1]}
if enum_files:
    enum_path = max(enum_files, key=os.path.getmtime)
    with open(enum_path) as f:
        enum_data = json.load(f)
    for rec in enum_data.get("results", []):
        c = rec["context"]
        key = f"{c['composition']}({c['facet']})_{c['reaction']}"
        enum_start_ads[key] = rec.get("starting_adsorbate", "*")
        smiles_map   = {}
        anchors_map  = {}
        for inter in rec.get("intermediates", []):
            lbl = inter["label"]
            if inter.get("smiles"):
                smiles_map[lbl] = inter["smiles"]
            if inter.get("anchor_indices"):
                anchors_map[lbl] = inter["anchor_indices"]
        enum_smiles[key]         = smiles_map
        enum_anchor_indices[key] = anchors_map
    print(f"Loaded enumerator starting adsorbates from: {enum_path}")

# ---------------------------------------------------------------------------
# Load pathway constructor output — restricts MLIP to pathway intermediates only
# ---------------------------------------------------------------------------

# Map (composition, facet, reaction) → set of labels that appear in at least
# one valid pathway.  reaction is part of the key because the same
# (composition, facet) can appear under multiple reactions in one run (e.g.
# Pt(111) HER and Pt(111) ORR) — without it, one reaction's entry silently
# overwrites the other's.  If eval_pathway_constructor.py has not been run
# yet (e.g. manual ad-hoc run), fall back to computing all pruner-kept
# intermediates with a warning.
pathway_labels_map: dict[tuple[str, str, str], set[str]] = {}
pc_files = glob.glob(os.path.join(RUN_DIR, "eval_pathway_constructor_*.json"))
if pc_files:
    pc_path = max(pc_files, key=os.path.getmtime)
    with open(pc_path) as f:
        pc_data = json.load(f)
    for rec in pc_data.get("results", []):
        c   = rec["context"]
        key = (c["composition"], c["facet"], c["reaction"])
        pathway_labels_map[key] = set(rec.get("pathway_labels", []))
    print(f"Loaded pathway labels from: {pc_path}")
else:
    print("WARNING: No eval_pathway_constructor_*.json found — "
          "computing all pruner-kept intermediates (run eval_pathway_constructor.py first).")

# ---------------------------------------------------------------------------
# Load MLIP — once, shared across all cases
# ---------------------------------------------------------------------------

print(f"\nLoading {MLIP_MODEL}...")
device = _best_device()
calculator = load_mlip(MLIP_MODEL, device=device)
print(f"  Calculator loaded on {device}.")

# ---------------------------------------------------------------------------
# Loop over all catalyst cases
# ---------------------------------------------------------------------------

_STATUS_LABELS = {
    "fairchem_db":        "fairchem_db   ",
    "smiles_table":       "smiles_table  ",
    "llm_needed":         "llm_needed    ",
    "defect_vacancy":     "defect_vacancy",
    "lattice_reference":  "lattice_ref   ",
    "defect_other":       "defect_other  ",
    "filtered_denticity": "filtered      ",
    "reference":          "reference     ",
}
_SKIP_STATUSES = {"defect_other", "filtered_denticity", "reference"}

all_results: dict[str, dict] = {}  # catalyst_key → per-catalyst results

for entry in pruner_data["results"]:
    ctx_dict = entry["context"]
    ctx = CatalystContext(
        composition=ctx_dict["composition"],
        facet=ctx_dict["facet"],
        reaction=ctx_dict["reaction"],
        pH=ctx_dict["pH"],
        U=ctx_dict["U"],
    )
    catalyst_key = f"{ctx.composition}({ctx.facet})_{ctx.reaction}"

    print(f"\n{'='*60}")
    print(f"{catalyst_key}  {ctx.reaction}")
    print(f"{'='*60}")

    start_ads = enum_start_ads.get(catalyst_key, "*")
    pc_key = (ctx.composition, ctx.facet, ctx.reaction)
    pathway_labels = pathway_labels_map.get(pc_key)

    kept_items = select_intermediates(entry["kept"], start_ads, MAX_DENTICITY, pathway_labels)
    if kept_items is None:
        print(f"  0 pathways found by pathway_constructor — skipping MLIP for {catalyst_key}")
        continue
    if pathway_labels is not None:
        print(f"  Pathway filter: {len(kept_items)} pruner-kept intermediates "
              "appear in valid pathways")

    # Pre-check: classify every kept intermediate before touching the MLIP.
    _status_groups = classify_intermediates(kept_items, MAX_DENTICITY)
    print(f"\n  Pre-check ({len(kept_items)} intermediates):")
    for status, labels in sorted(_status_groups.items()):
        tag = " [SKIP]" if status in _SKIP_STATUSES else ""
        print(f"    {_STATUS_LABELS.get(status, status)} ({len(labels)}): {', '.join(labels)}{tag}")
    print()

    case_result = run_adsorption_energy(
        ctx, kept_items, calculator, MAX_DENTICITY, LLM_MODEL,
        enum_smiles=enum_smiles.get(catalyst_key),
        enum_anchor_indices=enum_anchor_indices.get(catalyst_key),
    )

    if case_result.skipped:
        print(f"  ERROR building slab: {case_result.skip_reason} — skipping {catalyst_key}")
        continue

    print(f"  Slab: {case_result.slab_n_atoms} atoms")
    for item in case_result.item_statuses:
        if item.status == "ok":
            detail = f"{item.n_candidates} cands, {item.path_used}" if item.path_used else item.detail
            print(f"  {item.label:30s}  ΔE = {item.delta_e_ads:+.3f} eV  ({detail})")
        else:
            print(f"  {item.label:30s}  → SKIP ({item.detail})")

    # Save traj files — one folder per catalyst case
    safe_key = catalyst_key.replace("(", "").replace(")", "")
    out_dir  = os.path.join(RUN_DIR, "results", safe_key)
    os.makedirs(out_dir, exist_ok=True)

    traj_files: dict[str, str] = {}
    for label, relaxed in case_result.relaxed_structures.items():
        safe_label = label.replace("*", "ads").replace(" ", "_").replace("/", "_")
        traj_path  = os.path.join(out_dir, f"{safe_label}.traj")
        ase.io.write(traj_path, relaxed)  # type: ignore[arg-type]
        traj_files[label] = traj_path

    all_results[catalyst_key] = {
        "context":        ctx_dict,
        "slab_n_atoms":   case_result.slab_n_atoms,
        "energy_results": case_result.energy_results,
        "traj_files":     traj_files,
    }

    print(f"  Computed {len(case_result.energy_results)-1} intermediates  "
          f"(+1 clean surface reference *)")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path  = os.path.join(RUN_DIR, f"eval_adsorption_energy_{MLIP_MODEL}_{timestamp}.json")

state = {
    "mlip_model":   MLIP_MODEL,
    "timestamp":    timestamp,
    "pruner_file":  pruner_path,
    "results":      all_results,
}

with open(out_path, "w") as f:
    json.dump(state, f, indent=2)

# Always overwrite the canonical "latest" file read by eval_che.py
with open(os.path.join(RUN_DIR, "eval_adsorption_energy_state.json"), "w") as f:
    json.dump(state, f, indent=2)

print(f"\nResults saved to {out_path}")
print(f"Canonical state written to {os.path.join(RUN_DIR, 'eval_adsorption_energy_state.json')}")
print("\nDone.")
