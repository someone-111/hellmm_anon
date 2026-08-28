"""Re-derive the reported descriptors from finished runs, without re-running.

Every pathway CHE considered is persisted with its per-step energies, so ΔG° is
recoverable per step and the reporting question can be answered offline:

    U_L            limiting potential, from electrochemical steps only
    dG_chem_max    largest potential-independent step in the same pathway

These are separate physical quantities and are reported separately.  Folding
them into a single overpotential is what produces η < 0 — the reaction
apparently running below its own equilibrium potential, which cannot happen.
It is an artefact of a descriptor that cannot see chemical steps, not a result:
Pt(111) HER reports η = -0.146 eV while its real bottleneck is *H2 desorption at
+0.342 eV.  So η is withheld wherever the pathway carries an uphill chemical
step, and the pair is printed instead.

Usage:
    python analyse_descriptors.py runs/<dir> [runs/<dir> ...]
    python analyse_descriptors.py            # defaults to every runs/*/ found
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys


def dg0(step: dict) -> float:
    """ΔG° at U = 0 for one step: ΔE + ΔZPE − TΔS."""
    return step["delta_e"] + step["delta_zpe"] - step["delta_ts"]


def descriptors(pathway: dict) -> dict:
    """U_L, largest chemical step, and the limiting electrochemical step."""
    echem = [s for s in pathway["step_energies"] if s["n_electrons"] != 0]
    chem = [s for s in pathway["step_energies"] if s["n_electrons"] == 0]

    worst_echem = max(echem, key=lambda s: dg0(s) / abs(s["n_electrons"])) if echem else None
    worst_chem = max(chem, key=dg0) if chem else None

    return {
        "U_onset": pathway["U_onset"],
        "dG_chem_max": dg0(worst_chem) if worst_chem else None,
        "chem_step": (f'{worst_chem["parent"]} → {worst_chem["product"]}'
                      if worst_chem else None),
        "echem_step": (f'{worst_echem["parent"]} → {worst_echem["product"]}'
                       if worst_echem else None),
        "eta_raw": pathway["overpotential"],
    }


def collect(run_dirs: list[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for d in sorted(run_dirs):
        files = [f for f in glob.glob(os.path.join(d, "eval_che_*.json"))
                 if "state" not in f]
        if not files:
            continue
        for entry in json.load(open(files[0]))["results"]:
            best = next((p for p in entry.get("all_pathways", []) if p.get("is_best")), None)
            if best is None:
                continue
            rec = descriptors(best)
            rec["run"] = os.path.basename(d).split("_")[-1][-2:]
            out.setdefault(entry["catalyst"], []).append(rec)
    return out


INFEASIBLE = 900.0


def main(argv: list[str]) -> None:
    dirs = argv[1:] or [d for d in glob.glob("runs/*") if os.path.isdir(d)]
    data = collect(dirs)
    if not data:
        print("no runs with all_pathways found — needs a run from the current code")
        return

    for catalyst, recs in sorted(data.items()):
        print(f"\n{'=' * 74}\n{catalyst}\n{'=' * 74}")
        print(f"{'run':>4} {'U_L (V)':>9} {'dG_chem,max':>12} {'eta (eV)':>10}   limiting")
        feasible = []
        for r in sorted(recs, key=lambda x: x["run"]):
            if abs(r["eta_raw"]) > INFEASIBLE:
                print(f"{r['run']:>4} {'—':>9} {'—':>12} {'infeasible':>10}")
                continue
            feasible.append(r)
            chem = r["dG_chem_max"]
            # Withhold eta where a chemical step carries positive ΔG: the number
            # is not comparable with one from an all-PCET pathway.
            withheld = chem is not None and chem > 0.0
            eta = "withheld" if withheld else f"{r['eta_raw']:+.3f}"
            chem_s = "—" if chem is None else f"{chem:+.3f}"
            lim = r["chem_step"] if withheld else r["echem_step"]
            note = "  [chemical-limited]" if withheld else ""
            print(f"{r['run']:>4} {r['U_onset']:>+9.3f} {chem_s:>12} {eta:>10}   {lim}{note}")

        if feasible:
            u = [r["U_onset"] for r in feasible]
            print(f"\n  feasible {len(feasible)}/{len(recs)}"
                  f" | U_L mean {statistics.mean(u):+.3f}"
                  f" spread {max(u) - min(u):.3f} V")
            nchem = sum(1 for r in feasible
                        if r["dG_chem_max"] is not None and r["dG_chem_max"] > 0)
            if nchem:
                print(f"  {nchem}/{len(feasible)} carry an uphill chemical step — "
                      "overpotential withheld for those")


if __name__ == "__main__":
    main(sys.argv)
