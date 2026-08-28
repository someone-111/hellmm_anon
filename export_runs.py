"""Rebuild results_per_run.csv and steps_per_run.csv from the run directories.

These two CSVs are the released artifacts behind every headline number, and
until now nothing regenerated them — they were produced once in an interactive
session, which makes the "re-derivable from released artifacts" claim
unfalsifiable in the wrong direction.  This script closes that.

Source of truth is runs/<timestamp>_<jobid>/eval_che_*.json, which carries
`all_pathways` (every candidate CHE scored, each with its own U_onset,
overpotential, is_best flag and per-step energies).

Usage:
    python export_runs.py                 # write both CSVs
    python export_runs.py --check         # verify against the files on disk,
                                          # write nothing, exit 1 on any diff

The reported batch is pinned below rather than globbed, because runs/ also
contains exploratory and pre-fix runs that must not enter the released data.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

# --------------------------------------------------------------------------
# The reported batch.  Pinned explicitly: runs/ contains earlier exploratory
# runs and pre-fix runs (10719*, 10722*, 10739*, bare-timestamp dirs) that are
# NOT part of any reported result.
#
#   deepseek-v3      n=10  primary arm
#   deepseek-v4-pro  n=3   resource-limited, see PAPER_BRIEF §2
#   deepseek-v3.2    n=9   one of ten failed on a malformed n_electrons type
#                          (job 10770861 produced no CHE output)
# --------------------------------------------------------------------------
BATCH: dict[str, list[str]] = {
    "deepseek-v3": [
        "10766639", "10766641", "10766642", "10766643", "10766644",
        "10766645", "10766646", "10766647", "10766648", "10766649",
    ],
    "deepseek-v4-pro": ["10767243", "10767247", "10767248"],
    "deepseek-v3.2": [
        "10770853", "10770854", "10770855", "10770856", "10770857",
        "10770858", "10770859", "10770860", "10770861", "10770862",
    ],
}

# The four systems every run attempts, from config.CASES.  Needed because the
# denominator of every reliability claim is *attempted* runs, and a run that
# produced no pathway leaves no trace in any downstream output — so the
# attempted set cannot be recovered from the run directories alone.
SYSTEMS = [("Pt", "111", "HER"), ("Pt", "111", "ORR"),
           ("Au", "111", "ORR"), ("Au", "110", "CO2RR")]

# Per-cell outcome.  Distinguishing these matters: "8/10 feasible" merges three
# different failures, and only `ok` and `infeasible` leave a row behind.
#   ok                   scored, admissible
#   infeasible           scored, then rejected by CHE (mixed polarity,
#                        counter-directional, or a chemical step > 2.0 eV)
#   no_pathway           pathway_constructor returned zero pathways.  This is
#                        the V4-Pro HER inversion: it proposes only *H, never
#                        *H2, so the only route is * -> *H -> * and there is no
#                        2-electron cycle to score.
#   incomplete_energies  pathways existed but at least one intermediate had no
#                        computable adsorption energy, so CHE skipped the case
#                        ("Skipping CHE — no complete pathways" in the log)
#   run_failed           the run produced no CHE output at all (V3.2 job
#                        10770861 died in the enumerator on a malformed
#                        n_electrons type)

INFEASIBLE = -999.0          # CHE's sentinel when every candidate was rejected
RESULTS_CSV = "results_per_run.csv"
STEPS_CSV = "steps_per_run.csv"

RESULTS_COLS = ["arm", "run", "catalyst", "status", "feasible", "U_onset",
                "eta", "limiting_step", "dG_chem_max", "pathway", "n_pathways"]
# First 14 columns are the original schema, unchanged and in the original
# order, so anything reading the old file keeps working.  U_onset and feasible
# are appended: steps_per_run.csv previously exported neither, which is why
# feasibility had to be re-derived downstream (see verify_claims.feasible).
STEPS_COLS = ["arm", "run", "catalyst", "pathway", "is_best", "step_idx",
              "parent", "product", "rule", "n_electrons",
              "delta_e", "delta_zpe", "delta_ts", "dG0",
              "U_onset", "feasible", "pathway_idx"]

# WHY pathway_idx EXISTS.  `pathway` is the species sequence, and it is NOT a
# unique key: two candidates can traverse the same nodes by different edges
# (e.g. *CO → * as `desorption` in one and `bond_dissociation` in another).
# 59 of the 412 candidate pathways in this batch collide that way.  Any
# group-by on (arm, run, catalyst, pathway) therefore merges distinct pathways
# and sums their steps — which manufactures, for example, a 4-electron CO2RR
# route out of two 2-electron ones.  pathway_idx is the index within that
# case's all_pathways list and makes the key unique.  The defect is present in
# the previously released steps_per_run.csv, which had no such column.


def r4(x) -> str:
    return "" if x is None else f"{round(float(x), 4)}"


def dg0(step: dict, u_op: float) -> float:
    """Potential-independent ΔG° for one step.

    The JSON stores ΔG at the *operating* potential of the case
    (ΔG(U) = ΔG° − n_e·U), so recovering ΔG° means adding n_e·U back.  This
    matters for every PCET step of every case run at U ≠ 0 — CO₂RR at
    U = 0.10 V and both ORR cases at U = 0.78 V.  Chemical steps have n_e = 0
    and are unaffected.
    """
    return step["delta_g"] + step["n_electrons"] * u_op


def chem_max(steps: list[dict], u_op: float) -> float | None:
    chem = [dg0(s, u_op) for s in steps if s["n_electrons"] == 0]
    return max(chem) if chem else None


def find_run_dir(job: str) -> str | None:
    hits = [d for d in glob.glob("runs/*/")
            if os.path.basename(d.rstrip("/")).split("_")[-1] == job]
    return hits[0] if len(hits) == 1 else (hits[0] if hits else None)


def arm_from_dir(d: str) -> str | None:
    """Infer the LLM arm from the enumerator output filename.

    CHE outputs are named by MLIP, not by LLM, so the arm is only recoverable
    from a sibling stage's filename.
    """
    enum = sorted(glob.glob(d + "eval_enumerator_*.json"))
    if not enum:
        return None
    return os.path.basename(enum[-1])[len("eval_enumerator_"):].rsplit("_", 2)[0]


def collect() -> tuple[list[dict], list[dict]]:
    results: list[dict] = []
    steps: list[dict] = []
    problems: list[str] = []

    for arm, jobs in BATCH.items():
        for job in jobs:
            d = find_run_dir(job)
            if not d:
                problems.append(f"{arm}/{job}: no run directory")
                continue
            found = arm_from_dir(d)
            # A run that died inside the enumerator has no enumerator output to
            # cross-check the arm against; that is expected, not a discrepancy.
            if found is not None and found != arm:
                problems.append(f"{arm}/{job}: enumerator says arm={found!r}")
            # How many pathways the constructor produced per system, used to
            # tell "found nothing" apart from "found something CHE could not
            # score".  Absent for runs that died before this stage.
            n_built: dict[str, int] = {}
            pcf = sorted(glob.glob(d + "eval_pathway_constructor_*.json"))
            if pcf:
                for rec in json.load(open(pcf[-1]))["results"]:
                    c = rec["context"]
                    n_built[f"{c['composition']}({c['facet']})_{c['reaction']}"] = \
                        rec["n_pathways"]

            che = sorted(glob.glob(d + "eval_che_*.json"))
            if not che:
                for comp, fac, rxn in SYSTEMS:
                    results.append({c: "" for c in RESULTS_COLS} | {
                        "arm": arm, "run": job,
                        "catalyst": f"{comp}({fac})_{rxn}",
                        "status": "run_failed", "feasible": "False",
                    })
                continue

            scored = set()
            for rec in json.load(open(che[-1]))["results"]:
                cat = rec["catalyst"]
                allp = rec.get("all_pathways") or []
                if not allp:
                    problems.append(f"{arm}/{job} {cat}: no all_pathways")
                u = rec.get("U_onset")
                u_op = float(rec.get("U", 0.0))
                cm = chem_max(rec.get("step_energies") or [], u_op)
                stored = rec.get("dG_chem_max")
                if stored is not None and cm is not None and abs(stored - cm) > 5e-4:
                    problems.append(f"{arm}/{job} {cat}: dG_chem_max stored "
                                    f"{stored:.4f} != derived {cm:.4f}")
                scored.add(cat)
                ok = u is not None and u > INFEASIBLE + 1
                results.append({
                    "arm": arm, "run": job, "catalyst": cat,
                    "status": "ok" if ok else "infeasible",
                    "feasible": str(ok),
                    "U_onset": r4(u), "eta": r4(rec.get("overpotential")),
                    "limiting_step": rec.get("limiting_step", ""),
                    "dG_chem_max": r4(cm),
                    "pathway": " → ".join(rec.get("best_pathway") or []),
                    "n_pathways": len(allp),
                })
                for pidx, p in enumerate(allp):
                    pw = " → ".join(p["intermediates"])
                    pu = p.get("U_onset")
                    feas = pu is not None and pu > INFEASIBLE + 1
                    for i, s in enumerate(p["step_energies"]):
                        steps.append({
                            "arm": arm, "run": job, "catalyst": cat,
                            "pathway": pw, "pathway_idx": pidx,
                            "is_best": str(bool(p.get("is_best"))),
                            "step_idx": i, "parent": s["parent"],
                            "product": s["product"], "rule": s["rule"],
                            "n_electrons": s["n_electrons"],
                            "delta_e": r4(s["delta_e"]),
                            "delta_zpe": r4(s["delta_zpe"]),
                            "delta_ts": r4(s["delta_ts"]),
                            "dG0": r4(dg0(s, u_op)),
                            "U_onset": r4(pu), "feasible": str(feas),
                        })

            # Cells that never reached CHE leave no row of their own; emit one
            # so the attempted denominator is present in the artifact.
            for comp, fac, rxn in SYSTEMS:
                cat = f"{comp}({fac})_{rxn}"
                if cat in scored:
                    continue
                built = n_built.get(cat)
                results.append({c: "" for c in RESULTS_COLS} | {
                    "arm": arm, "run": job, "catalyst": cat,
                    "status": ("no_pathway" if built == 0 else
                               "incomplete_energies" if built else "not_reached"),
                    "feasible": "False",
                    "n_pathways": "" if built is None else built,
                })
    for p in problems:
        print(f"  WARNING  {p}", file=sys.stderr)
    return results, steps


def write(path: str, cols: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")


def check(path: str, cols: list[str], rows: list[dict]) -> bool:
    """Compare against the file on disk over the columns that file actually has."""
    if not os.path.exists(path):
        print(f"  {path}: absent")
        return False
    on_disk = list(csv.DictReader(open(path)))
    shared = [c for c in cols if c in (on_disk[0].keys() if on_disk else [])]
    key = lambda r: tuple(str(r[c]) for c in shared)          # noqa: E731
    a, b = sorted(map(key, on_disk)), sorted(map(key, rows))
    if a == b:
        print(f"  {path}: MATCH ({len(a)} rows over {len(shared)} shared columns)")
        return True
    only_disk = [x for x in a if x not in set(b)]
    only_new = [x for x in b if x not in set(a)]
    print(f"  {path}: DIFF — {len(on_disk)} on disk, {len(rows)} regenerated; "
          f"{len(only_disk)} only-on-disk, {len(only_new)} only-regenerated")
    for x in only_disk[:3]:
        print(f"      only on disk : {x}")
    for x in only_new[:3]:
        print(f"      only new     : {x}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="compare against the CSVs on disk, write nothing")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="restrict to these arms (default: all in BATCH)")
    args = ap.parse_args()

    if args.arms:
        for a in list(BATCH):
            if a not in args.arms:
                del BATCH[a]

    results, steps = collect()
    n_runs = len({(r["arm"], r["run"]) for r in results})
    n_paths = len({(s["arm"], s["run"], s["catalyst"], s["pathway"]) for s in steps})
    print(f"\narms={sorted({r['arm'] for r in results})}  runs={n_runs}  "
          f"cases={len(results)}  pathways={n_paths}  steps={len(steps)}")

    if args.check:
        ok = check(RESULTS_CSV, RESULTS_COLS, results)
        ok &= check(STEPS_CSV, STEPS_COLS, steps)
        return 0 if ok else 1

    write(RESULTS_CSV, RESULTS_COLS, results)
    write(STEPS_CSV, STEPS_COLS, steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
