"""Independent verification of the mechanism-selection claims (§4b, §5.3).

Reads ONLY steps_per_run.csv and re-derives every claim from the step records,
with no reference to how those claims were originally reached.  It exists
because the analysis behind §4b and §5.3 was revised twice before settling, and
a claim that rests on an interactive session is not a claim a reader can check.

Run:  python verify_claims.py
Exit code 0 if every claim reproduces, 1 otherwise.

Each check prints the derivation, not just a verdict, so a disagreement shows
where it comes from.
"""
from __future__ import annotations

import collections
import csv
import sys

CSV = "steps_per_run.csv"
# Total electrons the reaction requires, from ReactionDefinition.
N_E = {"Pt(111)_HER": 2, "Pt(111)_ORR": 4, "Au(111)_ORR": 4, "Au(110)_CO2RR": 2}
# A chemical step above this is a blocking barrier no potential can drive.
BLOCKING_EV = 0.75

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if detail:
        print(f"         {detail}")
    if not ok:
        failures.append(label)


def load() -> dict:
    """steps_per_run.csv -> {(arm, run, catalyst, pathway): [step, ...]}

    The key carries `pathway_idx` when the file provides it, because `pathway`
    (the species sequence) is NOT unique: two candidates can traverse the same
    species by different edges — *CO → * as `desorption` in one and
    `bond_dissociation` in another — and 59 of this batch's 412 candidates
    collide that way.  Keying on the species string alone merges them and sums
    their steps, which fabricates a 4-electron CO2RR route out of two
    2-electron ones and fails claim 1 for the wrong reason.  Files predating
    the pathway_idx column fall back to the old key.
    """
    paths: dict = collections.defaultdict(list)
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            for k in ("n_electrons", "step_idx"):
                r[k] = int(r[k])
            for k in ("delta_e", "delta_zpe", "delta_ts", "dG0"):
                r[k] = float(r[k])
            r["is_best"] = str(r["is_best"]).lower() == "true"
            uid = r.get("pathway_idx", "")
            paths[(r["arm"], r["run"], r["catalyst"],
                   r["pathway"] if uid == "" else f"{r['pathway']}#{uid}")].append(r)
    return paths


def u_onset(steps: list[dict]) -> float | None:
    """U_L for a cathodic reaction: -max(dG0 per electron) over echem steps."""
    ech = [s for s in steps if s["n_electrons"] != 0]
    if not ech:
        return None
    return -max(s["dG0"] / abs(s["n_electrons"]) for s in ech)


def worst_chem(steps: list[dict]) -> float | None:
    chem = [s["dG0"] for s in steps if s["n_electrons"] == 0]
    return max(chem) if chem else None


def feasible(steps: list[dict], cathodic: bool = True) -> bool:
    """Re-implement CHE's admissibility test from the step records alone.

    `is_best` cannot be used for this: CHE flags a best *candidate* even when
    every candidate was rejected, so an infeasible run still carries an
    is_best=True pathway.  steps_per_run.csv does not export U_onset or the
    infeasibility sentinel, so feasibility has to be re-derived here — which is
    the point of the exercise.
    """
    ech = [s for s in steps if s["n_electrons"] != 0]
    if not ech:
        return False
    cath = [s for s in ech if s["n_electrons"] < 0]
    an = [s for s in ech if s["n_electrons"] > 0]
    if cath and an:
        return False                      # mixed polarity
    if (an if cathodic else cath):
        return False                      # counter-directional
    chem = [s["dG0"] for s in steps if s["n_electrons"] == 0]
    if chem and max(chem) > 2.0:          # config.CHE_CHEM_STEP_MAX_DG
        return False
    return True


def main() -> int:
    paths = load()
    # The reported result per case = the admissible pathway with the highest
    # U_L (cathodic: highest U_L needs the least overpotential).
    by_case: dict = collections.defaultdict(list)
    for k, v in paths.items():
        if feasible(v):
            by_case[k[:3]].append((k, v))
    selected = {}
    for case, cands in by_case.items():
        k, v = max(cands, key=lambda kv: u_onset(kv[1]))
        selected[k] = v
    flagged = sum(1 for v in paths.values() if any(s["is_best"] for s in v))
    print(f"loaded {len(paths)} pathways; {flagged} carry is_best, "
          f"{len(selected)} are admissible results across {len(by_case)} cases\n")

    # ---------------------------------------------------------------- claim 1
    # Every SELECTED pathway is a valid route: electron count matches the
    # reaction definition, and no electrochemical step runs the wrong way.
    print("CLAIM 1 — every selected pathway is a valid n-electron cathodic route")
    bad = []
    for (arm, run, cat, pw), steps in selected.items():
        ne = sum(abs(s["n_electrons"]) for s in steps)
        anodic = [s for s in steps if s["n_electrons"] > 0]
        if ne != N_E[cat] or anodic:
            bad.append((arm, run, cat, ne, len(anodic), pw))
    check("all selected pathways have correct |n_e| and no anodic steps",
          not bad,
          "offenders: " + "; ".join(f"{a}/{r} {c} n_e={n} anodic={x}"
                                    for a, r, c, n, x, _ in bad[:5]) if bad else
          f"checked {len(selected)} selected pathways")

    # ---------------------------------------------------------------- claim 2
    # The ORR "second attractor" is a DIFFERENT valid mechanism, not a
    # truncation: it omits *O->*OH yet still carries 4 electrons.
    print("\nCLAIM 2 — ORR alternatives are complete mechanisms, not truncations")
    orr = [(k, v) for k, v in selected.items() if k[2] == "Pt(111)_ORR"]
    viaO = [(k, v) for k, v in orr
            if any(s["parent"] == "*O" and s["product"] == "*OH" for s in v)]
    alt = [(k, v) for k, v in orr if (k, v) not in viaO]
    alt_ok = all(sum(abs(s["n_electrons"]) for s in v) == 4 for _, v in alt)
    check("every ORR alternative still carries exactly 4 electrons", alt_ok,
          f"{len(viaO)} selected via *O->*OH, {len(alt)} via an alternative route; "
          f"alternatives: " + ", ".join(sorted({k[3].split(' → ')[2] for k, _ in alt})))

    ul_viaO = sorted({round(u_onset(v), 3) for _, v in viaO})
    ul_alt = sorted({round(u_onset(v), 3) for _, v in alt})
    check("the alternative route has the HIGHER U_L (needs less overpotential)",
          bool(ul_alt) and bool(ul_viaO) and min(ul_alt) > min(ul_viaO),
          f"U_L via *O->*OH: {ul_viaO}   via alternative: {ul_alt}")

    # ---------------------------------------------------------------- claim 3
    # Selecting the alternative is CORRECT, not a descriptor failure — the
    # alternatives carry no blocking chemical step.
    print("\nCLAIM 3 — ORR alternatives carry no blocking chemical step")
    worst = [(k[1], worst_chem(v)) for k, v in alt]
    check(f"no selected ORR alternative has a chemical step > {BLOCKING_EV} eV",
          all(w is None or w <= BLOCKING_EV for _, w in worst),
          "worst chemical step per run: " + ", ".join(f"{r}:{w:+.3f}" for r, w in worst))

    # ---------------------------------------------------------------- claim 4
    # The descriptor DOES fail, but rarely: count selected pathways whose U_L
    # ignores a blocking chemical step.
    print("\nCLAIM 4 — the descriptor failure is real but rare")
    blocked = [(k[0], k[1], k[2], worst_chem(v), u_onset(v))
               for k, v in selected.items()
               if (worst_chem(v) or 0) > BLOCKING_EV]
    check("at most 2 selected pathways are blocked yet still scored",
          len(blocked) <= 2,
          "blocked selections: " + "; ".join(
              f"{a}/{r} {c} chem={w:+.3f} U_L={u:+.3f}" for a, r, c, w, u in blocked)
          if blocked else "none found")

    # ---------------------------------------------------------------- claim 5
    # HER: the bottleneck is a chemical step, in EVERY run and arm.
    print("\nCLAIM 5 — HER is chemical-step limited in every run")
    her = [(k, v) for k, v in selected.items() if k[2] == "Pt(111)_HER"]
    chem_lim = [(k[0], k[1], worst_chem(v), u_onset(v)) for k, v in her
                if (worst_chem(v) or -9) > max(s["dG0"] for s in v if s["n_electrons"] != 0)]
    check("every HER run's worst chemical step exceeds its worst echem step",
          len(chem_lim) == len(her),
          f"{len(chem_lim)}/{len(her)} runs; "
          f"chem steps: {sorted({round(w,3) for _,_,w,_ in chem_lim})}, "
          f"U_L: {sorted({round(u,3) for _,_,_,u in chem_lim})}")

    print()
    if failures:
        print(f"{len(failures)} CLAIM(S) FAILED — do not publish these as written:")
        for f in failures:
            print(f"   - {f}")
        return 1
    print("all claims reproduce from steps_per_run.csv alone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
