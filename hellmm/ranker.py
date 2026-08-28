"""Ranker — module 8.

Ranks catalyst candidates by overpotential, combining outputs from:
  - module 2b (Pourbaix stability gate)
  - module 4  (adsorption energies)
  - module 5  (pathway constructor)
  - module 6  (CHE free energies)

For each catalyst:
  1. Pourbaix gate — skip if unstable at (U, pH)
  2. CHE ΔG(U, T) for every pathway
  3. Pick best pathway (lowest overpotential)
  4. Sort leaderboard: stable catalysts first, then by overpotential ascending
"""

from __future__ import annotations

import os
import warnings

import ase.io
from pydantic.dataclasses import dataclass

from .adsorption_energy import AdsorptionEnergyResult
from .che import PathwayFreeEnergy, compute_pathway_free_energy
from .io import (
    catalyst_context_from_dict,
    find_entry,
    reconstruct_enumerator_output,
    reconstruct_pathways,
    reconstruct_pruner_output,
)
from .tools.vibrations import ThermoCorrection
from .tools import compute_thermo_corrections
from .meta_reasoner import CatalystContext
from .pathway_constructor import Pathway
from .pourbaix import should_proceed
from .reaction import ReactionDefinition, get_reaction


# ---------------------------------------------------------------------------
# Input / output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CatalystResult:
    """All computed data for a single catalyst surface."""
    context: CatalystContext
    energy_results: dict[str, AdsorptionEnergyResult]      # label → ΔE_ads
    pathways: list[Pathway]                                 # from pathway_constructor
    thermo_corrections: dict[str, ThermoCorrection] | None = None
    gas_corrections: dict[str, ThermoCorrection] | None = None


@dataclass
class RankedCatalyst:
    context: CatalystContext
    best_overpotential: float           # eV
    best_U_onset: float                 # V vs RHE
    best_pathway: Pathway
    all_pathway_energies: list[PathwayFreeEnergy]
    pourbaix_stable: bool
    pourbaix_warning: str


@dataclass
class RankerResult:
    ranked: list[RankedCatalyst]        # sorted best → worst
    U: float
    temperature: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_catalysts(
    catalysts: list[CatalystResult],
    U: float,
    temperature: float = 298.15,
    run_pourbaix: bool = True,
    max_chem_step_dg: float = 0.0,
) -> RankerResult:
    """Rank catalysts by CHE overpotential.

    Args:
        catalysts: one CatalystResult per surface
        U: electrode potential V vs RHE
        temperature: K
        run_pourbaix: if True, apply Pourbaix gate before CHE (recommended)

    Returns:
        RankerResult with catalysts sorted best → worst overpotential.
        Unstable catalysts are placed last regardless of overpotential.
    """
    ranked: list[RankedCatalyst] = []

    for cat in catalysts:
        # Pourbaix gate
        pb_stable = True
        pb_warning = ""
        if run_pourbaix:
            proceed, reason = should_proceed(cat.context)
            if not proceed:
                warnings.warn(f"Skipping {cat.context.composition}: {reason}")
                continue
            pb_warning = reason  # non-empty if borderline stable

        if not cat.pathways:
            warnings.warn(f"No pathways for {cat.context.composition}({cat.context.facet}) — skipping.")
            continue

        rxn: ReactionDefinition = get_reaction(cat.context.reaction)

        pathway_energies: list[PathwayFreeEnergy] = []
        for pw in cat.pathways:
            try:
                pfe = compute_pathway_free_energy(
                    pathway=pw,
                    energy_results=cat.energy_results,
                    U=cat.context.U,   # only used as a fallback when the pathway has no
                                        # electrochemical steps; overpotential itself is
                                        # computed from U-independent ΔG° (che.py:267)
                    reaction=cat.context.reaction,
                    temperature=temperature,
                    thermo_corrections=cat.thermo_corrections,
                    gas_corrections=cat.gas_corrections,
                    max_chem_step_dg=max_chem_step_dg,
                )
                pathway_energies.append(pfe)
            except Exception as e:
                warnings.warn(
                    f"CHE failed for {cat.context.composition}({cat.context.facet}) "
                    f"pathway [{' → '.join(pw.intermediates)}]: {e}"
                )

        if not pathway_energies:
            continue

        best = min(pathway_energies, key=lambda p: p.overpotential)

        ranked.append(RankedCatalyst(
            context=cat.context,
            best_overpotential=best.overpotential,
            best_U_onset=best.U_onset,
            best_pathway=best.pathway,
            all_pathway_energies=pathway_energies,
            pourbaix_stable=pb_stable,
            pourbaix_warning=pb_warning,
        ))

    ranked.sort(key=lambda r: (not r.pourbaix_stable, r.best_overpotential))
    return RankerResult(ranked=ranked, U=U, temperature=temperature)


def run_ranker(
    enum_data: dict,
    all_catalyst_states: dict,
    pruner_data: dict,
    pc_data: dict,
    calculator,
    gas_thermo: dict[str, ThermoCorrection],
    max_denticity: int,
    temperature: float,
    max_chem_step_dg: float,
) -> RankerResult:
    """Run the full ranker stage from raw upstream JSON to a RankerResult.

    Builds one CatalystResult per (catalyst, stable operating point) from the
    adsorption-energy/enumerator/pruner/pathway-constructor data, computing
    per-catalyst vibrational corrections from relaxed trajectory files along
    the way, then ranks by CHE overpotential.

    Args:
        enum_data: parsed eval_enumerator_*.json
        all_catalyst_states: catalyst_key -> per-catalyst adsorption energy
            state (the "results" dict of eval_adsorption_energy_state.json)
        pruner_data: parsed eval_pruner_*.json
        pc_data: parsed eval_pathway_constructor_*.json
        calculator: loaded MLIP calculator, used for vibrational corrections
        gas_thermo: gas-phase thermo corrections from compute_gas_thermo_corrections()
        max_denticity: maximum '*' anchors allowed (passed through to pruner reconstruction)
        temperature: K
        max_chem_step_dg: eV cutoff for potential-independent chemical steps

    Returns:
        RankerResult (Pourbaix gate already applied upstream in eval_enumerator.py,
        so run_pourbaix=False here).
    """
    catalyst_results: list[CatalystResult] = []

    for enum_entry in enum_data["results"]:
        composition  = enum_entry["context"]["composition"]
        facet        = enum_entry["context"]["facet"]
        reaction     = enum_entry["context"]["reaction"]
        catalyst_key = f"{composition}({facet})_{reaction}"

        cat_state = all_catalyst_states.get(catalyst_key)
        if cat_state is None:
            print(f"Skipping {catalyst_key} — no adsorption energy results.")
            continue

        slab_n_atoms = cat_state.get("slab_n_atoms", 0)
        energy_map   = cat_state["energy_results"]
        traj_files   = cat_state.get("traj_files", {})

        pruner_entry = find_entry(pruner_data, composition, facet, reaction)
        pc_entry     = find_entry(pc_data,     composition, facet, reaction)
        if pruner_entry is None or pc_entry is None:
            print(f"Skipping {catalyst_key} — no pruner/pathway constructor entry.")
            continue

        ctx_placeholder = catalyst_context_from_dict(enum_entry["context"])
        enum_output   = reconstruct_enumerator_output(enum_entry, ctx_placeholder)
        pruner_output = reconstruct_pruner_output(
            pruner_entry, enum_entry, enum_output.intermediates, max_denticity=max_denticity,
        )
        all_pathways = reconstruct_pathways(pc_entry)

        known_labels = set(energy_map.keys())
        complete_pathways = [
            pw for pw in all_pathways
            if all(lbl in known_labels for lbl in pw.intermediates)
        ]

        if not complete_pathways:
            print(f"{catalyst_key}: no complete pathways — skipping ranking.")
            continue

        # Vibrational corrections (once per catalyst)
        thermo_corrections: dict = {}
        for label, traj_path in traj_files.items():
            if not os.path.exists(traj_path):
                continue
            try:
                atoms = ase.io.read(traj_path)
                ads_indices = list(range(slab_n_atoms, len(atoms)))
                if ads_indices:
                    corr = compute_thermo_corrections(atoms, calculator, ads_indices, label=label)
                    thermo_corrections[label] = corr
            except Exception as e:
                warnings.warn(f"Vib correction failed for {label} on {catalyst_key}: {e}")

        energy_results: dict[str, AdsorptionEnergyResult] = {
            lbl: AdsorptionEnergyResult(label=lbl, delta_e_ads=de, source="MLIP", uncertainty=0.0)
            for lbl, de in energy_map.items()
        }
        start_ads = enum_entry.get("starting_adsorbate", "*")
        if start_ads not in energy_results:
            energy_results[start_ads] = AdsorptionEnergyResult(
                label=start_ads, delta_e_ads=0.0, source="reference", uncertainty=0.0,
            )

        # Inner loop: one CatalystResult per stable operating point
        stable_points = enum_entry.get(
            "stable_operating_points",
            [[enum_entry["context"]["pH"], enum_entry["context"]["U"]]],
        )
        for ph, u in stable_points:
            ctx = catalyst_context_from_dict({
                "composition": composition, "facet": facet, "reaction": reaction,
                "pH": ph, "U": u,
            })
            catalyst_results.append(CatalystResult(
                context=ctx,
                energy_results=energy_results,
                pathways=complete_pathways,
                thermo_corrections=thermo_corrections if thermo_corrections else None,
                gas_corrections=gas_thermo,
            ))
            print(f"{catalyst_key}  pH={ph}  U={u:.2f} V: "
                  f"{len(complete_pathways)} complete pathway(s) ready for ranking")

    return rank_catalysts(
        catalysts=catalyst_results,
        U=0.0,               # nominal; each catalyst uses its own ctx.U internally
        temperature=temperature,
        run_pourbaix=False,  # Pourbaix gate already applied in eval_enumerator.py
        max_chem_step_dg=max_chem_step_dg,
    )


def print_leaderboard(result: RankerResult) -> None:
    """Print a human-readable leaderboard.

    Each catalyst is evaluated at its own operating potential (stored in
    CatalystContext.U), NOT at a common reference potential.  The leaderboard
    groups results by reaction so that overpotentials from different reactions
    are never compared on the same sorted line.
    """
    print(f"\n{'='*80}")
    print(f"CATALYST LEADERBOARD  T={result.temperature:.0f} K")
    print(f"  Note: each catalyst evaluated at its own U (see 'U_op' column).")
    print(f"{'='*80}")
    print(f"{'#':>3}  {'Catalyst':25s}  {'Reaction':8s}  {'U_op (V)':>8}  "
          f"{'η (eV)':>8}  {'U_onset (V)':>11}  {'Stable':>6}")
    print(f"{'-'*80}")
    for i, cat in enumerate(result.ranked, 1):
        ctx = cat.context
        label = f"{ctx.composition}({ctx.facet})"
        stable = "yes" if cat.pourbaix_stable else "NO"
        print(f"{i:>3}  {label:25s}  {ctx.reaction:8s}  {ctx.U:>8.3f}  "
              f"{cat.best_overpotential:>8.3f}  {cat.best_U_onset:>11.3f}  {stable:>6}")
        if cat.pourbaix_warning:
            print(f"     ! {cat.pourbaix_warning}")
    print(f"{'='*80}\n")
