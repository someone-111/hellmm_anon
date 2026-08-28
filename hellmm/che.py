"""CHE module — module 6.

Computes Gibbs free energy changes ΔG(U, pH) for each elementary step using
the Computational Hydrogen Electrode (CHE) formalism:

  ΔG = ΔE_ads + ΔZPE − TΔS − n_e · eU

where:
  ΔE_ads  — adsorption energy from module 4 (MLIP)
  ΔZPE    — zero-point energy correction from harmonic vibrational analysis
  TΔS     — entropic correction at temperature T
  n_e·eU  — electrochemical driving force (RHE scale, pH-independent)

Vibrational corrections (ThermoCorrection, compute_thermo_corrections,
compute_gas_thermo_corrections) live in hellmm.tools.vibrations.

CHE reference:  H⁺ + e⁻  ↔  ½ H₂(g)   at U=0 V vs RHE, all pH.
"""

from __future__ import annotations

import os
import warnings

import ase.io
from pydantic.dataclasses import dataclass

from .adsorption_energy import AdsorptionEnergyResult
from .chemistry import parse_formula as _parse_formula
from .io import (
    catalyst_context_from_dict,
    find_entry,
    reconstruct_enumerator_output,
    reconstruct_pathways,
    reconstruct_pruner_output,
)
from .pathway_constructor import Pathway, ReactionStep
from .reaction import get_reaction
from .tools import compute_thermo_corrections
from .tools.vibrations import ThermoCorrection


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StepFreeEnergy:
    step: ReactionStep
    delta_e: float      # ΔE_ads in eV
    delta_zpe: float    # ΔZPE = ZPE(product) − ZPE(parent) in eV
    delta_ts: float     # ΔTS  = TS(product)  − TS(parent)  in eV
    delta_g: float      # ΔG = ΔE + ΔZPE − ΔTS − n_e·eU in eV


@dataclass
class PathwayFreeEnergy:
    pathway: Pathway
    step_energies: list[StepFreeEnergy]
    limiting_step: ReactionStep
    overpotential: float          # eV — U_onset − U_ideal
    U_onset: float                # V vs RHE — minimum U at which all steps are downhill
    missing_energies: list[str]   # labels that defaulted to ΔE=0; non-empty = unreliable result



# ---------------------------------------------------------------------------
# Step 6a — gas-phase ZPE/TS reference decomposition
# ---------------------------------------------------------------------------

def _gas_correction_for_label(
    label: str,
    gas_corrections: dict[str, ThermoCorrection],
) -> tuple[float, float]:
    """Compute net ΔZPE and ΔTS contributions from gas references for a label.

    Uses the same CHE reference decomposition as compute_gas_references():
      H  → ½ ZPE(H₂),   ½ TS(H₂)
      O  → ZPE(H₂O) − ZPE(H₂),   TS(H₂O) − TS(H₂)
      C  → ZPE(CO) − [ZPE(H₂O) − ZPE(H₂)],  ...
      N  → ½ ZPE(N₂),  ½ TS(N₂)
    """
    gc = gas_corrections
    ref_zpe = {
        "H": 0.5 * gc["H2"].zpe,
        "O": gc["H2O"].zpe - gc["H2"].zpe,
        "C": gc["CO"].zpe - (gc["H2O"].zpe - gc["H2"].zpe),
        "N": 0.5 * gc["N2"].zpe,
    }
    ref_ts = {
        "H": 0.5 * gc["H2"].ts,
        "O": gc["H2O"].ts - gc["H2"].ts,
        "C": gc["CO"].ts - (gc["H2O"].ts - gc["H2"].ts),
        "N": 0.5 * gc["N2"].ts,
    }
    counts = _parse_formula(label)
    for elem in counts:
        if elem not in ref_zpe:
            warnings.warn(
                f"Element '{elem}' in '{label}' has no gas-phase ZPE/TS reference. "
                "Correction set to 0.0 eV. Add a reference gas molecule to "
                "_GAS_GEOMETRY and recompute gas_corrections."
            )
    zpe_ref = sum(n * ref_zpe.get(elem, 0.0) for elem, n in counts.items())
    ts_ref  = sum(n * ref_ts.get(elem, 0.0) for elem, n in counts.items())
    return zpe_ref, ts_ref


def _max_chem_dg(pfe: "PathwayFreeEnergy") -> float | None:
    """Largest ΔG° among the pathway's potential-independent (n_e = 0) steps.

    Returns None when the pathway has no chemical steps, which is the case the
    CHE limiting-potential descriptor was designed for and the only case where
    the overpotential alone fully describes the pathway.
    """
    chem = [se.delta_e + se.delta_zpe - se.delta_ts
            for se in pfe.step_energies if se.step.n_electrons == 0]
    return max(chem) if chem else None


def _serialize_steps(pfe: "PathwayFreeEnergy") -> list[dict]:
    """Per-step energies for one pathway, in JSON-ready form."""
    return [
        {
            "parent":      se.step.parent,
            "product":     se.step.product,
            "n_electrons": se.step.n_electrons,
            "rule":        se.step.rule,
            "delta_e":     se.delta_e,
            "delta_zpe":   se.delta_zpe,
            "delta_ts":    se.delta_ts,
            "delta_g":     se.delta_g,
        }
        for se in pfe.step_energies
    ]


# ---------------------------------------------------------------------------
# Step 6b — CHE free energy per pathway
# ---------------------------------------------------------------------------

def _che_delta_g(
    delta_e: float,
    delta_zpe: float,
    delta_ts: float,
    n_electrons: int,
    U: float,
) -> float:
    """ΔG = ΔE + ΔZPE − ΔTS − n_e · eU  (RHE scale, pH-independent).

    Standard CHE sign convention (Nørskov 2004): each H⁺+e⁻ released lowers
    ΔG by eU. Applying positive U drives oxidative steps (OER) downhill.
    For reduction (ORR/HER) the same formula holds — downhill steps occur at
    U below the equilibrium potential.
    """
    return delta_e + delta_zpe - delta_ts - n_electrons * U


def compute_pathway_free_energy(
    pathway: Pathway,
    energy_results: dict[str, AdsorptionEnergyResult],
    U: float,
    reaction: str,
    temperature: float = 298.15,
    thermo_corrections: dict[str, ThermoCorrection] | None = None,
    gas_corrections: dict[str, ThermoCorrection] | None = None,
    max_chem_step_dg: float = 0.0,
) -> PathwayFreeEnergy:
    """Compute ΔG(U, T) for every step in the pathway using CHE.

    Args:
        pathway: pathway from pathway_constructor
        energy_results: adsorption energies keyed by label (from module 4)
        U: electrode potential vs RHE in V
        reaction: "OER" | "HER" | "CO2RR" | "ORR" | "NRR"
        temperature: temperature in K (default 298.15 K)
        thermo_corrections: optional dict label → ThermoCorrection for adsorbed
            species. If None, ZPE and TS are set to 0 for all adsorbates.
        gas_corrections: optional dict formula → ThermoCorrection for gas refs.
            If None, gas ZPE/TS corrections are not applied.
        max_chem_step_dg: maximum ΔG (eV) permitted for chemical steps (n_e=0).
            Chemical steps are potential-independent; ΔG above this threshold
            means the step is thermodynamically blocked at any U. The pathway
            is then marked infeasible (U_onset = 999 V). Default 0.0 eV
            (strict CHE thermodynamics).

    Returns:
        PathwayFreeEnergy with per-step ΔG, limiting step, and overpotential.
        Infeasible pathways (uphill chemical step) have U_onset = 999.0 V.
    """
    rxn = get_reaction(reaction)
    u_ideal_rxn = rxn.U_ideal
    n_e_total = rxn.n_electrons_total
    target_set = set(rxn.target_labels)
    start_node = pathway.intermediates[0]

    # Sentinels for infeasible pathways.  The onset sentinel is signed by
    # direction — an anodic reaction becomes impossible as U → +∞, a cathodic
    # one as U → −∞ — so that it stays consistent with the direction-dependent
    # overpotential formula below.  A flat +999 onset would, under the cathodic
    # formula η = U_ideal − U_onset, yield a large *negative* overpotential and
    # sort infeasible pathways to the top of the ranking.
    _INFEASIBLE_U = -999.0 if rxn.cathodic else 999.0
    _INFEASIBLE_ETA = 999.0

    tc = thermo_corrections or {}
    gc = gas_corrections

    step_energies = []
    all_missing: list[str] = []
    for step in pathway.steps:
        # ΔE from adsorption energies — warn if an intermediate is missing.
        # Missing energies default to 0.0 which can produce spuriously low
        # overpotentials; callers should ensure all intermediates are computed.
        missing = [s for s in (step.parent, step.product)
                   if s not in energy_results and s != "*"]
        if missing:
            all_missing.extend(m for m in missing if m not in all_missing)
            warnings.warn(
                f"Missing adsorption energy for {missing} — defaulting to 0.0 eV. "
                "Overpotential estimate for this pathway is unreliable."
            )
        e_prod  = energy_results[step.product].delta_e_ads  if step.product  in energy_results else 0.0
        e_par   = energy_results[step.parent].delta_e_ads   if step.parent   in energy_results else 0.0
        delta_e = e_prod - e_par

        # CHE thermodynamic closure for cyclic reactions (OER, ORR).
        # Under the CHE reference (Man et al. 2011):
        #   G(O₂)_CHE = 2·G(H₂O) − 2·G(H₂) + n_e_total·U_ideal
        # The adsorption-energy ladder puts the bare surface at 0 and never
        # accounts for the gas O₂ entering or leaving, so n_e_total·U_ideal has
        # to be applied where the gas molecule crosses the boundary — and that
        # end depends on the reaction direction:
        #
        #   anodic  (OER):  O₂ is *released* by the terminal step returning to
        #                   the start → add   n_e·U_ideal to the terminal step
        #   cathodic (ORR): O₂ is *consumed* by the first step leaving the
        #                   start → subtract n_e·U_ideal from the first step
        #
        # Applying the anodic placement to a reduction puts the whole +4.92 eV
        # on the wrong end: it inflated ORR's terminal *OH → * from −0.74 to
        # +4.18 eV and left the real first step (* + O₂ → *OO) reading +4.22 eV
        # instead of ≈ −0.70 eV, which then tripped the chemical-step block and
        # made every reductive ORR pathway "infeasible".
        # Both n_e_total and U_ideal come from ReactionDefinition — no free parameters.
        # Gated on rxn.cyclic (not "start_node in target_set") so vacancy-initiated
        # runs (start_node="*V_O") trigger correctly — "*V_O" is not in target_labels
        # but the cycle is still cyclic and requires the closure correction.
        # CO2RR/NRR (cyclic=False) are excluded: their closure is via explicit
        # product_desorption edges and their U_ideal contribution is already zero.
        if rxn.cyclic:
            if rxn.cathodic:
                if step.parent == start_node and step.product != start_node:
                    delta_e -= n_e_total * u_ideal_rxn
            else:
                if step.product == start_node and step.parent != start_node:
                    delta_e += n_e_total * u_ideal_rxn

        # ΔZPE and ΔTS from vibrational corrections (adsorbate − gas ref)
        if gc is not None:
            prod_zpe_ref, prod_ts_ref = _gas_correction_for_label(step.product, gc)
            par_zpe_ref,  par_ts_ref  = _gas_correction_for_label(step.parent,  gc)
        else:
            prod_zpe_ref = par_zpe_ref = prod_ts_ref = par_ts_ref = 0.0

        prod_zpe = tc[step.product].zpe - prod_zpe_ref if step.product in tc else 0.0
        par_zpe  = tc[step.parent].zpe  - par_zpe_ref  if step.parent  in tc else 0.0
        prod_ts  = tc[step.product].ts  - prod_ts_ref  if step.product in tc else 0.0
        par_ts   = tc[step.parent].ts   - par_ts_ref   if step.parent  in tc else 0.0

        delta_zpe = prod_zpe - par_zpe
        delta_ts  = prod_ts  - par_ts

        delta_g = _che_delta_g(delta_e, delta_zpe, delta_ts, step.n_electrons, U)
        step_energies.append(StepFreeEnergy(
            step=step,
            delta_e=delta_e,
            delta_zpe=delta_zpe,
            delta_ts=delta_ts,
            delta_g=delta_g,
        ))

    # Mixed-polarity feasibility check.
    # A pathway containing both cathodic (n_e < 0) and anodic (n_e > 0)
    # electrochemical steps is physically infeasible: no single fixed electrode
    # potential can simultaneously drive steps of opposite polarity.  Cathodic
    # steps require U < U_threshold; anodic steps require U > U_threshold' with
    # U_threshold' > 0 > U_threshold.  The U_onset formula with abs(n_e) would
    # silently undercount the difficulty of such paths.
    cathodic_steps = [se for se in step_energies if se.step.n_electrons < 0]
    anodic_steps   = [se for se in step_energies if se.step.n_electrons > 0]
    if cathodic_steps and anodic_steps:
        worst = max(step_energies, key=lambda se: se.delta_g)
        warnings.warn(
            f"Mixed-polarity pathway rejected: {len(cathodic_steps)} cathodic "
            f"(n_e<0) and {len(anodic_steps)} anodic (n_e>0) steps — "
            "no fixed electrode potential can simultaneously drive both."
        )
        return PathwayFreeEnergy(
            pathway=pathway,
            step_energies=step_energies,
            limiting_step=worst.step,
            overpotential=_INFEASIBLE_ETA,
            U_onset=_INFEASIBLE_U,
            missing_energies=all_missing,
        )

    # Directional validity check.
    # A pathway whose electrochemical steps all run counter to the reaction's
    # own direction is not a poor pathway for this reaction — it is a different
    # reaction.  Observed in practice: an all-anodic ladder
    # * → *OH → *O → *OOH → *OO → * satisfies ORR's 4-electron count and passes
    # the mixed-polarity check above (its polarity is uniform, just backwards),
    # and was scored as an ORR result with U_onset = 2.46 V vs RHE — a potential
    # at which oxygen reduction cannot occur at all.  rxn.cathodic is a reaction
    # boundary condition, like U_ideal and n_electrons_total; it constrains
    # nothing about which intermediates exist or how they connect.
    wrong_way = anodic_steps if rxn.cathodic else cathodic_steps
    if wrong_way:
        worst = max(step_energies, key=lambda se: se.delta_g)
        direction = "reduction" if rxn.cathodic else "oxidation"
        opposite  = "oxidative" if rxn.cathodic else "reductive"
        warnings.warn(
            f"Counter-directional pathway rejected: {reaction} is a net {direction}, "
            f"but this pathway's {len(wrong_way)} electrochemical step(s) are "
            f"{opposite}. This is a different reaction, not a poor pathway for this one."
        )
        return PathwayFreeEnergy(
            pathway=pathway,
            step_energies=step_energies,
            limiting_step=worst.step,
            overpotential=_INFEASIBLE_ETA,
            U_onset=_INFEASIBLE_U,
            missing_energies=all_missing,
        )

    # Thermodynamic feasibility check.
    # Chemical steps (n_e = 0) are potential-independent: no electrode potential
    # can drive them downhill. A pathway with any such step endergonic beyond
    # max_chem_step_dg is thermodynamically blocked and has no valid U_onset.
    blocking = [
        se for se in step_energies
        if se.step.n_electrons == 0 and se.delta_g > max_chem_step_dg
    ]
    if blocking:
        worst = max(blocking, key=lambda se: se.delta_g)
        warnings.warn(
            f"Pathway blocked: chemical step '{worst.step.parent} → "
            f"{worst.step.product}' has ΔG = {worst.delta_g:+.3f} eV "
            f"> {max_chem_step_dg:.2f} eV (max_chem_step_dg). "
            f"No electrode potential can drive this step — pathway infeasible."
        )
        return PathwayFreeEnergy(
            pathway=pathway,
            step_energies=step_energies,
            limiting_step=worst.step,
            overpotential=_INFEASIBLE_ETA,
            U_onset=_INFEASIBLE_U,
            missing_energies=all_missing,
        )

    # U_onset: the potential at which every electrochemical step is downhill.
    #
    #   ΔG(step, U) = ΔG°(step) − n_e·eU
    #
    # The sign of n_e inverts the inequality, so the two directions need
    # different formulas — using one for both silently corrupts the other:
    #
    #   anodic  (n_e = +1, OER):  downhill ⟺ U ≥ ΔG°   → U_onset = max(ΔG°)
    #                             η = U_onset − U_ideal  (drive harder = higher U)
    #   cathodic (n_e = −1, HER/ORR/CO2RR/NRR):
    #                             downhill ⟺ U ≤ −ΔG°  → U_onset = −max(ΔG°)
    #                             η = U_ideal − U_onset  (drive harder = lower U)
    #
    # The previous implementation used max(ΔG°/|n_e|) and η = U_onset − U_ideal
    # unconditionally.  For cathodic reactions that inverts both quantities: it
    # reported ORR onsets above U_ideal (2.46 V vs RHE, where ORR cannot run)
    # and, for HER, U_onset = −0.137 V where the physical onset is +0.137 V.
    # The two sign errors happen to cancel in η only when U_ideal = 0, which is
    # why HER's overpotential looked plausible while its U_onset did not.
    u_ideal = u_ideal_rxn
    u_onset_candidates: list[tuple[float, StepFreeEnergy]] = []
    for se in step_energies:
        if se.step.n_electrons != 0:
            delta_g0 = se.delta_e + se.delta_zpe - se.delta_ts   # ΔG° at U=0
            u_onset_candidates.append((delta_g0 / abs(se.step.n_electrons), se))

    if u_onset_candidates:
        # The limiting step is the hardest one in either convention: the step
        # with the largest ΔG° per electron.  Only the reported sign differs.
        worst_dg0, limiting = max(u_onset_candidates, key=lambda x: x[0])
        U_onset = -worst_dg0 if rxn.cathodic else worst_dg0
    else:
        U_onset = U
        limiting = max(step_energies, key=lambda s: s.delta_g)

    # Overpotential: extra driving force needed beyond the thermodynamic
    # minimum, defined so that positive always means "harder than ideal"
    # regardless of direction.
    overpotential = (u_ideal - U_onset) if rxn.cathodic else (U_onset - u_ideal)

    return PathwayFreeEnergy(
        pathway=pathway,
        step_energies=step_energies,
        limiting_step=limiting.step,
        overpotential=overpotential,
        U_onset=U_onset,
        missing_energies=all_missing,
    )


# ---------------------------------------------------------------------------
# Full CHE stage — all catalysts, all stable operating points
# ---------------------------------------------------------------------------

def run_che(
    enum_data: dict,
    all_catalyst_states: dict,
    pruner_data: dict,
    pc_data: dict,
    calculator,
    gas_thermo: dict[str, ThermoCorrection],
    max_denticity: int,
    temperature: float,
    max_chem_step_dg: float,
) -> list[dict]:
    """Run the full CHE stage from raw upstream JSON to serializable summary rows.

    For every catalyst in enum_data with adsorption-energy results, computes
    adsorbate vibrational corrections from relaxed trajectory files, loads
    pre-computed pathways, and calls compute_pathway_free_energy() for every
    stable (pH, U) operating point. Prints detailed per-step diagnostics
    (ΔE/ΔZPE/ΔTS/ΔG, top-3 rate-limiting steps, leaderboard) along the way.

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
        List of JSON-serializable summary-row dicts, one per (catalyst,
        stable operating point), sorted by overpotential ascending.
    """
    summary_rows: list[dict] = []

    for enum_entry in enum_data["results"]:
        composition  = enum_entry["context"]["composition"]
        facet        = enum_entry["context"]["facet"]
        reaction     = enum_entry["context"]["reaction"]
        catalyst_key = f"{composition}({facet})_{reaction}"

        cat_state = all_catalyst_states.get(catalyst_key)
        if cat_state is None:
            print(f"\n{catalyst_key}: no adsorption energy results — skipping.")
            continue

        slab_n_atoms = cat_state.get("slab_n_atoms", 0)
        energy_map   = cat_state["energy_results"]
        traj_files   = cat_state.get("traj_files", {})

        pruner_entry = find_entry(pruner_data, composition, facet, reaction)
        pc_entry     = find_entry(pc_data,     composition, facet, reaction)
        if pruner_entry is None or pc_entry is None:
            print(f"\n{catalyst_key}: no pruner/pathway constructor entry — skipping.")
            continue

        # Placeholder ctx for reconstruction helpers (pH/U not used here)
        ctx_placeholder = catalyst_context_from_dict(enum_entry["context"])
        enum_output   = reconstruct_enumerator_output(enum_entry, ctx_placeholder)
        pruner_output = reconstruct_pruner_output(
            pruner_entry, enum_entry, enum_output.intermediates, max_denticity=max_denticity,
        )

        print(f"\n{'='*60}")
        print(f"{catalyst_key}  {reaction}")
        print(f"  {len(enum_output.intermediates)} total intermediates, "
              f"{len(pruner_output.kept)} kept")
        print(f"{'='*60}")

        # Step 1b — adsorbate vibrational corrections (once per catalyst)
        print(f"\n  STEP 1b — adsorbate vibrational corrections")
        thermo_corrections: dict = {}
        for label, traj_path in traj_files.items():
            if not os.path.exists(traj_path):
                continue
            try:
                atoms = ase.io.read(traj_path)
                ads_indices = list(range(slab_n_atoms, len(atoms)))
                if not ads_indices:
                    continue
                corr = compute_thermo_corrections(atoms, calculator, ads_indices, label=label)
                thermo_corrections[label] = corr
                zpe_ref, ts_ref = _gas_correction_for_label(label, gas_thermo)
                net_zpe = corr.zpe - zpe_ref
                net_ts  = corr.ts  - ts_ref
                print(f"  {label}: ZPE={corr.zpe:.4f}  TΔS={corr.ts:.4f}  "
                      f"net_ΔZPE={net_zpe:+.4f}  net_ΔTS={net_ts:+.4f}")
            except Exception as e:
                print(f"  {label}: vibrational correction failed: {e}")

        # Step 2 — load pre-computed pathways (once per catalyst)
        print(f"\n  STEP 2 — loading pathways")
        rxn = get_reaction(reaction)
        pathways = reconstruct_pathways(pc_entry)
        print(f"  Loaded {len(pathways)} pathway(s)")

        energy_results: dict[str, AdsorptionEnergyResult] = {
            lbl: AdsorptionEnergyResult(label=lbl, delta_e_ads=de, source="MLIP", uncertainty=0.0)
            for lbl, de in energy_map.items()
        }
        known_labels = set(energy_results.keys())
        complete_pathways = [
            pw for pw in pathways
            if all(lbl in known_labels for lbl in pw.intermediates)
        ]
        print(f"  {len(complete_pathways)} pathway(s) with all energies computed "
              f"(out of {len(pathways)} total)")
        if not complete_pathways and pathways:
            print("  WARNING: no pathway has all intermediates computed. "
                  "Re-run eval_adsorption_energy.py.")

        if not complete_pathways:
            print("  Skipping CHE — no complete pathways.")
            continue

        # Step 3 — CHE: one run per stable operating point (inner loop)
        stable_points = enum_entry.get(
            "stable_operating_points",
            [[enum_entry["context"]["pH"], enum_entry["context"]["U"]]],
        )

        for ph, u in stable_points:
            print(f"\n  STEP 3 — CHE ΔG(U={u:.2f} V, pH={ph}, T={temperature:.0f} K)")
            u_ideal = rxn.U_ideal
            all_pfe = [
                compute_pathway_free_energy(
                    pw, energy_results, u, reaction,
                    thermo_corrections=thermo_corrections, gas_corrections=gas_thermo,
                    max_chem_step_dg=max_chem_step_dg,
                    temperature=temperature,
                )
                for pw in complete_pathways
            ]
            # Best = lowest overpotential, which is now direction-correct for
            # both conventions (positive always means "harder than ideal").
            # This previously pre-filtered on `U_onset >= u_ideal`, an anodic
            # assumption that discarded every valid cathodic pathway: HER's
            # correct Volmer–Heyrovsky path (U_onset below U_ideal, as any
            # reduction must be) was dropped, leaving only an infeasible
            # pathway, so CHE reported η = 999 while the ranker — which has no
            # such filter — reported the real value from the same run.  Both
            # now use the same rule.
            best = min(all_pfe, key=lambda p: p.overpotential)

            for i, (pw, pfe) in enumerate(zip(complete_pathways, all_pfe)):
                is_best = (pfe is best)
                tag = "  ★ BEST" if is_best else ""
                print(f"\n  Pathway [{i}]: {' → '.join(pw.intermediates)}{tag}")
                for se in pfe.step_energies:
                    flag = " ← LIMITING" if se.step == pfe.limiting_step else ""
                    print(f"    {se.step.parent:18s} → {se.step.product:18s}  "
                          f"ΔE={se.delta_e:+.3f}  ΔZPE={se.delta_zpe:+.4f}  "
                          f"ΔTS={se.delta_ts:+.4f}  ΔG={se.delta_g:+.3f} eV{flag}")

                # Top-3 rate-limiting steps (by ΔG° = ΔE+ΔZPE−ΔTS at U=0)
                electrochemical = [
                    se for se in pfe.step_energies if se.step.n_electrons != 0
                ]
                top3 = sorted(
                    electrochemical,
                    key=lambda se: se.delta_e + se.delta_zpe - se.delta_ts,
                    reverse=True,
                )[:3]
                print(f"  Top-3 rate-limiting steps (by ΔG°, U-independent):")
                for rank, se in enumerate(top3, 1):
                    marker = " ← #1 (limiting)" if rank == 1 else ""
                    dg0 = se.delta_e + se.delta_zpe - se.delta_ts
                    print(f"    #{rank}  {se.step.parent:18s} → {se.step.product:18s}  "
                          f"ΔG°={dg0:+.3f} eV  (ΔE={se.delta_e:+.3f}){marker}")

                print(f"  → U_onset       : {pfe.U_onset:.3f} V vs RHE")
                print(f"  → Overpotential : {pfe.overpotential:.3f} eV")

            print(f"\n  SUMMARY  {catalyst_key}  pH={ph}  U={u:.2f} V  ({reaction})")
            print(f"  U_ideal       : {u_ideal:.2f} V vs RHE")
            print(f"  Best U_onset  : {best.U_onset:.3f} V vs RHE")
            print(f"  Overpotential : {best.overpotential:.3f} eV")
            print(f"  Limiting step : {best.limiting_step.parent} → {best.limiting_step.product}")

            # Top-3 for the best pathway
            best_elec = [se for se in best.step_energies if se.step.n_electrons != 0]
            best_top3 = sorted(best_elec, key=lambda se: se.delta_e + se.delta_zpe - se.delta_ts, reverse=True)[:3]
            print(f"  Top-3 steps   : " + " | ".join(
                f"#{r} {se.step.parent}→{se.step.product} ΔG°={se.delta_e+se.delta_zpe-se.delta_ts:+.3f} eV"
                for r, se in enumerate(best_top3, 1)
            ))

            summary_rows.append({
                "catalyst":      catalyst_key,
                "reaction":      reaction,
                "pH":            ph,
                "U":             u,
                "U_ideal":       u_ideal,
                "U_onset":       best.U_onset,
                "overpotential": best.overpotential,
                "limiting_step": f"{best.limiting_step.parent} → {best.limiting_step.product}",
                "top3_steps": [
                    {
                        "rank":    r,
                        "parent":  se.step.parent,
                        "product": se.step.product,
                        "dg0":     round(se.delta_e + se.delta_zpe - se.delta_ts, 4),
                    }
                    for r, se in enumerate(best_top3, 1)
                ],
                "best_pathway":  best.pathway.intermediates,
                # Largest potential-independent step in the best pathway, and a
                # flag for whether one exists at all.
                #
                # U_onset is defined over electrochemical steps only, which is
                # correct — no electrode potential drives an n_e=0 step.  But the
                # cycle's thermodynamics still has to balance, so cost pushed
                # into a chemical step leaves the electrochemical ones shallower
                # and U_onset stops reflecting the real barrier.  Taken to its
                # limit this yields η < 0, i.e. the reaction apparently running
                # below its own equilibrium potential, which is impossible: HER
                # reports η = -0.146 eV while its actual bottleneck is *H2
                # desorption at +0.342 eV, invisible to U_onset by construction.
                #
                # Reported as a separate number rather than folded into η,
                # because they are independent physical quantities and no single
                # scalar can express "easy to drive electrically, but blocked by
                # a chemical step". `overpotential` is left unchanged so existing
                # consumers keep working; reporting code should prefer the pair.
                "dG_chem_max":      _max_chem_dg(best),
                "chemical_limited": _max_chem_dg(best) is not None
                                    and _max_chem_dg(best) > 0.0,
                "step_energies": _serialize_steps(best),
                # Every pathway considered, not only the winner — including ones
                # rejected as mixed-polarity, counter-directional or chemically
                # blocked, which carry the sentinel U_onset/overpotential.
                #
                # Storing only `best` made the descriptor question un-answerable
                # after the fact: whether to rank by overpotential or by the
                # largest chemical step, and where to put max_chem_step_dg, both
                # decide which pathway wins, and the alternatives were gone by
                # the time the JSON was written.  Each of those questions then
                # cost a full ~3 h pipeline re-run to explore.
                #
                # With per-step delta_e/zpe/ts and n_electrons kept for every
                # pathway, ΔG° is recoverable per step, so U_onset under any sign
                # convention, the chemical bottleneck, any max_chem_step_dg
                # threshold, and any ranking rule can all be re-derived offline
                # from a completed run.  The extra volume is small next to the
                # enumerator's raw LLM responses.
                "all_pathways": [
                    {
                        "intermediates":    pfe.pathway.intermediates,
                        "U_onset":          pfe.U_onset,
                        "overpotential":    pfe.overpotential,
                        "limiting_step":    f"{pfe.limiting_step.parent} → {pfe.limiting_step.product}",
                        "missing_energies": sorted(pfe.missing_energies),
                        "is_best":          pfe is best,
                        "step_energies":    _serialize_steps(pfe),
                    }
                    for pfe in all_pfe
                ],
            })

    summary_rows.sort(key=lambda r: r["overpotential"])
    return summary_rows
