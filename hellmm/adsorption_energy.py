"""Adsorption energy — module 4.

Defines the AdsorptionEnergyResult output type, the formula for computing
adsorption energy from relaxed energies and CHE gas references, and the
per-catalyst orchestration (slab building, candidate resolution, relaxation,
sanity cutoffs) that drives the MLIP stage of the pipeline.

All portable heavy lifting (MLIP loading, slab building, structure placement,
relaxation, gas reference computation) lives in hellmm.tools; this module owns
the hellmm-specific decisions of *which* candidates to try, in what order, and
what counts as a physically sane result.
"""

from __future__ import annotations

import warnings

from dataclasses import dataclass as _plain_dataclass, field as _field

from pydantic.dataclasses import dataclass

from .chemistry import parse_formula as _parse_formula
from .meta_reasoner import CatalystContext
from .tools import (
    acat_surface_name,
    build_slab,
    check_adsorbate_geometry,
    adsorbate_is_intact,
    check_computability,
    compute_gas_references,
    compute_vacancy_configs,
    generate_adsorption_configs,
    generate_adsorption_configs_acat,
    generate_adsorption_configs_acat_bidentate,
    label_to_bidentate_smiles,
    label_to_fairchem_ids,
    label_to_smiles,
    relax_structure,
)


@dataclass
class AdsorptionEnergyResult:
    label: str          # adsorbate label, e.g. "*OH"
    delta_e_ads: float  # adsorption energy in eV
    source: str         # "MLIP" | "DFT" | "manual"
    uncertainty: float  # eV (0.0 if unknown)


def compute_adsorption_energy(
    e_slab_ads: float,
    e_slab: float,
    label: str,
    gas_refs: dict[str, float],
) -> float:
    """Compute ΔE_ads = E(slab+ads) − E(slab) − Σ nᵢ · E_ref_gas_i.

    Args:
        e_slab_ads: energy of relaxed slab+adsorbate (eV)
        e_slab: energy of relaxed clean slab (eV)
        label: adsorbate label, e.g. "*COOH" → C1H1O2
        gas_refs: per-element CHE refs from tools.compute_gas_references()

    Returns:
        ΔE_ads in eV
    """
    counts = _parse_formula(label)
    ref_correction = sum(n * gas_refs.get(elem, 0.0) for elem, n in counts.items())
    return e_slab_ads - e_slab - ref_correction


# ---------------------------------------------------------------------------
# Intermediate selection
# ---------------------------------------------------------------------------

def select_intermediates(
    kept_items: list[dict],
    start_ads: str,
    max_denticity: int,
    pathway_labels: set[str] | None,
) -> list[dict] | None:
    """Filter and augment pruner-kept items into the final MLIP compute list.

    Applies, in order: the denticity cap, injection of the starting adsorbate
    (needed as the reactant-state reference in the Peterson CHE convention
    even if the pruner didn't keep it), and restriction to labels that appear
    in at least one valid pathway.

    Args:
        kept_items: pruner "kept" entries, each a dict with at least "label"
        start_ads: starting adsorbate label for this catalyst case (e.g. "*CO2", or "*")
        max_denticity: maximum number of '*' anchors allowed
        pathway_labels: set of labels appearing in >=1 valid pathway, or None
            if pathway_constructor output isn't available (no filtering applied)

    Returns:
        Filtered list of items, or None if pathway_labels is an empty set
        (0 valid pathways found — caller should skip the whole case).
    """
    items = [k for k in kept_items if k["label"].count("*") <= max_denticity]

    if start_ads != "*":
        labels = {k["label"] for k in items}
        if start_ads not in labels:
            items = [{"label": start_ads, "parent": "*", "rule": "adsorption"}] + items

    if pathway_labels is not None:
        if not pathway_labels:
            return None
        items = [k for k in items if k["label"] in pathway_labels]

    return items


def classify_intermediates(kept_items: list[dict], max_denticity: int) -> dict[str, list[str]]:
    """Group kept-item labels by their check_computability() status.

    Used to print an upfront manifest of what will and won't be computed
    before touching the MLIP.
    """
    groups: dict[str, list[str]] = {}
    for item in kept_items:
        result = check_computability(item["label"], item.get("rule", ""), max_denticity)
        groups.setdefault(result["status"], []).append(item["label"])
    return groups


# ---------------------------------------------------------------------------
# Per-catalyst references
# ---------------------------------------------------------------------------

@_plain_dataclass
class CaseReferences:
    slab: object              # fairchem Slab
    gas_refs: dict[str, float]
    e_slab: float
    slab_n_atoms: int


def build_case_references(ctx: CatalystContext, calculator) -> CaseReferences:
    """Build the slab and CHE gas/clean-slab references for one catalyst case.

    Raises whatever build_slab() raises (e.g. missing MP API key, no MP
    entries, no slab for the requested miller index) — the caller decides
    whether that should skip the case.

    Prints a progress line, flushed immediately, before/after each sub-step.
    Added after a SLURM job hung for 3+ days inside this function with zero
    log output and no way to tell which of the three calls was stuck — these
    lines exist purely so a future hang is visible in the log instead of
    silent. Remove once build_slab/compute_gas_references/relax_structure all
    have their own timeouts and this is no longer needed as a diagnostic.
    """
    print(f"  [build_case_references] building slab for {ctx.composition}({ctx.facet})...", flush=True)
    slab = build_slab(ctx)
    print(f"  [build_case_references] slab built: {slab.atoms.get_chemical_formula()}, "
          f"{len(slab.atoms)} atoms — computing gas references...", flush=True)

    gas_refs = compute_gas_references(calculator)
    print(f"  [build_case_references] gas references computed — relaxing clean slab...", flush=True)

    _, e_slab = relax_structure(slab.atoms.copy(), calculator)
    print(f"  [build_case_references] clean slab relaxed: E={e_slab:.4f} eV", flush=True)

    return CaseReferences(slab=slab, gas_refs=gas_refs, e_slab=e_slab, slab_n_atoms=len(slab))


# ---------------------------------------------------------------------------
# Vacancy energy
# ---------------------------------------------------------------------------

@_plain_dataclass
class VacancyEnergyResult:
    status: str  # "ok" | "no_sites" | "all_failed" | "unphysical"
    e_vac: float | None = None
    relaxed: object | None = None
    n_sites: int = 0
    detail: str = ""


def compute_vacancy_energy(
    slab_atoms,
    calculator,
    gas_refs: dict[str, float],
    e_slab: float,
    max_abs_e_vac: float = 8.0,
) -> VacancyEnergyResult:
    """Compute oxygen-vacancy formation energy: E_vac = E(defect) + gas_refs["O"] - E(slab).

    Tries every vacancy site returned by compute_vacancy_configs() and keeps
    the global minimum. gas_refs["O"] = E(H2O) - E(H2) keeps this in the same
    CHE reference frame as all other ΔE_ads values.
    """
    try:
        vac_configs = compute_vacancy_configs(slab_atoms)
    except RuntimeError as e:
        return VacancyEnergyResult(status="no_sites", detail=str(e))

    best_e_vac = float("inf")
    best_relaxed = None
    for vcand in vac_configs:
        try:
            relaxed_v, e_defect = relax_structure(vcand, calculator)
            e_vac = e_defect + gas_refs["O"] - e_slab
            if e_vac < best_e_vac:
                best_e_vac = e_vac
                best_relaxed = relaxed_v
        except Exception:
            continue

    if best_relaxed is None:
        return VacancyEnergyResult(status="all_failed", n_sites=len(vac_configs))

    if abs(best_e_vac) > max_abs_e_vac:
        return VacancyEnergyResult(
            status="unphysical", e_vac=best_e_vac, n_sites=len(vac_configs),
            detail=f"|E_vac|={abs(best_e_vac):.2f} eV > {max_abs_e_vac} eV",
        )

    return VacancyEnergyResult(
        status="ok", e_vac=best_e_vac, relaxed=best_relaxed, n_sites=len(vac_configs),
    )


# ---------------------------------------------------------------------------
# Single-intermediate adsorption energy
# ---------------------------------------------------------------------------

@_plain_dataclass
class IntermediateEnergyResult:
    label: str
    status: str  # "ok" | "skipped"
    delta_e_ads: float | None = None
    relaxed: object | None = None
    path_used: str = ""
    n_candidates: int = 0
    detail: str = ""


def compute_intermediate_energy(
    label: str,
    parent: str,
    rule: str,
    slab,
    calculator,
    gas_refs: dict[str, float],
    e_slab: float,
    ctx: CatalystContext,
    stored_smiles: str | None,
    stored_anchors: tuple[int, int] | None,
    max_denticity: int,
    llm_model: str,
) -> IntermediateEnergyResult:
    """Resolve candidate geometries for one adsorbate label, relax, and score.

    Candidate resolution order: exact/formula match in the fairchem
    adsorbate DB, then ACAT+rdkit2ase (monodentate or bidentate) as fallback,
    using stored SMILES/anchors from the enumerator when available to avoid
    an LLM call. Every candidate geometry is relaxed and the global energy
    minimum is kept. A size-dependent unphysical-ΔE cutoff rejects likely
    bad placements: ACAT-placed geometries are unvalidated and more prone to
    collapsed/buried geometries than the curated fairchem DB, so they get a
    tighter cutoff (4 eV for ≤2 heavy atoms, 6 eV for ≥3 — large adsorbates
    like *OOOH legitimately reach 4-6 eV on oxides); fairchem DB geometries
    get 8 eV.

    Only called for "normal" adsorbate labels — the bare surface ("*"),
    "*O_lattice", and vacancy-rule labels are handled directly by the caller.
    """
    candidates = None
    path_used = "?"

    try:
        db_ids = label_to_fairchem_ids(label, rule)
        candidates = []
        for db_id in db_ids:
            candidates.extend(generate_adsorption_configs(slab, db_id))
        path_used = f"fairchem({len(db_ids)} geom)"
    except ValueError as e_fc:
        e_str = str(e_fc)
        if "surface defect" in e_str:
            return IntermediateEnergyResult(label=label, status="skipped", detail="surface defect")
        if "surface anchors" in e_str:
            if max_denticity < 2:
                return IntermediateEnergyResult(
                    label=label, status="skipped",
                    detail="bidentate; set max_denticity=2 to enable",
                )
            try:
                if stored_smiles and stored_anchors:
                    bi_smiles = stored_smiles
                    anchor_idx = tuple(stored_anchors)
                else:
                    bi_smiles, anchor_idx = label_to_bidentate_smiles(
                        label, parent, rule, ctx, model=llm_model
                    )
                candidates = generate_adsorption_configs_acat_bidentate(
                    slab.atoms, bi_smiles, anchor_idx
                )
                path_used = f"acat_bidentate({bi_smiles}, anchors={anchor_idx})"
            except Exception as e_bi:
                return IntermediateEnergyResult(
                    label=label, status="skipped", detail=f"bidentate acat: {e_bi}",
                )
        else:
            try:
                smiles = stored_smiles or label_to_smiles(label, parent, rule, ctx, model=llm_model)
                candidates = generate_adsorption_configs_acat(
                    slab.atoms, smiles, surface=acat_surface_name(slab),
                )
                path_used = f"acat({smiles})"
            except Exception as e_acat:
                return IntermediateEnergyResult(
                    label=label, status="skipped", detail=f"fairchem: {e_fc}; acat: {e_acat}",
                )
    except Exception as e:
        return IntermediateEnergyResult(
            label=label, status="skipped", detail=f"label_to_fairchem_ids error: {e}",
        )

    # Validate each relaxed geometry BEFORE it can win the energy minimisation.
    # A placement that collapses into the slab can have a spuriously low energy
    # and would otherwise be selected by min(). Geometry is checked per
    # candidate rather than only on the winner (which is what the old absolute
    # |ΔE| cutoff did, and why it had to be so aggressive).
    n_slab_atoms = len(slab.atoms)
    best_de = float("inf")
    best_relaxed = None
    n_bad_geometry = 0
    last_geom_reason = ""
    # Second tier, used only if no candidate stays chemically intact.  An
    # adsorbate that falls apart during relaxation is no longer the species its
    # label names, and scoring it anyway silently substitutes one intermediate
    # for another (see structure.adsorbate_is_intact).  But dissociation is
    # sometimes the correct surface chemistry — H2 on Pt(111) has no molecular
    # minimum — so preferring intact geometries is right where one exists,
    # while refusing outright would delete real intermediates.
    best_diss_de = float("inf")
    best_diss_relaxed = None
    n_dissociated = 0
    last_diss_reason = ""

    for cand in candidates:
        try:
            relaxed, e_ads = relax_structure(cand, calculator)
        except Exception:
            continue
        ok, reason = check_adsorbate_geometry(relaxed, n_slab_atoms)
        if not ok:
            n_bad_geometry += 1
            last_geom_reason = reason
            continue
        de = compute_adsorption_energy(e_ads, e_slab, label, gas_refs)
        intact, diss_reason = adsorbate_is_intact(relaxed, n_slab_atoms)
        if not intact:
            n_dissociated += 1
            last_diss_reason = diss_reason
            if de < best_diss_de:
                best_diss_de = de
                best_diss_relaxed = relaxed
            continue
        if de < best_de:
            best_de = de
            best_relaxed = relaxed

    if best_relaxed is None and best_diss_relaxed is not None:
        warnings.warn(
            f"{label}: every one of {n_dissociated} valid candidate(s) "
            f"{last_diss_reason}. Falling back to the lowest-energy dissociated "
            f"structure (ΔE={best_diss_de:+.3f} eV), which is the energy of the "
            "fragments rather than of the intermediate this label names."
        )
        best_de = best_diss_de
        best_relaxed = best_diss_relaxed

    if best_relaxed is None:
        detail = (
            f"all {n_bad_geometry} relaxed geometries invalid (last: {last_geom_reason})"
            if n_bad_geometry else "all candidates failed"
        )
        return IntermediateEnergyResult(
            label=label, status="skipped", path_used=path_used,
            n_candidates=len(candidates), detail=detail,
        )

    return IntermediateEnergyResult(
        label=label, status="ok", delta_e_ads=best_de, relaxed=best_relaxed,
        path_used=path_used, n_candidates=len(candidates),
    )


# ---------------------------------------------------------------------------
# Per-catalyst orchestration
# ---------------------------------------------------------------------------

@_plain_dataclass
class AdsorptionEnergyCaseResult:
    catalyst_key: str
    skipped: bool
    skip_reason: str
    slab_n_atoms: int
    energy_results: dict[str, float] = _field(default_factory=dict)
    relaxed_structures: dict[str, object] = _field(default_factory=dict)
    item_statuses: list[IntermediateEnergyResult] = _field(default_factory=list)


def run_adsorption_energy(
    ctx: CatalystContext,
    kept_items: list[dict],
    calculator,
    max_denticity: int,
    llm_model: str,
    enum_smiles: dict[str, str] | None = None,
    enum_anchor_indices: dict[str, tuple[int, int]] | None = None,
) -> AdsorptionEnergyCaseResult:
    """Compute adsorption energies for every kept intermediate of one catalyst case.

    Args:
        ctx: catalyst context (composition, facet, reaction, pH, U)
        kept_items: final list of items to compute, already filtered/augmented
            by select_intermediates()
        calculator: loaded MLIP calculator, shared across catalyst cases
        max_denticity: maximum '*' anchors allowed (only affects bidentate gating)
        llm_model: LLM model name for SMILES fallback when not in enum_smiles
        enum_smiles: label -> SMILES from the enumerator JSON (avoids LLM calls)
        enum_anchor_indices: label -> (i0, i1) bidentate anchor indices from
            the enumerator JSON

    Returns:
        AdsorptionEnergyCaseResult with per-label energies, relaxed structures
        (for the caller to write trajectories), and per-item status detail.
    """
    enum_smiles = enum_smiles or {}
    enum_anchor_indices = enum_anchor_indices or {}
    catalyst_key = f"{ctx.composition}({ctx.facet})_{ctx.reaction}"

    try:
        refs = build_case_references(ctx, calculator)
    except Exception as e:
        return AdsorptionEnergyCaseResult(
            catalyst_key=catalyst_key, skipped=True, skip_reason=str(e), slab_n_atoms=0,
        )

    energy_results: dict[str, float] = {"*": 0.0}
    relaxed_structures: dict[str, object] = {}
    item_statuses: list[IntermediateEnergyResult] = []

    for item in kept_items:
        label = item["label"]
        parent = item.get("parent", "*")
        rule = item.get("rule", "unknown")

        if label == "*":
            item_statuses.append(IntermediateEnergyResult(
                label=label, status="ok", delta_e_ads=0.0, detail="reference, skipped",
            ))
            continue

        # *O_lattice = healed vacancy = clean slab = CHE reference surface.
        # ΔE_ads(*O_lattice) = 0 by identity in this reference frame — see
        # eval_adsorption_energy.py history for the full derivation.
        if label == "*O_lattice":
            energy_results[label] = 0.0
            item_statuses.append(IntermediateEnergyResult(
                label=label, status="ok", delta_e_ads=0.0,
                detail="healed vacancy = clean slab = CHE reference",
            ))
            continue

        if rule == "lattice_o_release" or label == "*V_O":
            vac = compute_vacancy_energy(refs.slab.atoms, calculator, refs.gas_refs, refs.e_slab)
            if vac.status != "ok":
                item_statuses.append(IntermediateEnergyResult(
                    label=label, status="skipped", n_candidates=vac.n_sites,
                    detail=vac.detail or vac.status,
                ))
                continue
            energy_results[label] = vac.e_vac
            relaxed_structures[label] = vac.relaxed
            item_statuses.append(IntermediateEnergyResult(
                label=label, status="ok", delta_e_ads=vac.e_vac, path_used="vacancy",
                n_candidates=vac.n_sites,
            ))
            continue

        result = compute_intermediate_energy(
            label, parent, rule, refs.slab, calculator, refs.gas_refs, refs.e_slab, ctx,
            enum_smiles.get(label), enum_anchor_indices.get(label),
            max_denticity, llm_model,
        )
        item_statuses.append(result)
        if result.status == "ok":
            energy_results[label] = result.delta_e_ads
            relaxed_structures[label] = result.relaxed

    return AdsorptionEnergyCaseResult(
        catalyst_key=catalyst_key, skipped=False, skip_reason="",
        slab_n_atoms=refs.slab_n_atoms, energy_results=energy_results,
        relaxed_structures=relaxed_structures, item_statuses=item_statuses,
    )
