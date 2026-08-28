"""Surface structure generation — reusable across projects.

Covers the full pipeline from label → candidate slab+adsorbate structures:

  label → fairchem DB id(s)    — exact / formula match (deterministic)
        ↓
  DB id → Adsorbate             — pre-built 3D conformer + binding sites
        ↓
  bulk → Slab                   — MP API → fairchem Slab (auto-tagged)
        ↓
  Adsorbate + Slab → candidates — fairchem AdsorbateSlabConfig
        ↓  (fallback)
  label → SMILES → candidates   — ACAT + rdkit2ase (metals AND oxides)

Extension points: add SLI, defect structures, or alternative placement
strategies here without touching any other module.
"""

from __future__ import annotations

import time
import warnings


# Rules whose products are surface defects/states — not molecular adsorbates
# and have no fairchem DB entry.
_DEFECT_RULES: frozenset[str] = frozenset({
    "lattice_o_release",   # → *V_O
    "vacancy_healing",     # → *O_lattice
})


# ---------------------------------------------------------------------------
# Fairchem adsorbate DB — lazy-loaded label → id lookup
# ---------------------------------------------------------------------------

_FAIRCHEM_DB_LABELS: dict[str, int] | None = None


def _get_fairchem_db() -> dict[str, int]:
    """Return {label_str: db_id} mapping from fairchem adsorbate DB.

    Loaded once and cached in module-level variable.
    """
    global _FAIRCHEM_DB_LABELS
    if _FAIRCHEM_DB_LABELS is not None:
        return _FAIRCHEM_DB_LABELS

    import pickle
    from fairchem.data.oc.databases.pkls import ADSORBATE_PKL_PATH

    with open(ADSORBATE_PKL_PATH, "rb") as f:
        db = pickle.load(f)

    # db is {int: (Atoms, label_str, binding_indices, formation_eq)}
    _FAIRCHEM_DB_LABELS = {entry[1]: db_id for db_id, entry in db.items()}
    return _FAIRCHEM_DB_LABELS


def _get_db_formula_index() -> dict[str, list[str]]:
    """Return {formula_str: [db_label, ...]} grouping from fairchem DB."""
    import pickle
    from fairchem.data.oc.databases.pkls import ADSORBATE_PKL_PATH

    from hellmm.chemistry import formula_str as _formula_str
    from hellmm.chemistry import parse_formula as _parse_formula

    with open(ADSORBATE_PKL_PATH, "rb") as f:
        db = pickle.load(f)

    index: dict[str, list[str]] = {}
    for entry in db.values():
        atoms, label_str, _, _ = entry
        formula = atoms.get_chemical_formula(mode="all")
        counts = _parse_formula(formula)
        key = _formula_str(counts)
        index.setdefault(key, []).append(label_str)
    return index


_DB_FORMULA_INDEX: dict[str, list[str]] | None = None


def _get_formula_index() -> dict[str, list[str]]:
    global _DB_FORMULA_INDEX
    if _DB_FORMULA_INDEX is None:
        _DB_FORMULA_INDEX = _get_db_formula_index()
    return _DB_FORMULA_INDEX


# ---------------------------------------------------------------------------
# label → fairchem DB id(s)
# ---------------------------------------------------------------------------

def label_to_fairchem_ids(
    label: str,
    rule: str,
) -> list[int]:
    """Map an adsorbate label to ALL matching fairchem DB ids.

    Returns every same-formula DB entry so the caller can relax all candidate
    geometries and keep the lowest-energy one.  This replaces the previous LLM
    disambiguation step, making adsorption energies fully deterministic and
    reproducible across runs.

    Design rationale — why minimum-energy over LLM connectivity matching
    ------------------------------------------------------------------
    The fairchem adsorbate DB contains 86 pre-built 3D geometries.  For labels
    with no exact string match, multiple DB entries may share the same molecular
    formula (e.g. C1H1O2 covers both ``*OCHO`` (formate, O-bound) and ``*COOH``
    (carboxyl, C-bound)).

    A previous version used an LLM to pick the "best connectivity match" from
    same-formula candidates based on the label name, parent intermediate, and
    transformation rule.  This was motivated by wanting to respect the semantic
    intent of the label — e.g. ``*CO2H`` produced by the protonation rule on
    ``*CO2`` should be carboxyl-like, not formate-like.

    However, LLM-based selection introduced run-to-run non-determinism: the
    same label could map to different DB entries across jobs, causing ΔE_ads to
    shift by up to ~0.3 eV and pathway rankings to change.

    The minimum-energy approach is adopted instead for three reasons:

    1. **CHE convention**: the Computational Hydrogen Electrode uses the most
       stable accessible configuration of each surface intermediate.  Taking the
       minimum over all same-formula geometries is consistent with this
       convention and physically principled.

    2. **Determinism**: results are identical across runs, independent of LLM
       temperature or API state.  This is a prerequisite for benchmarking and
       reproducibility.

    3. **DB geometry quality**: fairchem DB geometries are pre-built from DFT
       and are not surface-specific.  Neither the LLM connectivity pick nor the
       minimum-energy pick is guaranteed to be the true minimum for a novel
       surface; the minimum-energy approach at least provides a lower bound.

    Strategy (in order):
    1. Early exit for surface-defect labels (vacancies, lattice sites).
    2. Early exit for bidentate / multi-site labels.
    3. Exact string match → single-element list (no ambiguity).
    4. All same-formula DB entries → list (may have >1 element).
    5. No formula match → raise ValueError (ACAT fallback handles these).

    Args:
        label: adsorbate label, e.g. "*OH", "*COOH"
        rule:  transformation rule that produced this intermediate

    Returns:
        list of fairchem DB indices (ints), always non-empty

    Raises:
        ValueError: if label is a surface defect, bidentate, or has no
                    same-formula DB entry
    """
    from hellmm.chemistry import formula_str as _formula_str
    from hellmm.chemistry import parse_formula as _parse_formula

    # 1. Surface-defect early exit.
    if rule in _DEFECT_RULES:
        raise ValueError(
            f"{label!r} is a surface defect/state produced by rule '{rule}' "
            "— no fairchem DB entry exists for it."
        )

    # 2. Bidentate / multi-site early exit.
    if label.count("*") > 1:
        raise ValueError(
            f"{label!r} has {label.count('*')} surface anchors — multi-site / C–C coupled "
            "species are not in the fairchem adsorbate DB. "
            "Skip and compute with DFT/NEB if needed."
        )

    db_labels = _get_fairchem_db()

    # 3. Exact string match.
    if label in db_labels:
        return [db_labels[label]]

    # 4. All same-formula DB entries — no LLM, try them all.
    label_formula = _formula_str(_parse_formula(label))
    candidates = _get_formula_index().get(label_formula, [])

    if not candidates:
        raise ValueError(
            f"No fairchem DB entry with formula {label_formula!r} for {label!r}. "
            "Use ACAT+rdkit2ase backend."
        )

    return [db_labels[c] for c in candidates]


# ---------------------------------------------------------------------------
# Computability pre-check
# ---------------------------------------------------------------------------

_VACANCY_RULES: frozenset[str] = frozenset({"lattice_o_release"})


def check_computability(
    label: str,
    rule: str,
    max_denticity: int = 1,
) -> dict[str, str]:
    """Check whether a kept intermediate can be computed without an LLM call.

    Runs in microseconds (no DB I/O on repeated calls — DB is cached after
    first load).  Designed to be called before the MLIP relaxation loop to
    produce an upfront manifest of what will and won't be computed.

    Returns:
        dict with keys "status" and "reason". Status values:
          "reference"          — bare surface (*), ΔE = 0 by definition
          "filtered_denticity" — more '*' anchors than max_denticity
          "defect_vacancy"     — lattice_o_release → needs E_vac (todo)
          "defect_other"       — other surface-defect rule → skip
          "fairchem_db"        — exact or formula match in fairchem adsorbate DB
          "smiles_table"       — in monodentate or bidentate SMILES lookup table
          "llm_needed"         — not in any table; LLM will be called at runtime
    """
    if label == "*":
        return {"status": "reference", "reason": "bare surface, ΔE = 0 by definition"}

    n_anchors = label.count("*")
    if n_anchors > max_denticity:
        return {
            "status": "filtered_denticity",
            "reason": f"{n_anchors} '*' anchors > MAX_DENTICITY={max_denticity}",
        }

    if label == "*O_lattice":
        return {
            "status": "lattice_reference",
            "reason": "healed vacancy = clean slab; ΔE = 0 in CHE reference frame",
        }
    if rule in _VACANCY_RULES or label == "*V_O":
        return {
            "status": "defect_vacancy",
            "reason": "lattice_o_release → E_vac via compute_vacancy_configs",
        }
    if rule in _DEFECT_RULES:
        return {
            "status": "defect_other",
            "reason": f"surface defect produced by rule '{rule}' — no adsorbate energy",
        }

    # Fairchem DB check (raises ValueError for defects/bidentate/no formula match)
    try:
        label_to_fairchem_ids(label, rule)
        return {"status": "fairchem_db", "reason": "exact or formula match in fairchem DB"}
    except ValueError:
        pass

    # SMILES lookup tables — deterministic, no LLM
    if label in _LABEL_SMILES_TABLE:
        return {
            "status": "smiles_table",
            "reason": f"SMILES in monodentate table: {_LABEL_SMILES_TABLE[label]}",
        }
    if label in _BIDENTATE_TABLE:
        smiles, anchors = _BIDENTATE_TABLE[label]
        return {
            "status": "smiles_table",
            "reason": f"SMILES in bidentate table: {smiles}, anchors={anchors}",
        }

    return {
        "status": "llm_needed",
        "reason": "not in fairchem DB or any SMILES table — LLM will be called at runtime",
    }


# ---------------------------------------------------------------------------
# Relaxed-geometry validation
# ---------------------------------------------------------------------------

def check_adsorbate_geometry(
    atoms,
    n_slab_atoms: int,
    min_contact: float = 0.8,
    max_bond: float = 3.2,
    burial_tol: float = 0.5,
) -> tuple[bool, str]:
    """Check that a relaxed slab+adsorbate is still a physically sane surface species.

    Replaces the previous absolute |ΔE_ads| cutoff, which was calibrated on
    *OH/*O/*OOH (0–3 eV) and silently excluded whole species families: in the
    CHE reference frame used here (O ref = E(H₂O) − E(H₂)) a *free, unbound*
    O₂ molecule evaluates to ≈ +5.7 eV, so the 4 eV cutoff applied to
    two-heavy-atom adsorbates sat 1.7 eV *below* unbound O₂ and no O₂-bearing
    species could ever pass — bound, unbound, or perfectly relaxed.  The
    cutoff was always a proxy for "did the placement collapse or fly away",
    which is a geometric question, so ask it geometrically instead.  A sane
    geometry with an unexpected energy is a real result and is reported, not
    silently dropped.

    Assumes the slab-first atom ordering used throughout the pipeline
    (atoms[:n_slab_atoms] = slab, atoms[n_slab_atoms:] = adsorbate), the same
    convention che.py and ranker.py use to pick adsorbate vibrational modes.

    Threshold calibration (measured on relaxed Pt(111) structures from
    runs/20260806_153226): real closest adsorbate–slab contacts span 1.55 Å
    (Pt–H) to 2.18 Å (Pt–OH).  min_contact=0.8 Å therefore sits well below
    any genuine bond while still catching true overlaps (a deliberately
    collapsed test geometry gave 0.06 Å), and max_bond=3.2 Å sits above the
    longest real contact with room for weakly-bound species.  Defaults are
    deliberately permissive: the goal is rejecting pathological placements,
    not fine-grained quality control.

    Args:
        atoms: relaxed ASE Atoms (slab + adsorbate)
        n_slab_atoms: number of slab atoms
        min_contact: below this adsorbate–slab distance (Å) the geometry has
            collapsed into the surface
        max_bond: above this closest adsorbate–slab distance (Å) the adsorbate
            has detached rather than adsorbed
        burial_tol: how far (Å) the whole adsorbate may sit below the topmost
            slab atom before it counts as buried inside the slab

    Returns:
        (ok, reason) — reason is "" when ok.
    """
    import numpy as np

    if len(atoms) <= n_slab_atoms:
        return True, ""   # nothing adsorbed (bare slab / vacancy) — not our concern

    ads = list(range(n_slab_atoms, len(atoms)))
    slab = list(range(n_slab_atoms))

    d = atoms.get_all_distances(mic=True)[np.ix_(ads, slab)]
    d_min = float(d.min())

    if d_min < min_contact:
        return False, f"collapsed into surface (closest contact {d_min:.2f} Å)"
    if d_min > max_bond:
        return False, f"adsorbate detached (closest contact {d_min:.2f} Å)"

    z_ads_max = float(atoms.positions[ads, 2].max())
    z_slab_top = float(atoms.positions[slab, 2].max())
    if z_ads_max < z_slab_top - burial_tol:
        return False, (f"adsorbate buried below surface "
                       f"(top adsorbate z={z_ads_max:.2f} Å < slab top {z_slab_top:.2f} Å)")

    return True, ""


def adsorbate_is_intact(
    atoms,
    n_slab_atoms: int,
    bond_tol: float = 1.2,
) -> tuple[bool, str]:
    """Check the relaxed adsorbate is still one connected fragment.

    Kept separate from check_adsorbate_geometry because it answers a different
    question, and because the answer must not be a hard rejection.  The checks
    there ask how the adsorbate sits relative to the slab; a fragment pair that
    *both* land properly on the surface passes every one of them.

    Measured on Au(110): *COOH dissociated in 4 of 5 repeat runs, H migrating
    onto the metal (Au-H 1.6-1.8 Å) and leaving physisorbed CO2 behind
    (symmetric C-O 1.181 Å, against 1.217/1.382 Å for intact carboxyl).  All
    four passed validation and were recorded as *COOH at ΔE = -0.95 eV — the
    energy of CO2 + H*, not of the intermediate.  The single run that kept the
    molecule together returned -0.545 eV and looked like the outlier.  Nothing
    downstream can notice: the label is carried as a string and CHE scores
    whatever energy arrives under it.

    Dissociation is not always an error, which is why this reports rather than
    rejects.  H2 on Pt(111) dissociates in every run, and that is correct
    surface chemistry — molecular H2 is not a stable adsorbate there, so the
    relaxed 2H* state is the physical one and its energy is the right quantity
    for a Tafel step.  Callers should prefer an intact candidate when one
    exists and fall back to a dissociated one, flagged, when none does.

    Two atoms count as bonded within `bond_tol` times the sum of their covalent
    radii.  1.2 is the usual convention and has wide margin here: the intact
    O-H bond sits at 1.02 of the radius sum, the dissociated one at 2.9.

    Returns:
        (intact, reason) — reason is "" when intact.
    """
    if len(atoms) <= n_slab_atoms:
        return True, ""

    from ase.data import covalent_radii

    ads = list(range(n_slab_atoms, len(atoms)))
    n_ads = len(ads)
    if n_ads <= 1:
        return True, ""

    z = atoms.get_atomic_numbers()
    dd = atoms.get_all_distances(mic=True)

    seen = {0}
    stack = [0]
    while stack:                           # flood-fill one fragment from atom 0
        i = stack.pop()
        for k in range(n_ads):
            if k in seen:
                continue
            gi, gk = ads[i], ads[k]
            limit = bond_tol * (covalent_radii[z[gi]] + covalent_radii[z[gk]])
            if dd[gi][gk] <= limit:
                seen.add(k)
                stack.append(k)

    if len(seen) < n_ads:
        loose = [atoms.get_chemical_symbols()[ads[k]]
                 for k in range(n_ads) if k not in seen]
        return False, (
            f"dissociated during relaxation — {len(seen)}/{n_ads} adsorbate "
            f"atoms remain bonded, detached: {loose}"
        )

    return True, ""


# ---------------------------------------------------------------------------
# Oxygen vacancy configurations
# ---------------------------------------------------------------------------

def compute_vacancy_configs(slab_atoms) -> list:
    """Generate defect slabs by removing each distinct surface oxygen atom.

    Surface oxygens are those within 3 Å of the highest O z-coordinate in
    the slab — this captures the topmost oxygen layer without relying on
    fairchem tag conventions.

    Each returned Atoms object has one surface O removed.  Relax all of them
    with the MLIP and keep the lowest-energy result; that energy feeds into:

        E_vac = E(defect_slab) + gas_refs["O"] − E(clean_slab)

    where gas_refs["O"] = E(H₂O) − E(H₂) is the CHE oxygen reference,
    keeping E_vac in the same reference frame as all other ΔE_ads values.

    Args:
        slab_atoms: plain ASE Atoms (clean slab)

    Returns:
        list of ASE Atoms, one per surface O site

    Raises:
        RuntimeError: if the slab contains no oxygen atoms
    """
    import numpy as np

    symbols  = slab_atoms.get_chemical_symbols()
    o_indices = [i for i, s in enumerate(symbols) if s == "O"]

    if not o_indices:
        raise RuntimeError(
            "No oxygen atoms found in slab — vacancy formation requires an oxide surface."
        )

    z        = slab_atoms.positions[:, 2]
    z_max_o  = max(z[i] for i in o_indices)
    surf_o   = [i for i in o_indices if z_max_o - z[i] <= 3.0]

    configs = []
    for idx in surf_o:
        defect = slab_atoms.copy()
        del defect[idx]
        configs.append(defect)

    return configs


# ---------------------------------------------------------------------------
# Build surface slab (MP API → fairchem Slab)
# ---------------------------------------------------------------------------

# Materials Project's client has no built-in request timeout, and a stalled
# connection blocks synchronously forever (observed: a SLURM job hung 3+ days
# on this call with no error, no CPU activity, no way to distinguish "slow"
# from "dead" without node access — even though the API key and endpoint were
# confirmed working). A hard wall-clock alarm turns that into a clear, fast
# failure that the caller's normal exception handling can act on.
_MP_API_TIMEOUT_S = 60


def _get_mp_entries_with_timeout(mpr, composition: str, timeout_s: int = _MP_API_TIMEOUT_S):
    import signal

    def _on_timeout(signum, frame):
        raise TimeoutError(
            f"Materials Project API call for '{composition}' did not respond "
            f"within {timeout_s}s — treating as a hung/unreachable request."
        )

    old_handler = signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(timeout_s)
    try:
        return mpr.get_entries(composition, inc_structure=True)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def build_slab(context):
    """Build a fairchem Slab for the given catalyst.

    Fetches the ground-state bulk structure from the Materials Project API
    (requires MY_MP_API_KEY environment variable), converts to ASE Atoms,
    then uses fairchem's slab generator which handles tiling and surface
    atom tagging automatically.

    Args:
        context: catalyst context with .composition and .facet (e.g. "110")

    Returns:
        fairchem Slab object (surface atoms tagged, sub-surface constrained)

    Raises:
        EnvironmentError: if MY_MP_API_KEY is not set
        TimeoutError: if the Materials Project API call hangs past _MP_API_TIMEOUT_S
        ValueError: if MP has no entries for the composition or no slab generated
    """
    import os
    from pymatgen.ext.matproj import MPRester
    from pymatgen.io.ase import AseAtomsAdaptor
    from fairchem.data.oc.core import Bulk, Slab

    api_key = os.environ.get("MY_MP_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "MY_MP_API_KEY is not set. Add it to your .env file:\n"
            "  MY_MP_API_KEY=your_key_here"
        )

    miller = tuple(int(x) for x in context.facet)

    with MPRester(api_key) as mpr:
        entries = _get_mp_entries_with_timeout(mpr, context.composition)
    if not entries:
        raise ValueError(f"No MP entries found for composition: {context.composition}")

    # Pick lowest-energy polymorph
    best = min(entries, key=lambda e: e.energy_per_atom)
    ase_bulk = AseAtomsAdaptor.get_atoms(best.structure)

    # Wrap in fairchem Bulk, then generate the specific-miller slab
    bulk = Bulk(bulk_atoms=ase_bulk)
    slabs = Slab.from_bulk_get_specific_millers(
        specific_millers=miller,
        bulk=bulk,
        min_ab=8.0,  # ensures supercell is large enough for site sampling
    )
    if not slabs:
        raise ValueError(
            f"No slab generated for {context.composition} {miller}. "
            "Try a different miller index."
        )

    return slabs[0]


# ---------------------------------------------------------------------------
# Adsorption structure configurations — fairchem DB backend
# ---------------------------------------------------------------------------

def generate_adsorption_configs(
    slab, db_id: int, num_sites: int = 20, num_augmentations_per_site: int = 3
) -> list:
    """Generate slab+adsorbate configurations at heuristic adsorption sites.

    Uses fairchem's Adsorbate (pre-built 3D geometry from the 86-entry DB)
    and AdsorbateSlabConfig to enumerate on-top, bridge, and hollow sites.
    Returns one ASE Atoms object per (site, orientation) pair — all should
    be relaxed and the lowest-energy configuration kept.

    Args:
        slab: fairchem Slab object (from build_slab)
        db_id: fairchem adsorbate DB id (from label_to_fairchem_ids)
        num_sites: number of sites to sample (default 20)
        num_augmentations_per_site: rotational orientations per site (default 3).
            Higher values reduce sensitivity to BFGS local minima on small unit cells
            (e.g. Cu(111) has only 4 unique high-symmetry sites; 3 augmentations
            gives 12 configurations instead of 4).

    Returns:
        list of ASE Atoms (slab + adsorbate at each sampled site × orientation)
    """
    from fairchem.data.oc.core import Adsorbate, AdsorbateSlabConfig

    adsorbate = Adsorbate(adsorbate_id_from_db=db_id)
    config = AdsorbateSlabConfig(
        slab=slab,
        adsorbate=adsorbate,
        num_sites=num_sites,
        num_augmentations_per_site=num_augmentations_per_site,
        mode="heuristic",
    )
    return config.atoms_list


# ---------------------------------------------------------------------------
# label → SMILES (lookup table + LLM fallback)
# ---------------------------------------------------------------------------

# Lookup table for common electrochemistry intermediates.
# Maps adsorbate label → SMILES of the free molecular fragment (no * anchor).
# Avoids an LLM call for well-characterised species.
_LABEL_SMILES_TABLE: dict[str, str] = {
    "*H":     "[H]",
    "*O":     "[O]",
    "*OH":    "O",
    "*OOH":   "OO",
    "*OO":    "[O][O]",
    "*O2":    "[O][O]",
    "*OOOH":  "[O]OO",    # hydrotrioxyl radical: •O-O-O-H, radical end binds surface (1H)
    "*V_OOOH": "[O]OO",  # same fragment as *OOOH; V_ prefix denotes vacancy site context
    "*OOHO":  "[OH][O][O]",  # HO-O-O•: radical end binds surface, H on proximal O (H1O3)
    # RuO2 cation-oxidation-state variants — adsorbate fragment is the same as the
    # base species; the [Ru^V]/[Ru^VI] suffix denotes the site oxidation state, which
    # the MLIP captures from the local geometry, not the SMILES.
    "*O[Ru^V]_lattice":   "[O]",
    "*O[Ru^VI]_lattice":  "[O]",
    "*OOH[Ru^V]_lattice": "OO",
    "*OOH[Ru^VI]_lattice":"OO",
    "*H2O":   "O",
    "*H2":    "[H][H]",
    "*CO":    "[C-]#[O+]",
    "*CO2":   "C(=O)=O",
    "*COOH":  "OC=O",
    "*CO2H":  "OC=O",
    "*COO":   "O=[C][O]", # carboxylate radical: C-radical binds surface, bent C1O2 (no H)
    "*OCHO":  "OC=O",
    "*HOOC":  "OC=O",
    "*CHO":   "C=O",
    "*COH":   "CO",
    "*CHOH":  "CO",
    "*CH2O":  "C=O",
    "*CH2OH": "CO",
    "*CH3OH": "CO",
    "*C":     "[C]",
    "*CH":    "[CH2]",
    "*CH2":   "[CH3]",
    "*CH3":   "C",
    "*C(OH)2": "O[C]O",     # gem-diol radical: C-radical (0 implicit H), two OH groups, C1H2O2
    "*COHOH":  "[C](O)O",   # same connectivity as *C(OH)2: C-radical, two OH groups, C1H2O2
    "*CH(OH)2": "[CH](O)O", # methanediol radical: C-radical (1H explicit), two OH groups, C1H3O2
    "*OCH2OH": "[O]CO",     # O-radical–CH2–OH: surface binds via O, C1H3O2
    "*HOCH2O": "OC[O]",     # HO–CH2–O-radical: surface binds via terminal O, C1H3O2
    "*CO2OH":   "OC(=O)[O]", # carbonate-like: CO2 + OH, C-radical with two O and one OH (C1H1O3)
    "*CO2(OH)": "OC(=O)[O]", # alias for *CO2OH
    "*HCO2OH":  "OC(=O)O",  # carbonic acid: H-C(=O)-OH with extra OH, H2CO3, C1H2O3
    "*C(O)(OH)": "C(=O)O",
    "*N":     "[N]",
    "*NH":    "[NH3]",
    "*NH2":   "N",
    "*NNH":   "N=N",        # N₂H₁ approximated as diazene (fairchem handles the real case)
    "*NNH2":  "N=N",       # N₂H₂: diazene HN=NH (correct formula; fairchem handles real geometry)
    "*N2":    "N#N",
    "*N2H4":  "NN",        # N₂H₄: hydrazine
    "*NHNH2": "[NH][NH2]",   # N2H3: NH bound to surface, NH2 on distal N
    "*NH2NH": "[NH2][NH]",   # N2H3: NH2 bound to surface, NH on distal N
}

_SMILES_SYSTEM_PROMPT = """\
You are an expert in surface chemistry and electrochemistry.
Convert an adsorbate label used in computational electrochemistry to a
SMILES string for the FREE molecular fragment — remove the surface anchor *,
keep only the molecular content.

Rules:
- Return JSON: {"smiles": "<SMILES>", "note": "<one sentence>"}
- SMILES must be parseable by RDKit
- Remove all * anchors; keep only atoms and bonds
- For atomic adsorbates (*O, *N, *C, *H): [O], [N], [C], [H]
- *OH → O (water SMILES; rdkit2ase will give H2O which is fine for site placement)
- *OOH → OO  |  *COOH → OC=O  |  *CO → [C-]#[O+]  |  *CHO → C=O
- Do NOT over-protonate: *O is not water; *OH is not H2O (use O for OH fragment)
"""

_SMILES_USER_TEMPLATE = """\
Catalyst: {composition}({facet}), {reaction}
Parent intermediate: {parent}
Formed via rule: {rule}

Adsorbate label: {label}

Convert to SMILES (free molecular fragment, no surface anchor *). Return JSON.
"""


def _smiles_formula(smiles: str) -> dict[str, int] | None:
    """Return element counts for a SMILES string using RDKit (with implicit H added).

    Returns None if RDKit is not installed or the SMILES is unparseable.
    Used to validate that LLM-generated SMILES match the formula implied by the label.
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        counts: dict[str, int] = {}
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            counts[sym] = counts.get(sym, 0) + 1
        return counts
    except ImportError:
        return None


def label_to_smiles(
    label: str,
    parent: str,
    rule: str,
    context,
    model: str = "deepseek-v3",
    max_retries: int = 3,
) -> str:
    """Convert an adsorbate label to a SMILES string for rdkit2ase.

    Checks a lookup table first; calls the LLM only for unknown labels.

    Args:
        label:      adsorbate label, e.g. "*OOH", "*COOH"
        parent:     parent intermediate (LLM context)
        rule:       transformation rule name (LLM context)
        context:    catalyst context (has .composition, .facet, .reaction)
        model:      LLM model alias
        max_retries: LLM retries on failure

    Returns:
        SMILES string (validated by RDKit when available)

    Raises:
        ValueError: if no valid SMILES can be produced after all retries
    """
    from hellmm.chemistry import parse_formula as _parse_formula
    from hellmm.llm import call_llm

    # Fast path: lookup table
    if label in _LABEL_SMILES_TABLE:
        return _LABEL_SMILES_TABLE[label]

    # LLM fallback
    user_msg = _SMILES_USER_TEMPLATE.format(
        composition=context.composition,
        facet=context.facet,
        reaction=context.reaction,
        label=label,
        parent=parent,
        rule=rule,
    )

    for attempt in range(max_retries):
        try:
            raw = call_llm(system=_SMILES_SYSTEM_PROMPT, user=user_msg, model=model)
            if not isinstance(raw, dict):
                raise ValueError(f"Expected dict, got {type(raw)}: {raw!r}")
            smiles = raw.get("smiles", "")
            if not smiles:
                raise ValueError(f"LLM returned empty SMILES for {label!r}")
            # Validate parsability and formula with RDKit if available
            try:
                from rdkit import Chem  # type: ignore[import]
                if Chem.MolFromSmiles(smiles) is None:
                    raise ValueError(f"RDKit cannot parse SMILES {smiles!r}")
                expected = _parse_formula(label)
                actual = _smiles_formula(smiles)
                if actual is not None and actual != expected:
                    raise ValueError(
                        f"SMILES {smiles!r} formula {actual} does not match "
                        f"label {label!r} formula {expected} — LLM used wrong H count "
                        "or wrong element. Retrying."
                    )
            except ImportError:
                pass
            return smiles
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                warnings.warn(
                    f"label_to_smiles failed for {label!r} "
                    f"(attempt {attempt + 1}): {exc}. Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                raise ValueError(
                    f"label_to_smiles failed for {label!r} "
                    f"after {max_retries} attempts: {exc}"
                )


# ---------------------------------------------------------------------------
# Bidentate adsorbate SMILES + anchor lookup
# ---------------------------------------------------------------------------

# Maps bidentate label → (SMILES_of_free_fragment, (anchor_heavy_atom_idx_0, anchor_heavy_atom_idx_1))
# Indices are into the RDKit heavy-atom list (0-based, before H addition) as
# produced by rdkit2ase.smiles2atoms — heavy-atom ordering matches SMILES traversal.
_BIDENTATE_TABLE: dict[str, tuple[str, tuple[int, int]]] = {
    "*CO2*":  ("O=C=O",  (0, 2)),   # CO2: O(0)=C(1)=O(2), both O bind surface
    "*OC*":   ("[C]=O",  (0, 1)),   # CO: C(0)=O(1), both atoms bind surface
    "*N2*":   ("N#N",    (0, 1)),   # N2: N(0)≡N(1), both N bind surface
    "*OCCO*": ("OCCO",   (0, 3)),   # oxalate bridge: O(0)-C-C-O(3), terminal O bind
    "*OCO*":  ("O=CO",   (0, 2)),   # formate bridge: O(0)=C(1)-O(2), both O bind
    "*OOH*":  ("OO",     (0, 1)),   # peroxo bridge: O(0)-O(1), both O bind
}

_BIDENTATE_SMILES_SYSTEM_PROMPT = """\
You are an expert in computational surface chemistry and electrochemistry.
Convert a bidentate adsorbate label (which has two * surface-anchors) to an
atom-mapped SMILES where the two binding atoms are tagged with :1 and :2.

Rules:
- Return JSON: {"smiles": "<atom-mapped SMILES>", "note": "<one sentence>"}
- Use standard SMILES with [atom:1] and [atom:2] for the two surface-binding atoms
- Remove all * anchors; tag only the atoms that were anchored to the surface
- Example: *CO2* (CO2, both O bind) → {"smiles": "[O:1]=C=[O:2]", ...}
- Example: *N2*  (N2, both N bind)  → {"smiles": "[N:1]#[N:2]", ...}
- SMILES must be parseable by RDKit
"""

_BIDENTATE_SMILES_USER_TEMPLATE = """\
Catalyst: {composition}({facet}), {reaction}
Parent intermediate: {parent}
Formed via rule: {rule}

Bidentate adsorbate label: {label}
(has two * anchors — both binding atoms must be tagged :1 and :2)

Convert to atom-mapped SMILES. Return JSON.
"""


def _extract_anchor_indices(mapped_smiles: str) -> tuple[str, tuple[int, int]]:
    """Parse atom-mapped SMILES → (clean_smiles, (idx_anchor_0, idx_anchor_1)).

    Returns clean SMILES (map numbers removed) and the heavy-atom indices of
    the two anchor atoms, matching the rdkit2ase heavy-atom ordering.
    """
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("RDKit is required for bidentate anchor extraction") from exc

    mol = Chem.MolFromSmiles(mapped_smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse atom-mapped SMILES: {mapped_smiles!r}")

    a0 = a1 = None
    for atom in mol.GetAtoms():
        m = atom.GetAtomMapNum()
        if m == 1:
            a0 = atom.GetIdx()
        elif m == 2:
            a1 = atom.GetIdx()

    if a0 is None or a1 is None:
        raise ValueError(
            f"SMILES {mapped_smiles!r} must have exactly two atoms tagged :1 and :2. "
            f"Got :1 → {a0}, :2 → {a1}."
        )

    # Remove map numbers; keep atom ordering canonical=False so indices match
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        atom.SetAtomMapNum(0)
    clean_smiles = Chem.MolToSmiles(rw.GetMol(), canonical=False)
    return clean_smiles, (a0, a1)


def label_to_bidentate_smiles(
    label: str,
    parent: str = "",
    rule: str = "",
    context=None,
    model: str = "deepseek-v3",
    max_retries: int = 3,
) -> tuple[str, tuple[int, int]]:
    """Convert a bidentate adsorbate label to (SMILES, (anchor_idx_0, anchor_idx_1)).

    Checks _BIDENTATE_TABLE first; calls the LLM with atom-mapped SMILES only
    for unknown labels. Atom indices are into the heavy-atom list produced by
    rdkit2ase.smiles2atoms (heavy atoms preserve SMILES traversal order).

    Args:
        label:       bidentate label, e.g. "*CO2*", "*N2*"
        parent:      parent intermediate (LLM context)
        rule:        transformation rule name (LLM context)
        context:     catalyst context (has .composition, .facet, .reaction)
        model:       LLM model alias
        max_retries: LLM retries on failure

    Returns:
        (smiles, (anchor_atom_idx_0, anchor_atom_idx_1))

    Raises:
        ValueError: if no valid SMILES+anchors can be produced
    """
    from hellmm.llm import call_llm

    if label in _BIDENTATE_TABLE:
        return _BIDENTATE_TABLE[label]

    if context is None:
        raise ValueError(
            f"No entry in _BIDENTATE_TABLE for {label!r} and no context provided for LLM fallback."
        )

    user_msg = _BIDENTATE_SMILES_USER_TEMPLATE.format(
        composition=context.composition,
        facet=context.facet,
        reaction=context.reaction,
        label=label,
        parent=parent,
        rule=rule,
    )

    for attempt in range(max_retries):
        try:
            raw = call_llm(system=_BIDENTATE_SMILES_SYSTEM_PROMPT, user=user_msg, model=model)
            if not isinstance(raw, dict):
                raise ValueError(f"Expected dict, got {type(raw)}: {raw!r}")
            mapped_smiles = raw.get("smiles", "")
            if not mapped_smiles:
                raise ValueError(f"LLM returned empty SMILES for {label!r}")
            clean_smiles, anchor_indices = _extract_anchor_indices(mapped_smiles)
            return clean_smiles, anchor_indices
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                warnings.warn(
                    f"label_to_bidentate_smiles failed for {label!r} "
                    f"(attempt {attempt + 1}): {exc}. Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                raise ValueError(
                    f"label_to_bidentate_smiles failed for {label!r} "
                    f"after {max_retries} attempts: {exc}"
                )


# ---------------------------------------------------------------------------
# Bidentate adsorption structure configurations — ACAT site-pair backend
# ---------------------------------------------------------------------------

def _rotation_matrix_align(v_from: "np.ndarray", v_to: "np.ndarray") -> "np.ndarray":
    """Rotation matrix that rotates v_from → v_to (vectors need not be unit length)."""
    import numpy as np
    v_from = v_from / (np.linalg.norm(v_from) + 1e-12)
    v_to   = v_to   / (np.linalg.norm(v_to)   + 1e-12)
    c  = float(np.dot(v_from, v_to))
    ax = np.cross(v_from, v_to)
    s  = float(np.linalg.norm(ax))
    if s < 1e-6:
        if c > 0:
            return np.eye(3)
        # Anti-parallel: 180° around any perpendicular axis
        perp = np.array([1.0, 0.0, 0.0]) if abs(v_from[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp -= np.dot(perp, v_from) * v_from
        perp /= np.linalg.norm(perp)
        return 2 * np.outer(perp, perp) - np.eye(3)
    vx = np.array([[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]])
    return np.eye(3) + vx + (vx @ vx) * (1.0 - c) / (s * s)


def _rotation_matrix_axis_angle(axis: "np.ndarray", angle: float) -> "np.ndarray":
    """Rodrigues rotation around a (unit) axis by angle (radians)."""
    import numpy as np
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    c, s  = np.cos(angle), np.sin(angle)
    ax, ay, az = axis
    return np.array([
        [c + ax*ax*(1-c),    ax*ay*(1-c) - az*s,  ax*az*(1-c) + ay*s],
        [ay*ax*(1-c) + az*s, c + ay*ay*(1-c),     ay*az*(1-c) - ax*s],
        [az*ax*(1-c) - ay*s, az*ay*(1-c) + ax*s,  c + az*az*(1-c)  ],
    ])


def _place_bidentate_config(
    slab_atoms,
    ads_atoms,
    anchor_indices: tuple[int, int],
    site1_pos: "np.ndarray",
    site2_pos: "np.ndarray",
    height: float,
    spin_angle: float,
):
    """Place bidentate adsorbate with anchor atoms above two surface sites.

    The adsorbate is rotated so its anchor-anchor vector aligns with the
    site1→site2 vector, then spun by spin_angle around that axis.  The
    midpoint of the two anchor atoms is placed height Å above the surface
    midpoint.

    Returns combined slab+adsorbate Atoms, or None if anchor indices are invalid.
    """
    import numpy as np

    a0, a1 = anchor_indices
    if a0 >= len(ads_atoms) or a1 >= len(ads_atoms):
        return None

    pos = ads_atoms.positions.copy().astype(float)
    anchor_mid = (pos[a0] + pos[a1]) / 2.0
    pos -= anchor_mid  # centre on anchor midpoint

    # 1. Align anchor-anchor vector to site-site vector
    ads_vec  = pos[a1] - pos[a0]
    surf_vec = np.array(site2_pos, dtype=float) - np.array(site1_pos, dtype=float)
    R_align  = _rotation_matrix_align(ads_vec, surf_vec)
    pos      = pos @ R_align.T

    # 2. Spin around site-site axis to sample out-of-plane orientations
    if abs(spin_angle) > 1e-6:
        axis   = surf_vec / (np.linalg.norm(surf_vec) + 1e-12)
        R_spin = _rotation_matrix_axis_angle(axis, spin_angle)
        pos    = pos @ R_spin.T

    # 3. Translate: anchor midpoint → surface midpoint + height in Z
    s1 = np.array(site1_pos, dtype=float)
    s2 = np.array(site2_pos, dtype=float)
    surf_mid        = (s1 + s2) / 2.0
    target          = surf_mid.copy()
    target[2]       = max(s1[2], s2[2]) + height
    pos            += target

    ads_placed           = ads_atoms.copy()
    ads_placed.positions = pos
    return slab_atoms + ads_placed


def generate_adsorption_configs_acat_bidentate(
    slab_atoms,
    smiles: str,
    anchor_indices: tuple[int, int],
    distance_tolerance: float = 0.8,
    n_rotations: int = 3,
    height: float = 2.0,
    max_pairs: int = 20,
) -> list:
    """Generate slab+adsorbate configurations for bidentate adsorbates.

    Enumerates pairs of ACAT adsorption sites whose mutual distance matches
    the adsorbate's anchor-anchor distance (within distance_tolerance).  For
    each valid pair the adsorbate is sampled at n_rotations spin angles around
    the site-site axis.

    All configurations should be relaxed with an MLIP and the lowest-energy
    one kept.  Post-relaxation, both anchor atoms should remain within ~2.5 Å
    of the surface (bidentate binding preserved).

    Args:
        slab_atoms:          plain ASE Atoms (e.g. fairchem Slab.atoms)
        smiles:              SMILES of free adsorbate fragment
        anchor_indices:      (idx0, idx1) — heavy-atom indices of the two
                             surface-binding atoms in the rdkit2ase conformer
        distance_tolerance:  max |d_site_pair − d_anchor_pair| in Å (default 0.8)
        n_rotations:         spin angles sampled around site-site axis (default 3)
        height:              anchor-midpoint above surface in Å (default 2.0)
        max_pairs:           maximum site pairs to sample (default 20)

    Returns:
        list of ASE Atoms (slab + adsorbate)

    Raises:
        ImportError: if acat or rdkit2ase is not installed
        RuntimeError: if no compatible site pairs found or all placements fail
    """
    import math
    import numpy as np

    try:
        from acat.adsorption_sites import SlabAdsorptionSites
        from acat.settings import CustomSurface
    except ImportError as exc:
        raise ImportError("acat is not installed: pip install acat") from exc

    try:
        from rdkit2ase import smiles2atoms
    except ImportError as exc:
        raise ImportError("rdkit2ase is not installed: pip install rdkit2ase") from exc

    ads_atoms = smiles2atoms(smiles)

    if max(anchor_indices) >= len(ads_atoms):
        raise ValueError(
            f"anchor_indices {anchor_indices} out of range for adsorbate with "
            f"{len(ads_atoms)} atoms (SMILES={smiles!r}). "
            "Check _BIDENTATE_TABLE or LLM output."
        )

    a0_pos      = ads_atoms.positions[anchor_indices[0]]
    a1_pos      = ads_atoms.positions[anchor_indices[1]]
    anchor_dist = float(np.linalg.norm(a1_pos - a0_pos))

    custom_surf = CustomSurface(slab_atoms, n_layers=None)
    sas         = SlabAdsorptionSites(slab_atoms, surface=custom_surf, allow_6fold=False)
    sites       = sas.get_sites()

    if not sites:
        raise RuntimeError("ACAT found 0 adsorption sites on the provided slab.")

    site_pos = [np.array(s["position"], dtype=float) for s in sites]

    # Find site pairs with compatible anchor-anchor distance
    n = len(site_pos)
    valid_pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(site_pos[j] - site_pos[i]))
            if abs(d - anchor_dist) <= distance_tolerance:
                valid_pairs.append((i, j))

    if not valid_pairs:
        all_dists = sorted(
            float(np.linalg.norm(site_pos[j] - site_pos[i]))
            for i in range(n) for j in range(i + 1, n)
        )
        dist_summary = (
            f"{all_dists[0]:.2f}–{all_dists[-1]:.2f} Å" if all_dists else "none"
        )
        raise RuntimeError(
            f"No site pairs within {distance_tolerance} Å of anchor distance "
            f"{anchor_dist:.2f} Å. Surface pair distances: {dist_summary}. "
            "Try increasing distance_tolerance."
        )

    if len(valid_pairs) > max_pairs:
        step       = max(1, len(valid_pairs) // max_pairs)
        valid_pairs = valid_pairs[::step][:max_pairs]

    candidates = []
    for i, j in valid_pairs:
        for rot_idx in range(n_rotations):
            spin   = rot_idx * (2.0 * math.pi / n_rotations)
            config = _place_bidentate_config(
                slab_atoms.copy(), ads_atoms, anchor_indices,
                site_pos[i], site_pos[j], height, spin,
            )
            if config is not None:
                candidates.append(config)

    if not candidates:
        raise RuntimeError(
            f"All bidentate placements failed for SMILES={smiles!r}, "
            f"anchor_indices={anchor_indices}."
        )

    return candidates


# ---------------------------------------------------------------------------
# Adsorption structure configurations — ACAT + rdkit2ase backend
# ---------------------------------------------------------------------------

# ACAT's built-in surface types.  Anything outside this set has no native
# handler and must fall back to CustomSurface.
_ACAT_SURFACES = frozenset({
    "fcc100", "fcc110", "fcc111", "fcc211", "fcc221", "fcc311", "fcc322",
    "fcc331", "fcc332",
    "bcc100", "bcc110", "bcc111", "bcc210", "bcc211", "bcc310",
    "hcp0001", "hcp10m10h", "hcp10m10t", "hcp10m11", "hcp10m12",
})


def acat_surface_name(slab) -> str | None:
    """Derive ACAT's native surface identifier (e.g. "fcc110") for a fairchem Slab.

    ACAT's generic ``CustomSurface`` detects adsorption sites by inferring
    atomic planes, and that inference fails outright on corrugated surfaces:
    on Au(110) — 8 planes of 6 atoms — it raised IndexError for every
    ``n_layers`` value tried, so no site could be generated and every
    adsorbate not present in the fairchem DB became uncomputable (including
    *CO2, the CO2RR starting adsorbate).  ACAT's native ``surface="fcc110"``
    handles the same slab correctly, returning the 4fold/5fold trough sites
    that characterise fcc(110).

    The identifier must be *derived* rather than guessed: passing the wrong
    one is silently wrong rather than an error — ``surface="fcc111"`` on this
    Au(110) slab also returns 72 sites, just with fcc(111) site types.  So the
    Bravais lattice is read from the bulk's space group and combined with the
    slab's own miller index; both are already carried on the Slab object.

    This encodes crystallographic site geometry only — the same category of
    information as the miller index itself.  It says nothing about the
    reaction, and no value here reaches an LLM prompt.

    Returns:
        An ACAT surface string, or None when the structure has no native ACAT
        handler (oxides, perovskites, unusual millers), in which case the
        caller should keep using CustomSurface.
    """
    bulk = getattr(slab, "bulk", None)
    millers = getattr(slab, "millers", None)
    if bulk is None or millers is None:
        return None

    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        bulk_atoms = getattr(bulk, "atoms", bulk)
        sga = SpacegroupAnalyzer(AseAtomsAdaptor.get_structure(bulk_atoms))
        system = sga.get_crystal_system()
        centring = sga.get_space_group_symbol()[0]
    except Exception:
        return None

    if system == "cubic" and centring == "F":
        lattice = "fcc"
    elif system == "cubic" and centring == "I":
        lattice = "bcc"
    elif system == "hexagonal":
        lattice = "hcp"
    else:
        return None      # not a simple metal lattice — CustomSurface handles it

    miller_str = "".join(str(int(m)) for m in millers)
    name = f"{lattice}{miller_str}"
    return name if name in _ACAT_SURFACES else None


def generate_adsorption_configs_acat(
    slab_atoms,
    smiles: str,
    n_rotations: int = 4,
    height: float = 2.0,
    max_sites: int = 20,
    surface: str | None = None,
) -> list:
    """Generate slab+adsorbate configurations via ACAT + rdkit2ase.

    Builds a 3D adsorbate conformer from SMILES (rdkit2ase), then places it
    at ACAT-detected adsorption sites on the slab. Uses ACAT's `CustomSurface`
    so the same code works on any surface topology — FCC/BCC/HCP metals,
    rutile/anatase oxides, perovskites — without requiring per-surface-type
    knowledge. Returns one ASE Atoms per (site, rotation) pair.

    Args:
        slab_atoms: plain ASE Atoms (e.g. fairchem Slab.atoms)
        smiles:     SMILES string for the free adsorbate fragment
        n_rotations: azimuthal orientations per site (default 4)
        height:     initial adsorbate–surface separation in Å (default 2.0)
        max_sites:  maximum sites to sample (default 20)
        surface:    optional ACAT native surface identifier (see
                    acat_surface_name).  When None, falls back to
                    CustomSurface, which is required for oxides/perovskites
                    but fails on corrugated metal surfaces such as fcc(110).

    Returns:
        list of ASE Atoms (slab + adsorbate at each site × rotation)

    Raises:
        ImportError: if acat or rdkit2ase is not installed
        RuntimeError: if ACAT finds no sites or all placements fail
    """
    try:
        from acat.adsorption_sites import SlabAdsorptionSites
        from acat.build.adlayer import add_adsorbate_to_site
        from acat.settings import CustomSurface
    except ImportError as exc:
        raise ImportError("acat is not installed: pip install acat") from exc

    try:
        from rdkit2ase import smiles2atoms
    except ImportError as exc:
        raise ImportError("rdkit2ase is not installed: pip install rdkit2ase") from exc

    # Build 3D adsorbate conformer from SMILES
    ads_atoms = smiles2atoms(smiles)

    # Prefer ACAT's native handler when the surface has one — CustomSurface's
    # plane inference fails on corrugated surfaces (see acat_surface_name).
    # CustomSurface remains the fallback so oxides/perovskites are unaffected.
    surf_ref = surface if surface in _ACAT_SURFACES else CustomSurface(slab_atoms, n_layers=None)
    sas = SlabAdsorptionSites(slab_atoms, surface=surf_ref, allow_6fold=False)
    sites = sas.get_sites()

    if not sites:
        raise RuntimeError("ACAT found 0 adsorption sites on the provided slab.")

    # Subsample sites if too many (uniform step to preserve coverage)
    if len(sites) > max_sites:
        step = max(1, len(sites) // max_sites)
        sites = sites[::step][:max_sites]

    candidates = []
    for site in sites:
        for rot_idx in range(n_rotations):
            angle = rot_idx * (360.0 / n_rotations)
            config = slab_atoms.copy()
            try:
                add_adsorbate_to_site(
                    config,
                    ads_atoms.copy(),
                    site,
                    height=height,
                    orientation=None,
                    tilt_angle=0.0,
                    n_rotation=angle,
                    to_initialize=10,
                )
                candidates.append(config)
            except Exception:
                continue

    if not candidates:
        raise RuntimeError(
            f"All ACAT placements failed for SMILES={smiles!r}."
        )

    return candidates
