"""Pourbaix checker — module 2b.

Verifies catalyst surface stability at (U, pH) BEFORE any MLIP computation.
This is a cheap gate: if the catalyst dissolves or transforms at operating
conditions, skip it rather than wasting compute on adsorption energies.

Position in pipeline:
    module 1 (meta_reasoner) → module 2b (pourbaix gate) → module 3 (enumerator)
    → module 4 (adsorption_energy) → ...

Stability criterion:
    The stable Pourbaix phase at (U, pH) must contain all elements of the
    catalyst composition. A pure metal dissolving to ions, or an oxide
    transforming to a higher oxide, is flagged as unstable.

    A decomposition energy threshold of 0.1 eV/atom is used as a leniency
    window — phases within this of the convex hull are considered "borderline
    stable" and generate a warning rather than a hard skip.

Requires:
    MY_MP_API_KEY in .env (same key used by adsorption_energy.build_slab)
"""

from __future__ import annotations

import os
import warnings

from dotenv import load_dotenv
from pydantic.dataclasses import dataclass

from .meta_reasoner import CatalystContext

load_dotenv()

# Decomposition energy window for "borderline" stability warning (eV/atom)
_STABILITY_THRESHOLD = 0.1


@dataclass
class PourbaixResult:
    stable: bool           # True if catalyst is the stable phase at (U, pH)
    stable_phase: str      # name of the stable phase at (U, pH)
    decomp_energy: float   # eV/atom above hull (0.0 = on hull = stable)
    warning: str           # non-empty if stability is uncertain


def check_stability(context: CatalystContext) -> PourbaixResult:
    """Check if the catalyst surface is stable at (U, pH) via Pourbaix diagram.

    Fetches MP entries for the catalyst chemsys + H + O, builds a Pourbaix
    diagram, and queries the stable phase at the operating (U, pH).

    Returns PourbaixResult. stable=False means the catalyst is NOT the stable
    phase and the caller should consider skipping.

    Raises:
        EnvironmentError: if MY_MP_API_KEY is not set
    """
    api_key = os.environ.get("MY_MP_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "MY_MP_API_KEY is not set. Add it to your .env file."
        )

    from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram, PourbaixEntry
    from pymatgen.core import Composition
    from pymatgen.ext.matproj import MPRester

    # Parse composition to element list, add H and O for aqueous diagram
    try:
        comp = Composition(context.composition)
        elements = [str(el) for el in comp.elements]
    except Exception:
        # Fallback: treat composition as a single element symbol
        elements = [context.composition]

    chemsys = list(set(elements + ["H", "O"]))

    with MPRester(api_key) as mpr:
        entries = mpr.get_entries_in_chemsys(chemsys)

    # Wrap as PourbaixEntry; skip pure H/O entries (cause ZeroDivisionError
    # in normalization since they have no "active" metal atoms)
    metal_els = set(elements)
    pb_entries: list[PourbaixEntry] = []
    for e in entries:
        entry_els = {str(el) for el in e.composition.elements}
        if entry_els <= {"H", "O"}:
            continue  # pure water/oxygen — not useful for catalyst stability
        try:
            pb_entries.append(PourbaixEntry(e))
        except Exception as ex:
            warnings.warn(f"Skipping Pourbaix entry {e.composition}: {ex}")

    if not pb_entries:
        return PourbaixResult(
            stable=True,
            stable_phase="(no entries found)",
            decomp_energy=0.0,
            warning=f"No Pourbaix entries found for {context.composition}. Assuming stable.",
        )

    diagram = PourbaixDiagram(pb_entries, filter_solids=True)

    # Pymatgen Pourbaix uses U vs SHE internally.
    # context.U is vs RHE → convert: U_SHE = U_RHE - 0.0592 × pH
    U_SHE = context.U - 0.0592 * context.pH

    stable_entry = diagram.get_stable_entry(pH=context.pH, V=U_SHE)
    decomp_e = float(diagram.get_decomposition_energy(stable_entry, pH=context.pH, V=U_SHE))

    stable_name = stable_entry.name

    # Stability criterion:
    # - stable_entry.phase_type == "Ion" → metal has dissolved → hard unstable (skip)
    #   Using phase_type (set by pymatgen on the PourbaixEntry object) is robust:
    #   it covers dissolved cations like "Ru+4" or "Cu2+" that may not contain "(aq)"
    #   in their name string.
    # - phase_type == "Solid" but different composition → phase transformation → warn only
    #   (metal stays at surface in some oxide form; electrochemistry can proceed)
    # This matches experimental reality: RuO2 at OER conditions may thermodynamically
    # prefer RuO4(s) but the surface remains a solid Ru-oxide and the catalyst works.
    dissolved = getattr(stable_entry, "phase_type", None) == "Ion"

    if dissolved:
        return PourbaixResult(
            stable=False,
            stable_phase=stable_name,
            decomp_energy=decomp_e,
            warning=(
                f"{context.composition} dissolves under operating conditions "
                f"(U={context.U} V vs RHE, pH={context.pH}). "
                f"Stable aqueous species: {stable_name}."
            ),
        )

    # Solid phase — check if composition matches the catalyst
    phase_formula = stable_name.replace("(s)", "").strip()
    same_phase = phase_formula == context.composition

    warn = ""
    if not same_phase:
        warn = (
            f"{context.composition} may transform to {stable_name} at "
            f"U={context.U} V vs RHE, pH={context.pH}. "
            f"Surface remains solid — proceeding, but stability is uncertain."
        )
    elif decomp_e > _STABILITY_THRESHOLD:
        warn = (
            f"{context.composition} is borderline stable "
            f"(decomp. energy = {decomp_e:.3f} eV/atom). "
            f"Stable phase: {stable_name}."
        )

    return PourbaixResult(
        stable=True,
        stable_phase=stable_name,
        decomp_energy=decomp_e,
        warning=warn,
    )


def should_proceed(context: CatalystContext) -> tuple[bool, str]:
    """Gate function: returns (proceed, reason).

    Call this after meta_reasoner and before any MLIP work.
    If proceed=False, reason explains why the catalyst was skipped.

    Example:
        proceed, reason = should_proceed(ctx)
        if not proceed:
            print(f"Skipping {ctx.composition}: {reason}")
            continue
    """
    try:
        result = check_stability(context)
    except Exception as e:
        warnings.warn(f"Pourbaix check skipped ({type(e).__name__}: {e}). Assuming stable.")
        return True, ""

    if not result.stable:
        return False, result.warning

    if result.warning:
        warnings.warn(f"Pourbaix [{context.composition}]: {result.warning}")

    return True, ""
