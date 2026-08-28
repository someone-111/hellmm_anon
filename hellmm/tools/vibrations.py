"""Vibrational ZPE and entropy corrections — reusable across projects.

Wraps ASE Vibrations + HarmonicThermo (adsorbed species) and IdealGasThermo
(gas-phase references) to compute zero-point energy and TΔS corrections for
use in the CHE free energy: ΔG = ΔE + ΔZPE − TΔS − n_e·eU.
"""

from __future__ import annotations

import warnings

from pydantic.dataclasses import dataclass


@dataclass
class ThermoCorrection:
    label: str
    zpe: float          # zero-point energy in eV
    ts: float           # T·S entropy term in eV (at temperature T)
    temperature: float  # K


# Gas-phase geometry types for IdealGasThermo: formula -> (geometry, sigma).
#
# Both entries are load-bearing and neither can be guessed:
#
#   geometry  selects how many vibrational modes IdealGasThermo keeps —
#             3N-5 for "linear", 3N-6 for "nonlinear".  Calling a diatomic
#             "nonlinear" leaves 3*2-6 = 0 modes, so its one real stretch is
#             discarded and the ZPE comes out as exactly zero.
#   sigma     is the rotational symmetry number; it enters the entropy as
#             -kT ln(sigma), i.e. 0.018 eV for a homonuclear diatomic and
#             0.028 eV for NH3 at 298 K.  Defaulting it to 1 silently
#             over-counts the entropy of every symmetric molecule.
#
# Neither failure raises, and both produce a plausible-looking number, so an
# unlisted molecule must be rejected rather than defaulted — see below.
_GAS_GEOMETRY = {
    "H2":  ("linear",    2),   # (geometry, symmetry number) — homonuclear, sigma=2
    "H2O": ("nonlinear", 2),
    "CO":  ("linear",    1),
    "N2":  ("linear",    2),
    "CO2": ("linear",    2),
    "O2":  ("linear",    2),
    "NH3": ("nonlinear", 3),
}

# Ceiling for a physically meaningful real vibrational mode, in eV.  Measured
# with this MLIP the stiffest real modes are H2 0.535, H2O 0.470, NH3 0.431 eV,
# so anything past ~0.6 is already unphysical.  The ceiling is deliberately set
# far above that: it exists to catch catastrophic Hessians from broken
# geometries — one adsorbate was seen at ZPE = 45.6 eV, entering a pathway as a
# +44 eV elementary step — not to police borderline frequencies.  At 2.0 eV
# (~16000 cm-1, ~4x the H2 stretch) a real mode can never trip it, while the
# observed failure is still caught 22x over.
#
# Note the 0.6-2.0 eV band passes silently.  That is acceptable while this only
# warns; if a trip is ever made to exclude a species or fail a pathway, revisit
# the value, since false positives stop being free at that point.
_MAX_MODE_EV = 2.0


def compute_thermo_corrections(
    atoms,
    calculator,
    adsorbate_indices: list[int],
    label: str,
    temperature: float = 298.15,
    gas_molecule: str | None = None,
) -> ThermoCorrection:
    """Compute ZPE and entropy for an adsorbed intermediate or gas molecule.

    For adsorbed species: uses ASE Vibrations (finite-difference, displaces
    only adsorbate atoms) + HarmonicThermo.

    For gas molecules: uses IdealGasThermo (includes translational, rotational,
    and vibrational partition functions).

    Args:
        atoms: relaxed ASE Atoms object (slab+adsorbate, or gas molecule)
        calculator: ASE Calculator (same MLIP as module 4)
        adsorbate_indices: indices of adsorbate atoms in the slab (or all
            atoms if gas_molecule is specified)
        label: adsorbate label (used for naming the Vibrations cache)
        temperature: temperature in K (default 298.15 K)
        gas_molecule: if not None, treat atoms as a gas molecule with this
            formula (e.g. "H2O"); uses IdealGasThermo instead of HarmonicThermo

    Returns:
        ThermoCorrection with ZPE and TS at the given temperature
    """
    import tempfile
    from ase.vibrations import Vibrations

    atoms = atoms.copy()
    atoms.calc = calculator
    atoms.pbc = True  # required by fairchem/MACE

    with tempfile.TemporaryDirectory() as tmpdir:
        vib = Vibrations(
            atoms,
            indices=adsorbate_indices,
            name=f"{tmpdir}/vib_{label.replace('*', 'ads_').replace('/', '_')}",
        )
        vib.run()
        vib_energies = vib.get_energies()

    # A broken geometry gives absurd force constants that neither ASE nor the
    # MLIP objects to, and the resulting ZPE flows straight into CHE as a ΔG
    # term.  Warn rather than drop the mode, so the affected species is
    # traceable in the log instead of quietly distorting a free-energy diagram.
    bad_modes = [e.real for e in vib_energies
                 if abs(e.imag) < 1e-9 and e.real > _MAX_MODE_EV]
    if bad_modes:
        warnings.warn(
            f"Unphysical vibrational mode(s) for {label!r}: "
            f"{[round(x, 3) for x in bad_modes]} eV exceed the {_MAX_MODE_EV} eV "
            "ceiling for real H/C/N/O chemistry. This indicates a broken geometry "
            "or unconverged relaxation, not a real frequency — the ZPE and entropy "
            "for this species are not trustworthy."
        )

    if gas_molecule is not None:
        # Gas phase: full partition function.
        # IdealGasThermo requires pbc=False on the atoms object, but MACE
        # requires pbc=True during the vibration run. Pass a non-periodic copy.
        from ase.thermochemistry import IdealGasThermo
        if gas_molecule not in _GAS_GEOMETRY:
            raise ValueError(
                f"No _GAS_GEOMETRY entry for {gas_molecule!r}. Add it as "
                "(geometry, symmetry_number). Both must be right: 'nonlinear' on "
                "a diatomic keeps 3N-6 = 0 modes and zeroes its ZPE, and a wrong "
                "symmetry number skews the entropy by kT*ln(sigma)."
            )
        geometry, symmetry = _GAS_GEOMETRY[gas_molecule]
        e_pot = atoms.get_potential_energy()
        atoms_nopbc = atoms.copy()
        atoms_nopbc.pbc = False
        thermo = IdealGasThermo(
            vib_energies=vib_energies,
            geometry=geometry,
            potentialenergy=e_pot,
            atoms=atoms_nopbc,
            symmetrynumber=symmetry,
            spin=0,
            ignore_imag_modes=True,
        )
        zpe = thermo.get_ZPE_correction()
        ts = thermo.get_entropy(temperature=temperature, pressure=101325.0, verbose=False) * temperature
    else:
        # Adsorbed species: harmonic approximation (frustrated translations/rotations included)
        from ase.thermochemistry import HarmonicThermo
        thermo = HarmonicThermo(vib_energies=vib_energies, ignore_imag_modes=True)
        zpe = thermo.get_ZPE_correction()
        ts = thermo.get_entropy(temperature=temperature, verbose=False) * temperature

    return ThermoCorrection(label=label, zpe=zpe, ts=ts, temperature=temperature)


def compute_gas_thermo_corrections(
    calculator,
    temperature: float = 298.15,
) -> dict[str, ThermoCorrection]:
    """Compute ZPE and entropy for the CHE gas references (H₂, H₂O, CO, N₂).

    Returns a dict keyed by molecule formula.
    """
    from ase.build import molecule

    corrections = {}
    for formula, (geometry, sym) in _GAS_GEOMETRY.items():
        atoms = molecule(formula)
        atoms.cell = [20.0, 20.0, 20.0]
        atoms.center()
        atoms.pbc = True
        indices = list(range(len(atoms)))
        try:
            corr = compute_thermo_corrections(
                atoms, calculator, indices, label=formula,
                temperature=temperature, gas_molecule=formula,
            )
            corrections[formula] = corr
        except Exception as e:
            warnings.warn(f"Vibrational correction failed for {formula}: {e}. Using ZPE=TS=0.")
            corrections[formula] = ThermoCorrection(label=formula, zpe=0.0, ts=0.0, temperature=temperature)
    return corrections
