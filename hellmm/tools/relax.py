"""Structure relaxation — reusable across projects.

Wraps ASE BFGS for MLIP-driven geometry optimization.
"""

from __future__ import annotations


def relax_structure(
    atoms,
    calculator,
    fmax: float = 0.05,
    steps: int = 300,
) -> tuple[object, float]:
    """Relax slab+adsorbate with a universal MLIP using ASE BFGS.

    Sub-surface atoms are already constrained (FixAtoms) by the fairchem
    slab generator, so only surface + adsorbate atoms move.

    Args:
        atoms: ASE Atoms object (slab + adsorbate from generate_candidates)
        calculator: ASE Calculator (e.g. FAIRChemCalculator from load_mlip)
        fmax: force convergence threshold in eV/Å (default 0.05)
        steps: maximum BFGS steps (default 300)

    Returns:
        (relaxed ASE Atoms, final potential energy in eV)
    """
    from ase.optimize import BFGS

    atoms = atoms.copy()
    # UMA/fairchem models require all-periodic boundary conditions.
    # For slabs, the vacuum gap in z is large enough that this is safe.
    atoms.pbc = True
    atoms.calc = calculator
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    e = atoms.get_potential_energy()
    return atoms, e
