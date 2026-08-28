"""CHE gas-phase reference energies — reusable across projects.

Computes per-element reference energies for the Computational Hydrogen
Electrode (CHE) convention using a universal MLIP.

CHE references (at U=0, pH=0, SHE):
  H : ½ E(H₂)
  O : E(H₂O) − E(H₂)
  C : E(CO)  − [E(H₂O) − E(H₂)]
  N : ½ E(N₂)

Why molecule-based refs work with MLIPs:
  The model predicts E_ML = E_DFT − Σ Nᵢ·εᵢ  (εᵢ are fitted linear refs).
  For H₂:  ½E_ML(H₂) = ½E_DFT(H₂) − ε_H   ← subtracts exactly ε_H per H
  So  ΔE_ads(*OH) = E_ML(slab+OH) − E_ML(slab) − [E_ML(H₂O) − ½E_ML(H₂)]
                  ≈ E_DFT(slab+OH) − E_DFT(slab) − [E_DFT(H₂O) − ½E_DFT(H₂)]
  The εᵢ corrections cancel, giving the same ΔE_ads the model was trained on.
"""

from __future__ import annotations

from .relax import relax_structure


def compute_gas_references(calculator) -> dict[str, float]:
    """Compute per-element CHE reference energies using gas-phase molecules.

    Relaxes each molecule with BFGS before taking the energy so that MLIP
    energies are evaluated at the MLIP minimum geometry, not at experimental
    bond lengths. ASE molecule() geometries (experimental) can differ from the
    MLIP minimum by 0.05–0.2 eV/molecule, introducing a systematic bias in all
    ΔE_ads values. Relaxation removes this bias with negligible extra cost.

    Args:
        calculator: ASE Calculator (from load_mlip)

    Returns:
        dict mapping element symbol → CHE reference energy per atom (eV):
          "H" → ½ E(H₂)
          "O" → E(H₂O) − E(H₂)
          "C" → E(CO)  − [E(H₂O) − E(H₂)]
          "N" → ½ E(N₂)
    """
    from ase.build import molecule

    def _gas_energy(formula: str) -> float:
        atoms = molecule(formula)
        atoms.cell = [20.0, 20.0, 20.0]
        atoms.center()
        atoms.pbc = True   # UMA requires consistent PBC; 20 Å vacuum is safe
        _, e = relax_structure(atoms, calculator)
        return e

    e = {mol: _gas_energy(mol) for mol in ("H2", "H2O", "CO", "N2")}

    return {
        "H": 0.5 * e["H2"],
        "O": e["H2O"] - e["H2"],
        "C": e["CO"] - (e["H2O"] - e["H2"]),
        "N": 0.5 * e["N2"],
    }
