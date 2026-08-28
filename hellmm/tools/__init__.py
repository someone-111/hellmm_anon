"""hellmm.tools — reusable MLIP, structure, and thermo utilities.

These modules are designed to be portable across projects. They have no
dependency on hellmm-specific domain logic (rules, enumerator, CHE).
"""

from .mlip import load_mlip, _best_device
from .relax import relax_structure
from .gas_references import compute_gas_references
from .vibrations import ThermoCorrection, compute_thermo_corrections, compute_gas_thermo_corrections
from .structure import (
    build_slab,
    check_adsorbate_geometry,
    adsorbate_is_intact,
    check_computability,
    compute_vacancy_configs,
    generate_adsorption_configs,
    acat_surface_name,
    generate_adsorption_configs_acat,
    generate_adsorption_configs_acat_bidentate,
    label_to_fairchem_ids,
    label_to_smiles,
    label_to_bidentate_smiles,
)

__all__ = [
    # MLIP
    "load_mlip",
    "_best_device",
    # Relaxation
    "relax_structure",
    # Gas references
    "compute_gas_references",
    # Vibrations
    "ThermoCorrection",
    "compute_thermo_corrections",
    "compute_gas_thermo_corrections",
    # Structure generation — monodentate
    "check_adsorbate_geometry",
    "adsorbate_is_intact",
    "check_computability",
    "compute_vacancy_configs",
    "build_slab",
    "generate_adsorption_configs",
    "acat_surface_name",
    "generate_adsorption_configs_acat",
    "label_to_fairchem_ids",
    "label_to_smiles",
    # Structure generation — bidentate
    "generate_adsorption_configs_acat_bidentate",
    "label_to_bidentate_smiles",
]
