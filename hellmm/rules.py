from pydantic.dataclasses import dataclass


@dataclass
class TransformationRule:
    name: str
    type: str           # "PCET", "reverse_PCET", "chemical"
    n_electrons: int    # electrons transferred (0 for chemical, 1 for PCET, -1 for reverse_PCET)
    description: str    # human-readable, used in LLM prompt


ALL_RULES: list[TransformationRule] = [
    TransformationRule(
        name="protonation",
        type="reverse_PCET",
        n_electrons=-1,
        description=(
            "Add H to a surface site or adsorbate via reverse PCET: "
            "H⁺ + e⁻ + *X → *XH (acidic) or H₂O + e⁻ + *X → *XH + OH⁻ (alkaline). "
            "Applies to bare sites, and to any C, O or N atom in the adsorbate."
        ),
    ),
    TransformationRule(
        name="deprotonation",
        type="PCET",
        n_electrons=1,
        description=(
            "Remove H from an adsorbate via PCET: "
            "*XH → *X + H⁺ + e⁻ (acidic) or *XH + OH⁻ → *X + H₂O + e⁻ (alkaline)."
        ),
    ),
    TransformationRule(
        name="hydroxylation",
        type="PCET",
        n_electrons=1,
        description=(
            "Add OH to a surface site or adsorbate via PCET: "
            "*X + H₂O → *XOH + H⁺ + e⁻ (acidic) or *X + OH⁻ → *XOH + e⁻ (alkaline). "
            "Applies to bare sites and to existing adsorbates."
        ),
    ),
    TransformationRule(
        name="dehydroxylation",
        type="chemical",
        n_electrons=0,
        description=(
            "Remove OH from an adsorbate as a chemical step, releasing it to solution: "
            "*XOH → *X + OH⁻ or *XOH + H⁺ → *X + H₂O."
        ),
    ),
    TransformationRule(
        name="bond_dissociation",
        type="chemical",
        n_electrons=0,
        description=(
            "Break any internal single bond within an adsorbate (C-C, C-O, C-H, O-H, etc.), "
            "producing two fragments that may both remain co-adsorbed or one may desorb. "
            "Applies whenever the adsorbate contains internal single bonds. "
            "As a chemical step (n_electrons=0), it transfers no electrons and is "
            "neither oxidative nor reductive — directionally neutral under any reaction. "
            "NOTE: do NOT use this rule when C–O (or N–O) cleavage is coupled with proton "
            "transfer producing H₂O. That is reductive_dehydration "
            "(n_electrons=−1), not bond_dissociation."
        ),
    ),
    TransformationRule(
        name="reductive_dehydration",
        type="reverse_PCET",
        n_electrons=-1,
        description=(
            "PCET C–O (or N–O) bond cleavage producing H₂O: "
            "A*–OH + H⁺ + e⁻ → A* + H₂O. "
            "Applies whenever an adsorbate carries a hydroxyl group whose departure from "
            "the adsorbate skeleton is coupled with proton uptake, releasing water as the "
            "condensed byproduct. "
            "n_electrons=−1 (one electron accepted per H₂O formed)."
        ),
    ),
    TransformationRule(
        name="desorption",
        type="chemical",
        n_electrons=0,
        description=(
            "Detach adsorbate from surface to gas or solution phase, regenerating the bare "
            "surface *. Apply this step whenever the adsorbate can chemically exist as a "
            "free molecule or dissolved species — do NOT apply potential-dependent or "
            "kinetic reasoning to decide whether to include it. Whether desorption is "
            "thermodynamically favoured at a given U is determined by the CHE analysis "
            "downstream; the enumerator only needs to confirm the step is chemically possible."
        ),
    ),
    TransformationRule(
        name="cc_coupling",
        type="chemical",
        n_electrons=0,
        description="Two co-adsorbed carbon-containing species form a C-C bond.",
    ),
    TransformationRule(
        name="oo_coupling",
        type="chemical",
        n_electrons=0,
        description=(
            "Two adjacent surface *O species couple directly to form O₂(g) and two bare "
            "surface sites, without forming *OOH — the Oxide Path Mechanism (OPM). "
            "In the single-site CHE framework this appears as *O → * + ½O₂(g). "
            "Apply only to *O; do NOT apply to *OH, *OOH, or any other adsorbate."
        ),
    ),
    TransformationRule(
        name="lattice_o_release",
        type="chemical",
        n_electrons=0,
        description=(
            "Surface lattice oxygen departs, creating an oxygen vacancy (*V_O). "
            "Treated as a chemical elementary step in the lattice oxygen mechanism (LOM) — "
            "charge balance is maintained by a separate coupled cation oxidation step, not within this rule itself. "
            "The bare surface site `*` on an oxide or perovskite is NOT empty — it is a metal cation "
            "surrounded by lattice oxygen atoms. Applying this rule to `*` always yields `*V_O`."
        ),
    ),
    TransformationRule(
        name="vacancy_healing",
        type="PCET",
        n_electrons=2,
        description=(
            "Oxygen vacancy healed by water via PCET: "
            "*V_O + H₂O → *O_lattice + 2H⁺ + 2e⁻ (acidic) or "
            "*V_O + 2OH⁻ → *O_lattice + H₂O + 2e⁻ (alkaline). "
            "Only applicable if oxygen vacancies (*V_O) are present at the interface."
        ),
    ),
    TransformationRule(
        name="cation_oxidation",
        type="PCET",
        n_electrons=1,
        description="Redox-active surface cation oxidation: M^n⁺ → M^(n+1)⁺ + e⁻.",
    ),
    TransformationRule(
        name="cation_reduction",
        type="reverse_PCET",
        n_electrons=-1,
        description="Redox-active surface cation reduction: M^(n+1)⁺ + e⁻ → M^n⁺.",
    ),
]
