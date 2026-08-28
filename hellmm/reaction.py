"""Reaction definitions — shared boundary conditions for all pipeline modules.

A ReactionDefinition encodes only what is fixed by physics and the experimental
setup, NOT the mechanism (which the enumerator discovers):

  - target_labels          : surface labels that terminate a pathway.  For
                             cyclic reactions (OER, ORR, HER) this is the clean
                             surface ["*"].  For linear reactions (CO2RR, NRR)
                             it is also ["*"] — the cycle closes after the
                             product desorbs via the product_surface_labels
                             injection (see below).
  - product_surface_labels : surface adsorbate labels that can desorb as the
                             desired product, closing the catalytic cycle.  The
                             pathway constructor injects a chemical desorption
                             edge (n_e=0, rule="product_desorption") from each
                             of these labels to "*" so that the DFS can find
                             linear-reaction pathways.  This is a reaction
                             boundary condition — it encodes *which product* the
                             reaction targets — not a mechanistic assumption.
                             The LLM's intermediate/edge discovery is untouched.
                             Leave empty for cyclic reactions.
  - n_electrons_total      : total electrons transferred per catalytic cycle
  - U_ideal                : thermodynamic equilibrium potential (V vs RHE)
  - min_steps              : minimum pathway length (filters trivial 1-step paths)

REACTION_TEMPLATES provides pre-built instances for the five supported reactions.
Callers look up by the reaction string from CatalystContext:

    rxn = REACTION_TEMPLATES[context.reaction]

If you add a new reaction, add an entry here — no other file needs to change.
"""

from __future__ import annotations

from dataclasses import field as _field

from pydantic.dataclasses import dataclass


@dataclass
class ReactionDefinition:
    name: str
    target_labels: list[str]    # surface labels that terminate a pathway
    n_electrons_total: int      # electrons per full catalytic cycle
    U_ideal: float              # V vs RHE — thermodynamic equilibrium potential
    min_steps: int              # minimum steps for a valid pathway
    # Surface labels whose desorption closes the catalytic cycle for linear
    # reactions.  The pathway constructor injects "*label → *" (n_e=0) for
    # each entry present in the pruned kept set.
    product_surface_labels: list[str] = _field(default_factory=list)
    # Whether the catalytic cycle closes back to the starting surface via a
    # universal terminal PCET step (True: OER, ORR, HER) rather than via a
    # specific named product-desorption edge (False: CO2RR→*CO→*, NRR→*NH3→*).
    # When True, pathway_constructor injects node→start for every kept node so
    # the DFS can close regardless of where the path ends up; che.py applies the
    # closure correction (∑ΔG°=n_e_total·U_ideal) on the terminal step.
    # When False, closure is handled by product_surface_labels injection — no
    # universal injection, no closure correction needed.
    cyclic: bool = False
    # Whether to filter pathways where all surface heavy-atom (tracked_element)
    # leaves at a non-terminal interior step.  True for reactions where the
    # target product must carry the heteroatom through to the desorption step
    # (CO2RR→CO, NRR→NH3).  False for reactions without heavy-atom tracking
    # (OER, ORR, HER).  This is a reaction boundary condition — it encodes
    # what the desired product is — not a mechanistic constraint on the LLM.
    filter_carbon_loss: bool = False
    # Element symbol to track when filter_carbon_loss=True.
    # "C" for CO2RR (carbon must reach the *CO desorption step).
    # "N" for NRR (nitrogen must reach the *NH3 desorption step).
    # Ignored when filter_carbon_loss=False.
    tracked_element: str = "C"
    # Hard lower bound on enumeration depth, flooring the LLM's suggestion.
    # Encodes the stoichiometric minimum: for NRR the shortest path from *N2
    # to *NH3 is *N2→*N→*NH→*NH2→*NH3 (4 steps), so depth<4 is mechanistically
    # impossible regardless of what the meta_reasoner suggests.
    min_depth: int = 1
    # Whether the reaction is a net reduction (electrons consumed, n_e<0 per
    # elementary step) rather than an oxidation.  True for HER/ORR/CO2RR/NRR,
    # False for OER.
    #
    # This is a reaction boundary condition in exactly the same category as
    # U_ideal and n_electrons_total — "ORR is a reduction" is the definition
    # of the reaction, not an assumption about its mechanism.  The LLM's
    # intermediate/edge discovery is untouched.
    #
    # Two things depend on it:
    #   1. Sign convention.  ΔG(U) = ΔG° − n_e·U, so a step runs downhill for
    #      U ≥ ΔG° when anodic but for U ≤ −ΔG° when cathodic.  U_onset and the
    #      overpotential therefore invert (see che.py).
    #   2. Directional validity.  A pathway whose electrochemical steps all run
    #      counter to the reaction direction is not a poor pathway for this
    #      reaction — it is a different reaction.  An all-oxidative ladder
    #      satisfying "ORR" by electron count is OER, and che.py rejects it.
    cathodic: bool = False


REACTION_TEMPLATES: dict[str, ReactionDefinition] = {
    "OER": ReactionDefinition(
        name="OER",
        target_labels=["*"],        # cyclic: returns to clean surface
        n_electrons_total=4,
        U_ideal=1.23,
        min_steps=3,                # at minimum * → *OH → *O → *OOH → *
        cyclic=True,
    ),
    "ORR": ReactionDefinition(
        name="ORR",
        target_labels=["*"],
        n_electrons_total=4,
        U_ideal=1.23,
        min_steps=3,
        # cyclic=False, unlike OER — ORR is run from the pre-adsorbed reactant
        # (start="*OO", see config.py), and `cyclic` means "close back to
        # start".  *OO is a consumed reactant, not the catalyst resting state,
        # so closing to it produced the bogus terminal step *OH → *OO
        # (rule="unknown", inferred, +3.02 eV) which then became the limiting
        # step and gave U_onset = -3.02 V.  Closure instead runs through
        # target_labels=["*"] via the *H2O product-desorption edge, exactly as
        # for CO2RR (*CO) and NRR (*NH3).
        cyclic=False,
        # 2 H2O per O2: the first leaves at *OOH → *O (reductive_dehydration),
        # the second via *OH → *H2O → * — 4 PCET steps total.
        product_surface_labels=["*H2O"],
        # O2 + 4H⁺ + 4e⁻ → 2H2O — a reduction.  Without this the all-oxidative
        # ladder * → *OH → *O → *OOH → *OO → * satisfies ORR on electron count
        # alone and is scored as if it were ORR; it is OER.
        cathodic=True,
        # NOTE: ORR must be run with start="*OO" (see config.py).  The rule set
        # has `desorption` but no inverse adsorption rule, so a gas-phase
        # reactant cannot enter the graph on its own — the same reason CO2RR
        # uses start="*CO2" and NRR uses start="*N2".  With start="*" the graph
        # has no route to *OO at all and the search wanders into H adsorption.
    ),
    "HER": ReactionDefinition(
        name="HER",
        target_labels=["*", "H2(g)"],
        n_electrons_total=2,
        U_ideal=0.00,
        min_steps=1,
        cyclic=True,
        # 2H⁺ + 2e⁻ → H2 — a reduction.
        cathodic=True,
        # *H2 desorbs as H2(g) — the reaction's product by definition.
        # Declared for the same reason as CO2RR's *CO and NRR's *NH3: the
        # pruner scores *H2 as implausible (it desorbs essentially instantly
        # on Pt), which is defensible chemistry but removes the only route
        # that closes the cycle with the correct electron count, leaving just
        # * → *H → * (Volmer immediately reversed, mixed-polarity, rejected).
        # This states what the product is, not which pathway reaches it.
        product_surface_labels=["*H2"],
    ),
    "CO2RR": ReactionDefinition(
        name="CO2RR",
        target_labels=["*"],
        n_electrons_total=2,
        # CO2 + 2H⁺ + 2e⁻ → CO + H₂O  ΔG° = +0.21 eV → U_eq = −0.104 V vs RHE.
        # U_ideal is the signed equilibrium potential on the RHE scale, not its
        # magnitude.  It was previously +0.10, justified by U_onset "always being
        # positive" — which stopped holding once che.py gained the cathodic-sign
        # fix and began reporting U_onset ≈ −0.17 V here.  With η = U_ideal −
        # U_onset for a cathodic reaction, the magnitude convention added 2·|U_eq|
        # ≈ 0.21 eV to every CO2RR overpotential.
        U_ideal=-0.104,
        min_steps=2,
        # *CO can release as CO(g), closing the 2e⁻ cycle.  The pathway
        # constructor injects *CO → * (n_e=0, product_desorption) so the DFS
        # finds the physically correct pathway *CO2 → *COOH → *CO → *.
        product_surface_labels=["*CO"],
        # CO must be carried through to the *CO desorption step.  Pathways
        # where C leaves at a non-terminal interior step (e.g. CO2 dissociation
        # to *O, releasing CO(g) chemically) are not electrocatalytic CO2RR.
        filter_carbon_loss=True,
        # CO2 + 2H⁺ + 2e⁻ → CO + H2O — a reduction (the U_eq comment above
        # already notes the cathodic sign; this makes it usable by che.py).
        cathodic=True,
    ),
    "NRR": ReactionDefinition(
        name="NRR",
        target_labels=["*"],
        n_electrons_total=6,
        U_ideal=0.00,
        min_steps=3,
        # *NH3 desorbs as NH3(g), closing the 6e⁻ cycle.
        # The pathway constructor injects *NH3 → * (n_e=0, product_desorption).
        product_surface_labels=["*NH3"],
        # Nitrogen must be carried through to the *NH3 desorption step.
        # Pathways where N leaves at a non-terminal interior step are not
        # valid NRR pathways.  tracked_element="N" tells the pathway
        # constructor which element to watch.
        filter_carbon_loss=True,
        tracked_element="N",
        # Shortest path *N2→*N→*NH→*NH2→*NH3 is 4 steps; depth<4 cannot
        # reach the product intermediate regardless of rule selection.
        min_depth=4,
        # N2 + 6H⁺ + 6e⁻ → 2NH3 — a reduction.
        cathodic=True,
    ),
}


def get_reaction(reaction: str) -> ReactionDefinition:
    """Look up a ReactionDefinition by name. Raises KeyError with a helpful message."""
    if reaction not in REACTION_TEMPLATES:
        supported = list(REACTION_TEMPLATES.keys())
        raise KeyError(
            f"Unknown reaction '{reaction}'. Supported: {supported}. "
            "Add a new entry to REACTION_TEMPLATES in reaction.py to extend."
        )
    return REACTION_TEMPLATES[reaction]
