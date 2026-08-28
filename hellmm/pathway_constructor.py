"""Pathway constructor — module 5.

Finds all elementary-step pathways from the starting adsorbate to any target
product label, traversing the pruned reaction graph produced by the enumerator
and pruner.

No LLM calls — pure graph traversal.

Design principles
-----------------
* The graph is built exclusively from ``EnumeratorOutput.edges`` — every edge
  has a backing ``IntermediateEdge`` with its authoritative per-step electron
  count (from the rule definition).  There is no separate graph field that can
  diverge from the edge list.

* Cyclic terminal step (OER, ORR, HER — ``reaction.cyclic=True``): the starting
  surface is never enumerated as a product, so the DFS cannot reach it without
  injection.  The constructor injects ``start`` as a potential neighbor of every
  kept node and infers the terminal electron count as ``n_e_total − Σ(prior)``.
  The guard 0 ≤ de ≤ 1 rejects non-elementary terminal steps.  Uses ``start``
  (not a hardcoded ``"*"``) so vacancy-initiated runs (``start="*V_O"``) close
  back to the real catalyst resting state, not the idealized bare surface.

* Linear product-desorption step (CO2RR, NRR): the enumerator discovers
  intermediates and transformation rules but cannot know which surface species
  are the desired *products* — that is a reaction boundary condition.
  ``ReactionDefinition.product_surface_labels`` names the surface adsorbates
  (e.g. ``*CO`` for CO₂→CO) that can desorb chemically to close the cycle.
  The constructor injects ``label → *`` (n_e=0, rule="product_desorption") for
  each such label that is present in the pruned kept set.  This edge is
  registered in both ``edge_electrons`` and ``edge_rule`` so no terminal
  inference is needed — the step is fully specified.

* Pathways are filtered to those whose total electron count equals
  ``reaction.n_electrons_total`` exactly.
"""

from pydantic.dataclasses import dataclass

from .chemistry import matches_species
from .enumerator import EnumeratorOutput
from .io import (
    catalyst_context_from_dict,
    find_entry,
    reconstruct_enumerator_output,
    reconstruct_pruner_output,
)
from .pruner import PrunerOutput
from .reaction import ReactionDefinition, get_reaction


@dataclass
class ReactionStep:
    parent: str       # adsorbate before step
    product: str      # adsorbate after step
    rule: str         # transformation rule name
    n_electrons: int  # electrons transferred in this step


@dataclass
class Pathway:
    steps: list[ReactionStep]
    intermediates: list[str]  # ordered labels from start to product
    n_electrons_total: int    # cumulative electrons transferred


def construct_pathways(
    enumerator_output: EnumeratorOutput,
    pruner_output: PrunerOutput,
    reaction: ReactionDefinition,
    max_steps: int = 10,
    filter_carbon_loss: bool = False,
) -> list[Pathway]:
    """Find all valid pathways from the starting adsorbate to any target label.

    Only edges backed by an ``IntermediateEdge`` (with a known rule and electron
    count) are traversed.  The graph is derived from those edges filtered to the
    pruned node set, so it is consistent by construction.

    Terminal step handling
    ~~~~~~~~~~~~~~~~~~~~~~
    When the path reaches a target node via an edge that is not in the edge
    list (i.e. the clean-surface node ``*`` which is the start, not a product),
    the electron count is inferred as ``n_e_total − Σ(prior)``.  The inferred
    value must satisfy ``0 ≤ de ≤ 1``; otherwise the pathway is discarded as
    non-elementary.

    Args:
        enumerator_output: full enumerator output with ``edges`` metadata
        pruner_output: which intermediates survived pruning
        reaction: ReactionDefinition with target_labels, n_electrons_total, min_steps
        max_steps: maximum pathway length (prevents runaway DFS)
        filter_carbon_loss: if True, remove pathways where the tracked element
            (reaction.tracked_element) leaves at a non-terminal interior step.
            Overridden by reaction.filter_carbon_loss when called from eval scripts.

    Returns:
        List of Pathway objects sorted by number of steps, filtered to exact
        electron count match.
    """
    # Build edge-metadata lookup from the authoritative IntermediateEdge list.
    # A (parent, label) pair can have more than one entry if the enumerator's
    # rules proposed the same product via different rules (e.g. both
    # "desorption" and "deprotonation" producing *H → *). Both are kept as
    # distinct options rather than one silently overwriting the other, so
    # that CHE — not enumeration order — decides which interpretation is
    # physically valid.
    edge_options: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for edge in enumerator_output.edges:
        key = (edge.parent, edge.label)
        edge_options.setdefault(key, []).append((edge.rule, edge.n_electrons))

    # Pruned node set — only edges where both endpoints survived are traversed.
    kept = {s.intermediate.label for s in pruner_output.kept}
    start = enumerator_output.starting_adsorbate
    kept.add(start)

    # Build graph from edges (consistent: every adjacency has backing metadata).
    graph: dict[str, list[str]] = {}
    for edge in enumerator_output.edges:
        if edge.parent in kept and edge.label in kept:
            neighbors = graph.setdefault(edge.parent, [])
            if edge.label not in neighbors:
                neighbors.append(edge.label)

    # For cyclic reactions the closure target IS start (whatever the run began at —
    # "*" for standard OER, "*V_O" for vacancy-initiated LOM).  reaction.target_labels
    # always contains "*", but "*V_O" ∉ {"*"}, so without this override the DFS
    # would inject node→start edges and then never accept start as a valid endpoint.
    # For linear reactions (CO2RR, NRR) target_labels stays as-is; closure is via
    # product_desorption edges to "*".
    target_set = {start} if reaction.cyclic else set(reaction.target_labels)

    # Cyclic reactions (OER, ORR, HER): inject the start node as a potential
    # terminal neighbor of every kept node.  The enumerator never emits the
    # starting surface as a product, so without injection the DFS cannot close
    # the cycle.  The terminal electron count is inferred from stoichiometry
    # inside the DFS; the guard 0 ≤ de ≤ 1 rejects non-elementary closures.
    # Gated on reaction.cyclic (not "start in target_set") so CO2RR/NRR are
    # never affected — they close via product_surface_labels instead.
    # Injects node → start (not node → "*") so vacancy-initiated runs
    # (start="*V_O") close back to the real catalyst resting state, not the
    # idealized bare surface "*" which may not be physically accessible.
    clean_surface = "*"
    if reaction.cyclic:
        for node in kept:
            if node != start:
                neighbors = graph.setdefault(node, [])
                if start not in neighbors:
                    neighbors.append(start)

    # Linear reactions (CO2RR, NRR): inject product desorption edges.
    # The enumerator discovers intermediates and transformation rules but cannot
    # determine which surface species are the intended *products* — that is
    # encoded in ReactionDefinition.product_surface_labels as a reaction
    # boundary condition (analogous to U_ideal or n_electrons_total).
    # For each product label that survived pruning, inject label → * with
    # n_e=0 (chemical desorption, no electron transfer) and
    # rule="product_desorption".  Registering in edge_options means the DFS
    # uses the explicit metadata; no terminal inference is needed.
    for prod_label in reaction.product_surface_labels:
        if clean_surface not in target_set:
            continue
        # Match by species, not exact spelling — the LLM's spelling of the
        # product varies between runs ("*H2O" vs "*OH2"), and an exact-string
        # test silently injected nothing, leaving the reaction with no route to
        # the clean surface and therefore zero pathways.
        match = next((k for k in kept if matches_species(k, prod_label)), None)
        if match is None:
            continue
        neighbors = graph.setdefault(match, [])
        if clean_surface not in neighbors:
            neighbors.append(clean_surface)
        key = (match, clean_surface)
        if key not in edge_options:          # don't overwrite if enumerator already has it
            edge_options[key] = [("product_desorption", 0)]

    n_e_total = reaction.n_electrons_total
    min_steps = reaction.min_steps
    pathways: list[Pathway] = []

    def step_variants(path: list[str]) -> list[list[ReactionStep]]:
        """Enumerate every combination of rule/electron-count interpretation
        for a fixed sequence of intermediate labels, branching wherever a step
        has more than one discovered (rule, n_electrons) option (see
        edge_options above). Each combination becomes a separate candidate
        Pathway, scored independently downstream by CHE — this function makes
        no judgment about which interpretation is physically correct.

        ReactionStep.n_electrons keeps the SIGNED value from the rule so that
        the CHE formula ΔG = ΔE − n_e·eU is directionally correct (n_e = −1
        for reductive/cathodic steps, +1 for oxidative/anodic).
        """
        results: list[list[ReactionStep]] = []

        def recurse(i: int, steps_so_far: list[ReactionStep], e_running_abs: int) -> None:
            if i == len(path) - 1:
                results.append(steps_so_far)
                return
            p, c = path[i], path[i + 1]
            is_last = i == len(path) - 2

            if (p, c) in edge_options:
                options = edge_options[(p, c)]
            elif is_last and c in target_set:
                # Terminal edge: the clean surface (*) was never enumerated as
                # a product, so it has no backing IntermediateEdge. Infer |de|
                # as the unsigned residual needed to reach n_e_total. Valid
                # values: 0 (chemical desorption) or 1 (elementary PCET); any
                # other value means this path is non-elementary — dies here.
                de_abs = n_e_total - e_running_abs
                if not (0 <= de_abs <= 1):
                    return
                if de_abs == 0:
                    options = [("unknown", 0)]
                else:
                    # Sign matches the direction of preceding electrochemical
                    # steps (majority vote): OER/ORR (n_e>0) -> +1, HER/CO2RR
                    # (n_e<0) -> -1.
                    prior_signs = [s.n_electrons for s in steps_so_far if s.n_electrons != 0]
                    de_signed = (-1 if prior_signs and sum(prior_signs) < 0 else 1) * de_abs
                    options = [("unknown", de_signed)]
            else:
                # Interior edge with missing metadata — treat as 0 e⁻ and let
                # the electron-count filter handle it.
                options = [("unknown", 0)]

            for rule, de_signed in options:
                step = ReactionStep(parent=p, product=c, rule=rule, n_electrons=de_signed)
                recurse(i + 1, steps_so_far + [step], e_running_abs + abs(de_signed))

        recurse(0, [], 0)
        return results

    def dfs(node: str, path: list[str], visited: set[str], abs_e_cum: int) -> None:
        if len(path) > max_steps + 1:
            return
        n_steps = len(path) - 1

        if node in target_set and n_steps >= min_steps:
            for steps in step_variants(path):
                pathways.append(Pathway(
                    steps=steps,
                    intermediates=list(path),
                    n_electrons_total=sum(abs(s.n_electrons) for s in steps),
                ))
            return

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                opts = edge_options.get((node, neighbor), [])
                abs_de = abs(opts[0][1]) if opts else 0
                dfs(neighbor, path + [neighbor], visited, abs_e_cum + abs_de)
                visited.remove(neighbor)

    # Cyclic reactions (OER/HER): the start node is also the terminal target —
    # leave it out of visited so the DFS can return to it.  Linear reactions
    # (CO2RR/NRR): start is never a valid mid-path node; add it to visited to
    # block revisiting and prevent spurious loops like *CO2 → *COOH → *CO2 → ....
    initial_visited: set[str] = set() if reaction.cyclic else {start}
    dfs(start, [start], initial_visited, 0)

    # Keep only pathways whose total electron count matches the reaction exactly.
    valid = [pw for pw in pathways if pw.n_electrons_total == n_e_total]

    # Optional: warn about carbon-loss pathways (where a non-terminal step loses
    # all C from the surface) but keep them — set filter_carbon_loss=True to
    # remove them.  Printed as warnings so the discovery is not silently hidden.
    from .chemistry import parse_formula as _parse_formula

    tracked = reaction.tracked_element  # "C" for CO2RR, "N" for NRR

    def _heteroatom_loss_step(pw: Pathway) -> str | None:
        """Return the offending step label if the pathway loses all surface
        tracked_element at a non-terminal interior step, else None."""
        for step in pw.steps[:-1]:
            if step.product in target_set:
                continue
            src_n = _parse_formula(step.parent).get(tracked, 0)
            dst_n = _parse_formula(step.product).get(tracked, 0)
            if src_n > 0 and dst_n == 0:
                return f"{step.parent} → {step.product}"
        return None

    import warnings as _warnings
    filtered: list[Pathway] = []
    for pw in valid:
        offender = _heteroatom_loss_step(pw)
        if offender:
            _warnings.warn(
                f"Pathway {'→'.join(pw.intermediates)!r} loses all surface "
                f"{tracked} at non-terminal step {offender!r}. "
                f"Pass filter_carbon_loss=True to construct_pathways to remove these.",
                UserWarning,
                stacklevel=2,
            )
            if filter_carbon_loss:
                continue
        filtered.append(pw)
    valid = filtered

    valid.sort(key=lambda p: len(p.steps))
    return valid


def find_trapped_intermediates(
    enumerator_output: EnumeratorOutput,
    pruner_output: PrunerOutput,
    reaction: ReactionDefinition,
) -> list[str]:
    """Find pruner-kept nodes with no outgoing edges in the real reaction graph.

    These are intermediates the catalyst can reach but never escape via any
    enumerator-discovered rule within the kept set — candidates for off-cycle
    trapping or surface poisoning.

    Uses only edges from the enumerator (no pathway_constructor injections such
    as cyclic closure or product_desorption). An intermediate is "trapped" if
    every enumerator-discovered next step leads to a node the pruner discarded,
    or if no next step was found at all.

    Scope: surface adsorbate dead-ends only. Does not model dissolution,
    amorphization, or competing reactions — those are the Pourbaix gate's job.

    Args:
        enumerator_output: full enumerator output
        pruner_output: which intermediates survived pruning
        reaction: ReactionDefinition with target_labels and product_surface_labels

    Returns:
        Sorted list of trapped intermediate labels.
    """
    kept = {s.intermediate.label for s in pruner_output.kept}
    start = enumerator_output.starting_adsorbate
    kept.add(start)

    # Collect nodes that have at least one outgoing edge to another kept node.
    # Done with a set rather than a full graph dict — O(E) and allocates less.
    has_outgoing: set[str] = set()
    for edge in enumerator_output.edges:
        if edge.parent in kept and edge.label in kept:
            has_outgoing.add(edge.parent)

    # Valid terminal states — reaching these is the goal, not a trap.
    not_trapped = {start, "*"}
    not_trapped.update(reaction.target_labels)
    not_trapped.update(reaction.product_surface_labels)

    return sorted(node for node in kept
                  if node not in not_trapped and node not in has_outgoing)


# ---------------------------------------------------------------------------
# Full pathway-construction stage — all catalyst cases
# ---------------------------------------------------------------------------

def run_pathway_constructor(
    pruner_data: dict,
    enum_data: dict,
    max_denticity: int,
    max_steps: int = 10,
) -> list[dict]:
    """Run pathway construction for every catalyst case in pruner_data.

    Pure graph traversal, no MLIP or LLM calls — catches 0-pathway cases
    before any GPU time is spent. Prints per-catalyst pathway/trapped-node
    diagnostics (matches the original eval_pathway_constructor.py output).

    Args:
        pruner_data: parsed eval_pruner_*.json
        enum_data: parsed eval_enumerator_*.json
        max_denticity: maximum '*' anchors allowed (passed through to pruner reconstruction)
        max_steps: maximum pathway length passed to construct_pathways()

    Returns:
        List of JSON-serializable per-catalyst records: context,
        starting_adsorbate, n_pathways, pathway_labels (union of labels
        across valid pathways — used by eval_adsorption_energy.py to restrict
        MLIP relaxations), trapped_intermediates, and full pathway/step detail.
    """
    records: list[dict] = []

    for entry in pruner_data["results"]:
        ctx_dict = entry["context"]
        ctx      = catalyst_context_from_dict(ctx_dict)

        enum_entry = find_entry(enum_data, ctx.composition, ctx.facet, ctx.reaction)
        if enum_entry is None:
            print(f"\n  {ctx.composition}({ctx.facet}) {ctx.reaction}: no enumerator entry — skipping")
            continue

        enum_output   = reconstruct_enumerator_output(enum_entry, ctx)
        pruner_output = reconstruct_pruner_output(
            entry, enum_entry, enum_output.intermediates, max_denticity=max_denticity,
        )
        rxn = get_reaction(ctx.reaction)

        print(f"\n{'='*60}")
        print(f"{ctx.composition}({ctx.facet})  {ctx.reaction}  "
              f"start={entry['starting_adsorbate']}")
        print(f"  pruner kept: {len(pruner_output.kept)} intermediates")
        print(f"{'='*60}")

        pathways = construct_pathways(
            enum_output, pruner_output, reaction=rxn,
            max_steps=max_steps,
            filter_carbon_loss=rxn.filter_carbon_loss,
        )

        print(f"  Found {len(pathways)} pathway(s)")
        for pw in pathways:
            print(f"    {'→'.join(pw.intermediates)}  ({pw.n_electrons_total} e⁻)")
            for s in pw.steps:
                print(f"       {s.parent:20s} → {s.product:20s}  "
                      f"rule={s.rule:25s}  n_e={s.n_electrons:+d}")

        if not pathways:
            print(f"  WARNING: 0 pathways for {ctx.composition}({ctx.facet}) — "
                  "no MLIP relaxations will be run for this case.")
            # Diagnostic: try without carbon-loss filter to give a hint
            unfiltered = construct_pathways(
                enum_output, pruner_output, reaction=rxn,
                max_steps=max_steps, filter_carbon_loss=False,
            )
            if unfiltered:
                print(f"  (Without filter_carbon_loss: {len(unfiltered)} pathway(s) — "
                      "check product_surface_labels or filter settings)")
            else:
                print("  (No pathways even without filter — check graph connectivity "
                      "and electron count)")

        # Union of intermediate labels across all valid pathways.
        # Excludes "*" — it is the CHE reference (ΔE=0) and never needs MLIP relaxation.
        pathway_labels: set[str] = set()
        for pw in pathways:
            pathway_labels.update(lbl for lbl in pw.intermediates if lbl != "*")

        # Trapped intermediates — kept nodes with no outgoing edges in the real
        # pruned graph (enumerator edges only, no pathway_constructor injections).
        # Nodes that appear in valid pathways are excluded: their only "exit" is via
        # the closure injection (e.g. O₂ release from *OO), which is correct
        # chemistry — they are useful terminal species, not dead-end traps.
        all_graph_dead_ends = find_trapped_intermediates(enum_output, pruner_output, rxn)
        trapped = [t for t in all_graph_dead_ends if t not in pathway_labels]
        if trapped:
            print(f"\n  Trapped intermediates ({len(trapped)}) — "
                  "kept but never appear in a valid pathway and have no exit in the pruned graph:")
            for t in trapped:
                print(f"    {t}")
        else:
            print(f"\n  No trapped intermediates.")

        records.append({
            "context":               ctx_dict,
            "starting_adsorbate":    entry["starting_adsorbate"],
            "n_pathways":            len(pathways),
            "pathway_labels":        sorted(pathway_labels),
            "trapped_intermediates": trapped,
            "pathways": [
                {
                    "intermediates":     pw.intermediates,
                    "n_electrons_total": pw.n_electrons_total,
                    "steps": [
                        {
                            "parent":      s.parent,
                            "product":     s.product,
                            "rule":        s.rule,
                            "n_electrons": s.n_electrons,
                        }
                        for s in pw.steps
                    ],
                }
                for pw in pathways
            ],
        })

    return records
