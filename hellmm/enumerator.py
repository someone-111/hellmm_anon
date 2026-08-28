"""Rule enumerator — second LLM touchpoint.

Applies each selected rule exhaustively to a starting adsorbate, expanding
level by level (breadth-first) up to a given depth. Runs the full expansion
n_runs times and takes the union, filtering by confidence.
"""

import logging
import re
import time
import warnings
from collections import defaultdict
from pydantic.dataclasses import dataclass

from .chemistry import canonicalization_key, matches_species, parse_formula
from .llm import call_llm
from .meta_reasoner import CatalystContext, RulesetSelection
from .pourbaix import should_proceed as _pourbaix_gate
from .reaction import REACTION_TEMPLATES
from .rules import ALL_RULES, TransformationRule
from .tools.structure import _BIDENTATE_TABLE, _LABEL_SMILES_TABLE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a surface chemistry enumerator. "
    "Apply the given transformation rule EXHAUSTIVELY to the given adsorbate. "
    "List every chemically valid product — one per possible application site. "
    "Do not skip any eligible atom. "
    "Do not reason about mechanism plausibility — just apply the rule mechanically. "
    "IMPORTANT: The bare surface site `*` on an oxide or perovskite catalyst is NOT empty — "
    "it represents a metal cation site surrounded by lattice oxygen atoms. "
    "For example, applying lattice_o_release to `*` on RuO2 always yields `*V_O`. "
    "Use canonical labels from the computational electrochemistry literature: "
    "*COOH (not *HO2C, *HOCO, *OCOH), "
    "*CHO (not *HCO, *OCH), "
    "*CO (not *OC), "
    "*OH (not *HO). "
    "Always place the surface anchor * at the start of the label."
)

USER_PROMPT_TEMPLATE = """\
Catalyst: {composition} ({facet}), {reaction} conditions, U={U}V vs RHE, pH={pH}

Adsorbate: {adsorbate}

Rule: {rule_name} — {rule_description}
Electrons transferred per application: {n_electrons}

List ALL products from applying this rule to every eligible site on the adsorbate.
For each product output:
- "label": product label using * for the surface anchor
- "site": which atom or bond the rule acted on
- "n_electrons": electrons transferred (must equal {n_electrons})
- "reasoning": one sentence

Return a JSON array of product objects. If no valid application exists, return [].
"""

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Intermediate:
    label: str                    # e.g. "*COOH", "*OH", "*V_O"
    parent: str                   # adsorbate this was generated from
    rule: str                     # rule name that produced it
    depth: int                    # steps from starting adsorbate
    n_electrons_cumulative: int   # total electrons transferred to reach this
    reasoning: str


@dataclass
class IntermediateEdge:
    """A directed edge in the mechanism graph: one elementary transformation step.

    Separated from Intermediate so that the same product label can appear as the
    child of multiple parents (e.g. *OOH reachable from both *OH and *O via
    hydroxylation). Intermediate stores unique *nodes* for pruning; IntermediateEdge
    stores every *transition* with its authoritative per-step electron count.
    """
    parent: str        # label before transformation
    label: str         # label after transformation
    rule: str          # rule name that produced this transition
    n_electrons: int   # electrons transferred in this step (from rule definition)


@dataclass
class EnumeratorOutput:
    starting_adsorbate: str
    catalyst_context: CatalystContext
    intermediates: list[Intermediate]     # unique surface species (for pruning)
    edges: list[IntermediateEdge]         # every (parent→child) transition with metadata
    graph: dict[str, list[str]]           # adjacency list, derived from edges
    depth_reached: int
    n_llm_calls: int
    raw_responses: list[dict]             # (prompt, raw response) pairs for debugging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALIASES: dict[str, str] = {
    # CO2RR label variants
    "*HO2C":  "*COOH",
    "*HOCO":  "*COOH",
    "*OCOH":  "*COOH",
    # -CO2H is standard organic notation for the carboxyl group, i.e. the same
    # species as *COOH; it is not formate (*OCHO), which keeps its own label.
    # Left unaliased it becomes a second node for one intermediate, scored with
    # its own adsorption energy — a split of ~0.2 eV was measured across repeat
    # runs, large enough to change which step CHE reports as rate-limiting.
    # canonicalization_key cannot merge these: it refuses any label with more
    # than one heavy element, because *CHO/*COH and *COOH/*OCHO are genuine
    # isomers that must stay distinct.  Spelling variants of a carbon species
    # therefore have to be named here, as the three above already are.
    "*CO2H":  "*COOH",
    "*HCO":   "*CHO",
    "*OCH":   "*CHO",
    "*OC":    "*CO",
    "*HO":    "*OH",
    "*OHOH":  "*(OH)2",
    "*HOOH":  "*OOH",
    "*HOO":   "*OOH",
    # Formate.  *CHO2 and *OCHO share formula CHO2 with *COOH, so this is an
    # isomer question rather than a spelling one and was checked on the relaxed
    # geometries rather than assumed: *CHO2 and *OCHO converge to the same
    # structure (C-H 1.11 Å, symmetric C-O 1.27 Å, O-bound at 2.27 Å — bidentate
    # formate), while *COOH is genuinely different (H on O at 0.99 Å, C-bound,
    # asymmetric C-O 1.23/1.37 Å) and keeps its own label.  Left unaliased the
    # two formate spellings are scored as separate species; they differed by
    # 0.026 eV purely because one drew 42 candidate placements and the other 21.
    "*CHO2":  "*OCHO",
    # Formic acid aliases — all represent HCOOH; normalise to one canonical label
    # so they share a single adsorption energy calculation.
    "*HCO2H": "*HCOOH",
    "*HOCHO": "*HCOOH",
    "*OCHOH": "*HCOOH",
    # HER
    "*HH":    "*H2",
    "*H₂":    "*H2",
    # OER oxide/perovskite label variants
    "*(OOH)":               "*OOH",
    "*O_{ads}":             "*O",
    "*_{empty}":            "*",
    "*O_lattice (unchanged)": "*O_lattice",
    # Bare-surface variants: LLM sometimes writes "(*)" or "( * )" for the
    # clean surface instead of canonical "*".
    "(*)":   "*",
    "( *)":  "*",
    "( * )": "*",
    "(*) ":  "*",
}


# Strips cation oxidation-state site annotations produced by the cation_redox
# rule, e.g. *O[Ru^V]_lattice → *O, *OOH[Ru^VI]_lattice → *OOH.
# The MLIP captures charge redistribution implicitly from geometry, so these
# annotated labels reduce to the same adsorbate as their base form.
# Pattern: [ElementSymbol^RomanNumeral]_lattice at end of label.
_CATION_REDOX_RE = re.compile(r'\[[A-Z][a-z]?\^[IVX]+\]_lattice$')


# Exact formula change implied by each rule, for the rules whose stoichiometry
# is fully determined by the rule itself.
#
# `_validate_products` already enforces the electron count the LLM returns, but
# nothing enforced the *atoms*: the LLM would return e.g. *OO -> *OOH under
# rule "hydroxylation" (which must add O and H) when the product only adds H.
# That is not a cosmetic mislabel — the edge then carries hydroxylation's
# n_e=+1 while the actual chemistry (adding H) is n_e=-1, so the electron
# count is sign-flipped and feeds corrupted values into U_onset and the
# polarity/direction checks in che.py.  Measured on one Pt(111) ORR run,
# 16/34 hydroxylation, 13/23 dehydroxylation, 8/30 protonation and 6/18
# deprotonation edges violated their own rule's stoichiometry; one such edge
# (*OOH -> *O(OH)2, labelled protonation but adding OH) inserted a spurious
# intermediate into the best pathway and inflated the overpotential from
# 0.97 to 1.48 eV.
#
# Rules with genuinely variable stoichiometry — bond_dissociation, desorption,
# the coupling rules, and the lattice/vacancy/cation rules — are deliberately
# absent here and remain unchecked.
_RULE_FORMULA_DELTA: dict[str, dict[str, int]] = {
    "protonation":           {"H": +1},
    "deprotonation":         {"H": -1},
    "hydroxylation":         {"O": +1, "H": +1},
    "dehydroxylation":       {"O": -1, "H": -1},
    "reductive_dehydration": {"O": -1, "H": -1},
}


def _formula_delta(parent: str, child: str) -> dict[str, int]:
    """Per-element change in atom counts going from `parent` to `child`."""
    a, b = parse_formula(parent), parse_formula(child)
    return {
        k: b.get(k, 0) - a.get(k, 0)
        for k in set(a) | set(b)
        if b.get(k, 0) != a.get(k, 0)
    }


def _normalize(label: str) -> str:
    """Canonical form for deduplication: strip spaces, ensure * prefix, apply aliases."""
    label = label.strip().replace("* ", "*")
    # Move trailing * to front: "H*" → "*H", "CO2*" → "*CO2"
    if not label.startswith("*") and label.endswith("*"):
        label = "*" + label[:-1]
    # Strip cation oxidation-state annotations: *OOH[Ru^V]_lattice → *OOH
    label = _CATION_REDOX_RE.sub("", label)
    return _ALIASES.get(label, label)


def _apply_rule(
    adsorbate: str,
    rule: TransformationRule,
    context: CatalystContext,
    model: str,
) -> tuple[list[dict], dict]:
    """One LLM call: apply rule to adsorbate.
    Returns (products list, raw_log dict with prompt + response).
    """
    user_msg = USER_PROMPT_TEMPLATE.format(
        composition=context.composition,
        facet=context.facet,
        reaction=context.reaction,
        U=context.U,
        pH=context.pH,
        adsorbate=adsorbate,
        rule_name=rule.name,
        rule_description=rule.description,
        n_electrons=rule.n_electrons,
    )
    max_retries = 3
    raw = None
    for attempt in range(max_retries):
        try:
            raw = call_llm(system=SYSTEM_PROMPT, user=user_msg, model=model)
            break
        except (ValueError, Exception) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                warnings.warn(f"LLM failed for {adsorbate} + {rule.name}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                # All retries exhausted
                warnings.warn(f"LLM failed {max_retries} times for {adsorbate} + {rule.name}: {e}")
                log = {"adsorbate": adsorbate, "rule": rule.name, "prompt": user_msg, "response": None, "error": str(e), "retries": max_retries}
                return [], log

    log = {"adsorbate": adsorbate, "rule": rule.name, "prompt": user_msg, "response": raw}

    if isinstance(raw, list):
        return raw, log
    for v in raw.values():
        if isinstance(v, list):
            return v, log
    return [], log


def _validate_products(
    products: list[dict],
    rule: TransformationRule,
    parent: str,
    depth: int,
    parent_electrons: int,
) -> list[Intermediate]:
    """Filter and convert raw LLM product dicts to Intermediate objects."""
    # Precondition guards — discard all products if the parent is incompatible
    if rule.name == "vacancy_healing" and parent != "*V_O":
        return []  # no vacancy to heal
    if rule.name == "lattice_o_release" and parent != "*":
        return []  # lattice O only departs from clean surface site
    if rule.name == "oo_coupling" and parent != "*O":
        return []  # OPM coupling only applies to the *O intermediate

    result = []
    for p in products:
        if not isinstance(p, dict):
            continue  # LLM returned a string instead of a product dict — skip
        label = str(p.get("label", "")).strip()
        # Strip anything after the first space — the LLM occasionally appends
        # prose after the label (e.g. "*CH2O containers=node").
        label = label.split()[0] if label else ""
        if not label or "*" not in label:
            continue  # discard malformed labels

        # Accept the product if the LLM returned the correct signed value OR
        # the correct magnitude (LLM sometimes omits the sign for reductive rules).
        # The authoritative electron count always comes from rule.n_electrons.
        returned_ne = p.get("n_electrons")
        if returned_ne != rule.n_electrons:
            # int() must be guarded, not just null-checked.  The field is present
            # and non-None but need not be a scalar: a model returned a *list*
            # here, and the bare int() raised TypeError out of the enumerator,
            # killing the whole run.  Same class as a bare list arriving where the
            # meta-reasoner expects an object, or a missing score in the pruner —
            # the right field with the wrong type.  A value we cannot interpret is
            # not a match, so fall through to the discard branch below; the
            # authoritative count comes from rule.n_electrons regardless.
            try:
                same_magnitude = abs(int(returned_ne)) == abs(rule.n_electrons)
            except (TypeError, ValueError):
                same_magnitude = False
            if same_magnitude:
                warnings.warn(
                    f"LLM returned n_electrons={returned_ne} for {label!r} under "
                    f"rule '{rule.name}' (expected {rule.n_electrons}). "
                    "Accepting product; using rule-defined electron count."
                )
            else:
                continue  # wrong magnitude — discard

        reasoning = str(p.get("reasoning", ""))
        e_cum = parent_electrons + rule.n_electrons

        # bond_dissociation can produce two co-adsorbed fragments: "*CO + *OH"
        # split into separate intermediates so each is explored in the next depth level
        fragments = [f.strip() for f in label.split("+") if "*" in f]
        if not fragments:
            continue

        for fragment in fragments:
            norm = _normalize(fragment)
            if norm == parent:
                continue  # discard self-loops

            # Mass balance: the product must differ from the parent by exactly
            # the atoms this rule adds or removes (see _RULE_FORMULA_DELTA).
            expected_delta = _RULE_FORMULA_DELTA.get(rule.name)
            if expected_delta is not None:
                actual_delta = _formula_delta(parent, norm)
                if actual_delta != expected_delta:
                    warnings.warn(
                        f"Discarding {parent!r} -> {norm!r} under rule "
                        f"'{rule.name}': formula change {actual_delta} does not "
                        f"match the rule's required {expected_delta}. The step "
                        "is not the elementary reaction the rule describes, so "
                        "its electron count would be wrong."
                    )
                    continue

            result.append(Intermediate(
                label=norm,
                parent=parent,
                rule=rule.name,
                depth=depth,
                n_electrons_cumulative=e_cum,
                reasoning=reasoning,
            ))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SAFETY_CAP = 100


def enumerate_intermediates(
    starting_adsorbate: str,
    context: CatalystContext,
    ruleset: RulesetSelection,
    depth: int = 2,
    n_runs: int = 3,
    model: str = "tencent-hy3-preview",
) -> EnumeratorOutput:
    # counts[normalized_label] = number of runs that produced it (for confidence filter)
    counts: dict[str, int] = defaultdict(int)
    # Unique nodes: one representative Intermediate per label (first seen wins)
    seen: dict[str, Intermediate] = {}
    # Unique edges: keyed by (parent, label, rule) — allows the same product
    # from multiple parents (*O → *OOH and *OH → *OOH both get entries), AND
    # keeps multiple rules that happen to propose the same (parent, label)
    # transition as separate edges (e.g. both "desorption" and "deprotonation"
    # producing *H → *) rather than one silently overwriting the other.
    # pathway_constructor treats these as distinct mechanistic options and
    # lets CHE — not enumeration order — decide which is physically valid.
    all_edges: dict[tuple[str, str, str], IntermediateEdge] = {}

    if n_runs < 2:
        warnings.warn(
            f"n_runs={n_runs}: confidence filter is disabled (min_count=1). "
            "All LLM-proposed intermediates will be kept regardless of reproducibility. "
            "Use n_runs >= 3 for benchmarks and publication-quality results."
        )

    n_llm_calls = 0
    depth_reached = 0
    raw_responses: list[dict] = []

    _rxn_def = REACTION_TEMPLATES.get(context.reaction)

    for run_idx in range(n_runs):
        print(f"  run {run_idx + 1}/{n_runs}", flush=True)
        # BFS state: (label, cumulative_electrons)
        frontier: list[tuple[str, int]] = [(_normalize(starting_adsorbate), 0)]
        visited_this_run: set[str] = {_normalize(starting_adsorbate)}

        for current_depth in range(1, depth + 1):
            next_frontier: list[tuple[str, int]] = []

            for parent_label, parent_electrons in frontier:
                for rule in ruleset.selected_rules:
                    if len(seen) >= SAFETY_CAP:
                        warnings.warn(
                            f"Safety cap of {SAFETY_CAP} intermediates reached. "
                            "Stopping enumeration early."
                        )
                        break

                    print(f"    [{n_llm_calls+1}] {parent_label} + {rule.name} ...", flush=True)
                    products_raw, log = _apply_rule(parent_label, rule, context, model)
                    raw_responses.append(log)
                    n_llm_calls += 1

                    intermediates = _validate_products(
                        products_raw, rule, parent_label, current_depth, parent_electrons
                    )

                    for inter in intermediates:
                        norm = inter.label

                        # --- Node tracking (deduplicated by label) ---
                        if norm not in seen:
                            seen[norm] = inter
                            print(f"      → {norm}  (e_cum={inter.n_electrons_cumulative})", flush=True)
                        counts[norm] += 1

                        # --- Edge tracking (deduplicated by (parent, label, rule)) ---
                        # Records every distinct parent→child transition with the
                        # authoritative per-step electron count from the rule definition.
                        edge_key = (inter.parent, norm, rule.name)
                        all_edges[edge_key] = IntermediateEdge(
                            parent=inter.parent,
                            label=norm,
                            rule=rule.name,
                            n_electrons=rule.n_electrons,
                        )

                        # --- Frontier management (per run, label-based) ---
                        if norm not in visited_this_run:
                            visited_this_run.add(norm)
                            next_frontier.append((norm, inter.n_electrons_cumulative))

            if not next_frontier:
                break

            frontier = next_frontier
            depth_reached = max(depth_reached, current_depth)

    start_norm = _normalize(starting_adsorbate)

    # Canonicalise label spellings BEFORE the confidence filter.
    # The LLM writes one species many ways — adsorbed O2 has been observed as
    # *O2, *OO, *O=O and *O(O) within a single run.  Left alone each spelling
    # is a separate graph node: pathways fragment, identical MLIP relaxations
    # repeat, and — critically — the reproducibility vote below splits across
    # variants, so a species proposed in *every* run can still fall under the
    # threshold and be deleted.  Merge only where formula provably determines
    # the species (see chemistry.canonicalization_key); everything else is
    # left untouched.
    groups: dict[str, list[str]] = defaultdict(list)
    for norm in seen:
        key = canonicalization_key(norm)
        if key is not None:
            groups[key].append(norm)

    # product_surface_labels must survive canonicalisation under exactly the
    # spelling ReactionDefinition uses, because downstream code looks them up
    # by string: pathway_constructor injects the product-desorption edge and
    # the pruner force-keeps them.  The representative below is otherwise
    # chosen partly by vote count, which is not stable between runs — so
    # *NH3 could silently become a different spelling in one run, both lookups
    # would miss, and NRR would lose every pathway with no error reported.
    _protected = {start_norm}
    if _rxn_def:
        _protected |= {_normalize(l) for l in _rxn_def.product_surface_labels}

    rename: dict[str, str] = {}
    for _key, variants in groups.items():
        if len(variants) < 2:
            continue
        _prot = sorted(v for v in variants if v in _protected)
        if _prot:
            rep = _prot[0]            # never rename a label named by config/ReactionDefinition
        else:
            rep = min(variants, key=lambda l: (
                l not in _LABEL_SMILES_TABLE,   # prefer a directly resolvable spelling
                -counts[l],                     # then the most reproduced
                len(l),                         # then the shortest
                l,                              # then alphabetical — fully deterministic
            ))
        for v in variants:
            if v != rep:
                rename[v] = rep

    if rename:
        merged_counts: dict[str, int] = defaultdict(int)
        for norm, c in counts.items():
            merged_counts[rename.get(norm, norm)] += c
        counts = merged_counts

        merged_seen: dict[str, Intermediate] = {}
        for norm, inter in seen.items():
            rep = rename.get(norm, norm)
            if rep not in merged_seen:
                merged_seen[rep] = Intermediate(
                    label=rep,
                    parent=rename.get(inter.parent, inter.parent),
                    rule=inter.rule,
                    depth=inter.depth,
                    n_electrons_cumulative=inter.n_electrons_cumulative,
                    reasoning=inter.reasoning,
                )
        seen = merged_seen

        merged_edges: dict[tuple[str, str, str], IntermediateEdge] = {}
        for e in all_edges.values():
            p = rename.get(e.parent, e.parent)
            l = rename.get(e.label, e.label)
            if p == l:
                continue              # self-loop created by the merge — drop it
            merged_edges[(p, l, e.rule)] = IntermediateEdge(
                parent=p, label=l, rule=e.rule, n_electrons=e.n_electrons,
            )
        all_edges = merged_edges

        print(f"  canonicalised {len(rename)} label variant(s) into "
              f"{len(set(rename.values()))} node(s)", flush=True)

    # Rule disambiguation on transitions the LLM has already proposed.
    #
    # Mass balance cannot separate every rule.  dehydroxylation (n_e=0) and
    # reductive_dehydration (n_e=-1) both declare {O:-1, H:-1}, so for a
    # transition such as *OOH -> *O the LLM's choice of rule *label* alone
    # decides whether the step is chemical or electrochemical — and that choice
    # is not reproducible.  Across repeat runs of the same case the *OOH -> *O
    # transition was seen carrying reductive_dehydration in some runs and only
    # n_e=0 rules in others, changing the pathway's total electron count and
    # hence whether it is admissible as the target reaction at all.
    #
    # What this does NOT do is invent chemistry.  A (parent, product) pair is
    # considered only if the LLM already placed it in the graph; the pair's own
    # formula delta is then matched against the rules the meta-reasoner selected
    # for this catalyst, and every rule consistent with that stoichiometry is
    # made available to the CHE step-variant branch.  Nothing is injected for a
    # transition the LLM never proposed: a step the model failed to find stays
    # missing and the pathway stays unfound.  That distinction is the point —
    # supplying an absent mechanism would be answering the question the
    # framework exists to ask.
    #
    # It is also deliberately symmetric.  Every stoichiometrically-consistent
    # rule is added, not the electrochemical one preferentially; adding only the
    # reductive member would bias every affected pathway toward a lower
    # overpotential.  In the current rule book exactly one ambiguous pair exists
    # (the two above), so in practice this adds the n_e=0 reading as often as
    # the n_e=-1 one.  Each addition is logged so the count is reportable.
    _selected_by_name = {r.name: r for r in ruleset.selected_rules}
    _disambiguated: list[tuple[str, str, str]] = []
    for _p, _l, _ in list(all_edges):
        _delta = _formula_delta(_p, _l)
        for _name, _expected in _RULE_FORMULA_DELTA.items():
            if _name not in _selected_by_name or _expected != _delta:
                continue
            if (_p, _l, _name) in all_edges:
                continue
            all_edges[(_p, _l, _name)] = IntermediateEdge(
                parent=_p,
                label=_l,
                rule=_name,
                n_electrons=_selected_by_name[_name].n_electrons,
            )
            _disambiguated.append((_p, _l, _name))

    if _disambiguated:
        print(f"  rule disambiguation: {len(_disambiguated)} equivalent "
              f"interpretation(s) added to LLM-proposed transitions", flush=True)
        for _p, _l, _name in _disambiguated:
            print(f"      {_p} -> {_l}  [{_name}]", flush=True)

    # Confidence filter: keep only intermediates reproduced by a MAJORITY of
    # independent runs.  The previous round(0.4·n_runs) threshold evaluated to
    # 1 at n_runs=3 (the configured value), i.e. a label proposed in a single
    # run entered the graph and the filter did nothing at all.
    # Exclude the starting adsorbate and the bare surface "*".
    # The bare surface is always a boundary node (start/end of the catalytic
    # cycle), never a scored intermediate — it has no adsorbate to evaluate
    # and its adsorption energy is 0 by definition (the CHE reference).
    # pathway_constructor.py handles "*" explicitly as a terminal neighbor.
    #
    # The reaction's declared product is exempt from the count requirement.
    # Reproducibility turns out not to correlate with correctness here: the LLM
    # regenerates the same over-hydroxylation junk (*O(OH)(OH)(OH)OH, *OOOOOH,
    # …) in every run, while proposing the actual product only occasionally —
    # ORR's water was proposed exactly once in three runs, so a 2-of-3 majority
    # deleted the one species the reaction cannot close without, and kept 34
    # junk labels.  The product is already a declared boundary condition and is
    # already force-kept in the pruner for exactly this reason; a statistical
    # filter should not be able to remove it either.  Note this only protects a
    # product the LLM *did* find — nothing is injected if it never appears.
    min_count = n_runs // 2 + 1 if n_runs > 1 else 1
    _product_found = {
        norm for norm in seen
        if _rxn_def and any(matches_species(norm, t)
                            for t in _rxn_def.product_surface_labels)
    }
    final_nodes = [
        inter for norm, inter in seen.items()
        if (counts[norm] >= min_count or norm in _product_found)
        and norm != start_norm and norm != "*"
    ]

    # Keep only edges where both endpoints survived the confidence filter.
    #
    # "*" is a boundary node, never a scored intermediate — final_nodes above
    # excludes it deliberately (it has no adsorbate to evaluate and its
    # adsorption energy is 0 by definition).  It must still be admissible as an
    # edge *target*, or the both-endpoints test below silently deletes every
    # transition that closes the cycle on the bare surface.  That is not
    # hypothetical: in most repeat runs of ORR the LLM proposed *OH -> * under
    # reductive_dehydration (n_e=-1) — the textbook direct water-release closure,
    # and mass-balance-valid ({O:-1, H:-1}) — and every one was discarded here
    # without a warning.  Runs that also happened to propose *OH -> *H2O
    # survived on that spare route; a run offering only the direct closure lost
    # its sole path to the product and reported 0 pathways, a false negative
    # produced by this filter rather than by the model.
    #
    # Admitting the edge does not admit miscounted pathways: the DFS keeps only
    # pathways whose summed |n_e| equals reaction.n_electrons_total exactly
    # (pathway_constructor.py), so a chemical *OH -> * desorption edge closing
    # ORR at 3 electrons is still rejected downstream.
    surviving = {inter.label for inter in final_nodes}
    surviving.add(start_norm)
    surviving.add("*")
    final_edges = [
        e for e in all_edges.values()
        if e.parent in surviving and e.label in surviving
    ]

    # Build graph from surviving edges (consistent by construction — no edge
    # exists in the graph without a backing IntermediateEdge).
    clean_graph: dict[str, list[str]] = {}
    for e in final_edges:
        if e.label not in clean_graph.get(e.parent, []):
            clean_graph.setdefault(e.parent, []).append(e.label)

    return EnumeratorOutput(
        starting_adsorbate=start_norm,
        catalyst_context=context,
        intermediates=final_nodes,
        edges=final_edges,
        graph=clean_graph,
        depth_reached=depth_reached,
        n_llm_calls=n_llm_calls,
        raw_responses=raw_responses,
    )


# ---------------------------------------------------------------------------
# Full enumerator stage — all catalyst cases
# ---------------------------------------------------------------------------

def run_enumerator(
    catalyst_cases: list[tuple[str, CatalystContext]],
    operating_points_map: dict[tuple[str, str, str, str], list[tuple[float, float]]],
    meta_data: dict,
    model: str,
    max_depth: int,
    n_runs: int,
) -> list[dict]:
    """Run enumeration for every catalyst case, gated by the Pourbaix stability check.

    For each (starting_adsorbate, CatalystContext) case: filters configured
    (pH, U) operating points to those where the catalyst is stable (skips the
    case entirely if none are stable), looks up the meta_reasoner's selected
    ruleset for this (composition, facet, reaction), then enumerates
    intermediates via enumerate_intermediates().

    Args:
        catalyst_cases: one (starting_adsorbate, CatalystContext) per unique
            (composition, facet, reaction, start) — e.g. config.unique_catalyst_cases()
        operating_points_map: (composition, facet, reaction, start) -> list of
            (pH, U) operating points to check for Pourbaix stability — e.g.
            built from config.operating_points_for() per case
        meta_data: parsed eval_meta_reasoner output (eval_results_*.json)
        model: LLM model name
        max_depth: safety cap on enumeration depth (overrides meta_reasoner's
            suggested_depth if larger)
        n_runs: LLM voting rounds for the confidence filter

    Returns:
        List of JSON-serializable per-catalyst records: context, starting
        adsorbate, stable operating points, selected rules, intermediates
        (with SMILES/anchor indices pre-resolved from the lookup tables where
        available), edges, graph, and enumeration metadata.

    Raises:
        KeyError: if a catalyst case has no matching meta_reasoner result —
            the meta_reasoner stage must be re-run to include it.
    """
    known_rules = {r.name: r for r in ALL_RULES}
    meta_lookup: dict[tuple, RulesetSelection] = {}
    for entry in meta_data["results"]:
        ctx = entry["context"]
        key = (ctx["composition"], ctx["facet"], ctx["reaction"])
        ruleset = RulesetSelection(
            selected_rules=[known_rules[n] for n in entry["selected_rules"] if n in known_rules],
            reasoning=entry.get("reasoning", {}),
            catalyst_class=entry.get("catalyst_class", "unknown"),
            chain_of_thought=entry.get("chain_of_thought", ""),
            suggested_depth=int(entry.get("suggested_depth", 3)),
        )
        meta_lookup[key] = ruleset
        print(f"  loaded: {key} → {[r.name for r in ruleset.selected_rules]}")

    records: list[dict] = []

    for starting_adsorbate, ctx in catalyst_cases:
        print(f"\n{'='*60}")
        print(f"{ctx.composition}({ctx.facet})  {ctx.reaction}  start={starting_adsorbate}")
        print(f"{'='*60}")

        # Pourbaix stability gate — check each configured (pH, U) operating point.
        # Points where the catalyst is thermodynamically unstable are filtered out;
        # only stable points proceed to CHE/ranker. If ALL points are unstable,
        # skip enumeration entirely. Falls back to stable=True if MY_MP_API_KEY
        # is not set.
        all_points = operating_points_map.get(
            (ctx.composition, ctx.facet, ctx.reaction, starting_adsorbate), []
        )
        stable_operating_points: list[list[float]] = []
        for ph, u in all_points:
            ctx_point = CatalystContext(
                composition=ctx.composition, facet=ctx.facet, reaction=ctx.reaction, pH=ph, U=u,
            )
            proceed, reason = _pourbaix_gate(ctx_point)
            if proceed:
                stable_operating_points.append([ph, u])
            else:
                print(f"  pH={ph}, U={u}: UNSTABLE — {reason}")

        if not stable_operating_points:
            print(f"  All operating points unstable for {ctx.composition}({ctx.facet}) — skipping.")
            continue

        if len(stable_operating_points) < len(all_points):
            n_skip = len(all_points) - len(stable_operating_points)
            print(f"  {n_skip} unstable operating point(s) filtered; "
                  f"{len(stable_operating_points)} remaining.")

        cache_key = (ctx.composition, ctx.facet, ctx.reaction)
        if cache_key not in meta_lookup:
            raise KeyError(
                f"No meta_reasoner result for {cache_key}. "
                f"Add this case to eval_meta_reasoner.py and re-run it."
            )
        ruleset = meta_lookup[cache_key]
        rxn_def   = REACTION_TEMPLATES.get(ctx.reaction)
        min_depth = rxn_def.min_depth if rxn_def else 1
        depth     = max(min_depth, min(ruleset.suggested_depth, max_depth))
        print(f"rules: {[r.name for r in ruleset.selected_rules]}")
        print(f"depth : {depth} (suggested={ruleset.suggested_depth}, floor={min_depth}, cap={max_depth})")

        # The meta-reasoner's chain_of_thought is deliberately NOT forwarded to
        # the enumerator.
        #
        # It used to be injected into every enumeration prompt as "why this rule
        # was selected".  In practice the model justifies a rule by working an
        # example, so that text carried the mechanism itself: for CO2RR it read
        # "protonation: CO2RR involves protonation steps (e.g. *CO2 -> *COOH)"
        # and "bond_dissociation: ... (e.g. *COOH -> *CO + OH)" — which is the
        # pathway the pipeline then reports as discovered, restated in the
        # context of all ~200 calls that discover it.  One model's guess became
        # a prior for the model doing the enumeration, and any claim about
        # discovery would have been unsupportable.
        #
        # Nothing is lost that the enumerator needs: it receives each rule's
        # name and full description, and its system prompt already tells it to
        # apply the rule mechanically rather than reason about which mechanism
        # is plausible.  The selected rule *set* remains the meta-reasoner's
        # contribution; its reasoning stays in the meta-reasoner's own output
        # for inspection.
        print("enumerating intermediates...", flush=True)
        output = enumerate_intermediates(
            starting_adsorbate=starting_adsorbate,
            context=ctx,
            ruleset=ruleset,
            depth=depth,
            n_runs=n_runs,
            model=model,
        )

        print(f"depth reached  : {output.depth_reached}")
        print(f"llm calls      : {output.n_llm_calls}")
        print(f"intermediates  : {len(output.intermediates)} unique nodes")
        for inter in sorted(output.intermediates, key=lambda x: (x.depth, x.label)):
            print(f"  depth={inter.depth}  {inter.label:20s}  via {inter.rule:20s}  e_cum={inter.n_electrons_cumulative}")
        print(f"edges          : {len(output.edges)} transitions")
        for e in sorted(output.edges, key=lambda x: (x.parent, x.label)):
            print(f"  {e.parent:20s} → {e.label:20s}  rule={e.rule:20s}  n_e={e.n_electrons}")
        print("graph:")
        for src, dsts in sorted(output.graph.items()):
            print(f"  {src} → {dsts}")

        records.append({
            "context": {
                "composition": ctx.composition,
                "facet": ctx.facet,
                "reaction": ctx.reaction,
                "pH": ctx.pH,
                "U": ctx.U,
            },
            "starting_adsorbate": starting_adsorbate,
            "stable_operating_points": stable_operating_points,
            "selected_rules": [r.name for r in ruleset.selected_rules],
            "intermediates": [
                {
                    "label": i.label,
                    "parent": i.parent,
                    "rule": i.rule,
                    "depth": i.depth,
                    "n_electrons_cumulative": i.n_electrons_cumulative,
                    "reasoning": i.reasoning,
                    # SMILES stored at enumeration time so eval_adsorption_energy.py
                    # doesn't need an LLM call for labels in the lookup tables.
                    "smiles": (
                        _LABEL_SMILES_TABLE.get(i.label)
                        or (_BIDENTATE_TABLE[i.label][0] if i.label in _BIDENTATE_TABLE else None)
                    ),
                    "anchor_indices": (
                        list(_BIDENTATE_TABLE[i.label][1]) if i.label in _BIDENTATE_TABLE else None
                    ),
                }
                for i in output.intermediates
            ],
            "edges": [
                {
                    "parent": e.parent,
                    "label": e.label,
                    "rule": e.rule,
                    "n_electrons": e.n_electrons,
                }
                for e in output.edges
            ],
            "graph": output.graph,
            "depth_reached": output.depth_reached,
            "n_llm_calls": output.n_llm_calls,
            "raw_responses": output.raw_responses,
        })

    return records
