"""JSON I/O utilities — load pipeline artifacts and reconstruct domain objects.

Consolidates boilerplate that was duplicated across eval_che.py,
eval_ranker.py, and eval_pathway_constructor.py:
  - glob + mtime selection of latest JSON
  - _find_entry (context lookup by composition+facet)
  - EnumeratorOutput reconstruction from dict
  - PrunerOutput reconstruction from dict
  - Pathway reconstruction from eval_pathway_constructor_*.json
"""

from __future__ import annotations

import glob
import json
import os


def load_latest_json(pattern: str) -> tuple[str, dict]:
    """Return (path, data) for the most recently modified file matching pattern.

    Args:
        pattern: glob pattern, e.g. "eval_enumerator_*.json"

    Returns:
        (resolved path str, parsed JSON dict)

    Raises:
        FileNotFoundError: if no files match the pattern
    """
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found. "
            "Make sure the upstream eval script has been run."
        )
    path = max(files, key=os.path.getmtime)
    with open(path) as f:
        data = json.load(f)
    return path, data


def find_entry(data: dict, composition: str, facet: str, reaction: str) -> dict | None:
    """Find the result entry matching a given (composition, facet, reaction) triple.

    reaction is required because the same (composition, facet) can appear
    under multiple reactions in one run (e.g. Pt(111) HER and Pt(111) ORR) —
    matching on (composition, facet) alone silently returns whichever
    reaction's entry happens to be first, corrupting the other's lookup.

    Args:
        data: parsed JSON dict with a "results" list, each entry having
              a "context" sub-dict with "composition", "facet", "reaction" fields.
        composition: e.g. "Cu"
        facet: e.g. "111"
        reaction: e.g. "OER"

    Returns:
        The matching entry dict, or None if not found.
    """
    for entry in data.get("results", []):
        ctx = entry.get("context", {})
        if (ctx.get("composition") == composition
                and ctx.get("facet") == facet
                and ctx.get("reaction") == reaction):
            return entry
    return None


def catalyst_context_from_dict(d: dict):
    """Build a CatalystContext from a context sub-dict.

    Args:
        d: dict with keys composition, facet, reaction, pH, U

    Returns:
        CatalystContext
    """
    from .meta_reasoner import CatalystContext
    return CatalystContext(
        composition=d["composition"],
        facet=d["facet"],
        reaction=d["reaction"],
        pH=d["pH"],
        U=d["U"],
    )


def reconstruct_enumerator_output(enum_entry: dict, ctx):
    """Reconstruct an EnumeratorOutput from a serialised JSON entry.

    Args:
        enum_entry: one entry from eval_enumerator_*.json["results"]
        ctx: CatalystContext for this entry

    Returns:
        EnumeratorOutput
    """
    from .enumerator import EnumeratorOutput, Intermediate, IntermediateEdge

    intermediates = [
        Intermediate(
            label=i["label"],
            parent=i["parent"],
            rule=i["rule"],
            depth=i["depth"],
            n_electrons_cumulative=i["n_electrons_cumulative"],
            reasoning=i.get("reasoning", ""),
        )
        for i in enum_entry["intermediates"]
    ]
    # Edges may be stored in either the pruner or enumerator JSON.
    # Callers should pass the enumerator entry; pruner edges are preferred
    # in reconstruct_pruner_output.
    edges = [
        IntermediateEdge(
            parent=e["parent"],
            label=e["label"],
            rule=e["rule"],
            n_electrons=e["n_electrons"],
        )
        for e in enum_entry.get("edges", [])
    ]
    return EnumeratorOutput(
        starting_adsorbate=enum_entry["starting_adsorbate"],
        catalyst_context=ctx,
        intermediates=intermediates,
        edges=edges,
        graph=enum_entry["graph"],
        depth_reached=enum_entry.get("depth_reached", 0),
        n_llm_calls=enum_entry.get("n_llm_calls", 0),
        raw_responses=[],
    )


def reconstruct_pruner_output(
    pruner_entry: dict,
    enum_entry: dict,
    intermediates,
    max_denticity: int = 1,
):
    """Reconstruct a PrunerOutput from serialised JSON entries.

    Args:
        pruner_entry: one entry from eval_pruner_*.json["results"]
        enum_entry: corresponding entry from eval_enumerator_*.json["results"]
            (used as fallback for edge data when pruner entry lacks edges)
        intermediates: list[Intermediate] already reconstructed (from
            reconstruct_enumerator_output)
        max_denticity: maximum number of '*' anchors allowed in kept
            intermediates (default 1 = monodentate only).  Set to 2 to
            include bidentate species once bidentate structure generation
            is supported in eval_adsorption_energy.py.

    Returns:
        PrunerOutput
    """
    from .enumerator import Intermediate
    from .pruner import PrunerOutput, ScoredIntermediate

    inter_lookup = {i.label: i for i in intermediates}

    kept_entries = pruner_entry["kept"]
    kept_entries = [k for k in kept_entries if k["label"].count("*") <= max_denticity]

    kept_scored = [
        ScoredIntermediate(
            intermediate=inter_lookup.get(
                k["label"],
                Intermediate(
                    label=k["label"], parent="?", rule=k.get("rule", "?"),
                    depth=k.get("depth", 0),
                    n_electrons_cumulative=k.get("n_electrons_cumulative", 0),
                    reasoning="",
                ),
            ),
            score=k["score"],
            keep=True,
            reasoning=k.get("reasoning", ""),
        )
        for k in kept_entries
    ]
    discarded_scored = [
        ScoredIntermediate(
            intermediate=inter_lookup.get(
                d["label"],
                Intermediate(
                    label=d["label"], parent="?", rule="?",
                    depth=0, n_electrons_cumulative=0, reasoning="",
                ),
            ),
            score=d["score"],
            keep=False,
            reasoning=d.get("reasoning", ""),
        )
        for d in pruner_entry["discarded"]
    ]
    return PrunerOutput(
        kept=kept_scored,
        discarded=discarded_scored,
        threshold=pruner_entry.get("threshold", 3.0),
        n_runs=pruner_entry.get("n_runs", 1),
        n_llm_calls=0,
        consensus_scores=pruner_entry.get("consensus_scores", {}),
    )


def reconstruct_pathways(pc_entry: dict) -> list:
    """Reconstruct Pathway objects from a pathway_constructor JSON entry.

    Args:
        pc_entry: one entry from eval_pathway_constructor_*.json["results"]

    Returns:
        list[Pathway]
    """
    from .pathway_constructor import Pathway, ReactionStep

    pathways = []
    for pw_dict in pc_entry.get("pathways", []):
        steps = [
            ReactionStep(
                parent=s["parent"],
                product=s["product"],
                rule=s["rule"],
                n_electrons=s["n_electrons"],
            )
            for s in pw_dict["steps"]
        ]
        pathways.append(Pathway(
            steps=steps,
            intermediates=pw_dict["intermediates"],
            n_electrons_total=pw_dict["n_electrons_total"],
        ))
    return pathways
