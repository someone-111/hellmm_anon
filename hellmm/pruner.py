import warnings
from pydantic.dataclasses import dataclass

from .chemistry import matches_species
from .enumerator import EnumeratorOutput, Intermediate
from .io import catalyst_context_from_dict, reconstruct_enumerator_output
from .llm import call_llm
from .meta_reasoner import CatalystContext
from .reaction import REACTION_TEMPLATES

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Reframed from a plausibility/stability check to a chemical-validity check.
# The stability framing systematically discarded reaction products and
# rate-limiting intermediates for the correct-sounding wrong reason: *H2
# scored 0.0 ("not a plausible surface intermediate") on Pt HER in every one
# of three independent runs, *NH3 scored 2.2 on Fe(100) NRR and disconnected
# the graph (0 pathways), *H2O2 scored 2.0 ("desorbs quickly"). In each case
# the LLM's chemistry was correct — these species genuinely don't bind
# strongly — but stability is exactly what the downstream MLIP+CHE stage
# computes, so scoring it here duplicates that stage and gets it wrong by
# penalising the species whose energetics actually determine the result.
# A validity check (is this species constructible at all) doesn't have this
# failure mode: *H2 and *NH3 are valid molecules regardless of how weakly
# they bind, and over-coordinated junk like *(OH)3 is invalid regardless of
# what it would score energetically.
SYSTEM_PROMPT = """\
You are an electrochemical surface chemistry expert. Judge whether each \
intermediate is a CHEMICALLY VALID surface species — one that could exist \
even transiently, even if short-lived or high in energy. You are NOT judging \
thermodynamic stability, abundance, or how well-established a species is in \
the literature — that is computed separately, downstream, from first-principles \
energies. Guessing at stability here would discard exactly the species that \
determine the result: reaction products (which by definition desorb readily) \
and high-energy rate-limiting intermediates. Reject ONLY species that are \
chemically impossible — impossible valence or coordination, nonsensical \
connectivity, or elements not present in this system.
"""

USER_PROMPT_TEMPLATE = """\
Catalyst: {composition} ({facet}), {reaction}, U={U}V vs RHE, pH={pH}

Score each of the following surface intermediates 0–10 for CHEMICAL VALIDITY \
(not stability):
  0  = chemically impossible (impossible valence/coordination, wrong elements)
  5  = unusual but chemically constructible
  10 = well-formed, ordinary surface species

Do NOT lower the score because a species desorbs readily, is short-lived, is \
high in energy, or is unfamiliar / not established in the literature — those \
are determined downstream by MLIP energetics, not by this check.

{label_list}

For each, return: label, score (0–10 integer), one-sentence reasoning.
Output as a JSON array of objects with keys "label", "score", "reasoning".
"""

# Max intermediates per LLM call — keeps response JSON small and parseable.
# At ~97 intermediates a single prompt reliably produces truncated JSON.
_CHUNK_SIZE = 25

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ScoredIntermediate:
    intermediate: Intermediate
    score: float          # 0–10, LLM plausibility score
    keep: bool            # True if consensus_score >= threshold
    reasoning: str        # LLM explanation


@dataclass
class PrunerOutput:
    kept: list[ScoredIntermediate]
    discarded: list[ScoredIntermediate]
    threshold: float
    n_runs: int
    n_llm_calls: int
    consensus_scores: dict[str, float]        # label → mean score across runs
    pruned_graph: dict[str, list[str]] | None = None  # adjacency list with discarded nodes removed; None if loaded from old JSON


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _score_batch(
    intermediates: list[Intermediate],
    context: CatalystContext,
    model: str,
    n_runs: int,
) -> tuple[dict[str, list[float]], dict[str, str], int]:
    """Score one batch of intermediates across n_runs LLM calls.

    Returns:
        score_accumulator: label → list of scores from successful runs
        reasoning_last:    label → reasoning string from most recent run
        n_llm_calls:       number of successful LLM calls made
    """
    label_list = "\n".join(
        f"{i+1}. {inter.label}"
        for i, inter in enumerate(intermediates)
    )
    user_msg = USER_PROMPT_TEMPLATE.format(
        composition=context.composition,
        facet=context.facet,
        reaction=context.reaction,
        U=context.U,
        pH=context.pH,
        label_list=label_list,
    )

    score_accumulator: dict[str, list[float]] = {inter.label: [] for inter in intermediates}
    reasoning_last: dict[str, str] = {}
    n_llm_calls = 0

    for run_idx in range(n_runs):
        try:
            raw = call_llm(system=SYSTEM_PROMPT, user=user_msg, model=model)
            n_llm_calls += 1
        except Exception as e:
            warnings.warn(f"Pruner LLM failed on run {run_idx + 1}: {e} — skipping run")
            continue

        if not isinstance(raw, list):
            warnings.warn(f"Pruner expected JSON array, got {type(raw)} — skipping run")
            continue

        unmatched: list[str] = []
        for entry in raw:
            label = str(entry.get("label", "")).strip()
            if label not in score_accumulator:
                # The model routinely answers with a spelling variant of a label
                # it was given (*OOH2 for a batch entry written *H2O2, and so
                # on).  Recover those through the same formula canonicalisation
                # the enumerator uses to merge variants, before treating the
                # entry as unmatched — a dropped score is not neutral, it sends
                # the intermediate to 0.0 and gets it discarded as though the
                # model had rejected it.  matches_species only merges where the
                # formula provably determines the species, so genuine isomers
                # still fail to match and are reported below.
                recovered = next(
                    (known for known in score_accumulator
                     if matches_species(label, known)),
                    None,
                )
                if recovered is None:
                    unmatched.append(label)
                    continue
                label = recovered
            # A missing or unparseable score is not a zero.  Defaulting it to
            # 0.0 records a vote that the species is chemically impossible, and
            # since the consensus is a mean over runs, one malformed entry can
            # drag an otherwise-kept intermediate below the threshold and delete
            # it — the same silent-discard failure as an unmatched label, but
            # reached by a different route and not covered by the recovery
            # above.  Skip the entry and say so instead.
            raw_score = entry.get("score")
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                warnings.warn(
                    f"Pruner run {run_idx + 1}: entry for {label!r} has no usable "
                    f"score (got {raw_score!r}) — skipping this vote rather than "
                    "recording it as 0.0, which would count as 'chemically "
                    "impossible' and could discard the intermediate."
                )
                continue
            score_accumulator[label].append(score)
            reasoning_last[label] = str(entry.get("reasoning", ""))

        # A response whose labels do not match the batch is indistinguishable,
        # downstream, from one where the model genuinely scored everything 0:
        # both leave score_accumulator empty, both yield consensus 0.0 with
        # blank reasoning, and every affected intermediate is then discarded.
        # Observed in practice: one 25-intermediate chunk recorded no scores at
        # all across every run and its species — including the reaction's own
        # *O/*OH/*OOH — were dropped, taking the only route to the product with
        # them.  An empty `raw` and a fully-unmatched `raw` are different faults
        # and are reported separately so the log distinguishes them.
        if unmatched:
            warnings.warn(
                f"Pruner run {run_idx + 1}: {len(unmatched)} label(s) in the "
                f"response matched no intermediate in this batch and were "
                f"ignored: {unmatched[:8]}{' ...' if len(unmatched) > 8 else ''}. "
                "Their scores are lost; affected intermediates fall back to 0.0 "
                "and will be discarded as if the model had rejected them."
            )
        elif not raw:
            warnings.warn(
                f"Pruner run {run_idx + 1}: model returned an empty array — no "
                "scores recorded for this batch. If this repeats across runs, "
                "every intermediate in the chunk falls back to 0.0 and is "
                "discarded without the model having rejected anything."
            )

    return score_accumulator, reasoning_last, n_llm_calls


def _heuristic_scores(intermediates: list[Intermediate]) -> dict[str, float]:
    """Depth-based fallback scores when all LLM calls fail.

    Keeps depth ≤ 2 intermediates (score=5.0) and discards deeper ones
    (score=0.0). Shallow intermediates are the most physically grounded;
    deep ones are speculative and expensive to compute.
    """
    return {
        inter.label: 5.0 if inter.depth <= 2 else 0.0
        for inter in intermediates
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prune(
    enumerator_output: EnumeratorOutput,
    context: CatalystContext,
    threshold: float = 3.0,
    n_runs: int = 3,
    model: str = "tencent-hy3-preview",
    max_denticity: int = 1,
    force_keep_labels: set[str] | frozenset[str] = frozenset(),
) -> PrunerOutput:
    """
    LLM-based plausibility scorer over enumerator intermediates.

    Splits intermediates into chunks of _CHUNK_SIZE to keep each LLM
    response small and JSON-parseable. Runs n_runs times per chunk,
    averages scores per intermediate, then keeps those above threshold.

    Fallback behaviour:
      - If the LLM fails on some chunks but not all: missing scores default
        to 0.0 (conservative — those intermediates are discarded).
      - If ALL LLM calls fail across all chunks: applies a depth-based
        heuristic (keep depth ≤ 2) rather than keeping everything.
    """
    # Pre-filter: discard intermediates above the max_denticity threshold.
    # By default (max_denticity=1) bidentate labels (>1 '*') are dropped here
    # because single-site MLIP methods (fairchem DB or ACAT) cannot place them.
    # Set max_denticity=2 once generate_adsorption_configs_acat_bidentate is
    # wired into eval_adsorption_energy.py to let these pass through to MLIP.
    pre_filtered: list[Intermediate] = []
    for inter in enumerator_output.intermediates:
        if inter.label.count("*") > max_denticity:
            print(f"  auto-discard (denticity>{max_denticity}) {inter.label}", flush=True)
        else:
            pre_filtered.append(inter)
    intermediates = pre_filtered

    # Split into chunks
    chunks = [
        intermediates[i : i + _CHUNK_SIZE]
        for i in range(0, len(intermediates), _CHUNK_SIZE)
    ]
    n_chunks = len(chunks)

    # score_accumulator[label] = list of scores from each successful run
    score_accumulator: dict[str, list[float]] = {inter.label: [] for inter in intermediates}
    reasoning_last: dict[str, str] = {}
    n_llm_calls = 0

    for chunk_idx, chunk in enumerate(chunks):
        print(
            f"  pruner chunk {chunk_idx + 1}/{n_chunks}  ({len(chunk)} intermediates)",
            flush=True,
        )
        chunk_scores, chunk_reasoning, chunk_calls = _score_batch(
            chunk, context, model, n_runs
        )
        n_llm_calls += chunk_calls
        min_runs = (n_runs + 1) // 2  # ceil(n_runs / 2): e.g. 3/5, 2/3
        if chunk_calls < min_runs:
            # Fewer than half the runs succeeded — not enough consensus to
            # trust the scores.  Fall back to depth-based heuristic for this
            # chunk so a single bad LLM response cannot discard key species.
            warnings.warn(
                f"Pruner chunk {chunk_idx + 1}: only {chunk_calls}/{n_runs} runs "
                f"succeeded (need ≥ {min_runs}). Applying depth-based heuristic "
                f"for this chunk."
            )
            for inter in chunk:
                score_accumulator[inter.label].append(
                    _heuristic_scores(chunk)[inter.label]
                )
        else:
            for label, scores in chunk_scores.items():
                score_accumulator[label].extend(scores)
            reasoning_last.update(chunk_reasoning)

    # Compute consensus scores
    any_scored = any(len(scores) > 0 for scores in score_accumulator.values())
    consensus_scores: dict[str, float] = {}

    if not any_scored:
        # Complete LLM failure across all chunks — apply depth-based heuristic
        warnings.warn(
            f"All pruner LLM calls failed ({n_chunks} chunk(s), {n_runs} run(s) each). "
            f"Applying heuristic fallback: keeping depth ≤ 2 intermediates only."
        )
        consensus_scores = _heuristic_scores(intermediates)
        effective_threshold = 3.0
    else:
        for label, scores in score_accumulator.items():
            consensus_scores[label] = sum(scores) / len(scores) if scores else 0.0

        # Threshold fallback: if everything scored is below threshold, use 50th percentile
        effective_threshold = threshold
        all_scores = list(consensus_scores.values())
        if all_scores and all(s < threshold for s in all_scores):
            sorted_scores = sorted(all_scores)
            p50 = sorted_scores[len(sorted_scores) // 2]
            warnings.warn(
                f"All intermediates scored below threshold {threshold}. "
                f"Falling back to 50th-percentile threshold {p50:.1f}."
            )
            effective_threshold = p50

    # Build output
    kept: list[ScoredIntermediate] = []
    discarded: list[ScoredIntermediate] = []

    for inter in intermediates:
        score = consensus_scores[inter.label]
        keep = score >= effective_threshold
        scored = ScoredIntermediate(
            intermediate=inter,
            score=score,
            keep=keep,
            reasoning=reasoning_last.get(inter.label, ""),
        )
        if keep:
            kept.append(scored)
            print(f"  keep  {inter.label:35s}  score={score:.1f}  {scored.reasoning}", flush=True)
        else:
            discarded.append(scored)
            print(f"  disc  {inter.label:35s}  score={score:.1f}  {scored.reasoning}", flush=True)

    # Force-keep reaction-critical intermediates (e.g. product_surface_labels).
    # The LLM may score these low on an unfamiliar facet but they are required
    # for pathway closure — discarding them silently kills all valid paths.
    if force_keep_labels:
        still_discarded: list[ScoredIntermediate] = []
        for s in discarded:
            # Match by species, not exact spelling: the LLM writes water as
            # "*H2O" in one run and "*OH2" in the next, and an exact-string
            # comparison here silently failed to rescue it.
            if any(matches_species(s.intermediate.label, t) for t in force_keep_labels):
                forced = ScoredIntermediate(
                    intermediate=s.intermediate,
                    score=s.score,
                    keep=True,
                    reasoning=f"[force-kept: product_surface_label] {s.reasoning}",
                )
                kept.append(forced)
                print(
                    f"  FORCE-KEEP {s.intermediate.label:30s}  "
                    f"score={s.score:.1f}  (product_surface_label)",
                    flush=True,
                )
            else:
                still_discarded.append(s)
        discarded = still_discarded

    # Build a pruned copy of the graph without mutating the input.
    discarded_labels = {s.intermediate.label for s in discarded}
    pruned_graph = {
        src: [dst for dst in dsts if dst not in discarded_labels]
        for src, dsts in enumerator_output.graph.items()
        if src not in discarded_labels
    }

    return PrunerOutput(
        kept=kept,
        discarded=discarded,
        threshold=effective_threshold,
        n_runs=n_runs,
        n_llm_calls=n_llm_calls,
        consensus_scores=consensus_scores,
        pruned_graph=pruned_graph,
    )


# ---------------------------------------------------------------------------
# Full pruner stage — all catalyst cases
# ---------------------------------------------------------------------------

def run_pruner(
    enum_data: dict,
    model: str,
    threshold: float,
    n_runs: int,
    max_denticity: int,
) -> list[dict]:
    """Run the pruner on every catalyst case in enum_data.

    Reaction-critical intermediates (ReactionDefinition.product_surface_labels)
    are force-kept regardless of LLM score, since discarding them silently
    kills all valid pathways downstream.

    Args:
        enum_data: parsed eval_enumerator_*.json
        model: LLM model name
        threshold: minimum consensus score (0-10) to keep an intermediate
        n_runs: LLM voting rounds per chunk
        max_denticity: maximum '*' anchors allowed

    Returns:
        List of JSON-serializable per-catalyst records: context,
        starting_adsorbate, kept/discarded intermediates with scores, the
        enumerator graph/edges (carried through for downstream reconstruction).
    """
    records: list[dict] = []

    for entry in enum_data["results"]:
        ctx_dict = entry["context"]
        ctx      = catalyst_context_from_dict(ctx_dict)
        enumerator_output = reconstruct_enumerator_output(entry, ctx)
        intermediates = enumerator_output.intermediates

        print(f"\n{'='*60}")
        print(f"{ctx.composition}({ctx.facet})  {ctx.reaction}  "
              f"{len(intermediates)} intermediates in")
        print(f"{'='*60}")

        rxn_def = REACTION_TEMPLATES.get(ctx.reaction)
        force_keep = set(rxn_def.product_surface_labels) if rxn_def else set()

        output = prune(
            enumerator_output=enumerator_output,
            context=ctx,
            threshold=threshold,
            n_runs=n_runs,
            model=model,
            max_denticity=max_denticity,
            force_keep_labels=force_keep,
        )

        print(f"\nkept      : {len(output.kept)}")
        print(f"discarded : {len(output.discarded)}")
        print(f"threshold : {output.threshold}")
        print(f"llm calls : {output.n_llm_calls}")

        print("\nKept intermediates (by score):")
        for s in sorted(output.kept, key=lambda x: -x.score):
            print(f"  {s.score:4.1f}  {s.intermediate.label:35s}  {s.reasoning}")

        records.append({
            "context":            ctx_dict,
            "starting_adsorbate": entry["starting_adsorbate"],
            "threshold":          output.threshold,
            "n_runs":             output.n_runs,
            "kept": [
                {
                    "label":                  s.intermediate.label,
                    "score":                  s.score,
                    "depth":                  s.intermediate.depth,
                    "rule":                   s.intermediate.rule,
                    "n_electrons_cumulative": s.intermediate.n_electrons_cumulative,
                    "reasoning":              s.reasoning,
                }
                for s in output.kept
            ],
            "discarded": [
                {
                    "label":     s.intermediate.label,
                    "score":     s.score,
                    "reasoning": s.reasoning,
                }
                for s in output.discarded
            ],
            "consensus_scores": output.consensus_scores,
            "graph":            enumerator_output.graph,
            "edges": [
                {
                    "parent":      e.parent,
                    "label":       e.label,
                    "rule":        e.rule,
                    "n_electrons": e.n_electrons,
                }
                for e in enumerator_output.edges
            ],
        })

    return records
