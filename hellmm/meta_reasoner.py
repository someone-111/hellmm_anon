import warnings
from dataclasses import asdict
from pydantic.dataclasses import dataclass

from .llm import call_llm
from .rules import ALL_RULES, TransformationRule

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert electrochemical surface chemist. Your task is to select which \
transformation rules are physically relevant for a given heterogeneous catalyst and reaction.

SELECTION CRITERIA — include a rule only if ALL of the following hold:
1. It can occur as an elementary mechanistic step under the given potential and pH
2. It involves species actually present at the interface (e.g. do not invoke C-C coupling \
if no carbon intermediates exist)
3. It is consistent with the catalyst material class (metal, oxide, perovskite)
4. It is directionally consistent with the reaction (oxidative reaction → favour oxidative \
steps; reductive reaction → favour reductive steps)

GOOD REASONING EXAMPLES:
- "desorption: product gases must leave the surface in every catalytic cycle → include"
- "cc_coupling: no C-containing intermediates in HER → exclude"
- "lattice_o_release: catalyst is a pure metal, no lattice oxygen → exclude"

BAD REASONING EXAMPLES (do not reason this way):
- "deprotonation: this is a HER step, not relevant here" \
— deprotonation is a general PCET step, not HER-specific
- "hydroxylation: not the main pathway" \
— vague; evaluate whether it can occur, not whether it is dominant
"""

USER_PROMPT_TEMPLATE = """\
Catalyst: {composition} ({facet})
Reaction: {reaction}
pH: {pH}
Potential U vs RHE: {U} V

Available transformation rules:
{rule_list}

Go through each rule one by one and decide if it passes all four selection criteria above.

Return a JSON object with these keys in order:
- "chain_of_thought": string — your step-by-step evaluation of each rule
- "selected_rules": list of rule name strings for all rules that pass
- "reasoning": dict mapping each selected rule name to one sentence justification
- "catalyst_class": one of "metal", "oxide", or "perovskite"
- "suggested_depth": integer — minimum BFS depth needed to reach the final product \
(e.g. OER: 3 steps * → *OH → *O → *OOH; HER: 2 steps * → *H → H₂; CO2RR to CO: 2 steps)
"""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CatalystContext:
    composition: str    # e.g. "RuO2", "Cu", "SrRuO3"
    facet: str          # e.g. "110", "111"
    reaction: str       # e.g. "OER", "CO2RR", "HER"
    pH: float
    U: float            # V vs RHE


@dataclass
class RulesetSelection:
    selected_rules: list[TransformationRule]
    reasoning: dict[str, str]   # rule_name → explanation
    catalyst_class: str         # "metal", "oxide", "perovskite"
    chain_of_thought: str = ""
    suggested_depth: int = 3    # BFS depth cap suggested by meta_reasoner


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_ruleset(
    context: CatalystContext,
    model: str = "claude-sonnet-4-20250514",
) -> RulesetSelection:
    known = {r.name: r for r in ALL_RULES}
    rule_list = "\n".join(
        f"- {r.name}: {r.description} [type={r.type}, n_electrons={r.n_electrons}]"
        for r in ALL_RULES
    )

    raw = call_llm(
        system=SYSTEM_PROMPT,
        user=USER_PROMPT_TEMPLATE.format(
            composition=context.composition,
            facet=context.facet,
            reaction=context.reaction,
            pH=context.pH,
            U=context.U,
            rule_list=rule_list,
        ),
        model=model,
    )

    # Some models answer this prompt with just the rule list rather than the
    # full object — kimi-k2 returned ['protonation', 'bond_dissociation',
    # 'desorption'] for one case out of four, having returned the object for the
    # other three.  The list carries the only field that changes what the
    # pipeline does: `selected_rules`.  `chain_of_thought` is deliberately unused
    # downstream (it leaked mechanism into the enumerator, see enumerator.py),
    # `reasoning` is documentation, and `suggested_depth` has a defined default
    # and is clamped by config in any case.  So accept the list, warn, and let
    # the defaults fill in — refusing would discard a usable answer over
    # formatting, and this failure is not specific to one model.
    if isinstance(raw, list):
        warnings.warn(
            f"meta_reasoner returned a bare list for {context.composition}"
            f"({context.facet}) {context.reaction} instead of the full object: "
            f"{raw}. Treating it as selected_rules; catalyst_class and "
            "suggested_depth fall back to defaults."
        )
        raw = {"selected_rules": raw}
    elif not isinstance(raw, dict):
        raise ValueError(f"Expected dict from meta_reasoner LLM, got {type(raw)}: {raw}")
    selected_names: list[str] = raw.get("selected_rules", [])
    selection = RulesetSelection(
        selected_rules=[known[n] for n in selected_names if n in known],
        reasoning={k: v for k, v in raw.get("reasoning", {}).items() if k in known},
        catalyst_class=raw.get("catalyst_class", "unknown"),
        chain_of_thought=str(raw.get("chain_of_thought", "")),
        suggested_depth=int(raw.get("suggested_depth", 3)),
    )
    _validate(selection, context)
    return selection


def _validate(selection: RulesetSelection, context: CatalystContext) -> None:
    names = {r.name for r in selection.selected_rules}
    reaction = context.reaction.upper()

    if not names:
        warnings.warn("No rules selected — check the LLM response.")

    if reaction == "OER":
        missing = {"deprotonation", "hydroxylation"} - names
        if missing:
            warnings.warn(f"OER system missing expected rules: {missing}")
        if "cc_coupling" in names:
            warnings.warn("cc_coupling selected for OER — unexpected.")

    # Relaxed constraint: perovskite rules can apply to oxides (e.g. lattice_o_release on RuO2)
    # Only warn if B-site cation rules selected on a pure metal
    if selection.catalyst_class == "metal" and names & {"cation_oxidation", "cation_reduction", "vacancy_healing"}:
        warnings.warn(
            f"B-site cation rules selected for pure metal catalyst: "
            f"{names & {'cation_oxidation', 'cation_reduction', 'vacancy_healing'}}"
        )


# ---------------------------------------------------------------------------
# Full meta-reasoner stage — all catalyst cases
# ---------------------------------------------------------------------------

def run_meta_reasoner(
    catalyst_cases: list[tuple[str, CatalystContext]],
    model: str,
) -> list[dict]:
    """Run rule selection for every catalyst case.

    Args:
        catalyst_cases: one (starting_adsorbate, CatalystContext) per unique
            catalyst — e.g. config.unique_catalyst_cases(). The starting
            adsorbate is unused here (rule selection is pH/U/start-independent)
            but kept for a consistent call shape with run_enumerator().
        model: LLM model name

    Returns:
        List of JSON-serializable per-catalyst records: context, catalyst
        class, suggested BFS depth, selected rules, and reasoning.
    """
    records: list[dict] = []

    for _, ctx in catalyst_cases:
        print(f"\n{'='*60}")
        print(f"{ctx.composition}({ctx.facet})  {ctx.reaction}  pH={ctx.pH}  U={ctx.U}V")
        print(f"{'='*60}")

        print("calling LLM...", flush=True)
        result = select_ruleset(ctx, model=model)

        print(f"catalyst_class : {result.catalyst_class}")
        print(f"suggested_depth: {result.suggested_depth}")
        print(f"selected_rules : {[r.name for r in result.selected_rules]}")
        if result.chain_of_thought:
            print(f"chain_of_thought:\n  {result.chain_of_thought}")
        print("reasoning:")
        for rule, reason in result.reasoning.items():
            print(f"  {rule:25s} {reason}")

        records.append({
            "context": asdict(ctx),
            "catalyst_class": result.catalyst_class,
            "suggested_depth": result.suggested_depth,
            "selected_rules": [r.name for r in result.selected_rules],
            "reasoning": result.reasoning,
            "chain_of_thought": result.chain_of_thought,
        })

    return records
