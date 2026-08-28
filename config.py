"""Central configuration for the hellmm pipeline.

Edit this file to change:
  - Which catalysts and reactions to study (CASES)
  - Which LLM and MLIP to use
  - Enumerator / pruner hyperparameters

All eval scripts read from here — changing one value updates the full pipeline.

CASES format
------------
Each entry is a dict with:
  start            : starting adsorbate label ("*", "*V_O", "*N2", "*CO2", ...)
  composition      : catalyst composition string (e.g. "NiOOH", "Fe", "CoOOH")
  facet            : Miller index string (e.g. "010", "100", "012")
  reaction         : reaction name — must match a key in REACTION_TEMPLATES
  operating_points : list of (pH, U_vs_RHE) tuples

The LLM and MLIP stages (meta_reasoner → enumerator → pruner → pathway_constructor
→ adsorption_energy) run ONCE per unique (composition, facet, reaction, start) —
they are pH/U-independent.  CHE and the ranker run once per operating point,
reusing the shared adsorption energies.  Use multiple operating_points entries
to evaluate one catalyst at several (pH, U) without redundant MLIP computation.

Pourbaix stability is checked per operating point inside eval_enumerator.py —
unstable (pH, U) pairs are filtered out before CHE runs.
"""

import os as _os

from hellmm.meta_reasoner import CatalystContext

# Output directory for all pipeline artifacts (JSONs, logs, trajectories).
# Override with HELLMM_RUN_DIR env var — submit_eval.sh sets this to
# runs/YYYYMMDD_HHMMSS so each SLURM job gets its own folder.
# Local runs default to runs/latest (always overwritten, never clutters root).
RUN_DIR: str = _os.environ.get("HELLMM_RUN_DIR", "runs/latest")
_os.makedirs(RUN_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Model arms for the reproducibility study.  Uncomment one.
#
# Benchmarked on one real enumerator call each before committing to a batch —
# the check a deepseek-v4-pro batch skipped, at the cost of 7 of 10 jobs killed
# at walltime with six still inside the enumerator:
#
#   model             s/call   out tok   $/1k calls   vendor
#   deepseek-v3          4.5       124         0.22   DeepSeek
#   deepseek-v3.2        5.8       150         0.16   DeepSeek
#   kimi-k2              4.7       165         0.61   Moonshot
#   qwen3-235b          92.3      6750        12.47   Qwen  (reasoning — avoid)
#   deepseek-v4-pro        —         —            —   DeepSeek (9.5-12 h/run)
#   gemini-2.5-flash       —         —            —   registered slug 400s
#
# Cross-vendor is deliberate for the second arm: two models from the same family
# agreeing says little about whether a failure mode is structural.

# Sustained throughput is not what a single benchmarked call suggests: kimi-k2
# measured 4.7 s/call in isolation but 26.2 s/call under load (OpenRouter
# throttling), which one job survives and ten concurrent likely would not.
# DeepSeek endpoints are the only ones demonstrated to sustain this pipeline at
# 10 concurrent jobs, by the primary arm itself.
#
# LLM_MODEL  = "deepseek-v3"       # primary arm, n=10, jobs 10766639-10766649
# LLM_MODEL  = "deepseek-v4-pro"   # abandoned: 9.5-12 h/run, 7/10 killed at walltime; n=3 salvaged
# LLM_MODEL  = "kimi-k2"           # cross-vendor, preferred scientifically; throttled under load
LLM_MODEL  = "deepseek-v3.2"       # second arm — different model, same family
MLIP_MODEL = "esen-sm-conserving-all-oc25"

# ---------------------------------------------------------------------------
# Enumerator
# ---------------------------------------------------------------------------

MAX_DEPTH         = 6   # safety cap — overrides meta_reasoner if it suggests more; NRR needs ≥6
ENUMERATOR_N_RUNS = 3   # LLM voting rounds (1 = fast/debug, 3 = confidence filter enabled)

# ---------------------------------------------------------------------------
# Pruner
# ---------------------------------------------------------------------------

PRUNER_THRESHOLD = 3.0   # minimum consensus score (0–10) to keep an intermediate
PRUNER_N_RUNS    = 5     # LLM voting rounds; needs ≥ ceil(n/2)=3 successes per chunk
MAX_DENTICITY    = 2     # 1 = monodentate only; 2 = include bidentate (*CO2*, *N2*, etc.)

# ---------------------------------------------------------------------------
# CHE / vibrations
# ---------------------------------------------------------------------------

#TEMPERATURE = 673.15   # K — 400°C, matching Fe NRR experimental conditions
TEMPERATURE = 298.15   # K — room temperature, for OER/HER/CO2RR (PSI CoOOH benchmark)

# Maximum ΔG (eV) allowed for chemical steps (n_e = 0) in a pathway.
# Chemical steps are potential-independent; ΔG > this threshold means the
# step is thermodynamically blocked regardless of electrode potential U,
# so the pathway has no valid U_onset and is excluded from ranking.
# 0.0 eV = strict CHE thermodynamics (recommended for publication).
CHE_CHEM_STEP_MAX_DG = 2.0   # eV

# ---------------------------------------------------------------------------
# Catalyst cases
# ---------------------------------------------------------------------------

CASES: list[dict] = [
    # Colleague-provided benchmark cases — Pt(111), TEMPERATURE=298.15 K
    # Easy: HER on Pt(111) — https://doi.org/10.1039/d1cp04134g
    # Onset ~0 V vs RHE (theoretically exact 0; Pt is measurement-limited).
    # Mechanism: Volmer step (* + H+ + e- -> *H) then Heyrovsky step (RDS).
    {
        "start":      "*",
        "composition": "Pt",
        "facet":      "111",
        "reaction":   "HER",
        "operating_points": [(0.0, 0.0)],
    },
    # Moderate: ORR on Pt(111) — https://doi.org/10.1021/jp047349j
    # Onset 0.78 V vs SHE = 0.78 V vs RHE at pH 0.
    # Mechanism is coverage-dependent.  At low O2 coverage — i.e. the initial
    # state — the reference gives the *dissociation* path:
    #   1/2 O2 + * -> *O ;  *O + (H+ + e-) -> *OH ;  *OH + (H+ + e-) -> * + H2O
    # At high O2 coverage (and on Au at any coverage) the *association* path
    # takes over: *OO -> *OOH -> *O + H2O -> *OH -> * + H2O.
    #
    # This case runs the association path only.  The dissociation path cannot
    # currently be expressed: it would need start="*O", but it closes in two
    # electrons per site, while REACTION_TEMPLATES["ORR"] sets
    # n_electrons_total=4 and pathway_constructor keeps only pathways matching
    # that total exactly, so every dissociative pathway would be discarded.
    # Running it would mean a separate 2-electron half-cycle definition, not a
    # config change.  HELLMM has no coverage model either way, so it cannot
    # predict which regime applies — flag this if the discovered pathway or
    # energy does not cleanly match one mechanism.
    # start="*OO", not "*": the rule set has `desorption` (adsorbate -> gas) but
    # no inverse adsorption rule, so a gas-phase *reactant* cannot enter the
    # graph on its own.  OER hides this (its reactant is H2O, supplied by
    # `hydroxylation`), but with start="*" the ORR graph had no route to *OO at
    # all and wandered into H adsorption (HER chemistry) instead.  Same
    # convention as NRR (start="*N2") and CO2RR (start="*CO2"): a reduction
    # consuming a gas molecule begins from the pre-adsorbed reactant.
    {
        "start":      "*OO",
        "composition": "Pt",
        "facet":      "111",
        "reaction":   "ORR",
        "operating_points": [(0.0, 0.78)],
    },
    # Moderate: ORR on Au(111) — https://doi.org/10.1021/jp047349j
    # Same reference as the Pt(111) case above, which treats both metals.  The
    # reference assigns Au the *association* mechanism regardless of coverage:
    #   O2 + * -> *OO -> *OOH -> *O + H2O -> *OH -> * + H2O
    # (on Pt the association path takes over only at high O2 coverage).  This is
    # the discriminating test of the pair: the same reaction, the same 4-electron
    # associative sequence, and the same start label on two metals whose O
    # binding differs strongly — Au is the weak binder, so the potential-limiting
    # step is expected to move to the first reduction (*OO -> *OOH) rather than
    # the *O/*OH end where it sits on Pt.  If HELLMM reproduces that shift it is
    # evidence the pipeline is tracking binding-energy trends and not just
    # recalling one textbook mechanism.
    #
    # Evaluated at the same (pH, U) as Pt(111) so the two are directly
    # comparable.  0.78 V is Pt's onset, not Au's — Au's is well below it — but
    # U_onset and the overpotential are computed from ΔG° and are U-independent
    # (see che.py), so this choice affects only the per-step ΔG printed in the
    # log and the Pourbaix stability gate.  Add a second operating point if an
    # Au-specific reporting potential is wanted; CHE reruns cheaply and the MLIP
    # stage is not repeated.
    #
    # Caveat to check before publishing this case: in acid, Au(111) is largely a
    # 2-electron ORR catalyst, reducing O2 to H2O2 rather than to water, and the
    # 4-electron route is mainly an alkaline (and facet-sensitive) phenomenon.
    # REACTION_TEMPLATES["ORR"] hardcodes n_electrons_total=4 to H2O, following
    # the reference's thermodynamic treatment.  A poor result here may therefore
    # be the reaction definition rather than the pipeline.
    {
        "start":      "*OO",
        "composition": "Au",
        "facet":      "111",
        "reaction":   "ORR",
        "operating_points": [(0.0, 0.78)],
    },
    # Moderate: CO2RR on Au(110) — https://doi.org/10.1021/acscatal.8b04852
    # Onset ~0.1 V vs RHE (close to Au(111) at low overpotential; Au(110)
    # pulls ahead above ~0.35 V). Path: CO2 -> *COOH -> *CO -> CO(g); RDS not
    # firmly established in the reference but likely *COOH formation.
    # U_ideal for CO2RR is already 0.10 V (see reaction.py) — this is close to
    # a limiting-case test: the reported onset is nearly the thermodynamic
    # ideal, i.e. Au(110) is reported as a near-minimal-overpotential catalyst
    # for this reaction.
    {
        "start":      "*CO2",
        "composition": "Au",
        "facet":      "110",
        "reaction":   "CO2RR",
        "operating_points": [(7.0, 0.10)],
    },
]

# ---------------------------------------------------------------------------
# Helper functions for eval scripts
# ---------------------------------------------------------------------------

def _make_ctx(case: dict, ph: float, u: float) -> CatalystContext:
    return CatalystContext(
        composition=case["composition"],
        facet=case["facet"],
        reaction=case["reaction"],
        pH=ph,
        U=u,
    )


def unique_catalyst_cases() -> list[tuple[str, CatalystContext]]:
    """One (start, CatalystContext) per unique (composition, facet, reaction, start).

    Used by LLM and MLIP stages (meta_reasoner → adsorption_energy) that are
    pH/U-independent.  pH/U are taken from the first operating point as a
    placeholder — they are not used by rule selection or intermediate enumeration.
    """
    seen: set = set()
    result: list[tuple[str, CatalystContext]] = []
    for case in CASES:
        key = (case["composition"], case["facet"], case["reaction"], case["start"])
        if key not in seen:
            seen.add(key)
            ph, u = case["operating_points"][0]
            result.append((case["start"], _make_ctx(case, ph, u)))
    return result


def operating_points_for(
    composition: str, facet: str, reaction: str, start: str
) -> list[tuple[float, float]]:
    """Return the configured (pH, U) operating points for a given catalyst."""
    for case in CASES:
        if (case["composition"] == composition
                and case["facet"] == facet
                and case["reaction"] == reaction
                and case["start"] == start):
            return list(case["operating_points"])
    return []
