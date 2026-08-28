"""Meta-reasoner prompt-leak ablation, n=15 per arm, per-rule frequency analysis.

See ablate_leak.py for the rationale.  The n=5 pilot showed that rule selection
varies substantially between repeats of the SAME arm, so a set-equality test
between arms is uninformative.  This version measures per-rule inclusion
frequency and tests each rule with Fisher's exact test, and separately reports
the within-arm instability as a result in its own right.
"""

import json
import os
import sys
import warnings
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore")

import hellmm.meta_reasoner as mr
from config import unique_catalyst_cases

MODEL = "deepseek-v3"
N_RUNS = 15
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ablation_raw_n15.json")

LEAK_TEMPLATE = mr.USER_PROMPT_TEMPLATE
LEAK_LINE = (
    '- "suggested_depth": integer — minimum BFS depth needed to reach the final product '
    "(e.g. OER: 3 steps * → *OH → *O → *OOH; HER: 2 steps * → *H → H₂; CO2RR to CO: 2 steps)\n"
)
CLEAN_LINE = (
    '- "suggested_depth": integer — minimum BFS depth needed to reach the final product\n'
)
assert LEAK_LINE in LEAK_TEMPLATE
CLEAN_TEMPLATE = LEAK_TEMPLATE.replace(LEAK_LINE, CLEAN_LINE)

cases = unique_catalyst_cases()
results: dict = {}

for arm, template in (("leak", LEAK_TEMPLATE), ("clean", CLEAN_TEMPLATE)):
    mr.USER_PROMPT_TEMPLATE = template
    for _, ctx in cases:
        key = f"{ctx.composition}({ctx.facet}) {ctx.reaction}"
        for run in range(N_RUNS):
            try:
                sel = mr.select_ruleset(ctx, model=MODEL)
                rec = {"rules": sorted(r.name for r in sel.selected_rules),
                       "depth": sel.suggested_depth, "class": sel.catalyst_class}
            except Exception as e:                      # noqa: BLE001
                rec = {"error": f"{type(e).__name__}: {e}"}
            results.setdefault(key, {}).setdefault(arm, []).append(rec)
            print(f"{arm:5s} {key:22s} {run+1:2d}/{N_RUNS}", flush=True)
        with open(OUT, "w") as f:
            json.dump({"model": MODEL, "n_runs": N_RUNS, "results": results}, f, indent=2)

print("\nDONE — raw at", OUT)
