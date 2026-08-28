"""Analyse the meta-reasoner prompt-leak ablation.

Reports, per system:
  * per-rule inclusion frequency in each arm + Fisher exact p (uncorrected)
  * within-arm instability (distinct rulesets over n repeats) -- the reference
    scale against which any between-arm difference must be judged
  * suggested_depth distribution
"""

import json
import sys
from collections import Counter

from scipy.stats import fisher_exact

PATH = sys.argv[1] if len(sys.argv) > 1 else "ablation_raw_n15.json"
d = json.load(open(PATH))
results, N = d["results"], d["n_runs"]

print("=" * 84)
print(f"META-REASONER PROMPT-LEAK ABLATION — {d['model']}, n={N} per arm")
print("  leak  = template as shipped (parenthetical names OER/HER ladders + CO2RR product)")
print("  clean = same template, parenthetical removed, nothing else changed")
print("=" * 84)

all_p = []
for key, arms in results.items():
    lk = [r for r in arms.get("leak", []) if "error" not in r]
    cl = [r for r in arms.get("clean", []) if "error" not in r]
    print(f"\n### {key}   (leak n={len(lk)}, clean n={len(cl)})")

    rules = sorted({r for rec in lk + cl for r in rec["rules"]})
    print(f"  {'rule':22s} {'leak':>9s} {'clean':>9s} {'p':>8s}")
    for rule in rules:
        a = sum(rule in r["rules"] for r in lk)
        b = sum(rule in r["rules"] for r in cl)
        _, p = fisher_exact([[a, len(lk) - a], [b, len(cl) - b]])
        all_p.append((key, rule, p))
        flag = "  <--" if p < 0.05 else ""
        print(f"  {rule:22s} {a:4d}/{len(lk):<4d} {b:4d}/{len(cl):<4d} {p:8.3f}{flag}")

    lk_sets = Counter(tuple(r["rules"]) for r in lk)
    cl_sets = Counter(tuple(r["rules"]) for r in cl)
    print(f"  within-arm instability: leak {len(lk_sets)} distinct rulesets in {len(lk)} runs"
          f"  |  clean {len(cl_sets)} in {len(cl)}")
    print(f"  modal ruleset reproduced: leak {lk_sets.most_common(1)[0][1]}/{len(lk)}"
          f"  |  clean {cl_sets.most_common(1)[0][1]}/{len(cl)}")
    ld = [r["depth"] for r in lk]
    cd = [r["depth"] for r in cl]
    print(f"  suggested_depth: leak {dict(Counter(ld))} mean {sum(ld)/len(ld):.2f}"
          f"  |  clean {dict(Counter(cd))} mean {sum(cd)/len(cd):.2f}")

print("\n" + "=" * 84)
sig = [x for x in all_p if x[2] < 0.05]
print(f"{len(sig)} of {len(all_p)} rule-level tests significant at uncorrected p<0.05")
if sig:
    for k, r, p in sig:
        print(f"   {k:22s} {r:22s} p={p:.4f}")
bonf = 0.05 / len(all_p)
print(f"Bonferroni threshold for {len(all_p)} tests: p<{bonf:.5f} — "
      f"{sum(1 for x in all_p if x[2] < bonf)} survive")
