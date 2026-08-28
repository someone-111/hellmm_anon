<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo/HELLMM_no_background.png">
    <img src="logo/HELLMM_white_background.png" width="240" alt="HELLMM logo">
  </picture>
</p>

# HELLMM — anonymous artifact

Code and per-run data for the submission *HELLMM: Heterogeneous Electrocatalysis
with LLM-driven Mechanism Elucidation*.

> **Status.** This repository accompanies a paper under review at the NeurIPS
> 2026 Workshop on Agentic Systems for Molecular Sciences. Author information is
> withheld for double-blind review and will be added on acceptance.

HELLMM is a seven-stage pipeline for electrocatalytic mechanism elucidation.
Three stages call a language model (rule selection, rule application, validity
scoring); four are deterministic (pathway construction, adsorption energies via
a universal MLIP, computational hydrogen electrode, ranking). Every quantitative
result is produced by the deterministic half.

---

## Verify the paper's claims without a GPU or an API key

Both commands read only files already in this repository and exit non-zero on
any mismatch. Neither calls a model or a potential.

```bash
python export_runs.py --check    # rebuild both CSVs from raw run outputs and diff
python verify_claims.py          # re-derive the five central claims from step records
```

`export_runs.py --check` walks `runs/*/eval_che_*.json`, reconstructs
`results_per_run.csv` and `steps_per_run.csv`, and compares them against the
copies committed here. `verify_claims.py` reads *only* `steps_per_run.csv` and
re-derives each claim from the step records, printing its derivation rather than
a verdict. It re-implements the admissibility test from scratch instead of
trusting the `is_best` flag, because that flag marks the best *candidate* even
when every candidate was rejected.

Requires Python 3.10+ and `scipy` (for the ablation analysis only).

## Method summary

**Systems.** HER on Pt(111); ORR on Pt(111) and Au(111); CO₂ reduction to CO on
Au(110). Slabs are cut from Materials Project bulk structures (lowest-energy
polymorph) with the fairchem slab generator.

**Energetics.** Adsorption energies from the `esen-sm-conserving-all-oc25`
universal MLIP, relaxing every candidate placement per species. Free energies
follow the computational hydrogen electrode, ΔG = ΔE + ΔZPE − TΔS − n_e·eU on
the RHE scale at T = 298.15 K, with the water reference μ_H = ½E(H₂),
μ_O = E(H₂O) − E(H₂), μ_C = E(CO) − μ_O. Adsorbate corrections use the harmonic
approximation over adsorbate-only modes; gas references use ideal-gas
thermochemistry with explicit geometry and rotational symmetry numbers.
U_L = −max(ΔG°/|n_e|) over electrochemical steps; η = U_ideal − U_L for a
reduction. **U_L and η are volts; ΔG is eV.**

**Structure resolution** proceeds in three tiers: the fairchem adsorbate
database, a 50-entry curated SMILES table, then an LLM fallback whose output
must parse in RDKit and reproduce the label's exact stoichiometry or the species
is skipped.

**Repeat sampling within a run.** The enumerator expands the graph three times
and keeps species appearing in at least two; the pruner scores each intermediate
in five calls and keeps those whose mean exceeds 3.0/10. BFS depth is capped at
6.

**Deterministic gates.** Every transformation is checked for mass balance and
electron-count consistency before entering the graph. Pathways must match the
reaction's total electron count exactly, must not mix anodic and cathodic steps
or run counter to the reaction direction, and must carry no chemical step above
2.0 eV. There is no LLM-judge stage.

## Data files

| file | contents |
|---|---|
| `results_per_run.csv` | one row per **attempted** (arm, run, system) cell — 92 rows — with a `status` column (`ok`, `infeasible`, `no_pathway`, `incomplete_energies`, `run_failed`). Non-`ok` rows carry no energies; filter on `status == 'ok'` before averaging. |
| `steps_per_run.csv` | every candidate pathway CHE scored, per step: ΔE, ΔZPE, −TΔS, ΔG°, electron count, rule. 2253 rows over 412 pathways. |
| `runs/<timestamp>_<id>/` | raw stage outputs for the 23 pipeline runs: enumerator (including every prompt and model response), pathway constructor, and CHE. |
| `ablation_prompt_leak/` | raw responses and analysis scripts for the meta-reasoner prompt-leak ablation (n = 15 per arm). |

Two traps worth knowing when working with `steps_per_run.csv`:

- **`pathway` is not a unique key.** Two candidates can traverse the same species
  by different edges (`*CO → *` as `desorption` in one, `bond_dissociation` in
  another); 59 of the 412 candidates collide this way. Group on
  `(arm, run, catalyst, pathway_idx)`.
- **`is_best` is not "the reported result."** See above.

`run` identifiers are scheduler job IDs, retained so every row traces to its
directory under `runs/`.

## Reproducing a run from scratch

Requires a GPU, an OpenRouter-compatible API key in `.env`
(`OPENROUTER_API_KEY=...`), and a Materials Project key (`MY_MP_API_KEY=...`)
for bulk structures and the Pourbaix stability gate.

```bash
pip install -r requirements.txt
python eval_meta_reasoner.py       # stage 1  (LLM)
python eval_enumerator.py          # stage 2  (LLM)
python eval_pruner.py              # stage 3  (LLM)
python eval_pathway_constructor.py # stage 4
python eval_adsorption_energy.py   # stage 5  (MLIP)
python eval_che.py                 # stage 6
python eval_ranker.py              # stage 7
```

Systems, models and thresholds are configured in `config.py`. The stages are
independent scripts reading each other's JSON output, so each can be re-run
without repeating the ones before it. Batch execution on a scheduler is left to
the user; the pipeline itself has no scheduler dependency.

Runs are stochastic by design — the paper's subject is how much the result moves
across repeats — so a fresh run will not reproduce a specific row. The
verification commands above are the reproducible path.

## Layout

```
hellmm/                  pipeline package
  meta_reasoner.py       stage 1 — transformation-rule selection      (LLM)
  enumerator.py          stage 2 — rule application, BFS graph        (LLM)
  pruner.py              stage 3 — chemical-validity scoring          (LLM)
  pathway_constructor.py stage 4 — DFS on exact electron count
  adsorption_energy.py   stage 5 — MLIP relaxation
  che.py                 stage 6 — free energies, onset potential
  ranker.py              stage 7 — cross-catalyst ranking
  reaction.py            reaction definitions (withheld from all prompts)
  rules.py               13 transformation rules
  pourbaix.py            aqueous stability gate
  tools/                 structure building, vibrations, MLIP loading
export_runs.py           rebuild the CSVs from run outputs
verify_claims.py         independent re-derivation of the five claims
config.py                systems, model, thresholds
```

`hellmm/reaction.py` holds the eleven per-reaction constraints — total electron
count, equilibrium potential, terminating labels, direction. None is rendered
into any prompt; the model must produce a mechanism satisfying constraints it
cannot read, and pathways are checked against them afterwards. The prompt
templates are module-level constants in the three LLM stages and can be printed
directly:

```python
import hellmm.meta_reasoner as mr, hellmm.enumerator as en, hellmm.pruner as pr
for m in (mr, en, pr):
    print(m.SYSTEM_PROMPT, m.USER_PROMPT_TEMPLATE)
```

## License

MIT.
