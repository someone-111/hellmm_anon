"""Shared chemistry utilities used across pipeline modules.

Kept minimal — only functions that are genuinely used by more than one module.
"""

from __future__ import annotations

import re

# A "plain" adsorbate spelling: element symbols, counts, parentheses and bond
# marks only.  Anything containing "_", "[", "]" or "^" carries site/defect
# semantics (vacancy, lattice oxygen, cation oxidation state) that a chemical
# formula does not capture — see canonicalization_key.
_PLAIN_ADSORBATE_RE = re.compile(r"[A-Za-z0-9()=]*")


def parse_formula(label: str) -> dict[str, int]:
    """Count elements in an adsorbate label string.

    Strips all '*' anchors and vacancy/lattice modifiers, then parses
    element symbols and their counts. Handles parenthesised groups with
    a trailing multiplier, e.g. C(OH)2 → C1O2H2.

    Examples:
        "*OCHOH"   → {"O": 2, "C": 1, "H": 2}
        "*CO*CO"   → {"C": 2, "O": 2}
        "*OOH"     → {"O": 2, "H": 1}
        "*C(OH)2"  → {"C": 1, "O": 2, "H": 2}
        "*C(O)(OH)"→ {"C": 1, "O": 2, "H": 1}
        "*V_O"     → {}   (vacancy, no adsorbate atoms)
        "*[ox]"    → {"O": 1}
    """
    clean = label.replace("*", "").replace("V_", "").replace("_lattice", "")
    clean = clean.replace("[ox]", "O")
    # Strip cation-site annotations produced by the cation_oxidation rule,
    # e.g. "_Ru5" in "*_Ru5OH" (means "OH on 5-coord Ru site", not "5 Ru atoms").
    clean = re.sub(r"_[A-Z][a-z]?\d+", "", clean)
    # Strip cation oxidation-state annotations like "Ru^5" or "Ru^(5)" that the
    # LLM emits for oxide/perovskite surfaces (e.g. "*Ru^5OH" → adsorbate = OH).
    clean = re.sub(r"[A-Z][a-z]?\^\(?\d+\)?", "", clean)

    counts: dict[str, int] = {}

    def _add(formula_str: str, multiplier: int = 1) -> None:
        for elem, n_str in re.findall(r"([A-Z][a-z]?)(\d*)", formula_str):
            if elem:
                counts[elem] = counts.get(elem, 0) + (int(n_str) if n_str else 1) * multiplier

    # Expand parenthesised groups before flat parsing.
    # Repeat until no more groups remain (handles nested-like cases).
    expanded = clean
    for _ in range(5):
        new = re.sub(
            r"\(([^()]+)\)(\d*)",
            lambda m: m.group(1) * (int(m.group(2)) if m.group(2) else 1),
            expanded,
        )
        if new == expanded:
            break
        expanded = new

    _add(expanded)
    return counts


def formula_str(counts: dict[str, int]) -> str:
    """Canonical formula string, e.g. {"C":1,"H":2,"O":2} → 'C1H2O2'."""
    return "".join(f"{e}{n}" for e, n in sorted(counts.items()))


def canonicalization_key(label: str) -> str | None:
    """Formula key for labels whose formula uniquely determines the species.

    The enumerator's LLM writes the same species many different ways —
    adsorbed O2 alone has been observed as ``*O2``, ``*OO``, ``*O=O`` and
    ``*O(O)`` in a single run.  Each spelling becomes a separate graph node,
    which fragments pathways, splits the reproducibility vote across
    variants, and repeats identical MLIP relaxations.

    Merging by formula is only safe where no isomerism is possible, so this
    returns a key **only** when both hold:

      * every heavy (non-H) atom is the same element — heteronuclear pairs
        such as ``*CHO`` (formyl) and ``*COH`` (hydroxycarbyne) share the
        formula CHO but are genuinely different species, and the SMILES
        table already treats them as distinct;
      * there are at most two heavy atoms — beyond that, distinct skeletons
        with the same formula become possible.

    Returns None when merging would not be provably safe, in which case the
    caller must leave the label untouched.

    Examples:
        "*OO"        → key shared with "*O2", "*O=O", "*O(O)"
        "*HOH"       → key shared with "*H2O"
        "*CHO"       → None  (two different heavy elements)
        "*OOOH"      → None  (three heavy atoms)
        "*V_O"       → None  (defect marker — see below)
        "*O_lattice" → None  (lattice-site marker — see below)
    """
    # Defect and lattice-site labels must never be merged.  parse_formula
    # strips the "V_" and "_lattice" markers, so *V_O (an oxygen vacancy),
    # *O_lattice (a healed lattice site, ΔE = 0 by identity in the CHE frame)
    # and *O (an adsorbed O adatom) all reduce to the same O1 formula despite
    # being three physically different states handled by three different code
    # paths.  Only plain adsorbate spellings — element symbols, counts,
    # parentheses and bond marks — are eligible.
    if not _PLAIN_ADSORBATE_RE.fullmatch(label.lstrip("*")):
        return None

    counts = parse_formula(label)
    heavy = {e: n for e, n in counts.items() if e != "H"}
    if len(heavy) != 1:
        return None
    (_elem, n_heavy), = heavy.items()
    if n_heavy > 2:
        return None
    return formula_str(counts)


def matches_species(candidate: str, target: str) -> bool:
    """True if `candidate` names the same surface species as `target`.

    ReactionDefinition hand-writes labels such as "*H2O" and "*NH3", but the
    spelling the LLM produces varies between runs: water was enumerated as
    "*H2O" in one run and "*OH2" in the next.  An exact string comparison then
    silently fails — in that second run the product-desorption edge was never
    injected and *OH2 was never force-kept, so ORR found zero pathways and
    dropped out of the results entirely with no error reported.

    Falls back to the canonicalisation key, which is spelling-independent
    wherever formula provably determines the species.  Where it does not
    (labels with no key, e.g. "*CO" or "*H2"), behaviour is unchanged from a
    plain string comparison and _ALIASES in enumerator.py remains responsible.
    """
    if candidate == target:
        return True
    key = canonicalization_key(target)
    return key is not None and canonicalization_key(candidate) == key
