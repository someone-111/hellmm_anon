import pytest
from hellmm import select_ruleset, CatalystContext


def test_oer_ruleset():
    # OER on RuO2 → deprotonation, hydroxylation, bond_dissociation, desorption
    # NOT cc_coupling, NOT any perovskite rules
    ...


def test_her_ruleset():
    # HER on Pt(111) → only protonation, deprotonation, desorption
    ...


def test_perovskite_ruleset():
    # OER on SrRuO3 → includes lattice_o_release, vacancy_healing, cation_oxidation
    ...


def test_metal_no_perovskite_rules():
    # Cu(111) CO2RR → no perovskite rules
    ...


def test_output_schema():
    # output always has selected_rules, reasoning, catalyst_class
    ...


def test_consistency():
    # same context 5×, selected_rules overlap ≥ 80%
    ...
