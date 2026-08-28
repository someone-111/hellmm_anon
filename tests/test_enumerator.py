import pytest
from hellmm.enumerator import enumerate_intermediates
from hellmm.meta_reasoner import CatalystContext, RulesetSelection
from hellmm.rules import ALL_RULES


def test_oer_depth1():
    # *OH (hydroxylation) and *H (protonation) must appear; no perovskite intermediates
    ...


def test_oer_depth2():
    # *OH, *O, *OOH must appear (textbook OER set)
    ...


def test_her_depth1():
    # starting from * on Pt(111), depth=1 → must find *H
    ...


def test_co2rr_au_depth2():
    # *COOH and *CO must appear; *OCCO must NOT (Au doesn't dimerize CO)
    ...


def test_graph_structure():
    # every edge A→B has B in intermediates list
    ...


def test_electron_counting():
    # cumulative electron count correctly tracked through depth-2 chain
    ...


def test_union_across_runs():
    # n_runs=3 produces >= intermediates than n_runs=1
    ...


def test_confidence_filter():
    # intermediates appearing in only 1/5 runs are discarded
    ...


def test_safety_cap():
    # >100 intermediates triggers warning without crash
    ...


def test_malformed_labels():
    # products without * anchor are silently discarded
    ...
