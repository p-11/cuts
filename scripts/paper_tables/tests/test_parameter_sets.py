import json
from pathlib import Path

import pytest
from paper_tables.parameter_sets import (
    BY_NAME,
    COUNTER_BITS,
    N_BYTES,
    PARAMETER_SETS,
)

from cuts import CUTS
from cuts.scheme import widest_index_bits

HERE = Path(__file__).parents[1]
PROOFS = json.loads((HERE / "lambda2_bounds.json").read_text())
SEARCH = json.loads((HERE / "confirmed_rows.json").read_text())["rows"]


def test_names_are_unique_and_both_schemes_are_present():
    assert len(BY_NAME) == len(PARAMETER_SETS) == 10
    assert {p.scheme for p in PARAMETER_SETS} == {"CUTS", "CUTS+C"}


@pytest.mark.parametrize("p", PARAMETER_SETS, ids=lambda p: p.name)
def test_set_is_valid_and_gives_128_bits_for_one_use(p):
    scheme = CUTS(p.params)  # raises on invalid parameters
    assert p.params.n == N_BYTES
    assert scheme.one_use_security_bits >= 128
    if p.scheme == "CUTS+C":
        assert p.params.counter_bits == COUNTER_BITS
        assert p.params.index_bits == widest_index_bits(scheme.M, COUNTER_BITS)
    else:
        assert p.params.counter_bits == p.params.index_bits == 0


@pytest.mark.parametrize("p", PARAMETER_SETS, ids=lambda p: p.name)
def test_L_is_the_smallest_with_a_proof_reaching_the_target(p):
    proof = PROOFS[p.name]
    x = p.params
    assert (proof["h"], proof["w"], proof["L"], proof["S"]) == (x.h, x.w, x.L, x.S)
    assert proof["target"] == p.target_bits
    assert proof["two_use"] >= p.target_bits
    below = proof["two_use_at_L_minus_1"]  # None: L - 1 is not a valid set
    assert below is None or below < p.target_bits


@pytest.mark.parametrize("p", PARAMETER_SETS, ids=lambda p: p.name)
def test_shape_comes_from_the_search(p):
    x = p.params
    assert any(
        (r["scheme"], r["h"], r["w"], r["target"])
        == (p.scheme, x.h, x.w, p.target_bits)
        for r in SEARCH
    ), f"{p.name}: no confirmed search row with this shape and target"
