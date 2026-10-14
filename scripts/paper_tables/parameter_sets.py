"""The parameter sets used in the paper.

This is the frozen list. The table generator, the test vectors
and the toy example take their sets from here, so the paper
uses one list everywhere.

Every set uses n = 16 and a binary tree, and gives 128 bits for one use.
Each set's shape (h, w and the rule for S) comes from the search in
search_front.py and confirm_rows.py. Its L is the smallest for that shape
whose proven two-use security, the bits a key keeps after signing two
messages, reaches the set's target: the lambda_2 lower bound from
lambda2_bound.py, plus the grinding bits g = r - log2 M for CUTS+C. The
proofs are in lambda2_bounds.json, next to this file.

CUTS+C sets use a 16-bit counter and the widest index width r that counter
allows (widest_index_bits).
"""

from __future__ import annotations

from dataclasses import dataclass

from cuts import CUTS, CUTSParams
from cuts.scheme import widest_index_bits

N_BYTES = 16
COUNTER_BITS = 16


@dataclass(frozen=True)
class ParameterSet:
    name: str  # stable label for tables and test vectors
    target_bits: int  # two-use target the set was confirmed against
    role: str  # why the set is in the paper
    params: CUTSParams

    @property
    def scheme(self) -> str:
        return "CUTS+C" if self.params.counter_bits else "CUTS"


def cuts(h: int, w: int, L: int, S: int) -> CUTSParams:
    return CUTSParams(N_BYTES, h, w, L, S)


def cuts_c(h: int, w: int, L: int, S: int) -> CUTSParams:
    M = CUTS(cuts(h, w, L, S)).M
    r = widest_index_bits(M, COUNTER_BITS)
    return CUTSParams(N_BYTES, h, w, L, S, r, COUNTER_BITS)


PARAMETER_SETS: tuple[ParameterSet, ...] = (
    # Near one-time size: WOTS+C's signature is 530 B.
    ParameterSet(
        "cuts-c-48-cheap",
        48,
        "same size as WOTS+C, cheap keygen",
        cuts_c(10, 5, 32, 64),
    ),
    ParameterSet(
        "cuts-c-48-smallest",
        48,
        "smallest signature overall",
        cuts_c(14, 5, 31, 62),
    ),
    # 80 bits after two signatures.
    ParameterSet(
        "cuts-c-80-smallest",
        80,
        "smallest signature at 80 bits",
        cuts_c(15, 5, 47, 118),
    ),
    ParameterSet(
        "cuts-c-80-cheap",
        80,
        "smallest at 80 bits with cheap keygen",
        cuts_c(11, 4, 55, 103),
    ),
    ParameterSet(
        "cuts-c-80-fastest",
        80,
        "lowest verification at 80 bits",
        cuts_c(15, 1, 73, 0),
    ),
    # 128 bits after two signatures.
    ParameterSet(
        "cuts-c-128-smallest",
        128,
        "smallest signature at 128 bits",
        cuts_c(15, 5, 81, 162),
    ),
    ParameterSet(
        "cuts-c-128-cheap",
        128,
        "smallest at 128 bits with cheap keygen",
        cuts_c(11, 4, 92, 138),
    ),
    ParameterSet(
        "cuts-c-128-fastest",
        128,
        "lowest verification at 128 bits",
        cuts_c(15, 1, 123, 0),
    ),
    ParameterSet(
        "cuts-128-smallest",
        128,
        "smallest at 128 bits without a counter",
        cuts(15, 5, 88, 176),
    ),
    ParameterSet(
        "cuts-128-fastest",
        128,
        "lowest verification at 128 bits without a counter",
        cuts(15, 1, 132, 0),
    ),
)

BY_NAME = {p.name: p for p in PARAMETER_SETS}
