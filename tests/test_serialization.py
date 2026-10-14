"""The paper's serialization appendix against the reference implementation.

Every hash input is built byte for byte as the appendix's table gives it,
hashed with SHAKE256, and compared with what scheme.py computes, for CUTS and
CUTS+C. The layout is written out here, so this catches the code drifting
from the appendix; an edit to the appendix needs the same edit here.
"""

import hashlib

import pytest

from cuts import CUTS, CUTSParams
from cuts.scheme import widest_index_bits

N = 16
BASE = CUTSParams(N, 3, 3, 4, 4)
PLUS_C = CUTSParams(N, 3, 3, 4, 4, widest_index_bits(CUTS(BASE).M, 16), 16)
SEED = bytes(range(N))
MESSAGE = b"serialization"


def shake(x: bytes, length: int) -> bytes:
    return hashlib.shake_256(x).digest(length)


def enc(a: int) -> bytes:
    return a.to_bytes(8, "big")


def pid(p: CUTSParams) -> bytes:
    """Seven fields; r = s = 0 for CUTS."""
    fields = (p.n, p.h, p.w, p.L, p.S, p.index_bits, p.counter_bits)
    return b"".join(enc(x) for x in fields)


@pytest.mark.parametrize("p", [BASE, PLUS_C], ids=["CUTS", "CUTS+C"])
def test_tree_hashes(p: CUTSParams):
    scheme = CUTS(p)
    x_l, x_r = bytes(N), bytes([1]) * N
    leaf = shake(b"CUTS/LEAF/v0" + pid(p) + enc(9) + SEED, N)
    merge = shake(b"CUTS/MERGE/v0" + pid(p) + enc(5) + x_l + x_r, N)
    chain = shake(b"CUTS/CHAIN/v0" + pid(p) + enc(5) + enc(1) + x_l, N)
    assert scheme._leaf(SEED, 9) == leaf
    assert scheme._merge(5, x_l, x_r) == merge
    assert scheme._advance(5, x_l, 1, 2) == chain


@pytest.mark.parametrize("p", [BASE, PLUS_C], ids=["CUTS", "CUTS+C"])
def test_keys_randomizer_selection_and_signature(p: CUTSParams):
    scheme = CUTS(p)
    sk, pk = scheme.keygen(SEED)
    sig = scheme.sign(sk, MESSAGE)
    assert len(pk) == N and sk.to_bytes() == SEED + pk

    rho = shake(b"CUTS/RAND/v0" + pid(p) + SEED + pk + MESSAGE, N)
    assert sig[:N] == rho

    tail = pk + rho + MESSAGE
    if not p.counter_bits:
        beta = ((scheme.M - 1).bit_length() + 256 + 7) // 8
        bound = (2 ** (8 * beta) // scheme.M) * scheme.M
        c = 0
        while (
            u := int.from_bytes(
                shake(b"CUTS/HMSG/v0" + pid(p) + enc(c) + tail, beta), "big"
            )
        ) >= bound:
            c += 1
        rank, counter_field = u % scheme.M, b""
        assert len(sig) == (p.L + 1) * N
    else:
        s_b, r_b = (p.counter_bits + 7) // 8, (p.index_bits + 7) // 8
        c = 0
        while (
            rank := int.from_bytes(
                shake(b"CUTS/HMSG/v0" + pid(p) + c.to_bytes(s_b, "big") + tail, r_b),
                "big",
            )
            >> (8 * r_b - p.index_bits)
        ) >= scheme.M:
            c += 1
        counter_field = c.to_bytes(s_b, "big")
        assert sig[N : N + s_b] == counter_field
        assert len(sig) == (p.L + 1) * N + s_b

    frontier, positions = scheme.unrank(rank)
    start = N + len(counter_field)
    values = [sig[start + i * N : start + (i + 1) * N] for i in range(p.L)]
    for v, a, z in zip(frontier, positions, values, strict=True):
        assert z == scheme._advance(v, scheme._chain_start(SEED, v), 0, a)
    assert scheme.verify(pk, MESSAGE, sig)


def test_pid_has_a_fixed_length_in_both_modes():
    # A CUTS identifier must not be a prefix of a CUTS+C one, or inputs that
    # end in the message could be read in either mode.
    assert len(BASE.pid) == len(PLUS_C.pid) == 56
    assert BASE.pid[:40] == PLUS_C.pid[:40] and BASE.pid != PLUS_C.pid
