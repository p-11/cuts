"""Checks the CUTS core against the specification and a small worked family."""

import itertools
import math

import pytest

from cuts.scheme import (
    CUTS,
    DST_HMSG,
    GRIND_FAILURE_LOG2_MAX,
    MESSAGE_DSTS,
    CUTSParams,
    SecretKey,
    acceptance_probability,
    decoration_counts,
    frontier_counts,
    grind_failure_log2,
    widest_index_bits,
)

SMALL = CUTSParams(n=16, h=3, w=2, L=3, S=1)

# A spread of shapes: w = 1 (no chains), S at both ends, wide chains.
INSTANCES = [
    SMALL,
    CUTSParams(n=16, h=2, w=1, L=2, S=0),
    CUTSParams(n=16, h=4, w=3, L=5, S=0),
    CUTSParams(n=16, h=4, w=3, L=5, S=10),
    CUTSParams(n=32, h=6, w=5, L=12, S=24),
    CUTSParams(n=16, h=8, w=2, L=20, S=10),
]


def plus_c(n, h, w, L, S, grinding, counter_bits=16):
    """CUTS+C parameters with r = ceil(log2 M) + grinding."""
    M = CUTS(CUTSParams(n, h, w, L, S)).M
    return CUTSParams(n, h, w, L, S, (M - 1).bit_length() + grinding, counter_bits)


# CUTS+C instances: S at the top of its range, and a counter that is not a
# whole number of bytes.
GRIND_INSTANCES = [
    plus_c(16, 3, 2, 3, 1, grinding=2),
    plus_c(16, 4, 3, 5, 10, grinding=4),
    plus_c(16, 6, 5, 12, 32, grinding=6),
    plus_c(16, 8, 4, 20, 36, grinding=4, counter_bits=12),
]
ALL_INSTANCES = INSTANCES + GRIND_INSTANCES


def is_frontier(nodes, h):
    """True when `nodes` meets every root-to-leaf path exactly once."""
    for leaf in range(1 << h, 1 << (h + 1)):
        path = {leaf >> i for i in range(h + 1)}
        if len(path & set(nodes)) != 1:
            return False
    return True


def test_worked_encoding_appendix():
    """The numbers stated in the text of the paper's worked-encoding appendix."""
    scheme = CUTS(CUTSParams(n=16, h=2, w=3, L=3, S=2))
    assert frontier_counts(2, 3)[2][3] == 2
    assert decoration_counts(3, 3, 2)[3][2] == 6
    assert scheme.M == 12
    assert divmod(9, 6) == (1, 3)
    assert scheme.unrank(9) == ((4, 5, 3), (1, 0, 1))
    assert scheme.params.verification_hashes == 8


def test_small_example_ordering():
    """The complete rank table of the small family (h=3, w=2, L=3, S=1)."""
    scheme = CUTS(SMALL)
    assert scheme.M == 6
    expected = [
        ((2, 6, 7), (0, 0, 1)),
        ((2, 6, 7), (0, 1, 0)),
        ((2, 6, 7), (1, 0, 0)),
        ((4, 5, 3), (0, 0, 1)),
        ((4, 5, 3), (0, 1, 0)),
        ((4, 5, 3), (1, 0, 0)),
    ]
    assert [scheme.unrank(r) for r in range(6)] == expected


@pytest.mark.parametrize(
    "params",
    [
        CUTSParams(n=16, h=3, w=3, L=4, S=3),
        CUTSParams(n=16, h=4, w=2, L=6, S=2),
        CUTSParams(n=16, h=3, w=1, L=5, S=0),
    ],
)
def test_unrank_is_a_bijection_onto_the_family(params):
    """Every rank gives a distinct, valid encoding, and M counts them all."""
    scheme = CUTS(params)
    seen = set()
    for r in range(scheme.M):
        frontier, positions = scheme.unrank(r)
        assert len(frontier) == len(positions) == params.L
        assert is_frontier(frontier, params.h)
        assert all(0 <= a < params.w for a in positions)
        assert sum(positions) == params.S
        seen.add((frontier, positions))
    assert len(seen) == scheme.M

    # Cross-check M by brute force over all position vectors.
    decorations = sum(
        1
        for vector in itertools.product(range(params.w), repeat=params.L)
        if sum(vector) == params.S
    )
    assert scheme.M % decorations == 0


@pytest.mark.parametrize("params", ALL_INSTANCES)
def test_sign_verify_roundtrip(params):
    scheme = CUTS(params)
    sk, pk = scheme.keygen()
    for msg in [b"", b"hello", bytes(range(256))]:
        sig = scheme.sign(sk, msg)
        assert len(sig) == params.signature_bytes
        assert scheme.verify(pk, msg, sig)


@pytest.mark.parametrize("params", ALL_INSTANCES)
def test_rejects_forgeries(params):
    scheme = CUTS(params)
    sk, pk = scheme.keygen(bytes([1]) * params.n)
    sig = scheme.sign(sk, b"pay alice 1")

    other_pk = scheme.keygen(bytes([2]) * params.n)[1]
    assert not scheme.verify(other_pk, b"pay alice 1", sig)
    assert not scheme.verify(pk, b"pay alice 1", sig[:-1])
    assert not scheme.verify(pk, b"pay alice 1", sig + b"\x00")
    values_start = params.n + params.counter_bytes  # after rho and any counter
    for i in range(values_start, len(sig), params.n):  # flip one bit in each Z_j
        tampered = bytearray(sig)
        tampered[i] ^= 1
        assert not scheme.verify(pk, b"pay alice 1", bytes(tampered))

    # A different message, or a different rho, only changes the encoding the
    # verifier selects. The old encoding comes back with probability 1/M, so
    # these checks only mean something when M is large.
    if scheme.M >= 2**32:
        assert not scheme.verify(pk, b"pay alice 2", sig)
        assert not scheme.verify(pk, b"pay alice 1", bytes([sig[0] ^ 1]) + sig[1:])


def count_tree_hashes(scheme, fn):
    """Run fn() and count SHAKE calls, excluding message hashing."""
    calls = 0
    original = scheme._shake

    def counting(*parts, length=None):
        nonlocal calls
        if parts[0] not in MESSAGE_DSTS:
            calls += 1
        return original(*parts, length=length)

    scheme._shake = counting
    try:
        fn()
    finally:
        del scheme._shake
    return calls


@pytest.mark.parametrize("params", ALL_INSTANCES)
def test_hash_counts_match_section_9(params):
    scheme = CUTS(params)
    sk, pk = scheme.keygen()
    sig = scheme.sign(sk, b"m")
    assert count_tree_hashes(scheme, scheme.keygen) == params.keygen_hashes
    assert count_tree_hashes(scheme, lambda: scheme.verify(pk, b"m", sig)) == (
        params.verification_hashes
    )


def test_deterministic_and_domain_separated():
    seed = bytes(16)
    a = CUTS(CUTSParams(n=16, h=4, w=3, L=5, S=5))
    b = CUTS(CUTSParams(n=16, h=4, w=3, L=5, S=6))  # differs only in S

    sk1, pk1 = a.keygen(seed)
    sk2, pk2 = a.keygen(seed)
    assert pk1 == pk2
    assert a.sign(sk1, b"m") == a.sign(sk2, b"m")

    # PID enters every tree hash, so the same seed gives a different key.
    _, pk_b = b.keygen(seed)
    assert pk_b != pk1
    assert not b.verify(pk1, b"m", a.sign(sk1, b"m"))


def test_secret_key_serialisation():
    scheme = CUTS(SMALL)
    sk, pk = scheme.keygen()
    restored = SecretKey.from_bytes(sk.to_bytes(), SMALL.n)
    assert restored == sk
    assert scheme.verify(pk, b"x", scheme.sign(restored, b"x"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 16, "h": 3, "w": 2, "L": 3, "S": 1, "counter_bits": 65},  # too wide
        {"n": 16, "h": 3, "w": 2, "L": 3, "S": 1, "counter_bits": 16},  # no r
        {"n": 16, "h": 3, "w": 2, "L": 3, "S": 1, "index_bits": 8},  # no s
        {"n": 0, "h": 3, "w": 2, "L": 3, "S": 1},  # n below 1
        {"n": 16, "h": 0, "w": 2, "L": 2, "S": 1},  # h below 1
        {"n": 16, "h": 64, "w": 2, "L": 3, "S": 1},  # h above 63
        {"n": 16, "h": 3, "w": 0, "L": 3, "S": 0},  # w below 1
        {"n": 16, "h": 3, "w": 2, "L": 1, "S": 0},  # L below 2
        {"n": 16, "h": 3, "w": 2, "L": 9, "S": 1},  # L above 2^h
        {"n": 16, "h": 3, "w": 2, "L": 3, "S": 4},  # S above L(w - 1)
    ],
)
def test_invalid_params_rejected(kwargs):
    with pytest.raises(ValueError, match=r"^need "):
        CUTSParams(**kwargs)


# ---- CUTS+C --------------------------------------------------------------


def grind_one(scheme, message):
    """Keygen (fixed seed), sign `message`, return (pk, rho, signature, encoding)."""
    sk, pk = scheme.keygen(bytes([1]) * scheme.params.n)
    sig = scheme.sign(sk, message)
    rho = sig[: scheme.params.n]
    return pk, rho, sig, scheme.select(pk, rho, message)


@pytest.mark.parametrize("params", GRIND_INSTANCES)
def test_signer_takes_the_first_counter_that_lands(params):
    scheme = CUTS(params)
    pk, rho, sig, encoding = grind_one(scheme, b"m")
    counter = int.from_bytes(sig[params.n : params.n + params.counter_bytes], "big")
    assert counter == encoding.counter
    assert all(scheme.select(pk, rho, b"m", c) is None for c in range(counter))
    assert scheme.select(pk, rho, b"m", counter) == encoding
    assert sum(encoding.positions) == params.S


@pytest.mark.parametrize("params", GRIND_INSTANCES)
def test_rejects_a_changed_counter(params):
    """A changed counter verifies only if it selects the same encoding.

    Another accepted counter can select the same encoding (with probability
    1/M on small families), and the result is then a valid signature: the
    verifier does not require the first accepted counter.
    """
    scheme = CUTS(params)
    pk, rho, sig, encoding = grind_one(scheme, b"m")
    assert encoding.counter is not None  # every CUTS+C signature carries one
    n, size = params.n, params.counter_bytes
    for other in {encoding.counter ^ 1, (1 << params.counter_bits) - 1}:
        tampered = sig[:n] + other.to_bytes(size, "big") + sig[n + size :]
        selected = scheme.select(pk, rho, b"m", other)
        same = selected is not None and selected[:2] == encoding[:2]
        assert scheme.verify(pk, b"m", tampered) == same


def test_rejects_a_counter_outside_its_bit_width():
    params = GRIND_INSTANCES[-1]  # 12-bit counter in 2 bytes
    scheme = CUTS(params)
    pk, rho, sig, _ = grind_one(scheme, b"m")
    assert scheme.select(pk, rho, b"m", 1 << params.counter_bits) is None
    too_wide = sig[: params.n] + b"\xff\xff" + sig[params.n + 2 :]
    assert not scheme.verify(pk, b"m", too_wide)


@pytest.mark.parametrize("params", GRIND_INSTANCES)
def test_verification_hashes_the_message_once(params):
    """One message hash at the signature's counter, as in the specification."""
    scheme = CUTS(params)
    pk, _, sig, _ = grind_one(scheme, b"m")
    seen = []
    original = scheme._shake

    def recording(*parts, length=None):
        seen.append(parts[0])
        return original(*parts, length=length)

    scheme._shake = recording
    assert scheme.verify(pk, b"m", sig)
    del scheme._shake
    assert seen.count(DST_HMSG) == 1


def test_acceptance_rate_matches_p_accept():
    scheme = CUTS(plus_c(16, 6, 5, 12, 32, grinding=6))
    trials = 20_000
    hits = sum(
        scheme._grind(c.to_bytes(4, "big"), 0) is not None for c in range(trials)
    )
    p = scheme.p_accept
    assert p == scheme.M / 2**scheme.params.index_bits
    assert abs(hits - trials * p) < 5 * math.sqrt(trials * p * (1 - p))


def test_accepted_encodings_are_uniform_over_the_family():
    """The small family (M = 6), ground from 5-bit indexes."""
    scheme = CUTS(plus_c(16, 3, 2, 3, 1, grinding=2))
    family = {scheme.unrank(r) for r in range(scheme.M)}
    counts = dict.fromkeys(family, 0)
    for i in range(6_000):
        encoding = scheme.select(b"pk", b"rho", i.to_bytes(4, "big"))
        counts[(encoding.frontier, encoding.positions)] += 1
    assert set(counts) == family
    assert all(800 < count < 1200 for count in counts.values())  # mean 1000


def test_security_gains_the_grinding_bits():
    scheme = CUTS(plus_c(16, 6, 5, 12, 32, grinding=6))
    r = scheme.params.index_bits
    assert scheme.grinding_bits == pytest.approx(r - math.log2(scheme.M))
    assert scheme.one_use_security_bits == pytest.approx(min(128, r))


def test_matches_the_specification_instance_table():
    """Instance 4 of the CUTS+C page: (h, w, L, S, r, s) = (12, 4, 67, 123, 246, 16)."""
    scheme = CUTS(CUTSParams(16, 12, 4, 67, 123, 246, 16))
    assert scheme.params.signature_bytes - scheme.params.n == 1074  # page omits rho
    assert scheme.params.verification_hashes + 1 == 340
    assert scheme.params.keygen_hashes == 32_761
    assert round(2**scheme.grinding_bits) == 468  # mean grinding trials


def test_widest_index_bits_is_the_limit():
    M = CUTS(CUTSParams(16, 6, 5, 12, 32)).M
    r = widest_index_bits(M, 16)
    fail = grind_failure_log2(acceptance_probability(M, r), 16)
    wider = grind_failure_log2(acceptance_probability(M, r + 1), 16)
    assert fail <= GRIND_FAILURE_LOG2_MAX < wider


def test_widest_index_bits_rejects_a_counter_too_short_for_any_r():
    # M = 6, s = 1: even r = 3 leaves failure probability 2^-4.
    with pytest.raises(ValueError, match="too short"):
        widest_index_bits(6, 1)


def test_cs_and_plus_c_keys_are_domain_separated():
    cs = CUTS(CUTSParams(n=16, h=4, w=3, L=5, S=10))
    plus = CUTS(plus_c(16, 4, 3, 5, 10, grinding=4))
    seed = bytes(16)
    sk_cs, pk_cs = cs.keygen(seed)
    _, pk_c = plus.keygen(seed)
    assert pk_cs != pk_c
    assert not plus.verify(pk_cs, b"m", cs.sign(sk_cs, b"m"))


def test_cs_rejects_a_counter():
    scheme = CUTS(SMALL)
    with pytest.raises(ValueError, match="no counter"):
        scheme.select(b"pk", b"rho", b"m", 0)


def test_rejects_an_index_too_narrow_for_the_family():
    M = CUTS(CUTSParams(16, 4, 3, 5, 10)).M  # 14, so r must be at least 4
    with pytest.raises(ValueError, match="index_bits >= 4"):
        CUTS(CUTSParams(16, 4, 3, 5, 10, (M - 1).bit_length() - 1, 16))


def test_rejects_a_counter_too_short_to_grind():
    # 20 bits of grinding needs about 2^20 counters; 16 bits cannot supply them.
    with pytest.raises(ValueError, match="too short"):
        CUTS(plus_c(16, 4, 3, 5, 10, grinding=20))
