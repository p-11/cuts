import pytest
from paper_tables import hash_counts as hc
from paper_tables.parameter_sets import BY_NAME

from cuts.scheme import CUTS, DST_HMSG, DST_RAND, CUTSParams, SecretKey

SMALL = [
    CUTSParams(16, 3, 2, 3, 1),
    CUTSParams(16, 4, 1, 5, 0),
    CUTSParams(16, 4, 3, 5, 4),
    CUTSParams(16, 4, 3, 5, 10, 13, 16),
    CUTSParams(16, 5, 4, 7, 9, 22, 12),
]


@pytest.mark.parametrize(
    "in_bytes,out_bytes,perms",
    [
        (0, 16, 1),
        (135, 16, 1),
        (136, 16, 2),
        (271, 16, 2),
        (272, 16, 3),
        (10, 136, 1),
        (10, 137, 2),
        (10, 273, 3),
    ],
)
def test_shake256_permutations(in_bytes, out_bytes, perms):
    assert hc.shake256_permutations(in_bytes, out_bytes) == perms


@pytest.mark.parametrize(
    "in_bytes,out_bytes,compressions",
    [
        (0, 16, 1),
        (55, 32, 1),
        (56, 32, 2),
        (119, 16, 2),
        (120, 16, 3),
        (55, 33, 1 + 2),
        (100, 74, 2 + 3),
    ],
)
def test_sha256_compressions(in_bytes, out_bytes, compressions):
    assert hc.sha256_compressions(in_bytes, out_bytes) == compressions


@pytest.mark.parametrize("params", SMALL)
def test_keygen_calls_match_an_instrumented_keygen(params):
    scheme = CUTS(params)
    calls = hc.record(scheme, lambda: scheme.keygen(bytes(params.n)))
    counted = {dst: sum(c.dst == dst for c in calls) for dst in hc.TREE}
    assert counted == hc.keygen_calls(params)
    assert sum(counted.values()) == params.keygen_hashes


@pytest.mark.parametrize("params", SMALL)
def test_sign_and_verify_make_the_theorem_s_tree_calls(params):
    scheme = CUTS(params)
    sk = SecretKey(bytes(params.n), bytes(params.n))
    sig = scheme.sign(sk, hc.MESSAGE)
    sign = hc.record(scheme, lambda: scheme.sign(sk, hc.MESSAGE))
    verify = hc.record(scheme, lambda: scheme.verify(sk.public_key, hc.MESSAGE, sig))
    G, V = params.keygen_hashes, params.verification_hashes
    assert sum(c.dst in hc.TREE for c in sign) == G - V
    assert sum(c.dst == DST_RAND for c in sign) == 1
    assert sum(c.dst in hc.TREE for c in verify) == V
    assert sum(c.dst == DST_HMSG for c in verify) == 1
    assert sum(c.dst == DST_RAND for c in verify) == 0


def test_expected_message_hashes():
    cuts = CUTS(BY_NAME["cuts-128-fastest"].params)
    assert hc.expected_message_hashes(cuts) == 1.0
    cuts_c = CUTS(BY_NAME["cuts-c-48-cheap"].params)
    # About 2^g counters per signature; the cap at 2^s changes nothing here.
    assert hc.expected_message_hashes(cuts_c) == pytest.approx(
        2**cuts_c.grinding_bits, rel=1e-9
    )


@pytest.mark.parametrize("name", ["cuts-c-48-cheap", "cuts-128-fastest"])
def test_measure_adds_up(name):
    """KeyGen costs G tree calls; Verify costs V tree calls and one message hash.

    Only the tree inputs are assumed to fit one SHAKE256 block (checked here).
    The message hash is charged from its measured length, whatever the byte
    layout: with a 56-byte PID and a 32-byte message it spans two blocks.
    """
    p = BY_NAME[name].params
    scheme = CUTS(p)
    sk = SecretKey(bytes(p.n), bytes(p.n))
    sig = scheme.sign(sk, hc.MESSAGE)
    calls = hc.record(scheme, lambda: scheme.verify(sk.public_key, hc.MESSAGE, sig))
    tree = [c for c in calls if c.dst in hc.TREE]
    assert all(c.in_bytes < hc.SHAKE256_RATE for c in tree)
    (msg,) = [c for c in calls if c.dst == DST_HMSG]
    costs = hc.measure(p)
    assert costs.keygen.shake256 == p.keygen_hashes
    assert costs.verify.shake256 == p.verification_hashes + (
        hc.shake256_permutations(msg.in_bytes, msg.out_bytes)
    )


@pytest.mark.parametrize(
    "in_bytes,prefix,saved",
    [(100, 52, 0), (100, 63, 0), (100, 64, 1), (109, 69, 1), (200, 130, 2)],
)
def test_sha256_cached_saves_one_compression_per_whole_prefix_block(
    in_bytes, prefix, saved
):
    plain = hc.sha256_compressions(in_bytes, 16)
    assert hc.sha256_cached(in_bytes, 16, prefix) == plain - saved


def test_tree_prefix_is_the_tag_and_pid():
    """The fixed prefix depends on the parameter set only, not on the key."""
    for name in ("cuts-128-fastest", "cuts-c-48-cheap"):
        p = BY_NAME[name].params
        pid = len(CUTS(p)._pid)
        tags = {len(d) for d in hc.TREE}
        assert hc.tree_prefix_bytes(p) == sorted(t + pid for t in tags)


def test_cached_cost_only_drops_whole_prefix_blocks():
    for name in ("cuts-128-fastest", "cuts-c-48-cheap"):
        costs = hc.measure(BY_NAME[name].params)
        for cost in (costs.keygen, costs.sign, costs.verify):
            assert cost.sha256_cached <= cost.sha256
        if max(hc.tree_prefix_bytes(BY_NAME[name].params)) < hc.SHA256_BLOCK:
            assert costs.verify.sha256_cached == costs.verify.sha256
