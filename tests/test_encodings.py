"""Exhaustive checks of the paper's "Encodings and unranking" subsection.

For admissible (h, w, L, S) with h <= 4, w <= 4 and family size up to
MAX_FAMILY, the encoding family is enumerated independently of the code
under test, and then compared with the counting tables (Lemma counts), the
unranking (Theorem unrank, eq. unrank) and the canonical order (Definition
family). The sliding-window recurrence
from the implementation-cost paragraph is checked against the direct one.
"""

import itertools

import pytest

from cuts.scheme import CUTS, CUTSParams, decoration_counts, frontier_counts

# Families above this size are skipped to keep the suite fast: 631 of the
# 1190 admissible cases run, and every (h, w) below still has all L from 2
# to 2^h covered for some S.
MAX_FAMILY = 5_000
SHAPES = [(h, w) for h in range(1, 5) for w in range(1, 5)]


def leaves(v, h):
    """The leaves of T(v) in a height-h heap-ordered tree, as a range."""
    depth = v.bit_length() - 1
    t = h - depth
    return range(v << t, (v + 1) << t)


def frontiers_by_subsets(h):
    """All complete frontiers of T_h, by testing every vertex subset.

    Uses only the definition: meets every root-to-leaf path exactly once.
    """
    vertices = range(1, 1 << (h + 1))
    paths = [{leaf >> i for i in range(h + 1)} for leaf in range(1 << h, 1 << (h + 1))]
    out = []
    for size in range(1, (1 << h) + 1):
        for subset in itertools.combinations(vertices, size):
            chosen = set(subset)
            if all(len(path & chosen) == 1 for path in paths):
                out.append(frozenset(subset))
    return out


def frontiers_by_dyadic_cover(h):
    """All complete frontiers of T_h, as covers of the leaves by subtrees.

    A set meets every root-to-leaf path exactly once if and only if the
    leaf sets of its vertices partition the leaves. Used where subset
    enumeration is too slow (h = 4 has 2^31 subsets).
    """
    n_leaves = 1 << h
    out = []

    def extend(start, chosen):
        if start == n_leaves:
            out.append(frozenset(chosen))
            return
        # Subtrees whose leaf range starts at `start`: size 2^j with 2^j | start.
        for j in range(h + 1):
            size = 1 << j
            if start % size or start + size > n_leaves:
                continue
            v = (n_leaves + start) >> j
            extend(start + size, chosen + [v])

    extend(0, [])
    return out


def all_frontiers(h):
    return frontiers_by_subsets(h) if h <= 3 else frontiers_by_dyadic_cover(h)


def canonical_order(frontier, h):
    """Left to right: sort by the first leaf under each vertex."""
    return tuple(sorted(frontier, key=lambda v: leaves(v, h)[0]))


def decorations(w, L, S):
    """All vectors in [w]^L with sum S, in lexicographic order.

    Prunes prefixes whose sum can no longer reach S, so the cost is
    proportional to the output rather than to w^L.
    """
    if L == 0:
        return [()] if S == 0 else []
    return [
        (a, *tail)
        for a in range(min(w - 1, S) + 1)
        if S - a <= (L - 1) * (w - 1)
        for tail in decorations(w, L - 1, S - a)
    ]


def admissible(h, w):
    for L in range(2, (1 << h) + 1):
        for S in range(L * (w - 1) + 1):
            yield L, S


def frontier_key(frontier, v):
    """The order Algorithm UnrankFrontier ranks by, built from the text.

    Order by the size a of the left restriction, then the left restriction,
    then the right restriction; a single vertex is the only frontier of size 1.
    """
    if frontier == (v,):
        return ()
    left = tuple(u for u in frontier if is_descendant(u, 2 * v))
    right = tuple(u for u in frontier if is_descendant(u, 2 * v + 1))
    assert left + right == frontier
    return (len(left), frontier_key(left, 2 * v), frontier_key(right, 2 * v + 1))


def is_descendant(u, v):
    while u > v:
        u >>= 1
    return u == v


@pytest.fixture(scope="module")
def frontier_table():
    return {h: all_frontiers(h) for h in range(1, 5)}


def test_both_frontier_enumerations_agree():
    for h in range(1, 4):
        assert sorted(map(sorted, frontiers_by_subsets(h))) == sorted(
            map(sorted, frontiers_by_dyadic_cover(h))
        )


@pytest.mark.parametrize("h", range(1, 5))
def test_frontier_counts_match_enumeration(h, frontier_table):
    """N[t, l] for every t <= h and l <= 2^t, including the zero entries."""
    N = frontier_counts(h, 1 << h)
    for t in range(h + 1):
        sizes = [len(f) for f in frontier_table[t]] if t else [1]
        for size in range((1 << h) + 2):
            expected = sizes.count(size)
            got = N[t][size] if size < len(N[t]) else 0
            assert got == expected, (t, size)
        # Every size between 1 and 2^t occurs (the M >= 1 remark).
        assert set(sizes) == set(range(1, (1 << t) + 1))


@pytest.mark.parametrize("w", range(1, 5))
def test_decoration_counts_match_enumeration_and_sliding_window(w):
    L = 6
    S = L * (w - 1) + 2  # past the top of the range, where D must be zero
    D = decoration_counts(w, L, S)
    for k in range(L + 1):
        for s in range(S + 1):
            brute = sum(1 for a in itertools.product(range(w), repeat=k) if sum(a) == s)
            assert D[k][s] == brute, (k, s)
            # Every sum between 0 and k(w-1) occurs.
            assert (D[k][s] > 0) == (s <= k * (w - 1))
            if k >= 1:
                window = (
                    (D[k][s - 1] if s >= 1 else 0)
                    + D[k - 1][s]
                    - (D[k - 1][s - w] if s >= w else 0)
                )
                assert D[k][s] == window, (k, s)


def family_cases():
    for h, w in SHAPES:
        N = frontier_counts(h, 1 << h)
        for L, S in admissible(h, w):
            D = decoration_counts(w, L, S)
            if N[h][L] * D[L][S] <= MAX_FAMILY:
                yield h, w, L, S


@pytest.mark.parametrize("h,w,L,S", list(family_cases()))
def test_unrank_is_the_canonical_bijection(h, w, L, S, frontier_table):
    """Exhaustive check of Lemma counts, Theorem unrank and the stated order."""
    scheme = CUTS(CUTSParams(n=16, h=h, w=w, L=L, S=S))

    frontiers = [canonical_order(f, h) for f in frontier_table[h] if len(f) == L]
    decs = decorations(w, L, S)
    family = {(f, a) for f in frontiers for a in decs}

    # Equation (family-size): M = N[h, L] D_w[L, S] = |E|.
    assert scheme.M == len(frontiers) * len(decs) == len(family)

    ranked = [scheme.unrank(r) for r in range(scheme.M)]

    # A bijection [M] -> E: every output is in the family, and none repeats.
    assert set(ranked) == family
    assert len(set(ranked)) == len(ranked)

    # Frontiers come out in canonical (left-to-right) order.
    assert all(f == canonical_order(f, h) for f, _ in ranked)

    # Eq. (unrank): r = r_F * D + r_A, frontier-major.
    D = len(decs)
    assert all(ranked[r][0] == ranked[r - r % D][0] for r in range(scheme.M))

    # The orders the algorithms are described as using.
    frontier_seq = [ranked[i * D][0] for i in range(len(frontiers))]
    assert frontier_seq == sorted(frontier_seq, key=lambda f: frontier_key(f, 1))
    assert [ranked[r][1] for r in range(D)] == sorted(decs)  # lexicographic


@pytest.mark.parametrize("h,w,L,S", [(2, 3, 3, 2), (3, 2, 4, 2), (4, 2, 5, 3)])
def test_unrank_rejects_ranks_outside_the_family(h, w, L, S):
    scheme = CUTS(CUTSParams(n=16, h=h, w=w, L=L, S=S))
    for r in (-1, scheme.M):
        with pytest.raises(ValueError):
            scheme.unrank(r)


def test_toy_example_family_size():
    """The paper's worked encoding: (h, w, L, S) = (2, 3, 3, 2) has M = 2 * 6 = 12 > 9."""
    scheme = CUTS(CUTSParams(n=16, h=2, w=3, L=3, S=2))
    assert scheme.M == 12
