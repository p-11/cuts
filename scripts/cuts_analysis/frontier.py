"""CUTS frontier counting and canonical unranking."""

from __future__ import annotations

from typing import TypeAlias

Frontier: TypeAlias = tuple[int, ...]
Decoration: TypeAlias = tuple[int, ...]
Encoding: TypeAlias = tuple[Frontier, Decoration]


def build_frontier_counts(h: int, L: int) -> list[list[int]]:
    """Return N[t][ell] for 0 <= t <= h and 0 <= ell <= L."""

    N = [[0] * (L + 1) for _ in range(h + 1)]
    N[0][1] = 1

    for t in range(1, h + 1):
        N[t][1] = 1
        child_capacity = 1 << (t - 1)

        for ell in range(2, min(L, 1 << t) + 1):
            a_min = max(1, ell - child_capacity)
            a_max = min(child_capacity, ell - 1)
            N[t][ell] = sum(
                N[t - 1][a] * N[t - 1][ell - a] for a in range(a_min, a_max + 1)
            )

    return N


def build_decoration_counts(w: int, L: int, S: int) -> list[list[int]]:
    """Return D[k][s] for 0 <= k <= L and 0 <= s <= S."""

    D = [[0] * (S + 1) for _ in range(L + 1)]
    D[0][0] = 1

    # Sliding-window form of D[k][s] = sum_a D[k-1][s-a].
    for k in range(1, L + 1):
        window = 0
        for s in range(S + 1):
            window += D[k - 1][s]
            if s >= w:
                window -= D[k - 1][s - w]
            D[k][s] = window

    return D


def unrank_frontier(
    v: int,
    t: int,
    ell: int,
    rank: int,
    N: list[list[int]],
) -> Frontier:
    """Return the frontier with the specified canonical rank."""

    if not 0 <= rank < N[t][ell]:
        raise ValueError("frontier rank is out of range")
    if ell == 1:
        return (v,)

    child_capacity = 1 << (t - 1)
    a_min = max(1, ell - child_capacity)
    a_max = min(child_capacity, ell - 1)

    for a in range(a_min, a_max + 1):
        right_size = ell - a
        right_count = N[t - 1][right_size]
        block_size = N[t - 1][a] * right_count

        if rank < block_size:
            left_rank, right_rank = divmod(rank, right_count)
            left = unrank_frontier(2 * v, t - 1, a, left_rank, N)
            right = unrank_frontier(
                2 * v + 1,
                t - 1,
                right_size,
                right_rank,
                N,
            )
            return left + right

        rank -= block_size

    raise AssertionError("unreachable for a valid frontier rank")


def unrank_decoration(
    k: int,
    s: int,
    rank: int,
    w: int,
    D: list[list[int]],
) -> Decoration:
    """Return the decoration with the specified canonical rank."""

    if not 0 <= rank < D[k][s]:
        raise ValueError("decoration rank is out of range")
    if k == 0:
        return ()

    a_min = max(0, s - (k - 1) * (w - 1))
    a_max = min(w - 1, s)

    for a in range(a_min, a_max + 1):
        block_size = D[k - 1][s - a]

        if rank < block_size:
            tail = unrank_decoration(k - 1, s - a, rank, w, D)
            return (a,) + tail

        rank -= block_size

    raise AssertionError("unreachable for a valid decoration rank")


def unrank_encoding(
    rank: int,
    h: int,
    w: int,
    L: int,
    S: int,
    N: list[list[int]],
    D: list[list[int]],
) -> Encoding:
    """Return the decorated frontier with rank in 0, ..., M-1."""

    decoration_count = D[L][S]
    family_size = N[h][L] * decoration_count
    if not 0 <= rank < family_size:
        raise ValueError("encoding rank is out of range")

    frontier_rank, decoration_rank = divmod(rank, decoration_count)
    return (
        unrank_frontier(1, h, L, frontier_rank, N),
        unrank_decoration(L, S, decoration_rank, w, D),
    )
