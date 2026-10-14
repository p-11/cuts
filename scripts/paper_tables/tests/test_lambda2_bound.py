from collections import defaultdict

import pytest
from cuts_analysis import CUTSAnalyzer
from cuts_analysis import CUTSParams as AnalysisParams
from flint import arb, ctx
from paper_tables import lambda2_bound as lb

from cuts import CUTSParams

# Small families where cuts-analysis can count Gamma_2 by brute force.
SMALL = [
    (2, 2, 2, 1),
    (2, 2, 3, 1),
    (3, 2, 3, 1),
    (2, 3, 3, 2),
    (3, 2, 4, 2),
    (3, 3, 3, 3),
    (2, 3, 4, 4),
    (4, 2, 3, 2),
]


class InlinePool:
    def map(self, fn, jobs):
        return [fn(j) for j in jobs]


class Poly:
    """Polynomial in 6 variables, truncated above the target exponents.

    Truncation cannot change the target coefficient: exponents only add.
    """

    def __init__(self, terms, caps):
        self.t, self.caps = terms, caps

    def __add__(self, o):
        t = defaultdict(int, self.t)
        for k, v in o.t.items():
            t[k] += v
        return Poly(dict(t), self.caps)

    def __mul__(self, o):
        t = defaultdict(int)
        for k1, v1 in self.t.items():
            for k2, v2 in o.t.items():
                k = tuple(a + b for a, b in zip(k1, k2, strict=True))
                if all(a <= c for a, c in zip(k, self.caps, strict=True)):
                    t[k] += v1 * v2
        return Poly(dict(t), self.caps)


def exact(h, w, L, S):
    """(Gamma_2, lambda_2) by brute force over every pair of disclosures.

    lambda_2 is a ball, since comparisons with it must be exact.
    """
    a = CUTSAnalyzer(AnalysisParams(16, h, w, L, S))
    res = a.lambda2(method="exact", max_pairs=10**6)
    gamma = res.probability_numerator * a.M**3 // res.probability_denominator
    ctx.prec = lb.BALL_PREC
    return gamma, (3 * arb(a.M).log() - arb(gamma).log()) / arb(2).log()


@pytest.mark.parametrize("h,w,L,S", SMALL)
def test_recurrence_counts_covered_triples(h, w, L, S):
    caps = (L, L, L, S, S, S)
    v = [Poly({tuple(int(i == j) for j in range(6)): 1}, caps) for i in range(6)]
    R = lb.evaluate_R(
        h, w, v[0], v[1], v[2], v[3], v[4], v[5], Poly({(0,) * 6: 1}, caps)
    )
    assert R.t.get(caps, 0) == exact(h, w, L, S)[0]


@pytest.mark.parametrize("h,w,L,S", SMALL)
@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_certified_bound_never_exceeds_the_truth(h, w, L, S, m):
    p = CUTSParams(16, h, w, L, S)
    r = lb.choose_r(h, w, L, S)
    assert arb(lb.certified_bound(p, r, m, InlinePool())) < exact(h, w, L, S)[1]


@pytest.mark.parametrize("h,w,L,S", [*SMALL, (6, 4, 20, 30), (6, 1, 30, 0)])
@pytest.mark.parametrize("m", [1, 3, 5, 6])
def test_certified_bound_matches_the_float_one(h, w, L, S, m):
    p = CUTSParams(16, h, w, L, S)
    r = lb.choose_r(h, w, L, S)
    certified = lb.certified_bound(p, r, m, InlinePool())
    estimate = lb.filtered_bound(p, r, m, InlinePool())
    assert certified <= estimate + lb.AGREE_BITS
    assert abs(certified - estimate) <= lb.AGREE_BITS


@pytest.mark.parametrize("h,w,L,S", [(2, 2, 2, 1), (2, 2, 3, 1), (2, 2, 3, 2)])
def test_large_m_recovers_the_exact_value(h, w, L, S):
    # Exponents of R are at most 2^h in x, y and 2^h (w - 1) in z, u. With m
    # above that, only the target exponent survives the filter.
    m = (1 << h) * max(1, w - 1) + 1
    p = CUTSParams(16, h, w, L, S)
    r = lb.choose_r(h, w, L, S)
    bound = lb.certified_bound(p, r, m, InlinePool())
    truth = exact(h, w, L, S)[1]
    assert truth - 1e-5 < arb(bound) < truth


def test_m_1_matches_the_decimal_positive_evaluation():
    h, w, L, S = 4, 3, 6, 6
    r = lb.choose_r(h, w, L, S)
    p = CUTSParams(16, h, w, L, S)
    via_filter = lb.filtered_bound(p, r, 1, InlinePool())
    import math
    from decimal import Decimal

    theta = [Decimal(math.log(v)) for v in r]
    M = CUTSAnalyzer(AnalysisParams(16, h, w, L, S)).M
    via_decimal = 3 * math.log2(M) - float(lb.objective(h, w, L, S, theta)) / math.log(
        2
    )
    assert abs(via_filter - via_decimal) < 1e-5


@pytest.mark.parametrize("h,w,L,S", [(2, 3, 4, 4), (2, 2, 3, 1), (3, 2, 3, 2)])
@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_filter_matches_one_built_from_explicit_coefficients(h, w, L, S, m):
    import math

    from cuts import CUTS

    caps = (10**6,) * 6  # no truncation: the filter keeps high aliases too
    v = [Poly({tuple(int(i == j) for j in range(6)): 1}, caps) for i in range(6)]
    R = lb.evaluate_R(
        h, w, v[0], v[1], v[2], v[3], v[4], v[5], Poly({(0,) * 6: 1}, caps)
    )
    q = (L, L, L, S, S, S)
    r = lb.choose_r(h, w, L, S)
    point = (r[0], r[0], r[1], r[2], r[2], r[3])
    F = sum(
        c * math.prod(a**e for a, e in zip(point, k, strict=True))
        for k, c in R.t.items()
        if all((k[i] - q[i]) % m == 0 for i in range(6))
    )
    M = CUTS(CUTSParams(16, h, w, L, S)).M
    explicit = 3 * math.log2(M) + lb.log2_q_r(L, S, r) - math.log2(F)
    fast = lb.filtered_bound(CUTSParams(16, h, w, L, S), r, m, InlinePool())
    assert abs(fast - explicit) <= 1e-9


def test_smallest_meeting_steps_down_and_up():
    def meets(threshold, first_valid=0):
        return lambda L: None if L < first_valid else L >= threshold

    assert lb.smallest_meeting(50, meets(45)) == 45  # steps down
    assert lb.smallest_meeting(40, meets(45)) == 45  # steps up
    assert lb.smallest_meeting(50, meets(30, first_valid=48)) == 48  # stops at validity


def test_same_shape_keeps_h_w_and_the_S_rule():
    from paper_tables.parameter_sets import BY_NAME

    for name in ("cuts-c-80-smallest", "cuts-128-smallest", "cuts-c-128-fastest"):
        p = BY_NAME[name].params
        assert lb.same_shape(p, p.L) == p
        q = lb.same_shape(p, p.L + 3)
        assert q is not None
        assert (q.h, q.w, q.counter_bits, q.L) == (p.h, p.w, p.counter_bits, p.L + 3)
    # Below the one-use bound there is no valid set.
    assert lb.same_shape(BY_NAME["cuts-c-48-cheap"].params, 31) is None
