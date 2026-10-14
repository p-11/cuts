"""Brute-force checks of the coverage analysis in the paper's "Reuse security" subsection.

On every admissible toy family with h <= 3, w <= 3 and M <= MAX_FAMILY:

- the symbolic closure is computed literally from the three inference rules
  of the closure definition, by iterating them to a fixed point;
- Gamma_2, the number of covered ordered triples in E^3, is counted by brute
  force with that closure and compared with the coefficient formula of the
  exact triple-count theorem, evaluated from the subtree recurrence with exact
  integer polynomials, and with the exact lambda_2 of cuts_analysis;
- the coverage dynamic program of the conditional-count proposition, written
  from its statement, is compared with brute-force |Forge| for every ordered
  pair of sources, and with cuts_analysis.forgeable_target_count;
- the single-closure lemma, union preservation, the one-disclosure antichain
  property and the counterexample in the role-of-completeness remark are
  checked directly.

The family is enumerated independently of the code under test (see
test_encodings.py). Nothing here uses cuts_analysis except as a third party
to agree with.
"""

from collections import defaultdict
from functools import cache
from itertools import product

import pytest
from cuts_analysis import CUTSAnalyzer
from cuts_analysis import CUTSParams as AnalysisParams
from test_encodings import canonical_order, decorations, frontiers_by_dyadic_cover

from cuts.scheme import CUTS, CUTSParams

# Families above this size are skipped to keep the suite fast: 122 of the
# 171 admissible families with h <= 3 and w <= 3 run.
MAX_FAMILY = 60
PK = "pk"


def toy_families():
    for h, w in product((1, 2, 3), (1, 2, 3)):
        for L in range(2, (1 << h) + 1):
            for S in range(L * (w - 1) + 1):
                if CUTS(CUTSParams(16, h, w, L, S)).M <= MAX_FAMILY:
                    yield h, w, L, S


FAMILIES = list(toy_families())


@cache
def family(h, w, L, S):
    """All encodings (F, A) of the family, F in canonical order."""
    frontiers = [
        canonical_order(f, h) for f in frontiers_by_dyadic_cover(h) if len(f) == L
    ]
    return [(f, a) for f in frontiers for a in decorations(w, L, S)]


def disclosure(encoding):
    frontier, positions = encoding
    return set(zip(frontier, positions))


# ---- The closure definition, applied literally ----------------------------


def closure(disclosed, h, w):
    """Least forward-closed set containing the disclosure and pk.

    Iterates the chain rule, the non-root merge rule and the root rule until
    nothing new is derived.
    """
    known = set(disclosed) | {PK}
    internal = range(2, 1 << h)  # non-root internal vertices
    changed = True
    while changed:
        changed = False
        new = {(v, a + 1) for (v, a) in known - {PK} if a < w - 1}
        new |= {
            (v, 0)
            for v in internal
            if (2 * v, w - 1) in known and (2 * v + 1, w - 1) in known
        }
        if (2, w - 1) in known and (3, w - 1) in known:
            new.add(PK)
        if not new <= known:
            known |= new
            changed = True
    return frozenset(known)


def single_closure_formula(encoding, h, w):
    """The right-hand side of the single-closure lemma."""
    frontier, positions = encoding
    ancestors = {f >> k for f in frontier for k in range(1, f.bit_length())}
    out = {PK}
    out |= {(v, a) for v in ancestors - {1} for a in range(w)}
    out |= {(v, a) for v, p in zip(frontier, positions) for a in range(p, w)}
    return frozenset(out)


def forge(sources, targets, h, w):
    known = closure(set().union(*(disclosure(e) for e in sources)), h, w)
    return [t for t in targets if disclosure(t) <= known]


# ---- The conditional-count proposition, as stated -------------------------


def conditional_count(sources, h, w, L, S):
    """|Forge(E_1, ..., E_b)| = [x^L z^S] Q_1, with Q_v from the proposition."""

    def tau(v):
        best = w
        for frontier, positions in sources:
            if v in frontier:
                best = min(best, positions[frontier.index(v)])
            elif any(f >> k == v for f in frontier for k in range(1, h + 1)):
                best = 0
        return best

    def Q(v):
        poly = defaultdict(int)
        for a in range(tau(v), w):
            if a <= S:
                poly[(1, a)] += 1
        if v < 1 << h:  # internal
            for prod_key, c in mul2(Q(2 * v), Q(2 * v + 1)).items():
                poly[prod_key] += c
        return poly

    def mul2(p, q):
        out = defaultdict(int)
        for (x1, z1), c1 in p.items():
            for (x2, z2), c2 in q.items():
                if x1 + x2 <= L and z1 + z2 <= S:
                    out[(x1 + x2, z1 + z2)] += c1 * c2
        return out

    return mul2(Q(2), Q(3)).get((L, S), 0)


# ---- The subtree recurrence and the coefficient formula --------------------

# Variable order (x1, x2, y, z1, z2, u), as in the positive-evaluation bound.


def poly_mul(p, q, caps):
    out = defaultdict(int)
    for e1, c1 in p.items():
        for e2, c2 in q.items():
            e = tuple(a + b for a, b in zip(e1, e2))
            if all(a <= cap for a, cap in zip(e, caps)):
                out[e] += c1 * c2
    return dict(out)


def poly_add(*polys):
    out = defaultdict(int)
    for p in polys:
        for e, c in p.items():
            out[e] += c
    return dict(out)


def monomial(x1=0, x2=0, y=0, z1=0, z2=0, u=0):
    return (x1, x2, y, z1, z2, u)


def gamma2_by_recurrence(h, w, L, S):
    """Gamma_2 = [x1^L x2^L y^L z1^S z2^S u^S] C_{h-1}^2."""
    caps = (L, L, L, S, S, S)
    r = range(w)
    a1 = {monomial(x1=1, z1=a): 1 for a in r}
    a2 = {monomial(x2=1, z2=a): 1 for a in r}
    b1 = {monomial(x1=1, y=1, z1=a, u=c): 1 for a in r for c in r if a <= c}
    b2 = {monomial(x2=1, y=1, z2=a, u=c): 1 for a in r for c in r if a <= c}
    gamma = {
        monomial(1, 1, 1, a, b, c): 1 for a in r for b in r for c in r if min(a, b) <= c
    }
    d = {monomial(y=1, u=c): 1 for c in r}

    def sq(p):
        return poly_mul(p, p, caps)

    A1, A2, B1, B2, C = a1, a2, b1, b2, gamma
    for _ in range(1, h):
        A1s, A2s = sq(A1), sq(A2)
        A1, A2, B1, B2, C = (
            poly_add(a1, A1s),
            poly_add(a2, A2s),
            poly_add(b1, poly_mul(d, A1s, caps), sq(B1)),
            poly_add(b2, poly_mul(d, A2s, caps), sq(B2)),
            poly_add(
                gamma,
                poly_mul(
                    d,
                    poly_add(
                        poly_mul(a1, A2s, caps),
                        poly_mul(a2, A1s, caps),
                        poly_mul(A1s, A2s, caps),
                    ),
                    caps,
                ),
                poly_mul(a1, sq(B2), caps),
                poly_mul(a2, sq(B1), caps),
                sq(C),
            ),
        )
    # Only one coefficient of R = C^2 is needed: sum C[e] C[q - e].
    return sum(
        c * C.get(tuple(q - x for q, x in zip(caps, e)), 0) for e, c in C.items()
    )


# ---- Tests -----------------------------------------------------------------


@pytest.mark.parametrize("h,w,L,S", FAMILIES)
def test_single_closure_and_one_use(h, w, L, S):
    """The single-closure lemma, and Forge(E) = {E} (so p_1 = 1/M)."""
    fam = family(h, w, L, S)
    for e in fam:
        assert closure(disclosure(e), h, w) == single_closure_formula(e, h, w)
        assert forge([e], fam, h, w) == [e]


@pytest.mark.parametrize("h,w,L,S", FAMILIES)
def test_gamma2_brute_force_matches_the_coefficient_formula(h, w, L, S):
    """Gamma_2 three ways, plus union preservation and the coverage DP per pair."""
    fam = family(h, w, L, S)
    M = len(fam)
    assert M == CUTS(CUTSParams(16, h, w, L, S)).M
    singles = [closure(disclosure(e), h, w) for e in fam]
    analyzer = CUTSAnalyzer(AnalysisParams(n=16, h=h, w=w, L=L, S=S))

    gamma2 = 0
    for i, e1 in enumerate(fam):
        for j, e2 in enumerate(fam):
            known = closure(disclosure(e1) | disclosure(e2), h, w)
            # Union preservation for these two sources.
            assert known == singles[i] | singles[j]
            covered = sum(disclosure(t) <= known for t in fam)
            # The coverage DP, from the proposition and from cuts_analysis.
            assert conditional_count([e1, e2], h, w, L, S) == covered
            assert analyzer.forgeable_target_count((e1, e2)) == covered
            gamma2 += covered

    assert gamma2 == gamma2_by_recurrence(h, w, L, S)
    exact = analyzer.lambda2(method="exact", max_pairs=M * M)
    assert exact.probability_numerator == gamma2  # p_2 = Gamma_2 / M^3
    assert exact.probability_denominator == M**3


@pytest.mark.parametrize("h,w,L,S", [(2, 2, 3, 1), (3, 2, 4, 2), (3, 3, 5, 3)])
def test_coverage_dp_with_three_sources(h, w, L, S):
    """The proposition holds for any number b of sources, not only two."""
    fam = family(h, w, L, S)
    for e1, e2, e3 in zip(fam, fam[1:] + fam[:1], fam[2:] + fam[:2]):
        sources = [e1, e2, e3]
        assert conditional_count(sources, h, w, L, S) == len(forge(sources, fam, h, w))


@pytest.mark.parametrize("w", (2, 3))
def test_union_fails_for_non_frontier_disclosures(w):
    """The counterexample in the role-of-completeness remark (h = 2)."""
    d1, d2 = {(4, w - 1)}, {(5, w - 1)}
    assert (2, 0) in closure(d1 | d2, 2, w)
    assert (2, 0) not in closure(d1, 2, w)
    assert (2, 0) not in closure(d2, 2, w)
