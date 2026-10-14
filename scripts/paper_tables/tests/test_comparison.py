import cmath
import itertools
import json
import math
import multiprocessing
import random

import pytest
from paper_tables import comparison as cmp


def constant_sum(chains, w, total):
    return [v for v in itertools.product(range(w), repeat=chains) if sum(v) == total]


def exact_lambda2_wots(chains, w, total):
    """Brute force over C^3: a target c is covered when c >= min(a, b) per chain."""
    C = constant_sum(chains, w, total)
    gamma2 = sum(
        all(ci >= min(ai, bi) for ai, bi, ci in zip(a, b, c))
        for a in C
        for b in C
        for c in C
    )
    return 3 * math.log2(len(C)) - math.log2(gamma2)


@pytest.fixture(scope="module")
def pool():
    with multiprocessing.Pool(4) as p:
        yield p


def test_closed_form_K_matches_its_definition():
    rng = random.Random(1)
    for w in (1, 2, 4, 7):
        for _ in range(5):
            z1, z2, u = (
                cmath.rect(rng.uniform(0.5, 1.5), rng.uniform(0, 6)) for _ in "abc"
            )
            direct = sum(
                z1**a * z2**b * u**c
                for a in range(w)
                for b in range(w)
                for c in range(w)
                if min(a, b) <= c
            )
            assert cmp.K(z1, z2, u, w, 1 + 0j) == pytest.approx(direct, rel=1e-12)


@pytest.mark.parametrize("chains,w,total", [(3, 4, 4), (4, 3, 4), (5, 4, 7)])
def test_wots_c_bound_is_sound_and_converges_to_the_exact_value(pool, chains, w, total):
    exact = exact_lambda2_wots(chains, w, total)
    z, u = cmp.wots_c_choose_r(chains, w, total)
    small = cmp.wots_c_bound(z, u, 2, pool, chains=chains, w=w, total=total)
    # Past every exponent's range the filter keeps only the target coefficient.
    big = 2 * chains * (w - 1) + 1
    converged = cmp.wots_c_bound(z, u, big, pool, chains=chains, w=w, total=total)
    certified = cmp.wots_c_bound(
        z, u, big, pool, certify=True, chains=chains, w=w, total=total
    )
    assert small <= exact + 1e-9
    assert converged == pytest.approx(exact, abs=1e-9)
    # exact is a float log2 itself; allow its last-bit rounding.
    assert exact - 1e-6 <= certified <= exact + 1e-12


def test_committed_wots_c_bound_matches_a_fresh_computation(pool):
    saved = json.loads(cmp.BOUNDS.read_text())["wots-c"]
    z, u = saved["r"]
    fresh = cmp.wots_c_bound(z, u, saved["m"], pool)
    assert saved["lambda2"] == pytest.approx(fresh, abs=cmp.AGREE_BITS)
    # The bound has converged: a larger modulus does not move it.
    assert cmp.wots_c_bound(z, u, 3 * saved["m"] // 2, pool) == pytest.approx(
        fresh, abs=1e-6
    )


@pytest.mark.parametrize("k,a", [(1, 1), (2, 2), (3, 2), (2, 3)])
def test_fors_lambda2_matches_enumeration(k, a):
    leaves = list(itertools.product(range(2**a), repeat=k))
    covered = sum(
        all(t[i] in (s1[i], s2[i]) for i in range(k))
        for s1 in leaves
        for s2 in leaves
        for t in leaves
    )
    exact = 3 * math.log2(len(leaves)) - math.log2(covered)
    assert cmp.fors_lambda2(k, a) == pytest.approx(exact, abs=1e-12)


def test_fors_128s_matches_fips_205():
    row = cmp.fors_row("SLH-DSA-128s", *cmp.FORS_128S)
    assert row.bytes - cmp.N_BYTES == 14 * 13 * 16  # FORS part: k (a + 1) n
    assert row.lambda1 == 168


def test_smallest_fors_is_smallest():
    cap = 327_671
    k, a = cmp.smallest_fors(128, cap)
    best = cmp.fors_row("", k, a)
    assert best.lambda1 >= 128 and best.lambda2 >= 128 and best.keygen <= cap
    for aa in range(1, 40):
        for kk in range(1, 200):
            r = cmp.fors_row("", kk, aa)
            if r.lambda1 >= 128 and r.lambda2 >= 128 and r.keygen <= cap:
                assert (r.bytes, r.verify) >= (best.bytes, best.verify)


def test_wots_c_row_follows_shrincs():
    row = cmp.wots_c_row()
    assert row.bytes == 2 + 32 * 16 + 16  # counter, chains, randomizer
    assert row.verify == 32 * 15 - 240 + 1
    assert row.grinding == pytest.approx(128 - row.lambda1)
