"""Comparison of the paper's parameter sets with WOTS+C and FORS.

Every row is counted in the units of the cost theorem, so the schemes compare
directly: signature bytes including an n-byte randomizer (CUTS's rho, the R
that SHRINCS and SLH-DSA send), and tree primitive calls for verification (V)
and key generation (G), where a leaf derivation, a chain step and a merge each
count once and a final compression of several values into the public key
counts as one merge. lambda_1 and lambda_2 are the one- and two-use exponents
of the coverage analysis: -log2 of the chance that one or two signatures
reveal a uniform fresh target. The grinding gap g of CUTS+C and WOTS+C is a
separate column; it rests on a heuristic argument and is never added in.

WOTS+C, as in SHRINCS (draft BIP, commit a5891ec): 32 chains of 4 bits
(w = 16), constant sum 240, a 2-byte counter, and a 128-bit digest. It is a
one-level tree whose frontier is every leaf, so two sources a, b cover a
target c when c_i >= min(a_i, b_i) on every chain, and

    Gamma_2 = [z1^S z2^S u^S] K(z1, z2, u)^32,

with K the single-vertex polynomial of the subtree recurrence. lambda_2 is
bounded by positive evaluation and the congruence filter (the theorems of the
coverage section) at a modulus where the bound has converged, and certified
in ball arithmetic by `certify_wots_c`, which writes comparison_bounds.json.

FORS with k trees of height a: a signature reveals one leaf secret per tree,
so two signatures cover a uniform target leaf with probability
(2 - 2^-a) / 2^a in each tree, independently:
lambda_1 = k a and lambda_2 = k (a - log2(2 - 2^-a)), exactly. Rows: the FORS
of SLH-DSA-128s (FIPS 205: k = 14, a = 12), and the smallest FORS with
lambda_1, lambda_2 >= 128 whose key generation costs no more than the
costliest CUTS set's.

Usage, from the repository root (certifies the WOTS+C bound, about a minute):
    PYTHONPATH=scripts python scripts/paper_tables/comparison.py
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
import multiprocessing
from dataclasses import dataclass
from pathlib import Path

from flint import acb, arb, ctx

from cuts.scheme import decoration_counts

N_BYTES = 16
BOUNDS = Path(__file__).with_name("comparison_bounds.json")

# WOTS+C as in the SHRINCS draft BIP, commit a5891ec (Stateful Parameters).
WOTS_C_CHAINS = 32
WOTS_C_W = 16
WOTS_C_SUM = 240
WOTS_C_COUNTER_BYTES = 2
WOTS_C_DIGEST_BITS = 128

# FORS of SLH-DSA-128s (FIPS 205, Table 2).
FORS_128S = (14, 12)  # (k, a)

M_CERTIFY = 128  # the filter has converged here (see tests)
BALL_PREC = 128
AGREE_BITS = 1e-6


@dataclass(frozen=True)
class Row:
    scheme: str
    name: str
    bytes: int
    verify: int
    keygen: int
    lambda1: float
    lambda2: float
    grinding: float | None  # heuristic gap g, or None


# ---- WOTS+C ------------------------------------------------------------------------


def wots_c_family_size(
    chains: int = WOTS_C_CHAINS, w: int = WOTS_C_W, total: int = WOTS_C_SUM
) -> int:
    """|C|: digit vectors in [w]^chains with the constant sum."""
    return decoration_counts(w, chains, total)[chains][total]


def K(z1, z2, u, w: int, one):
    """sum over a, b, c < w with min(a, b) <= c of z1^a z2^b u^c, in O(w).

    Equals P(z1) P(z2) U(u) - sum_c u^c Q_c(z1) Q_c(z2), where Q_c(z) sums z^a
    over c < a < w. Works for floats, complex numbers and complex balls.
    """
    p1, p2 = [one], [one]
    for _ in range(w - 1):
        p1.append(p1[-1] * z1)
        p2.append(p2[-1] * z2)
    s1, s2 = [one * 0] * (w + 1), [one * 0] * (w + 1)
    for a in range(w - 1, -1, -1):
        s1[a] = s1[a + 1] + p1[a]
        s2[a] = s2[a + 1] + p2[a]
    out, uc = one * 0, one
    for c in range(w):
        out += uc * (s1[0] * s2[0] - s1[c + 1] * s2[c + 1])
        uc = uc * u
    return out


def _log2_q_r(total: int, z: float, u: float) -> float:
    return total * (2 * math.log2(z) + math.log2(u))


def wots_c_choose_r(chains: int, w: int, total: int) -> tuple[float, float]:
    """(z, u) maximising the m = 1 bound, by coordinate search in log space."""
    log2C = math.log2(wots_c_family_size(chains, w, total))

    def f(z, u):
        R = K(z, z, u, w, 1.0) ** chains
        return 3 * log2C + _log2_q_r(total, z, u) - math.log2(R)

    best, val, step = (1.0, 1.0), f(1.0, 1.0), 0.5
    while step > 1e-9:
        moved = False
        for dz, du in ((step, 0), (-step, 0), (0, step), (0, -step)):
            z, u = best[0] * math.exp(dz), best[1] * math.exp(du)
            v = f(z, u)
            if v > val:
                best, val, moved = (z, u), v, True
        if not moved:
            step /= 2
    return best


def _filter_part(job):
    """Real part of the filter's sum over one j1 (j2 >= j1 by symmetry)."""
    chains, w, total, z, u, m, j1, certify = job
    if certify:
        ctx.prec = BALL_PREC
        one = acb(1)
        om = [acb.exp_pi_i(acb(2 * k) / m) for k in range(m)]
        zb, ub = acb(arb(z)), acb(arb(u))
    else:
        one = 1.0 + 0j
        om = [cmath.exp(2j * math.pi * k / m) for k in range(m)]
        zb, ub = z, u
    zs = [zb * o for o in om]
    us = [ub * o for o in om]
    terms = []
    for j2 in range(j1, m):
        for j3 in range(m):
            jq = ((j1 + j2 + j3) * total) % m
            v = (om[-jq % m] * K(zs[j1], zs[j2], us[j3], w, one) ** chains).real
            terms.append(v if j1 == j2 else 2 * v)
    if certify:
        s = sum(terms[1:], terms[0])
        return tuple(_man_exp(e) for e in (s.mid(), s.rad()))
    return math.fsum(terms)


def _man_exp(v: arb) -> tuple[int, int]:
    mant, e = v.man_exp()
    return int(mant), int(e)


def wots_c_bound(
    z: float,
    u: float,
    m: int,
    pool,
    certify: bool = False,
    chains: int = WOTS_C_CHAINS,
    w: int = WOTS_C_W,
    total: int = WOTS_C_SUM,
) -> float:
    """Lower bound on lambda_2 from the filter at (m, (z, z, u)), in bits.

    In floats, or certified in ball arithmetic: then the lower end of the
    ball, rounded down to a double.
    """
    jobs = [(chains, w, total, z, u, m, j1, certify) for j1 in range(m)]
    parts = pool.map(_filter_part, jobs)
    C = wots_c_family_size(chains, w, total)
    if not certify:
        F = math.fsum(parts) / m**3
        return 3 * math.log2(C) + _log2_q_r(total, z, u) - math.log2(F)
    ctx.prec = BALL_PREC
    balls = [
        arb(arb(a) * arb(2) ** b, arb(c) * arb(2) ** d) for (a, b), (c, d) in parts
    ]
    F_up = arb((sum(balls[1:], balls[0]) / m**3).upper())
    lam = (
        3 * arb(C).log() + total * (2 * arb(z).log() + arb(u).log()) - F_up.log()
    ) / arb(2).log()
    return math.nextafter(float(lam.lower()), -math.inf)


def certify_wots_c(pool) -> dict:
    z, u = wots_c_choose_r(WOTS_C_CHAINS, WOTS_C_W, WOTS_C_SUM)
    approx = wots_c_bound(z, u, M_CERTIFY, pool)
    proven = wots_c_bound(z, u, M_CERTIFY, pool, certify=True)
    if abs(proven - approx) > AGREE_BITS:
        raise SystemExit(f"certified {proven} and float {approx} disagree")
    return {"m": M_CERTIFY, "r": [z, u], "lambda2": proven}


def wots_c_row() -> Row:
    C = wots_c_family_size()
    log2C = math.log2(C)
    lambda2 = json.loads(BOUNDS.read_text())["wots-c"]["lambda2"]
    return Row(
        "WOTS+C",
        "SHRINCS",
        WOTS_C_COUNTER_BYTES + WOTS_C_CHAINS * N_BYTES + N_BYTES,
        # Chain steps from the revealed positions, and one compression of
        # the chain ends into the public key.
        WOTS_C_CHAINS * (WOTS_C_W - 1) - WOTS_C_SUM + 1,
        WOTS_C_CHAINS + WOTS_C_CHAINS * (WOTS_C_W - 1) + 1,
        log2C,
        lambda2,
        WOTS_C_DIGEST_BITS - log2C,
    )


# ---- FORS --------------------------------------------------------------------------


def fors_lambda2(k: int, a: int) -> float:
    return k * (a - math.log2(2 - 2.0**-a))


def fors_row(name: str, k: int, a: int) -> Row:
    return Row(
        "FORS",
        name,
        k * (a + 1) * N_BYTES + N_BYTES,
        k * (a + 1) + 1,  # leaf hash, a merges per tree; one root compression
        k * (3 * 2**a - 1) + 1,  # leaf secret and hash, merges; compression
        float(k * a),
        fors_lambda2(k, a),
        None,
    )


def smallest_fors(target: float, max_keygen: int) -> tuple[int, int]:
    """Smallest signature with lambda_1, lambda_2 >= target and G <= max_keygen."""
    best = None
    for a in range(1, 40):
        k = max(math.ceil(target / a), 1)
        while fors_lambda2(k, a) < target:
            k += 1
        row = fors_row("", k, a)
        if row.keygen > max_keygen:
            continue
        key = (row.bytes, row.verify)
        if best is None or key < best[0]:
            best = (key, (k, a))
    return best[1]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.parse_args(argv)
    with multiprocessing.Pool() as pool:
        result = certify_wots_c(pool)
    BOUNDS.write_text(json.dumps({"wots-c": result}, indent=2) + "\n")
    print(f"WOTS+C lambda_2 >= {result['lambda2']:.4f} (m = {result['m']})")
    print(f"Wrote {BOUNDS.name}")


if __name__ == "__main__":
    main()
