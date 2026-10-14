"""Step 3: prove each parameter set's two-use security and fix its L.

Two disclosures cover a uniform target with probability p_2 = Gamma_2 / M^3,
where Gamma_2 is one coefficient of the polynomial R = C_{h-1}^2 given by the
subtree recurrence (paper, "Reuse security"), at q = (L, L, L, S, S, S). R has
nonnegative coefficients, so for any positive point r and integer m >= 1

    lambda_2 >= 3 log2 M + q . log2 r - log2 R_q^(m)(r)

where R_q^(m)(r) keeps the terms of R(r) whose exponents are congruent to q
mod m (the congruence-filtered bound; m = 1 is positive evaluation).
R_q^(m)(r) is the average of omega^(-j.q) R(r omega^j) over j in [m]^6,
with omega = exp(2 pi i / m).

For each set in parameter_sets.py:
  1. Choose r by maximising the m = 1 bound (convex in log r; by symmetry
     x1 = x2 and z1 = z2).
  2. Evaluate the filtered bound at that r for several m, and keep the
     best: m in {18, 21, 22, 24, 28} for w > 1, every m <= 128 for w = 1.
  3. Keep h, w, the S rule and the counter, and find the smallest L whose
     proven two-use security reaches the set's target: lambda_2 plus g for
     CUTS+C, each rounded down to 0.1 bit as in the paper's tables.
The result, with the bound at L - 1, goes to lambda2_bounds.json. If an L
differs from parameter_sets.py, the script says so and exits with an error.

Rigour. Floats and decimals only choose r and m; any r > 0 and m >= 1 give
a valid bound. The bound at the chosen (m, r) is then certified in ball
arithmetic (python-flint): every quantity, including the roots of unity, is
an interval guaranteed to contain the exact value. The script takes the
upper end of the filtered sum and the lower end of lambda_2, and rounds down
to 0.1 bit. It stops if the certified and float values differ by more than
1e-6 bits.

With w = 1 the polynomial does not depend on z1, z2 or u (and S = 0), so
filtering in those coordinates is the identity and is skipped.

Usage, from the repository root (about 2 hours 15 minutes on 4 cores):
    PYTHONPATH=scripts python scripts/paper_tables/lambda2_bound.py
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
import multiprocessing
import time
from dataclasses import dataclass
from decimal import ROUND_CEILING, Context, Decimal, localcontext
from pathlib import Path
from typing import TypeVar

from flint import acb, arb, ctx

from cuts import CUTS, CUTSParams

T = TypeVar("T")

# ---- the polynomial R, in parts ----------------------------------------------------
#
# These functions are generic over any values with + and *: Decimals for the
# positive evaluation, and complex numbers or complex balls for the filter.


def powers(z: T, w: int, one: T) -> list[T]:
    out = [one]
    for _ in range(w - 1):
        out.append(out[-1] * z)  # type: ignore[operator]
    return out


def total(xs: list[T]) -> T:
    s = xs[0]
    for v in xs[1:]:
        s = s + v  # type: ignore[operator]
    return s


def poly_P(pz: list[T]) -> T:
    """P(z) = sum over a < w of z^a."""
    return total(pz)


def poly_J(pz: list[T], pu: list[T]) -> T:
    """J(z, u) = sum over 0 <= a <= c < w of z^a u^c."""
    w = len(pz)
    return total([pz[a] * pu[c] for c in range(w) for a in range(c + 1)])  # type: ignore[operator]


def poly_K(p1: list[T], p2: list[T], pu: list[T]) -> T:
    """K(z1, z2, u) = sum over a, b, c < w with min(a, b) <= c of z1^a z2^b u^c."""
    w = len(pu)
    return total(
        [
            p1[a] * p2[b] * pu[c]  # type: ignore[operator]
            for a in range(w)
            for b in range(w)
            for c in range(w)
            if min(a, b) <= c
        ]
    )


def squares_A(a: T, h: int) -> list[T]:
    """A_t^2 for t < h - 1, where A_0 = a and A_{t+1} = a + A_t^2."""
    out, A = [], a
    for _ in range(h - 1):
        s = A * A  # type: ignore[operator]
        out.append(s)
        A = a + s  # type: ignore[operator]
    return out


def parts_B(b: T, d: T, As: list[T]) -> tuple[list[T], list[T]]:
    """(d A_t^2 + B_t^2, d A_t^2) for t < h - 1; B_{t+1} = b + d A_t^2 + B_t^2."""
    Q, dA, B = [], [], b
    for s in As:
        Bs = B * B  # type: ignore[operator]
        dAs = d * s  # type: ignore[operator]
        Q.append(dAs + Bs)
        dA.append(dAs)
        B = b + dAs + Bs  # type: ignore[operator]
    return Q, dA


def final_R(g, a1, a2, As2, Q1, Q2, dA1):
    """R = C_{h-1}^2, with C_{t+1} = g + a1 Q2 + a2 Q1 + d A1^2 A2^2 + C_t^2."""
    C = g
    for t in range(len(As2)):
        C = g + a1 * Q2[t] + a2 * Q1[t] + dA1[t] * As2[t] + C * C
    return C * C


def evaluate_R(h: int, w: int, x1, x2, y, z1, z2, u, one):
    """R at one point (the subtree recurrence)."""
    p1, p2, pu = powers(z1, w, one), powers(z2, w, one), powers(u, w, one)
    a1, a2 = x1 * poly_P(p1), x2 * poly_P(p2)
    b1, b2 = x1 * y * poly_J(p1, pu), x2 * y * poly_J(p2, pu)
    d = y * poly_P(pu)
    g = x1 * x2 * y * poly_K(p1, p2, pu)
    As1, As2 = squares_A(a1, h), squares_A(a2, h)
    Q1, dA1 = parts_B(b1, d, As1)
    Q2, _ = parts_B(b2, d, As2)
    return final_R(g, a1, a2, As2, Q1, Q2, dA1)


# ---- step 1: positive evaluation (m = 1), which also chooses r ------------------

UP = Context(prec=50, rounding=ROUND_CEILING, Emax=10**9, Emin=-(10**9))


def objective(h: int, w: int, L: int, S: int, theta: list[Decimal]) -> Decimal:
    """ln R(r) - q . ln r at r = exp(theta) = (x, x, y, z, z, u); chooses r."""
    with localcontext(UP):
        tx, ty, tz, tu = theta
        x, y, z, u = (t.exp() for t in theta)
        R = evaluate_R(h, w, x, x, y, z, z, u, Decimal(1))
        return R.ln() - (L * (2 * tx + ty) + S * (2 * tz + tu))


# The search keeps |ln r| <= THETA_MAX. The optimum is interior for real
# parameter sets; it runs to infinity only when the target sits on the edge
# of R's support (for example L = 2^h), where any large r is near-optimal.
THETA_MAX = 40


def minimise(f, dims: int) -> list[Decimal]:
    """Coordinate search with halving steps; converges for convex f."""
    theta = [Decimal(0)] * dims
    best = f(theta)
    step = Decimal(4)
    while step > Decimal("1e-6"):
        improved = False
        for i in range(dims):
            for sign in (1, -1):
                while True:
                    trial = list(theta)
                    trial[i] += sign * step
                    if abs(trial[i]) > THETA_MAX:
                        break
                    value = f(trial)
                    if value >= best:
                        break
                    theta, best, improved = trial, value, True
        if not improved:
            step /= 2
    return theta


def choose_r(h: int, w: int, L: int, S: int) -> tuple[float, float, float, float]:
    """The r maximising the m = 1 bound, as floats (x, y, z, u)."""
    if w > 1:
        theta = minimise(lambda t: objective(h, w, L, S, t), 4)
    else:  # z and u have no effect
        t2 = minimise(lambda t: objective(h, w, L, S, [*t, Decimal(0), Decimal(0)]), 2)
        theta = [*t2, Decimal(0), Decimal(0)]
    return tuple(math.exp(float(t)) for t in theta)  # type: ignore[return-value]


def log2_q_r(L: int, S: int, r: tuple[float, float, float, float]) -> float:
    x, y, z, u = r
    return L * (2 * math.log2(x) + math.log2(y)) + S * (2 * math.log2(z) + math.log2(u))


# ---- step 2: the congruence filter ------------------------------------------------

BALL_PREC = 128  # bits of working precision for the certified filter


def _ball(v: float) -> acb:
    return acb(arb(v))  # exact: every double is a dyadic rational


def _filter_chunk(job) -> list:
    """Real parts of the filter's terms, summed per (jy, ju) representative.

    In floats, or in balls if certify is set. A ball is returned as exact
    (midpoint, radius) pairs of (mantissa, exponent), since flint values
    cannot be pickled.
    """
    h, w, L, S, r, m, mz, reps, certify = job
    if certify:
        ctx.prec = BALL_PREC
        one = acb(1)
        om = [acb.exp_pi_i(acb(2 * k) / m) for k in range(m)]
        x, y, z, u = (_ball(v) for v in r)
    else:
        one = 1.0 + 0j
        om = [cmath.exp(2j * math.pi * k / m) for k in range(m)]
        x, y, z, u = r
    X, Y = [x * o for o in om], [y * o for o in om]
    Z = [z * om[k * (m // mz) % m] if mz > 1 else z for k in range(mz)]
    U = [u * om[k * (m // mz) % m] if mz > 1 else u for k in range(mz)]
    pz = [powers(v, w, one) for v in Z]
    pu = [powers(v, w, one) for v in U]
    PZ = [poly_P(p) for p in pz]
    # Index k = (jx, jz) for one signature's (x, z) coordinates.
    ks = [(jx, jz) for jx in range(m) for jz in range(mz)]
    a = {k: X[k[0]] * PZ[k[1]] for k in ks}
    As = {k: squares_A(a[k], h) for k in ks}
    out = []
    for (jy, ju), weight in reps:
        d = Y[jy] * poly_P(pu[ju])
        J = [poly_J(pz[jz], pu[ju]) for jz in range(mz)]
        K = [[poly_K(pz[i], pz[j], pu[ju]) for j in range(mz)] for i in range(mz)]
        QB = {k: parts_B(X[k[0]] * Y[jy] * J[k[1]], d, As[k]) for k in ks}
        terms = []
        for i, k1 in enumerate(ks):
            Q1, dA1 = QB[k1]
            for k2 in ks[i:]:
                Q2, _ = QB[k2]
                g = X[k1[0]] * X[k2[0]] * Y[jy] * K[k1[1]][k2[1]]
                R = final_R(g, a[k1], a[k2], As[k2], Q1, Q2, dA1)
                jq = (k1[0] + k2[0] + jy) * L + (k1[1] + k2[1] + ju) * S
                v = (om[-jq % m] * R).real
                terms.append(v if k1 == k2 else 2 * v)  # (k1, k2) and (k2, k1)
        if certify:
            v = weight * total(terms)
            out.append(tuple(_exact(e) for e in (v.mid(), v.rad())))
        else:
            out.append(weight * math.fsum(terms))
    return out


def _exact(v: arb) -> tuple[int, int]:
    m, e = v.man_exp()
    return int(m), int(e)


def _unpack(mid: tuple[int, int], rad: tuple[int, int]) -> arb:
    exact = [arb(m) * arb(2) ** e for m, e in (mid, rad)]
    return arb(exact[0], exact[1])


def _filter_sum(params: CUTSParams, r, m: int, pool, certify: bool):
    """The filtered sum R_q^(m)(r) as a float, or as a ball if certify is set."""
    h, w, L, S = params.h, params.w, params.L, params.S
    mz = m if w > 1 else 1
    # Conjugate symmetry: the j and -j terms are conjugates, so sum the real
    # parts over one (jy, ju) from each pair {(jy, ju), (-jy, -ju)}.
    reps = []
    for jy in range(m):
        for ju in range(mz):
            partner = (-jy % m, -ju % mz)
            if (jy, ju) <= partner:
                reps.append(((jy, ju), 1 if partner == (jy, ju) else 2))
    chunks = [reps[i :: len(reps)] for i in range(len(reps))]
    jobs = [(h, w, L, S, r, m, mz, c, certify) for c in chunks if c]
    sums = [v for part in pool.map(_filter_chunk, jobs) for v in part]
    n_terms = m**3 * mz**3
    if not certify:
        return math.fsum(sums) / n_terms
    ctx.prec = BALL_PREC
    return total([_unpack(*v) for v in sums]) / n_terms


def filtered_bound(params: CUTSParams, r, m: int, pool) -> float:
    """The filter's lower bound on lambda_2 at (m, r) in floats, for the search."""
    F = _filter_sum(params, r, m, pool, certify=False)
    if not math.isfinite(F) or F <= 0:
        return -math.inf  # no usable bound at this (m, r)
    M = CUTS(CUTSParams(params.n, params.h, params.w, params.L, params.S)).M
    shift = max(0, M.bit_length() - 64)  # log2 of a big int, accurately
    log2_M = shift + math.log2(M >> shift)
    return 3 * log2_M + log2_q_r(params.L, params.S, r) - math.log2(F)


def certified_bound(params: CUTSParams, r, m: int, pool) -> float:
    """Proven lower bound on lambda_2 from the filter at (m, r), in bits.

    The lower end of the ball, as a double rounded down.
    """
    F = _filter_sum(params, r, m, pool, certify=True)
    ctx.prec = BALL_PREC
    F_up = arb(F.upper())
    if not F_up.is_finite() or not F_up > 0:
        return -math.inf
    M = CUTS(CUTSParams(params.n, params.h, params.w, params.L, params.S)).M
    L, S = params.L, params.S
    x, y, z, u = (arb(v) for v in r)
    log_q_r = L * (2 * x.log() + y.log()) + S * (2 * z.log() + u.log())
    lam = (3 * arb(M).log() + log_q_r - F_up.log()) / arb(2).log()
    return math.nextafter(float(lam.lower()), -math.inf)


# Moduli tried. The bound is not monotone in m (a modulus can suit one L and
# S badly), so several are tried and the best kept; any (m, r) gives a valid
# bound. With w > 1 the cost grows as m^6 (m = 28 takes under a minute per
# set). With w = 1 the filter is three-dimensional, so every m up to 128 is
# cheap.
M_WIDE = (18, 21, 22, 24, 28)
M_W1 = range(2, 129)

AGREE_BITS = 1e-6  # how far the certified and float values may differ


def floor1(v: float) -> float:
    return math.floor(v * 10) / 10


@dataclass(frozen=True)
class Proof:
    params: CUTSParams
    lambda2: float  # certified lower bound on the family's lambda_2 (unrounded)
    grinding: float  # g for CUTS+C, 0 for CUTS
    m: int
    r: tuple[float, float, float, float]

    @property
    def two_use(self) -> float:
        """lambda_2 + g as the tables show them, each rounded down to 0.1 bit."""
        return floor1(self.lambda2) + floor1(self.grinding)


def prove(params: CUTSParams, pool) -> Proof:
    """The certified lambda_2 for one set, at the best of the moduli above."""
    h, w, L, S = params.h, params.w, params.L, params.S
    r = choose_r(h, w, L, S)
    best, best_m = -math.inf, 1
    for m in M_WIDE if w > 1 else M_W1:
        b = filtered_bound(params, r, m, pool)
        if b > best:
            best, best_m = b, m
    proven = certified_bound(params, r, best_m, pool)
    if not abs(proven - best) <= AGREE_BITS:
        raise SystemExit(f"certified {proven} and float {best} disagree for {params}")
    g = CUTS(params).grinding_bits if params.counter_bits else 0.0
    return Proof(params, proven, g, best_m, r)


S_FRACTIONS = (0.0, 0.25, 0.5, 0.75)  # the search's grid of S rules


def same_shape(params: CUTSParams, L: int) -> CUTSParams | None:
    """The set with the same h, w, S rule and counter at frontier size L.

    S is the same fraction of the upper half of its range; r is the widest
    the counter allows. None if the set gives under 128 bits for one use.
    """
    from paper_tables.search_front import make_key

    h, w, n = params.h, params.w, params.n
    f = 0.0
    if w > 1:
        mid = math.ceil(params.L * (w - 1) / 2)
        f = (params.S - mid) / (params.L * (w - 1) - mid)
        f = min(S_FRACTIONS, key=lambda g: abs(g - f))
    mid = math.ceil(L * (w - 1) / 2)
    S = mid + round(f * (L * (w - 1) - mid))
    if L < 2 or L > 1 << h:
        return None
    M = CUTS(CUTSParams(n, h, w, L, S)).M
    key = make_key(h, w, L, S, M, params.counter_bits)
    return key.params if key is not None else None


def smallest_meeting(start: int, meets) -> int:
    """Smallest L at which meets(L) holds, searching from start.

    meets(L) returns True, False, or None when L is not a valid set. Assumes
    meets is monotone in L, as two-use security is.
    """
    L = start
    if meets(L):
        while meets(L - 1):
            L -= 1
        return L
    while not meets(L):
        L += 1
    return L


OUT = Path(__file__).with_name("lambda2_bounds.json")


def main(argv=None) -> None:
    from paper_tables.parameter_sets import PARAMETER_SETS

    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--only", nargs="*", help="set names to run (default: all)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)

    sets = [s for s in PARAMETER_SETS if not args.only or s.name in args.only]
    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    changed = []
    with multiprocessing.Pool(args.workers) as pool:
        for s in sets:
            start = time.monotonic()
            proofs: dict[int, Proof | None] = {}

            def meets(L: int, s=s, proofs=proofs) -> bool | None:
                if L not in proofs:
                    params = same_shape(s.params, L)
                    proofs[L] = prove(params, pool) if params else None
                    pr = proofs[L]
                    shown = f"{floor1(pr.two_use):6.1f}" if pr else "invalid"
                    print(f"  {s.name:<20} L={L:<4} proven {shown}", flush=True)
                pr = proofs[L]
                return None if pr is None else pr.two_use >= s.target_bits

            L = smallest_meeting(s.params.L, meets)
            pr, below = proofs[L], proofs.get(L - 1)
            assert pr is not None
            results[s.name] = {
                "target": s.target_bits,
                "h": pr.params.h,
                "w": pr.params.w,
                "L": L,
                "S": pr.params.S,
                "lambda2": floor1(pr.lambda2),
                "grinding": round(pr.grinding, 2),
                "two_use": round(pr.two_use, 1),
                "two_use_at_L_minus_1": round(below.two_use, 1) if below else None,
                "m": pr.m,
                "r": list(pr.r),
            }
            order = [x.name for x in PARAMETER_SETS if x.name in results]
            ordered = {name: results[name] for name in order}
            args.out.write_text(json.dumps(ordered, indent=2) + "\n")
            note = (
                "" if L == s.params.L else f"  <- parameter_sets.py has L={s.params.L}"
            )
            print(
                f"{s.name:<20} L={L} proven {floor1(pr.two_use)} "
                f"({time.monotonic() - start:.0f}s){note}",
                flush=True,
            )
            if L != s.params.L:
                changed.append(s.name)
    print(f"Written to {args.out}.")
    if changed:
        raise SystemExit(f"Update parameter_sets.py for: {', '.join(changed)}")


if __name__ == "__main__":
    main()
