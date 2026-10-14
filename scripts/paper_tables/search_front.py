"""Step 1: probe the grid of binary CUTS and CUTS+C parameter sets.

For each security target and each column of the grid, find the smallest L
whose two-use security (the bits a key keeps after signing two messages)
reaches the target. The estimates go to a cache; confirm_rows.py then builds
each target's front from them, shortlists sets and confirms those.

Targets: CUTS at --cuts-targets, CUTS+C at --cuts-c-targets. For CUTS+C,
two-use security is the family's estimate plus the grinding bits
g = r - log2 M, with r the widest the --counter-bits counter allows. Every
key gives 128 bits for one use.

The grid is every (h, w, L, S) with h <= --max-h, w <= --max-w and the
signature within --max-bytes, at n = 16. A column fixes (h, w) and S as a
fraction of the upper half of its range (--s-fractions): 0 is the midpoint
L(w-1)/2 and 1 is L(w-1). S below the midpoint is left out; its mirror above
costs less to verify and has measured at least as secure. CUTS and CUTS+C
share the grid, so they share every estimate.

Two-use security grows with L, close to linearly, so in each column only the
smallest L reaching a target can be on that target's front. The search starts
at the column's smallest L, steps up by extrapolating from the estimates so
far (undershooting, where estimates are cheap) and closes the gap by
interpolation. All targets in a column share one set of probes, at
--probe-samples, using the point estimate.

Estimates are appended to the cache as they finish and loaded back on start,
so Ctrl-C and rerunning carries on without redoing anything.

Usage, from the repository root:
    PYTHONPATH=scripts python scripts/paper_tables/search_front.py --plan
    PYTHONPATH=scripts python scripts/paper_tables/search_front.py
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import queue
import signal
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from cuts_analysis import CUTSAnalyzer
from cuts_analysis import CUTSParams as AnalysisParams

from cuts.scheme import (
    GRIND_FAILURE_LOG2_MAX,
    CUTSParams,
    acceptance_probability,
    decoration_counts,
    frontier_counts,
    grind_failure_log2,
)

N_BYTES = 16
ONE_USE_BITS = 128
Z_95 = 1.96
CUTS, CUTS_C = "CUTS", "CUTS+C"
# First step of a search, in two-use bits per element. Above every rate
# measured (about 1.3 to 1.7), so the first step undershoots.
FIRST_STEP_SLOPE = 2.0
# A search is given up without estimating its largest L only if a secant over
# at least this many elements predicts it falls short by this many bits.
GIVE_UP_SPAN = 4
GIVE_UP_MARGIN = 3.0


# ---- keys and the grid ------------------------------------------------------------


@dataclass(frozen=True)
class Key:
    h: int
    w: int
    L: int
    S: int
    r: int = 0  # CUTS+C index width; 0 for CUTS
    counter_bits: int = 0
    log2_M: float = field(default=0.0, compare=False)

    @property
    def params(self) -> CUTSParams:
        return CUTSParams(
            N_BYTES, self.h, self.w, self.L, self.S, self.r, self.counter_bits
        )

    @property
    def scheme(self) -> str:
        return CUTS_C if self.counter_bits else CUTS

    @property
    def bytes(self) -> int:
        return self.params.signature_bytes

    @property
    def verify(self) -> int:
        """V, excluding the message hash."""
        return self.params.verification_hashes

    @property
    def keygen(self) -> int:
        return self.params.keygen_hashes

    @property
    def cost(self) -> tuple[int, int, int]:
        return self.bytes, self.verify, self.keygen

    @property
    def grinding(self) -> float:
        return self.r - self.log2_M if self.counter_bits else 0.0

    @property
    def family(self) -> str:
        """What two-use security is estimated over, whatever r and s are."""
        return f"h={self.h} w={self.w} L={self.L} S={self.S}"

    @property
    def label(self) -> str:
        text = f"({self.h}, {self.w}, {self.L}, {self.S})"
        return text + (f" r={self.r}" if self.counter_bits else "")


def log2_p_min(counter_bits: int) -> float:
    """log2 of the smallest acceptance chance a counter of this width allows."""
    return math.log2(
        -math.expm1(GRIND_FAILURE_LOG2_MAX * math.log(2) / 2**counter_bits)
    )


def widest_r(M: int, counter_bits: int) -> int:
    """widest_index_bits, starting from a float guess so it stays fast."""
    r = math.floor(math.log2(M) - log2_p_min(counter_bits))
    while grind_failure_log2(acceptance_probability(M, r), counter_bits) > (
        GRIND_FAILURE_LOG2_MAX
    ):
        r -= 1
    return r


def make_key(h: int, w: int, L: int, S: int, M: int, counter_bits: int) -> Key | None:
    """The key for this family in one scheme, or None if one use is below 128."""
    if M <= 1:
        return None
    log2_M = math.log2(M)
    if not counter_bits:
        return Key(h, w, L, S, 0, 0, log2_M) if log2_M >= ONE_USE_BITS else None
    if log2_M < ONE_USE_BITS - 32:
        return None
    r = widest_r(M, counter_bits)
    return Key(h, w, L, S, r, counter_bits, log2_M) if r >= ONE_USE_BITS else None


Column = tuple[int, int, float]  # (h, w, S fraction)


def build_columns(args) -> dict[tuple[str, Column], list[Key]]:
    """Every valid key, by scheme and column, sorted by consecutive L."""
    schemes = []
    if args.cuts_targets:
        schemes.append(0)
    if args.cuts_c_targets:
        schemes.append(args.counter_bits)
    max_L = args.max_bytes // N_BYTES - 1
    N = frontier_counts(args.max_h, max_L)
    out: dict[tuple[str, Column], list[Key]] = {}
    for w in range(1, args.max_w + 1):
        D = decoration_counts(w, max_L, max_L * (w - 1))
        fractions = [0.0] if w == 1 else args.s_fractions
        for h in range(1, args.max_h + 1):
            for L in range(2, min(max_L, 1 << h) + 1):
                mid = math.ceil(L * (w - 1) / 2)
                for f in fractions:
                    S = mid + round(f * (L * (w - 1) - mid))
                    M = N[h][L] * D[L][S]
                    for s in schemes:
                        key = make_key(h, w, L, S, M, s)
                        if key is None:
                            continue
                        if key.bytes > args.max_bytes or key.verify > args.max_verify:
                            continue
                        out.setdefault((key.scheme, (h, w, f)), []).append(key)
    for c, keys in out.items():
        run = [keys[0]]  # keep the first run of consecutive L
        for k in keys[1:]:
            if k.L != run[-1].L + 1:
                break
            run.append(k)
        out[c] = run
    return out


# ---- estimates, cached, on a worker pool ------------------------------------------


def estimate(job: tuple[Key, int, int]) -> tuple[Key, int, int, float, float]:
    """cuts-analysis two-use estimate over the family, and its standard error."""
    key, samples, seed = job
    analyzer = CUTSAnalyzer(AnalysisParams(N_BYTES, key.h, key.w, key.L, key.S))
    result = analyzer.lambda2(method="mc", samples=samples, seed=seed)
    return key, samples, seed, result.lambda_bits, result.standard_error_bits or 0.0


def ignore_interrupts() -> None:
    """Worker initialiser: Ctrl-C is handled by the main process alone."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class Cache:
    """Estimates on disk, one JSON line each, keyed by what was estimated."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[str, tuple[float, float]] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a run killed mid-write leaves a partial last line
                self.entries[entry["key"]] = (entry["lambda2"], entry["se"])

    @staticmethod
    def name(key: Key, samples: int, seed: int) -> str:
        return f"{key.family} samples={samples} seed={seed}"

    def get(self, key: Key, samples: int, seed: int) -> tuple[float, float] | None:
        return self.entries.get(self.name(key, samples, seed))

    def put(self, key: Key, samples: int, seed: int, value: tuple[float, float]):
        name = self.name(key, samples, seed)
        self.entries[name] = value
        with self.path.open("a") as f:
            f.write(
                json.dumps({"key": name, "lambda2": value[0], "se": value[1]}) + "\n"
            )


@dataclass
class Result:
    tag: int  # who asked
    key: Key
    samples: int
    seed: int
    lam: float
    se: float
    cached: bool


class Estimator:
    """Hands estimates to the pool as they are asked for, through the cache.

    Results come back in the order they finish. An estimate asked for again
    while it is running is computed once and handed to every asker.
    """

    def __init__(self, pool, cache: Cache) -> None:
        self.pool = pool
        self.cache = cache
        self.finished: queue.Queue = queue.Queue()  # filled by the pool
        self.ready: deque[Result] = deque()
        self.waiting: dict[str, list[int]] = {}
        self.pending = 0  # requests not yet returned by next()
        self.queued = 0  # estimates with the pool
        self.computed = 0
        self.start = time.monotonic()

    def request(self, key: Key, samples: int, seed: int, tag: int) -> None:
        self.pending += 1
        cached = self.cache.get(key, samples, seed)
        if cached is not None:
            self.ready.append(Result(tag, key, samples, seed, *cached, True))
            return
        name = self.cache.name(key, samples, seed)
        if name in self.waiting:
            self.waiting[name].append(tag)
            return
        self.waiting[name] = [tag]
        self.queued += 1
        self.pool.apply_async(
            estimate,
            ((key, samples, seed),),
            callback=self.finished.put,
            error_callback=self.finished.put,
        )

    def next(self) -> Result:
        if not self.ready:
            while True:
                try:
                    out = self.finished.get(timeout=1)
                    break
                except queue.Empty:
                    continue
            if isinstance(out, BaseException):
                raise out
            key, samples, seed, lam, se = out
            self.queued -= 1
            self.computed += 1
            self.cache.put(key, samples, seed, (lam, se))
            first, *others = self.waiting.pop(self.cache.name(key, samples, seed))
            self.ready.append(Result(first, key, samples, seed, lam, se, False))
            self.ready.extend(
                Result(t, key, samples, seed, lam, se, True) for t in others
            )
        self.pending -= 1
        return self.ready.popleft()

    def elapsed(self) -> str:
        return duration(time.monotonic() - self.start)


def duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


# ---- the search: smallest L per column and target ---------------------------------


@dataclass(eq=False)
class Search:
    """The smallest L reaching one target in one column of one scheme.

    curve holds the family's probe estimates by L. It is shared by every
    search on the same column, CUTS and CUTS+C alike.
    """

    keys: list[Key]
    target: float
    curve: dict[int, float]
    steps: int = 0
    answer: int | None = None  # index into keys of the smallest L found
    searched: bool = False

    @property
    def scheme(self) -> str:
        return self.keys[0].scheme

    @property
    def name(self) -> str:
        return f"{self.scheme} {self.target:g}"

    @property
    def key(self) -> Key:
        assert self.answer is not None
        return self.keys[self.answer]

    def known(self) -> dict[int, float]:
        """Two-use bits at every probed index of this search's keys."""
        first = self.keys[0].L
        return {
            L - first: lam + self.keys[L - first].grinding
            for L, lam in self.curve.items()
            if 0 <= L - first < len(self.keys)
        }

    def bracket(self, known: dict[int, float]) -> tuple[int, int | None]:
        """Largest index short of the target, smallest reaching it above that."""
        lo = max((i for i, v in known.items() if v < self.target), default=-1)
        hi = min(
            (i for i, v in known.items() if v >= self.target and i > lo), default=None
        )
        return lo, hi

    def next_probe(self) -> int | None:
        """The next index to probe, or None when the search is settled."""
        last = len(self.keys) - 1
        known = self.known()
        lo, hi = self.bracket(known)
        if hi is not None:
            if hi - lo <= 1:
                return None
            if lo == -1:
                return 0  # another search reached the target higher up
            if self.steps % 3 == 2 or known[hi] <= known[lo]:
                return (lo + hi) // 2
            t = (self.target - known[lo]) / (known[hi] - known[lo])
            i = lo + round(t * (hi - lo))
            return min(max(i, lo + 1), hi - 1)
        if lo == -1:
            return 0
        if lo == last:
            return None
        below = [j for j in known if j < lo]
        if below:
            j = max(below)
            slope = (known[lo] - known[j]) / (lo - j)
            # Growth is slower than linear, so a secant from below overestimates.
            reach = known[lo] + slope * (last - lo)
            if (
                lo - j >= GIVE_UP_SPAN
                and slope > 0
                and reach < self.target - GIVE_UP_MARGIN
            ):
                return None
            cap = lo + max(GIVE_UP_SPAN, 2 * (lo - j))
        else:
            slope, cap = FIRST_STEP_SLOPE, last
        if slope <= 0:
            nxt = cap
        else:
            nxt = lo + math.ceil((self.target - known[lo]) / slope)
        return min(max(nxt, lo + 1), cap, last)

    def settle(self) -> None:
        self.searched = True
        _, self.answer = self.bracket(self.known())


def make_searches(columns: dict[tuple[str, Column], list[Key]], args) -> list[Search]:
    """One search per (scheme, column, target), cheapest columns first."""
    curves: dict[Column, dict[int, float]] = {}
    targets = {CUTS: args.cuts_targets, CUTS_C: args.cuts_c_targets}
    out = []
    for (scheme, col), keys in sorted(columns.items(), key=lambda kv: kv[0][1][1::-1]):
        curve = curves.setdefault(col, {})
        for t in targets[scheme]:
            # Two-use security is at most log2 M + g.
            start = next(
                (i for i, k in enumerate(keys) if k.log2_M + k.grinding >= t), None
            )
            if start is not None:
                out.append(Search(keys[start:], t, curve))
    return out


def preload(searches: list[Search], cache: Cache, samples: int, seed: int) -> int:
    """Put every cached probe estimate into its column's curve; return the count."""
    n = 0
    for s in searches:
        for k in s.keys:
            if k.L not in s.curve and (v := cache.get(k, samples, seed)) is not None:
                s.curve[k.L] = v[0]
                n += 1
    return n


def probe_all(searches: list[Search], est: Estimator, args) -> None:
    """Settle every search on probe estimates."""
    total, settled = len(searches), 0

    def advance(n: int) -> bool:
        s = searches[n]
        i = s.next_probe()
        if i is None:
            s.settle()
            return False
        est.request(s.keys[i], args.probe_samples, args.seed, n)
        return True

    for n in range(total):
        if not advance(n):
            settled += 1
    print(f"{settled}/{total} searches settled from the cache.", flush=True)
    while est.pending:
        res = est.next()
        s = searches[res.tag]
        s.curve[res.key.L] = res.lam
        s.steps += 1
        if not advance(res.tag):
            settled += 1
        if res.cached:
            continue
        bits = res.lam + res.key.grinding
        if s.searched:
            note = f"settled at L={s.key.L}" if s.answer is not None else "out of reach"
        else:
            note = "reaches target" if bits >= s.target else "short"
        print(
            f"  {s.name:<11} {res.key.family:<26} {bits:6.1f} ±{Z_95 * res.se:4.1f}  "
            f"{note:<20} | searches {settled}/{total}, {est.queued} queued, "
            f"{est.elapsed()}",
            flush=True,
        )


# ---- fronts -------------------------------------------------------------------------

T = TypeVar("T")


def pareto(items: Iterable[T], cost: Callable[[T], tuple]) -> list[T]:
    """Items no other item matches or beats on every coordinate of cost."""

    def dominates(a: tuple, b: tuple) -> bool:
        return all(x <= y for x, y in zip(a, b)) and a != b

    kept: list[tuple[tuple, T]] = []
    # In sorted order every dominator comes first.
    for c, item in sorted(((cost(i), i) for i in items), key=lambda t: t[0]):
        if not any(dominates(kc, c) for kc, _ in kept):
            kept.append((c, item))
    return [item for _, item in kept]


def fronts(searches: list[Search]) -> dict[str, list[Search]]:
    """Each target's front over (bytes, V, G), from each search's answer."""
    by_name: dict[str, list[Search]] = {}
    for s in searches:
        if s.answer is not None:
            by_name.setdefault(s.name, []).append(s)
    return {n: pareto(group, lambda s: s.key.cost) for n, group in by_name.items()}


# ---- main -----------------------------------------------------------------------------


def s_fraction(text: str) -> float:
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"{text} is not in [0, 1]")
    return value


def grid_parser() -> argparse.ArgumentParser:
    """Options shared with confirm_rows.py, which must use the same grid."""
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("grid (must match between the two steps)")
    g.add_argument(
        "--cuts-targets",
        type=float,
        nargs="*",
        default=[128.0],
        help="CUTS two-use targets in bits; none to skip CUTS",
    )
    g.add_argument(
        "--cuts-c-targets",
        type=float,
        nargs="*",
        default=[48.0, 64.0, 80.0, 96.0, 112.0, 128.0],
        help="CUTS+C two-use targets in bits; none to skip CUTS+C",
    )
    g.add_argument("--counter-bits", type=int, default=16)
    g.add_argument("--max-bytes", type=int, default=2400)
    g.add_argument("--max-verify", type=int, default=10**9)
    g.add_argument("--max-h", type=int, default=15)
    g.add_argument("--max-w", type=int, default=5)
    g.add_argument(
        "--s-fractions", type=s_fraction, nargs="+", default=[0.0, 0.25, 0.5, 0.75]
    )
    g.add_argument("--probe-samples", type=int, default=16)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--cache", type=Path, default=Path("search_front_cache.jsonl"))
    return p


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0], parents=[grid_parser()]
    )
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--plan", action="store_true", help="print the grid and stop")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    columns = build_columns(args)
    searches = make_searches(columns, args)
    print(
        f"Grid: {sum(len(k) for k in columns.values())} keys in {len(columns)} "
        f"columns, {len(searches)} searches."
    )
    if not searches or args.plan:
        return
    cache = Cache(args.cache)
    print(
        f"Cache: {args.cache}, {preload(searches, cache, args.probe_samples, args.seed)}"
        " probes loaded."
    )
    pool = multiprocessing.Pool(args.workers, initializer=ignore_interrupts)
    est = Estimator(pool, cache)
    try:
        probe_all(searches, est, args)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        print(f"\nStopped. Estimates are saved in {args.cache}; rerun to carry on.")
        raise SystemExit(130) from None
    pool.close()
    pool.join()
    found = sum(s.answer is not None for s in searches)
    print(
        f"\nDone in {est.elapsed()}, {est.computed} new estimates. {found} of "
        f"{len(searches)} searches reach their target. Next: confirm_rows.py."
    )


if __name__ == "__main__":
    main()
