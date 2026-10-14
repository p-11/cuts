"""Search CUTS+C keys for the stateful path of a SHRINCS-shaped scheme.

SHRINCS signs statefully with many one-time keys under one root: its state
counter moves the signer to a fresh WOTS+C key for every signature. If state
fails and a key signs twice, WOTS+C falls to about 34 bits. CINCS keeps that
behaviour and swaps the one-time key for a CUTS+C key, so that a key signed
twice by mistake still keeps a usable level of security.

Whatever tree holds the keys adds the same authentication path and index to
every signature, whichever key it holds. So the key is chosen on its own
here, and compared with the WOTS+C key it replaces. Each CUTS+C key gives 128
bits for its one planned signature; "reuse" is its security if signed twice.

The search runs in three stages:

1. Exact. Size, verification, keygen, one-use security and grinding are
   closed-form, so the whole grid is filtered to a competitive envelope at
   once. Keys follow the CUTS+C specification: r is the widest the counter
   allows, so grinding adds its full g = r - log2 M bits. One-use security is
   then r whatever S is, so S can rise to cut verification while r >= 128.
   S is tried at a few points between the midpoint of its range and that
   top (--s-fractions); a higher S verifies faster but keeps a little less
   security under reuse.
2. Probe. Reuse security has no closed form. It is estimated with
   cuts-analysis at three key lengths L per (h, w, S point) and interpolated
   between, since it grows almost linearly in L.
3. Confirm. The keys the probes put on the front (no other key is at least
   as small, as fast to verify and as safe under reuse) are re-estimated
   with more samples, until every key on the front has been confirmed.

Every estimate is appended to a cache file as it finishes. Interrupt at any
time with Ctrl-C and run the same command again to carry on from there.

Costs are in SHA-256 compressions, as in the SHRINCS BIP, assuming CUTS is
hashed the way SHRINCS hashes (one compression per leaf, chain or merge hash).
The SHRINCS model is checked against the figures the BIP publishes.

Usage:
    PYTHONPATH=scripts python scripts/search_stateful/search_stateful.py
    PYTHONPATH=scripts python scripts/search_stateful/search_stateful.py \
        --max-key-bytes 800 --max-key-verify 250
    PYTHONPATH=scripts python scripts/search_stateful/search_stateful.py \
        --max-key-keygen 131072 --counter-bits 17
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import random
import signal
import time
from dataclasses import dataclass
from pathlib import Path

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

N_BYTES = 16  # hash output length, as in SHRINCS
SECURITY_BITS = 128

# ---- SHRINCS stateful path (BIP draft, SHRINCS/shrincs-bip, main) ---------

WOTS_C_CHAINS = 32
WOTS_C_W = 16  # 4 bits per chain
WOTS_C_SUM = 240
T_SF_COMPRESSIONS = 9  # 22-byte ADRS + 512 bytes of chain tips, after the midstate
MESSAGE_COMPRESSIONS = 4  # H_msg_sf, 32-byte message and empty ctx: 2 inner + 2 outer

# One WOTS+C key: counter and chains; chain completion, T_sf and one grinding
# hash to verify; its chains and T_sf to generate.
WOTS_C_KEY_BYTES = 2 + WOTS_C_CHAINS * 16
WOTS_C_KEY_VERIFY = WOTS_C_CHAINS * (WOTS_C_W - 1) - WOTS_C_SUM + T_SF_COMPRESSIONS + 1
WOTS_C_KEY_KEYGEN = WOTS_C_CHAINS + WOTS_C_CHAINS * (WOTS_C_W - 1) + T_SF_COMPRESSIONS


def ceildiv(a: int, b: int) -> int:
    return -(-a // b)


def first_signature_bytes(key_bytes: int) -> int:
    """First UXMSS signature: indicator, R, 1-byte index, key, one path node."""
    return 1 + 16 + 1 + key_bytes + 16


def first_signature_verify(key_verify: int) -> int:
    """The key, one merge up the path, and H_msg_sf."""
    return key_verify + 1 + MESSAGE_COMPRESSIONS


def tree_keygen(key_keygen: int, keys: int) -> int:
    """Every key, and one merge per internal node of the tree holding them."""
    return keys * key_keygen + keys - 1


def check_shrincs_model() -> None:
    """The model must reproduce the figures published in the BIP."""
    assert first_signature_bytes(WOTS_C_KEY_BYTES) == 548
    assert first_signature_verify(WOTS_C_KEY_VERIFY) == 255
    assert tree_keygen(WOTS_C_KEY_KEYGEN, 256) == 133_631  # UXMSS depth 255
    assert tree_keygen(WOTS_C_KEY_KEYGEN, 32) == 16_703  # UXMSS depth 31


# ---- candidate keys (stage 1: exact) --------------------------------------


@dataclass(frozen=True)
class Key:
    h: int
    w: int
    L: int
    S: int
    r: int  # index width; grinding adds r - log2 M bits
    counter_bits: int
    grinding: float

    @property
    def params(self) -> CUTSParams:
        return CUTSParams(
            N_BYTES, self.h, self.w, self.L, self.S, self.r, self.counter_bits
        )

    @property
    def bytes(self) -> int:
        """Counter and revealed values. rho takes the place of SHRINCS's R."""
        return self.L * N_BYTES + self.params.counter_bytes

    @property
    def verify(self) -> int:
        """Tree hashes, plus the message hash at the signature's counter."""
        return self.params.verification_hashes + 1

    @property
    def keygen(self) -> int:
        return self.params.keygen_hashes

    @property
    def family(self) -> str:
        """What reuse is estimated over: the family, whatever r and s are."""
        return f"h={self.h} w={self.w} L={self.L} S={self.S}"

    @property
    def label(self) -> str:
        return f"{self.family} r={self.r}"


def log2_p_min(counter_bits: int) -> float:
    """log2 of the smallest acceptance chance a counter of this width allows."""
    return math.log2(
        -math.expm1(GRIND_FAILURE_LOG2_MAX * math.log(2) / 2**counter_bits)
    )


def widest_r(M: int, counter_bits: int, floor: float) -> int:
    """Largest r with M / 2^r >= p_min, checked exactly after the float guess."""
    r = math.floor(math.log2(M) - floor)
    while grind_failure_log2(acceptance_probability(M, r), counter_bits) > (
        GRIND_FAILURE_LOG2_MAX
    ):
        r -= 1
    return r


def envelope(args) -> dict[tuple[int, int, float], list[Key]]:
    """Every key inside the size, verification and keygen limits.

    Grouped into columns of the same (h, w, S point), ordered by L.
    """
    max_L = args.max_key_bytes // N_BYTES
    floor = log2_p_min(args.counter_bits)
    # r >= 128 needs log2 M >= 128 + log2 p_min.
    min_log2_M = SECURITY_BITS + floor
    N = frontier_counts(args.max_h, max_L)
    columns: dict[tuple[int, int, float], list[Key]] = {}
    for w in range(2, args.max_w + 1):
        D = decoration_counts(w, max_L, max_L * (w - 1))
        for h in range(1, args.max_h + 1):
            if ((1 << (h + 1)) - 2) * w + 1 > args.max_key_keygen:
                break
            for L in range(2, min(max_L, 1 << h) + 1):
                mid = ceildiv(L * (w - 1), 2)
                if math.log2(N[h][L] * D[L][mid]) < min_log2_M:
                    continue
                top = mid
                while top < L * (w - 1) and math.log2(N[h][L] * D[L][top + 1]) >= (
                    min_log2_M
                ):
                    top += 1
                for f in args.s_fractions:
                    S = mid + round(f * (top - mid))
                    M = N[h][L] * D[L][S]
                    r = widest_r(M, args.counter_bits, floor)
                    if r < SECURITY_BITS:
                        continue
                    key = Key(h, w, L, S, r, args.counter_bits, r - math.log2(M))
                    if key.bytes > args.max_key_bytes:
                        continue
                    if key.verify > args.max_key_verify:
                        continue
                    columns.setdefault((h, w, f), []).append(key)
    return {c: list(dict.fromkeys(keys)) for c, keys in columns.items()}


# ---- reuse estimates (stages 2 and 3), cached -----------------------------


def estimate(job: tuple[Key, int, int]) -> tuple[Key, float, float]:
    """cuts-analysis two-use estimate over the family, and its standard error."""
    key, samples, seed = job
    analyzer = CUTSAnalyzer(AnalysisParams(N_BYTES, key.h, key.w, key.L, key.S))
    result = analyzer.lambda2(method="mc", samples=samples, seed=seed)
    return key, result.lambda_bits, result.standard_error_bits or 0.0


def ignore_interrupts() -> None:
    """Worker initialiser: Ctrl-C is handled by the main process alone."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def wots_c_reuse(samples: int, seed: int) -> tuple[float, float]:
    """Two-use security of one WOTS+C key signed twice (no error estimate).

    After two signatures the attacker knows each chain from the lower of the
    two revealed positions upward. A forgery is any constant-sum index set at
    or above those floors, and each grinding hash is uniform over 2^128 index
    sets, so the forgeable fraction is (count of such sets) / 2^128.
    """
    rng = random.Random(seed)

    def constant_sum_vector() -> list[int]:
        while True:
            v = [rng.randrange(WOTS_C_W) for _ in range(WOTS_C_CHAINS)]
            if sum(v) == WOTS_C_SUM:
                return v

    total = 0
    for _ in range(samples):
        a, b = constant_sum_vector(), constant_sum_vector()
        ways = {0: 1}  # partial sum -> number of index sets
        for floor in (min(x, y) for x, y in zip(a, b, strict=True)):
            step: dict[int, int] = {}
            for s, count in ways.items():
                for value in range(floor, WOTS_C_W):
                    if s + value <= WOTS_C_SUM:
                        step[s + value] = step.get(s + value, 0) + count
            ways = step
        total += ways.get(WOTS_C_SUM, 0)
    return -math.log2(total / samples / 2**128), 0.0


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
                    # A run killed mid-write leaves a partial last line.
                    print(f"Skipping an unreadable line in {path}.")
                    continue
                self.entries[entry["key"]] = (entry["lambda2"], entry["se"])

    @staticmethod
    def name(item: Key | str, samples: int, seed: int) -> str:
        label = item if isinstance(item, str) else item.family
        return f"{label} samples={samples} seed={seed}"

    def get(
        self, item: Key | str, samples: int, seed: int
    ) -> tuple[float, float] | None:
        return self.entries.get(self.name(item, samples, seed))

    def require(self, item: Key | str, samples: int, seed: int) -> tuple[float, float]:
        """The estimate for an item already run through `run`."""
        value = self.get(item, samples, seed)
        if value is None:
            raise KeyError(f"no estimate for {self.name(item, samples, seed)}")
        return value

    def put(
        self, item: Key | str, samples: int, seed: int, value: tuple[float, float]
    ) -> None:
        name = self.name(item, samples, seed)
        self.entries[name] = value
        with self.path.open("a") as f:
            f.write(
                json.dumps({"key": name, "lambda2": value[0], "se": value[1]}) + "\n"
            )


def run(
    stage: str, keys: list[Key], samples: int, seed: int, cache: Cache, pool
) -> None:
    """Estimate every key not in the cache, printing progress as each finishes."""
    todo = [k for k in dict.fromkeys(keys) if cache.get(k, samples, seed) is None]
    print(
        f"\n{stage}: {len(keys) - len(todo)} cached, {len(todo)} to estimate "
        f"({samples} samples each)",
        flush=True,
    )
    if not todo:
        return
    start = time.monotonic()
    jobs = [(k, samples, seed) for k in todo]
    for done, (key, lam, se) in enumerate(pool.imap_unordered(estimate, jobs), 1):
        cache.put(key, samples, seed, (lam, se))
        elapsed = time.monotonic() - start
        eta = elapsed / done * (len(todo) - done)
        print(
            f"  [{done:>3}/{len(todo)}] {key.family:<24} "
            f"reuse {lam + key.grinding:6.1f} bits   "
            f"{elapsed:5.0f}s elapsed, ~{eta:4.0f}s left",
            flush=True,
        )


def probe_points(column: list[Key]) -> list[Key]:
    """Shortest, middle and longest key in one (h, w) column."""
    return [column[0], column[len(column) // 2], column[-1]]


def interpolate(column: list[Key], known: dict[int, float]) -> dict[Key, float]:
    """Reuse bits for every key in a column, linear in L between probes."""
    Ls = sorted(known)
    out = {}
    for key in column:
        lo = max([x for x in Ls if x <= key.L], default=Ls[0])
        hi = min([x for x in Ls if x >= key.L], default=Ls[-1])
        if lo == hi:
            base = known[lo]
        else:
            t = (key.L - lo) / (hi - lo)
            base = known[lo] + t * (known[hi] - known[lo])
        out[key] = base + key.grinding
    return out


def front(reuse: dict[Key, float]) -> list[Key]:
    """Keys no other key matches or beats on size, verification and reuse.

    Keygen is a limit (--max-key-keygen), not something to trade here: a
    taller tree always costs more keygen and buys a little reuse, so almost
    every key would survive. Rerun with another limit to see that trade.
    """

    def beats(a: Key, b: Key) -> bool:
        no_worse = a.bytes <= b.bytes and a.verify <= b.verify and reuse[a] >= reuse[b]
        better = a.bytes < b.bytes or a.verify < b.verify or reuse[a] > reuse[b]
        return no_worse and better

    keys = list(reuse)
    kept = [k for k in keys if not any(beats(o, k) for o in keys)]
    return sorted(kept, key=lambda k: (k.bytes, k.verify))


def estimate_reuse(columns, args, cache: Cache, pool) -> dict[Key, float]:
    """Stages 2 and 3: reuse bits for every key, confirmed on the front."""
    probes = [k for c in columns.values() for k in probe_points(c)]
    run("Stage 2 (probe)", probes, args.probe_samples, args.seed, cache, pool)

    reuse: dict[Key, float] = {}
    for column in columns.values():
        known = {
            k.L: cache.require(k, args.probe_samples, args.seed)[0]
            for k in probe_points(column)
        }
        reuse.update(interpolate(column, known))

    # Confirm the front, replace its estimates, and repeat until it is stable.
    confirm_seed = args.seed + 1
    confirmed: set[Key] = set()
    while pending := [k for k in front(reuse) if k not in confirmed]:
        run(
            "Stage 3 (confirm)",
            pending,
            args.confirm_samples,
            confirm_seed,
            cache,
            pool,
        )
        for k in pending:
            lam, _ = cache.require(k, args.confirm_samples, confirm_seed)
            reuse[k] = lam + k.grinding
            confirmed.add(k)
    return reuse


# ---- report ---------------------------------------------------------------

COLUMNS = "  {:<31} {:>5} {:>6} {:>7} {:>9}  {:>7} {:>6} {:>10} {:>11}"


def print_header() -> None:
    print(
        COLUMNS.format(
            "key",
            "bytes",
            "verify",
            "keygen",
            "reuse",
            "1st sig",
            "verify",
            "keygen 32",
            "keygen 256",
        )
    )
    print("  " + "-" * 105)


def print_row(label, nbytes, verify, keygen, reuse, se) -> None:
    reuse_text = f"{reuse:.0f}" + (f" ±{1.96 * se:.0f}" if se else "")
    print(
        COLUMNS.format(
            label,
            nbytes,
            verify,
            keygen,
            reuse_text,
            first_signature_bytes(nbytes),
            first_signature_verify(verify),
            tree_keygen(keygen, 32),
            tree_keygen(keygen, 256),
        )
    )


def s_fraction(text: str) -> float:
    """An --s-fractions value: 0 is the midpoint of S's range, 1 its top."""
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"{text} is not in [0, 1]")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--max-key-bytes",
        type=int,
        default=1300,
        help=f"size limit per key (WOTS+C: {WOTS_C_KEY_BYTES})",
    )
    parser.add_argument(
        "--max-key-verify",
        type=int,
        default=350,
        help=f"verification limit per key (WOTS+C: {WOTS_C_KEY_VERIFY})",
    )
    parser.add_argument(
        "--max-key-keygen",
        type=int,
        default=1 << 15,
        help=f"keygen limit per key (WOTS+C: {WOTS_C_KEY_KEYGEN})",
    )
    parser.add_argument(
        "--counter-bits",
        type=int,
        default=16,
        help="grinding counter width s (SHRINCS: 16); each extra bit buys one bit",
    )
    parser.add_argument(
        "--s-fractions",
        type=s_fraction,
        nargs="+",
        default=[0.0, 0.5, 1.0],
        help="S points to try, from the midpoint (0) to the highest S (1)",
    )
    parser.add_argument("--max-w", type=int, default=16)
    parser.add_argument("--max-h", type=int, default=20)
    parser.add_argument("--probe-samples", type=int, default=32)
    parser.add_argument("--confirm-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--plan", action="store_true", help="print the stage 1 counts and stop"
    )
    parser.add_argument(
        "--cache", type=Path, default=Path("search_stateful_cache.jsonl")
    )
    args = parser.parse_args()

    check_shrincs_model()
    cache = Cache(args.cache)
    columns = envelope(args)
    total = sum(len(c) for c in columns.values())
    probes = len({k for c in columns.values() for k in probe_points(c)})
    print(
        f"Stage 1: {total} keys in {len(columns)} (h, w, S point) columns fit the "
        f"limits (bytes <= {args.max_key_bytes}, verify <= {args.max_key_verify}, "
        f"keygen <= {args.max_key_keygen}); {probes} probes to estimate."
    )
    if not columns or args.plan:
        return

    pool = multiprocessing.Pool(args.workers, initializer=ignore_interrupts)
    try:
        reuse = estimate_reuse(columns, args, cache, pool)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        print(
            f"\nInterrupted. Finished estimates are saved in {args.cache}; "
            "run the same command again to continue."
        )
        raise SystemExit(130) from None
    pool.close()
    pool.join()

    wots = cache.get("WOTS+C", args.confirm_samples, args.seed)
    if wots is None:
        wots = wots_c_reuse(args.confirm_samples, args.seed)
        cache.put("WOTS+C", args.confirm_samples, args.seed, wots)

    print("\nKey costs in SHA-256 compressions. Each key gives 128 bits for one use;")
    print("'reuse' is its security if signed twice (95% interval). The last four")
    print("columns put the key under a SHRINCS-style tree: first UXMSS signature,")
    print("and stateful keygen for 32 and 256 keys.\n")
    print_header()
    print_row(
        "SHRINCS WOTS+C", WOTS_C_KEY_BYTES, WOTS_C_KEY_VERIFY, WOTS_C_KEY_KEYGEN, *wots
    )
    print()
    for key in front(reuse):
        lam, se = cache.require(key, args.confirm_samples, args.seed + 1)
        print_row(key.label, key.bytes, key.verify, key.keygen, lam + key.grinding, se)


if __name__ == "__main__":
    main()
