"""Step 2: shortlist parameter sets from the probe cache and confirm them.

Reads the probe estimates search_front.py left in the cache, builds each
target's provisional front from them, and shortlists per target:

  smallest   the smallest signature
  fastest    the lowest verification V
  cheap      the smallest signature with keygen G <= --cheap-keygen

Confirming a set finds the smallest L in its column whose two-use estimate at
--samples reaches the target at the lower end of its 95% interval, with L - 1
falling short. Probe estimates run high, so this is usually 1 to 3 above the
probe's L. A window of --window consecutive L is estimated at once, in
parallel, and moved only when needed.

Writes confirmed_rows.json next to this file. Uses the same cache as step 1,
so Ctrl-C and rerunning carries on. Pass the same grid options as step 1.

Usage, from the repository root:
    PYTHONPATH=scripts python scripts/paper_tables/confirm_rows.py --plan
    PYTHONPATH=scripts python scripts/paper_tables/confirm_rows.py
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
from dataclasses import dataclass, field
from pathlib import Path

from paper_tables import search_front as sf

OUT = Path(__file__).with_name("confirmed_rows.json")

# Measured seconds per Monte Carlo sample by w, at L = 70 and L = 100; cost
# grows as a power of L between them. Used only for the --plan estimate.
SAMPLE_SECONDS = {
    1: (0.003, 0.005),
    2: (0.16, 0.82),
    3: (0.5, 2.8),
    4: (1.2, 6.5),
    5: (2.4, 12.5),
}


def seconds_per_sample(w: int, L: int) -> float:
    lo, hi = SAMPLE_SECONDS[min(w, 5)]
    return lo * (L / 70) ** (math.log(hi / lo) / math.log(100 / 70))


@dataclass(eq=False)
class Row:
    """One shortlisted set, and the confirm estimates tried in its column."""

    search: sf.Search
    picked_as: list[str]
    window: int
    tried: dict[int, tuple[float, float]] = field(default_factory=dict)
    result: sf.Key | None = None
    unreachable: bool = False

    @property
    def probe_L(self) -> int:
        return self.search.key.L

    def key_at(self, L: int) -> sf.Key:
        return self.search.keys[L - self.search.keys[0].L]

    def passes(self, L: int) -> bool:
        lam, se = self.tried[L]
        return lam + self.key_at(L).grinding - sf.Z_95 * se >= self.search.target

    def first_window(self) -> list[int]:
        last = self.search.keys[-1].L
        return list(range(self.probe_L, min(self.probe_L + self.window, last + 1)))

    def next_window(self) -> list[int]:
        """Settle if possible; otherwise return the L still to estimate."""
        first, last = self.search.keys[0].L, self.search.keys[-1].L
        passing = sorted(L for L in self.tried if self.passes(L))
        if passing:
            best = passing[0]
            if best == first or (best - 1 in self.tried and not self.passes(best - 1)):
                self.result = self.key_at(best)
                return []
            return [best - 1]
        top = max(self.tried)
        if top >= last:
            self.unreachable = True
            return []
        return list(range(top + 1, min(top + 1 + self.window, last + 1)))


def shortlist(searches: list[sf.Search], cheap_keygen: int, window: int) -> list[Row]:
    """Smallest, fastest and cheap-keygen set on each target's front."""
    rows: dict[int, Row] = {}
    for group in sf.fronts(searches).values():
        picks = {
            "smallest": min(group, key=lambda s: (s.key.bytes, s.key.verify)),
            "fastest": min(group, key=lambda s: (s.key.verify, s.key.bytes)),
        }
        cheap = [s for s in group if s.key.keygen <= cheap_keygen]
        if cheap:
            picks["cheap"] = min(cheap, key=lambda s: (s.key.bytes, s.key.verify))
        for why, s in picks.items():
            rows.setdefault(id(s), Row(s, [], window)).picked_as.append(why)
    return sorted(
        rows.values(),
        key=lambda r: (r.search.scheme, r.search.target, r.search.key.bytes),
    )


def row_dict(row: Row) -> dict:
    assert row.result is not None
    k = row.result
    lam, se = row.tried[k.L]
    two_use = lam + k.grinding
    return {
        "scheme": k.scheme,
        "target": row.search.target,
        "picked_as": row.picked_as,
        "h": k.h,
        "w": k.w,
        "L": k.L,
        "S": k.S,
        "r": k.r,
        "s": k.counter_bits,
        "bytes": k.bytes,
        "verify": k.verify,
        "keygen": k.keygen,
        "log2_M": round(k.log2_M, 2),
        "grinding": round(k.grinding, 2),
        "lambda2_family": round(lam, 2),
        "two_use": round(two_use, 2),
        "two_use_low95": round(two_use - sf.Z_95 * se, 2),
        "se": round(se, 3),
        "probe_L": row.probe_L,
    }


def confirm(rows: list[Row], est: sf.Estimator, samples: int, seed: int) -> None:
    pending = {n: r.first_window() for n, r in enumerate(rows)}
    round_no = 0
    while pending:
        round_no += 1
        jobs = [(n, L) for n, Ls in pending.items() for L in Ls]
        fresh = sum(
            est.cache.get(rows[n].key_at(L), samples, seed) is None for n, L in jobs
        )
        print(f"\nRound {round_no}: {len(jobs)} estimates, {fresh} to run.", flush=True)
        for n, L in jobs:
            est.request(rows[n].key_at(L), samples, seed, n)
        done = 0
        while est.pending:
            res = est.next()
            row = rows[res.tag]
            row.tried[res.key.L] = (res.lam, res.se)
            if res.cached:
                continue
            done += 1
            print(
                f"  [{done:>3}/{fresh}] {row.search.name:<11} {res.key.family:<26} "
                f"{res.lam + res.key.grinding:6.1f} ±{sf.Z_95 * res.se:4.1f}  "
                f"{'passes' if row.passes(res.key.L) else 'short '} | "
                f"{est.queued} queued, {est.elapsed()}",
                flush=True,
            )
        pending = {}
        for n, r in enumerate(rows):
            if r.result is None and not r.unreachable and (more := r.next_window()):
                pending[n] = more


def print_rows(rows: list[Row]) -> None:
    fmt = "  {:<11} {:<28} {:>5} {:>5} {:>8} {:>5} {:>12}  {}"
    print(
        fmt.format(
            "target", "(h, w, L, S)", "bytes", "V", "G", "g", "two-use", "picked as"
        )
    )
    for r in rows:
        if r.result is None:
            print(f"  {r.search.name:<11} {r.search.key.label}: no L within the limits")
            continue
        d = row_dict(r)
        print(
            fmt.format(
                r.search.name,
                r.result.label,
                d["bytes"],
                d["verify"],
                d["keygen"],
                f"{d['grinding']:.1f}" if d["s"] else "-",
                f"{d['two_use']:.1f} ±{d['two_use'] - d['two_use_low95']:.1f}",
                ", ".join(r.picked_as),
            )
        )


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0], parents=[sf.grid_parser()]
    )
    p.add_argument("--samples", type=int, default=128, help="samples per confirm")
    p.add_argument("--window", type=int, default=3, help="L estimated at once per set")
    p.add_argument(
        "--cheap-keygen",
        type=int,
        default=1 << 14,
        help="keygen limit for the 'cheap' pick",
    )
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument(
        "--plan",
        action="store_true",
        help="print the shortlist and a time estimate, then stop",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    seed = args.seed + 1  # independent of the probes
    cache = sf.Cache(args.cache)
    searches = sf.make_searches(sf.build_columns(args), args)
    sf.preload(searches, cache, args.probe_samples, args.seed)
    for s in searches:
        s.settle()
    rows = shortlist(searches, args.cheap_keygen, args.window)

    print(f"Shortlist: {len(rows)} sets.")
    for r in rows:
        k = r.search.key
        print(
            f"  {r.search.name:<11} {k.label:<28} probe L={r.probe_L:<4} "
            f"{k.bytes:>5} B  V={k.verify:<4} G={k.keygen:<7} {', '.join(r.picked_as)}"
        )
    todo = {(r.key_at(L), args.samples, seed) for r in rows for L in r.first_window()}
    todo = {j for j in todo if cache.get(*j) is None}
    workers = args.workers or multiprocessing.cpu_count()
    secs = sum(seconds_per_sample(k.w, k.L) * n for k, n, _ in todo) / workers
    print(
        f"First round: {len(todo)} estimates to run, about {sf.duration(secs)} "
        f"on {workers} workers (rough)."
    )
    if args.plan:
        return

    pool = multiprocessing.Pool(workers, initializer=sf.ignore_interrupts)
    est = sf.Estimator(pool, cache)
    try:
        confirm(rows, est, args.samples, seed)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        print(f"\nStopped. Estimates are saved in {args.cache}; rerun to carry on.")
        raise SystemExit(130) from None
    pool.close()
    pool.join()

    print(f"\nDone in {est.elapsed()}, {est.computed} new estimates.\n")
    print_rows(rows)
    out = [row_dict(r) for r in rows if r.result is not None]
    args.out.write_text(json.dumps({"rows": out}, indent=2) + "\n")
    print(f"\nWritten to {args.out}.")


if __name__ == "__main__":
    main()
