import math
from types import SimpleNamespace

import pytest
from paper_tables import search_front as sf

from cuts import CUTS


class InlinePool:
    """Runs each job at once, in-process, like a pool with one fast worker."""

    def apply_async(self, func, args, callback, error_callback):
        callback(func(*args))


ARGS = SimpleNamespace(probe_samples=16, seed=1)


def fake_estimator(monkeypatch, bits, se=0.0):
    """Replace the family estimate with bits(L), and record the calls."""
    calls = []

    def estimate(job):
        key, samples, seed = job
        calls.append((key.L, samples))
        return key, samples, seed, bits(key.L), se

    monkeypatch.setattr(sf, "estimate", estimate)
    return calls


def keys(lo=30, hi=149, h=15, grinding=0.0):
    """Consecutive-L keys; grinding > 0 makes them CUTS+C keys."""
    if not grinding:
        return [sf.Key(h, 2, L, L // 2, log2_M=2.0 * L) for L in range(lo, hi + 1)]
    return [
        sf.Key(h, 2, L, L // 2, 200, 16, log2_M=200 - grinding)
        for L in range(lo, hi + 1)
    ]


def run(searches, cache_path):
    est = sf.Estimator(InlinePool(), sf.Cache(cache_path))
    sf.probe_all(searches, est, ARGS)
    return est


def smallest(bits, target, lo=30, hi=149):
    return min(L for L in range(lo, hi + 1) if bits(L) >= target)


@pytest.mark.parametrize(
    "bits",
    [
        lambda L: 1.3 * L - 20,  # linear
        lambda L: 30 * math.sqrt(L) - 170,  # concave, like the real curve
        lambda L: 1.0 * L + 20,  # crosses near the start
    ],
)
def test_finds_smallest_L_cheaply(monkeypatch, tmp_path, bits):
    calls = fake_estimator(monkeypatch, bits)
    s = sf.Search(keys(), 128.0, {})
    run([s], tmp_path / "c.jsonl")
    assert s.key.L == smallest(bits, 128)
    assert len(calls) <= 10
    assert max(L for L, _ in calls) <= s.key.L + 4  # stays where estimates are cheap


def test_grinding_counts_toward_the_target(monkeypatch, tmp_path):
    fake_estimator(monkeypatch, lambda L: 1.3 * L - 20)
    s = sf.Search(keys(grinding=9.0), 128.0, {})
    run([s], tmp_path / "c.jsonl")
    assert s.key.L == 107  # 1.3 L - 20 + 9 >= 128


def test_targets_in_a_column_share_probes(monkeypatch, tmp_path):
    calls = fake_estimator(monkeypatch, lambda L: 1.3 * L - 20)
    curve: dict[int, float] = {}
    targets = (48, 64, 80, 96, 112, 128)
    ss = [sf.Search(keys(), float(t), curve) for t in targets]
    run(ss, tmp_path / "c.jsonl")
    for s, t in zip(ss, targets, strict=True):
        assert s.key.L == smallest(lambda L: 1.3 * L - 20, t)
    probes = [L for L, _ in calls]
    assert len(probes) == len(set(probes)) < 6 * 5


def test_gives_up_without_probing_the_top(monkeypatch, tmp_path):
    calls = fake_estimator(monkeypatch, lambda L: 0.5 * L)
    s = sf.Search(keys(), 128.0, {})
    run([s], tmp_path / "c.jsonl")
    assert s.searched and s.answer is None
    assert 149 not in [L for L, _ in calls]


def test_unreachable_column_is_settled(monkeypatch, tmp_path):
    fake_estimator(monkeypatch, lambda L: 1.3 * L - 20)
    s = sf.Search(keys(30, 100), 128.0, {})  # needs L = 114
    run([s], tmp_path / "c.jsonl")
    assert s.searched and s.answer is None


def test_shared_keys_are_estimated_once(monkeypatch, tmp_path):
    calls = fake_estimator(monkeypatch, lambda L: 1.3 * L - 20)
    a, b = sf.Search(keys(), 128.0, {}), sf.Search(keys(), 128.0, {})
    run([a, b], tmp_path / "c.jsonl")
    assert len(calls) == len(set(calls))
    assert a.key == b.key


def test_rerun_settles_from_the_cache(monkeypatch, tmp_path):
    calls = fake_estimator(monkeypatch, lambda L: 1.3 * L - 20)
    cache = tmp_path / "c.jsonl"
    first = sf.Search(keys(), 128.0, {})
    run([first], cache)
    n = len(calls)
    # A rerun, here with an extra target, reuses every cached probe.
    again = [sf.Search(keys(), 128.0, {})]
    again.append(sf.Search(again[0].keys, 112.0, again[0].curve))
    sf.preload(again, sf.Cache(cache), ARGS.probe_samples, ARGS.seed)
    run(again, cache)
    assert again[0].key == first.key
    assert len(calls) - n <= 2  # only what the new target needs


def test_pareto_keeps_ties():
    k = [sf.Key(1, 1, L, 0) for L in range(2, 7)]
    costs = {k[0]: (1, 5), k[1]: (2, 2), k[2]: (5, 1), k[3]: (3, 3), k[4]: (2, 2)}
    assert set(sf.pareto(costs, lambda key: costs[key])) == {k[0], k[1], k[2], k[4]}


def test_grid_keys_are_valid():
    a = sf.parse_args(["--max-h", "10", "--max-w", "4", "--max-bytes", "1700"])
    columns = sf.build_columns(a)
    assert {scheme for scheme, _ in columns} == {sf.CUTS, sf.CUTS_C}
    for (scheme, (h, w, _)), ks in columns.items():
        Ls = [key.L for key in ks]
        assert Ls == list(range(Ls[0], Ls[-1] + 1))
        for key in ks[:: max(1, len(ks) // 4)]:
            cuts = CUTS(key.params)  # validates the parameters
            assert (key.scheme, key.h, key.w) == (scheme, h, w)
            assert key.bytes <= 1700
            assert key.S >= math.ceil(key.L * (w - 1) / 2)
            assert math.isclose(key.log2_M, cuts.selector_entropy_bits)
            assert cuts.one_use_security_bits >= 128
            if scheme == sf.CUTS_C:
                assert key.counter_bits == 16 and key.r >= 128
                assert math.isclose(key.grinding, cuts.grinding_bits, abs_tol=1e-6)


def test_cuts_and_cuts_c_share_curves():
    a = sf.parse_args(["--max-h", "12", "--max-w", "3"])
    searches = sf.make_searches(sf.build_columns(a), a)
    by_curve: dict[int, set[str]] = {}
    for s in searches:
        by_curve.setdefault(id(s.curve), set()).add(s.scheme)
    assert any(schemes == {sf.CUTS, sf.CUTS_C} for schemes in by_curve.values())
