from paper_tables import confirm_rows as cr
from paper_tables import search_front as sf


def search(probe_L, h=15, w=2, lo=30, hi=60, target=50.0):
    keys = [
        sf.Key(h, w, L, L * (w - 1) // 2, log2_M=2.0 * L) for L in range(lo, hi + 1)
    ]
    s = sf.Search(keys, target, {})
    s.answer = probe_L - lo
    return s


def confirm(row, bits, se=0.5):
    """Drive a row's windows with bits(L), as the confirm rounds would."""
    windows = [row.first_window()]
    while windows[-1]:
        for L in windows[-1]:
            row.tried[L] = (bits(L), se)
        windows.append(row.next_window())
    return windows[:-1]


def test_steps_up_from_the_probe_answer():
    row = cr.Row(search(40), ["smallest"], window=3)
    # Lower bound 1.3 L - 2 - 0.98 >= 50 first at L = 41.
    windows = confirm(row, lambda L: 1.3 * L - 2)
    assert row.result is not None and row.result.L == 41
    assert windows == [[40, 41, 42]]


def test_moves_the_window_up_when_nothing_passes():
    row = cr.Row(search(40), ["smallest"], window=3)
    windows = confirm(row, lambda L: 1.3 * L - 8)  # passes from L = 46
    assert row.result is not None and row.result.L == 46
    assert windows == [[40, 41, 42], [43, 44, 45], [46, 47, 48]]


def test_checks_one_below_when_the_window_starts_too_high():
    row = cr.Row(search(40), ["smallest"], window=3)
    windows = confirm(row, lambda L: 1.3 * L + 2)  # passes from L = 38
    assert row.result is not None and row.result.L == 38
    assert windows == [[40, 41, 42], [39], [38], [37]]


def test_unreachable_within_the_column():
    row = cr.Row(search(58), ["smallest"], window=3)
    confirm(row, lambda L: 10.0)
    assert row.result is None and row.unreachable


def test_shortlist_picks_smallest_fastest_and_cheap():
    small = search(31, h=15, w=5)  # fewest bytes, big keygen
    fast = search(50, h=15, w=1)  # lowest V
    cheap = search(40, h=8, w=3)  # keygen under the limit
    rows = cr.shortlist([small, fast, cheap], cheap_keygen=1 << 14, window=3)
    picked = {tuple(r.picked_as): r.search for r in rows}
    assert picked[("smallest",)] is small
    assert picked[("fastest",)] is fast
    assert picked[("cheap",)] is cheap
