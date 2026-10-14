# Stateful-path search

`search_stateful.py` searches for the CUTS+C key to use in the stateful path of CINCS.

SHRINCS signs statefully with many one-time WOTS+C keys under one root: its state counter moves the signer to a fresh key for every signature. If state fails and a key signs twice, WOTS+C falls to about 34 bits. CINCS keeps that design and replaces each WOTS+C key with a CUTS+C key, which keeps more security if it is signed twice by mistake.

Whatever tree holds the keys adds the same authentication path and index to every signature, so the search compares key against key. Every key it considers gives 128 bits for its one planned signature. For each key it reports:

- size, verification and keygen cost, in SHA-256 compressions as in the SHRINCS BIP, assuming CUTS is hashed the way SHRINCS hashes;
- "reuse": two-use security, the bits left if the key signs two different messages.

Reuse is estimated with the copy of cuts-analysis in `scripts/cuts_analysis`.

## Running it

From the repository root:

```bash
PYTHONPATH=scripts python3 scripts/search_stateful/search_stateful.py --plan   # how much work the limits imply, then stop
PYTHONPATH=scripts python3 scripts/search_stateful/search_stateful.py
```

With the defaults a run from scratch takes about half an hour on four cores. A line is printed as each estimate finishes. Stop with Ctrl-C at any time; running the same command again carries on where it stopped.

Estimates are saved to `search_stateful_cache.jsonl` in the directory you run from (ignored by git). They are keyed by (h, w, L, S), so changing the limits or the counter width reuses earlier work.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--max-key-bytes` | 1300 | Size limit per key (WOTS+C: 514) |
| `--max-key-verify` | 350 | Verification limit per key, in compressions (WOTS+C: 250) |
| `--max-key-keygen` | 32768 | Keygen limit per key, in compressions (WOTS+C: 521) |
| `--counter-bits` | 16 | Counter width s, as in SHRINCS. Each extra bit adds about one bit of grinding |
| `--s-fractions` | 0 0.5 1 | S points tried, from the midpoint of its range (0) to the highest S that keeps r at 128 (1) |
| `--max-w`, `--max-h` | 16, 20 | Largest chain width and tree height considered |
| `--probe-samples`, `--confirm-samples` | 32, 128 | Monte Carlo samples per estimate in stages 2 and 3 |
| `--seed` | 1 | Monte Carlo seed |
| `--workers` | all cores | Worker processes |
| `--cache` | `search_stateful_cache.jsonl` | Where estimates are saved |
| `--plan` | | Print the stage 1 counts and stop |

For a quicker search closer to SHRINCS's size, use `--max-key-bytes 800 --max-key-verify 250`. For taller trees and one more bit of grinding, use `--max-key-keygen 131072 --counter-bits 17`.

## How it works

1. **Exact.** Size, verification, keygen, one-use security and grinding have closed forms, so the whole grid is filtered to the limits at once. Each key uses the widest r its counter allows, so grinding adds its full g = r − log2 M bits. One-use security is then r whatever S is, so S is tried at several points up to the highest value that keeps r at 128.
2. **Probe.** Reuse security has no closed form. It is estimated at three key lengths L in each (h, w, S point) column and interpolated in between, since it grows almost linearly in L.
3. **Confirm.** The keys the probes put on the front are re-estimated with more samples. This repeats until every key on the front has been confirmed.

## Reading the output

The final table lists the front: keys that no other key matches or beats on size, verification and reuse together. The first row is the SHRINCS WOTS+C key, for comparison.

| Column | Meaning |
|---|---|
| key | h, w, L, S and r of the CUTS+C key |
| bytes, verify, keygen | Cost of the key alone |
| reuse | Two-use security in bits, with its 95% interval |
| 1st sig, verify | The first SHRINCS stateful signature with this key in place of WOTS+C |
| keygen 32, keygen 256 | Stateful keygen for a tree of 32 or 256 keys |

To choose a key:

1. The row near 514 bytes shows what reuse security you get at WOTS+C's size.
2. For the reuse bound you would accept, find the smallest row whose reuse, less its interval, still clears it. Its extra bytes over 514 are what that safety costs per signature.
3. Check that row's verification (255 or less matches SHRINCS) and its keygen columns against the devices you target. Keygen is where CUTS costs most, because every key is a whole tree.

If nothing reaches the bound within the keygen limit, rerun with a higher `--max-key-keygen`. The cache keeps the earlier work.

## Caveats

- Reuse figures are Monte Carlo estimates plus the grinding bits. Small sample counts tend to err high, which is why the front is re-checked with more samples.
- The grinding bonus rests on the same heuristic argument as SPHINCS+C, not a proof.
- Costs assume SHA-256 hashing in the SHRINCS style. The reference implementation uses SHAKE256, so the hash function has to be settled before these become real counts.
