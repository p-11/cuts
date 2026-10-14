# CUTS

CUTS (Chained Uniform-sum Tree Signatures) is a few-time hash-based signature scheme.

This repository contains a reference implementation in Python, and parameter sets.

## Layout

    src/                    the scheme
    scripts/                parameter search, table generation, other tooling
    tests/                  tests and test vectors

### The scheme

`src/` is the only implementation of CUTS in this repository. It accepts any valid parameter set.



## Install

Requires Python 3.10 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Implementation

`src/scheme.py` is a single-file implementation of the CUTS specification: key generation, signing and verification. It supports both CUTS-cs and CUTS+C, as specified (see below). It uses only the Python standard library (`hashlib` for SHAKE256). No other packages are needed at runtime.


### Usage

```python
from cuts import CUTS, CUTSParams

scheme = CUTS(CUTSParams(n=16, h=14, w=2, L=49, S=24))

sk, pk = scheme.keygen()
sig = scheme.sign(sk, b"message")
assert scheme.verify(pk, b"message", sig)
```

`CUTSParams` rejects values outside the ranges in the specification. `CUTS(params)` precomputes the counting tables, so build it once per parameter set and reuse it. Keys serialise with `sk.to_bytes()` and `SecretKey.from_bytes(data, n)`. The public key and the signature are plain `bytes`.

Each instance also reports its exact costs: `params.signature_bytes`, `params.verification_hashes`, `params.keygen_hashes` and `scheme.selector_entropy_bits`. `scheme.one_use_security_bits` is its security for one signature.

### CUTS+C

Set `index_bits` (r) and `counter_bits` (s) to use it instead of CUTS-cs:

```python
from cuts.scheme import widest_index_bits

M = CUTS(CUTSParams(n=16, h=11, w=5, L=32, S=78)).M
r = widest_index_bits(M, 16)  # 128 for this family
scheme = CUTS(CUTSParams(n=16, h=11, w=5, L=32, S=78, index_bits=r, counter_bits=16))
```

For counters c = 0, 1, 2, … the signer hashes `DST_hmsg || PID || enc_s(c) || PK || rho || m`, keeps the first r bits as an integer u, and stops at the first u < M. It unranks u exactly as CUTS-cs does and sends c in `ceil(s/8)` bytes: the signature is `rho || c || Z_0 || ... || Z_{L-1}`. The verifier checks that one counter, so it hashes the message once, as in CUTS-cs.

A forger has to grind as well, so every security figure gains g = r − log2 M bits (`scheme.grinding_bits`), and one-use security is min(8n, r) whatever S is. S can therefore rise to cut verification while r stays at 128 or more. `widest_index_bits(M, s)` gives the largest r a counter of s bits allows; each counter bit buys about one bit of g, up to g ≈ s − 6.5.

- r and s are part of the parameter identifier, so a CUTS-cs key and a CUTS+C key with the same (n, h, w, L, S) never share a tree.
- `CUTS(params)` rejects a parameter set whose signer would run out of counters with probability above 2^-128. If it happens anyway, signing raises an error; it never produces an invalid signature.

`counter_bits=0`, the default, is CUTS-cs.

### Demo

```bash
python3 scripts/demo.py
```

The demo prints the metrics for one parameter set, then runs keygen, sign and verify. It also shows that a signature fails for a different message and under a different parameter set. It ends with a CUTS+C instance, showing the counter the signer found and the bits grinding adds.

## Tests

```bash
pytest
```

Test vectors for each named parameter set go in `tests/vectors/`.

## Linting and formatting

The code is linted and formatted with [ruff](https://docs.astral.sh/ruff/), which `pip install -e ".[dev]"` installs. Run both from the repository root:

```bash
ruff check .            # lint
ruff format --check .   # formatting; lists files that would change
```

To fix what ruff can fix on its own, then reformat:

```bash
ruff check --fix .
ruff format .
```

Run both checks, and `pytest`, before committing.

## Status

Research code. Not audited. Do not use it to protect anything.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this repository, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
