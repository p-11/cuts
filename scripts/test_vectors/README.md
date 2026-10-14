# Test vectors

`make_vectors.py` generates known-answer vectors for CUTS and CUTS+C from the reference implementation in `src/scheme.py`, verifying each one before writing it.

It writes `tests/vectors/vectors.json`: 12 vectors, one per mode on the small family (n, h, w, L, S) = (16, 2, 3, 3, 2), and one for each set in `scripts/paper_tables/parameter_sets.py`. `tests/test_vectors.py` checks every one against the implementation.

Each vector gives the parameters, seed, message, public key and signature, and the encoding it opens (frontier `F`, positions `A`, and the counter for CUTS+C). Byte strings are hexadecimal. The seed is the first n bytes of SHAKE256 over `"CUTS test vector seed: "` followed by the vector's name, and the message is `"CUTS test vector"`.

From the repository root:

```bash
PYTHONPATH=scripts python scripts/test_vectors/make_vectors.py
PYTHONPATH=scripts python scripts/test_vectors/make_vectors.py --check
```

Rerun it after any change to the scheme that alters a byte; `--check` (run in CI) fails until then.
