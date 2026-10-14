# CUTS analysis

A small, standard-library-only package for:

- counting the CUTS decorated-frontier family;
- computing selector entropy and the costs specified in section 9;
- computing or estimating two-use reuse security
  `lambda_2 = -log2(p_2)`.

Here `p_2` is the probability that an independent uniformly selected target
encoding is derivable after observing two independent uniformly selected CUTS
encodings.

This package analyzes the combinatorial encoding family. It does not implement
keys, hashes, signing, or verification.

## Python API

This copy has no command line; import it from the repository root with
`scripts` on the path (`PYTHONPATH=scripts`, as pytest is configured to do).

```python
from cuts_analysis import CUTSAnalyzer, CUTSParams

params = CUTSParams(n=32, h=3, w=2, L=3, S=1)
cuts = CUTSAnalyzer(params)

metrics = cuts.metrics()
reuse = cuts.lambda2(method="auto", samples=10_000, seed=7)

print(metrics.selector_entropy)
print(metrics.signature_bytes)
print(metrics.verification_total)
print(reuse.lambda_bits)
```

The Monte Carlo method samples only the two observed encodings. For every
sampled pair, it uses a dynamic program to count all forgeable targets. This is
far more useful for small probabilities than sampling a third target and
waiting for rare successful forgeries.

Exact lambda_2 averages over all `M^2` ordered observation pairs. The
`max_pairs` argument prevents an accidentally huge exact run.
