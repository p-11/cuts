# Paper parameter sets

The parameter sets used in the paper, how they were chosen, and the tables built from them. The tables and the per-set test vectors take their sets from `parameter_sets.py`.

| File | What it is |
|---|---|
| `parameter_sets.py` | The 10 frozen sets. |
| `search_front.py`, `confirm_rows.py` | Steps 1 and 2: the search that picked each set's shape. |
| `confirmed_rows.json` | The search's result. |
| `lambda2_bound.py` | Step 3: proves each set's two-use security and fixes its L. |
| `lambda2_bounds.json` | The proofs. |
| `make_tables.py` | Writes the paper's LaTeX tables to `build/tables/` (or `--out DIR`): the parameter tables, hash counts, the comparison, and the worked-encoding appendix. |

## How the sets were chosen

All sets are binary, with n = 16, and give 128 bits for one use. Two-use security is the bits a key keeps after signing two messages; for CUTS+C (16-bit counter) it includes the grinding bits g = r − log₂ M.

1. **Search** (`search_front.py`). Over h ≤ 15, w ≤ 5, signatures up to 2,400 B, and S at four points in the upper half of its range, find for each two-use target the smallest L reaching it, using Monte Carlo estimates from `scripts/cuts_analysis`. Targets: 128 for CUTS; 48, 64, 80, 96, 112 and 128 for CUTS+C.
2. **Shortlist** (`confirm_rows.py`). For each target, take the smallest-signature, lowest-verification and cheap-keygen (G ≤ 16,384) sets and confirm them with 128-sample estimates. Ten of these were chosen for the paper. Their **shape**, meaning h, w and the rule for S, is fixed from here.
3. **Proof** (`lambda2_bound.py`). For each of the ten, prove a lower bound on λ₂ with the congruence-filtered bound of the paper's "Reuse security" subsection, and set **L** to the smallest value for that shape whose proven two-use security reaches the target (λ₂ and g each rounded down to 0.1 bit, as in the tables). `lambda2_bounds.json` records the bound at L and at L − 1.

The Monte Carlo L in `confirmed_rows.json` can differ from the final L, in either direction: the proof decides.

## Reproducing

From the repository root, with the virtualenv from the main README active:

```bash
PYTHONPATH=scripts python scripts/paper_tables/search_front.py
PYTHONPATH=scripts python scripts/paper_tables/confirm_rows.py
PYTHONPATH=scripts python scripts/paper_tables/lambda2_bound.py
PYTHONPATH=scripts python scripts/paper_tables/make_tables.py
```

On 10 cores, steps 1, 2 and 4 take about 3 hours, 20 minutes and seconds. Step 3 takes about 2 hours 15 minutes on 4 cores, most of it certifying each bound in ball arithmetic. Steps 1 and 2 cache their estimates in `search_front_cache.jsonl` (not committed) and can be stopped and rerun. `lambda2_bound.py` exits with an error if an L in `parameter_sets.py` is not the smallest proven one.
