"""Correctness tests for the CUTS analysis package."""

import math
import unittest

from cuts_analysis import CUTSAnalyzer, CUTSParams

from cuts import scheme


class MetricsTests(unittest.TestCase):
    def test_specification_metrics(self) -> None:
        cuts = CUTSAnalyzer(CUTSParams(n=32, h=3, w=2, L=3, S=1))
        metrics = cuts.metrics()

        self.assertEqual(metrics.family_size, 6)
        self.assertAlmostEqual(metrics.selector_entropy, math.log2(6))
        self.assertEqual(metrics.signature_bytes, 128)
        self.assertEqual(metrics.public_key_bytes, 32)
        self.assertEqual(metrics.secret_key_bytes, 64)
        self.assertEqual(metrics.keygen_leaf_derivations, 8)
        self.assertEqual(metrics.keygen_chain_hashes, 14)
        self.assertEqual(metrics.keygen_merge_hashes, 7)
        self.assertEqual(metrics.keygen_total, 29)
        self.assertEqual(metrics.verification_revealed_chain_hashes, 2)
        self.assertEqual(metrics.verification_merge_hashes, 2)
        self.assertEqual(metrics.verification_reconstructed_chain_hashes, 1)
        self.assertEqual(metrics.verification_total, 5)


class Lambda2Tests(unittest.TestCase):
    def test_conditional_count_matches_target_enumeration(self) -> None:
        cuts = CUTSAnalyzer(CUTSParams(n=32, h=3, w=2, L=3, S=1))
        family = tuple(cuts.encoding(rank) for rank in range(cuts.M))

        for first in family:
            for second in family:
                observed = (first, second)
                brute_count = sum(
                    cuts.is_forgeable(observed, target) for target in family
                )
                self.assertEqual(
                    cuts.forgeable_target_count(observed),
                    brute_count,
                )

    def test_exact_small_family(self) -> None:
        cuts = CUTSAnalyzer(CUTSParams(n=32, h=2, w=2, L=2, S=1))
        result = cuts.lambda2(method="exact")

        self.assertEqual(result.method, "exact")
        self.assertEqual(result.probability_fraction.numerator, 3)
        self.assertEqual(result.probability_fraction.denominator, 4)
        self.assertAlmostEqual(result.lambda_bits, -math.log2(0.75))

    def test_auto_selects_exact_for_small_family(self) -> None:
        cuts = CUTSAnalyzer(CUTSParams(n=32, h=2, w=2, L=2, S=1))
        self.assertEqual(cuts.lambda2().method, "exact")

    def test_conditional_monte_carlo_is_reproducible(self) -> None:
        cuts = CUTSAnalyzer(CUTSParams(n=32, h=3, w=2, L=3, S=1))
        first = cuts.lambda2(method="mc", samples=1000, seed=7)
        second = cuts.lambda2(method="mc", samples=1000, seed=7)

        self.assertEqual(first, second)
        self.assertIsNotNone(first.standard_error_bits)

    def test_exact_limit(self) -> None:
        cuts = CUTSAnalyzer(CUTSParams(n=32, h=3, w=2, L=3, S=1))
        with self.assertRaises(ValueError):
            cuts.lambda2(method="exact", max_pairs=35)


class SchemeAgreementTests(unittest.TestCase):
    """The counting tables here are a copy of cuts.scheme's; keep them in step."""

    def test_family_matches_scheme(self) -> None:
        for h, w, L, S in [
            (3, 2, 3, 1),
            (4, 3, 5, 4),
            (6, 4, 9, 13),
            (10, 16, 20, 150),
        ]:
            with self.subTest(h=h, w=w, L=L, S=S):
                analyzer = CUTSAnalyzer(CUTSParams(n=16, h=h, w=w, L=L, S=S))
                cuts = scheme.CUTS(scheme.CUTSParams(16, h, w, L, S))
                self.assertEqual(analyzer.M, cuts.M)
                self.assertEqual(analyzer.N, cuts._N)
                self.assertEqual(analyzer.D, cuts._D)

    def test_unranking_matches_scheme(self) -> None:
        analyzer = CUTSAnalyzer(CUTSParams(n=16, h=4, w=3, L=5, S=4))
        cuts = scheme.CUTS(scheme.CUTSParams(16, 4, 3, 5, 4))
        for rank in range(analyzer.M):
            self.assertEqual(analyzer.encoding(rank), cuts.unrank(rank))


if __name__ == "__main__":
    unittest.main()
