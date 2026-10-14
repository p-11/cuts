"""Standard CUTS metrics and two-use reuse analysis."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

from .frontier import (
    Encoding,
    build_decoration_counts,
    build_frontier_counts,
    unrank_encoding,
)


@dataclass(frozen=True)
class CUTSParams:
    n: int
    h: int
    w: int
    L: int
    S: int

    def __post_init__(self) -> None:
        values = (self.n, self.h, self.w, self.L, self.S)
        if any(type(value) is not int for value in values):
            raise TypeError("n, h, w, L, and S must be integers")
        if not 1 <= self.n < (1 << 64):
            raise ValueError("n must satisfy 1 <= n < 2**64")
        if not 1 <= self.h <= 63:
            raise ValueError("h must satisfy 1 <= h <= 63")
        if not 1 <= self.w < (1 << 64):
            raise ValueError("w must satisfy 1 <= w < 2**64")
        if not 2 <= self.L <= (1 << self.h):
            raise ValueError("L must satisfy 2 <= L <= 2**h")
        if not 0 <= self.S < (1 << 64):
            raise ValueError("S must satisfy 0 <= S < 2**64")
        if self.S > self.L * (self.w - 1):
            raise ValueError("S must satisfy S <= L*(w-1)")


@dataclass(frozen=True)
class StandardMetrics:
    family_size: int
    selector_entropy: float
    signature_bytes: int
    public_key_bytes: int
    secret_key_bytes: int
    keygen_leaf_derivations: int
    keygen_chain_hashes: int
    keygen_merge_hashes: int
    keygen_total: int
    verification_revealed_chain_hashes: int
    verification_merge_hashes: int
    verification_reconstructed_chain_hashes: int
    verification_total: int


@dataclass(frozen=True)
class Lambda2Result:
    method: str
    lambda_bits: float
    probability_numerator: int
    probability_denominator: int
    observation_pairs: int
    standard_error_bits: float | None = None

    @property
    def probability_fraction(self) -> Fraction:
        return Fraction(self.probability_numerator, self.probability_denominator)

    @property
    def probability(self) -> float:
        """Return p_2 as a float; extremely small values can underflow to zero."""

        return float(self.probability_fraction)

    @property
    def confidence_interval_95(self) -> tuple[float, float] | None:
        """Approximate 95% interval for Monte Carlo lambda_2."""

        if self.standard_error_bits is None:
            return None
        radius = 1.96 * self.standard_error_bits
        return self.lambda_bits - radius, self.lambda_bits + radius


class CUTSAnalyzer:
    """Metrics and lambda_2 analysis for one fixed CUTS parameter set."""

    def __init__(self, params: CUTSParams):
        if not isinstance(params, CUTSParams):
            raise TypeError("params must be CUTSParams")

        self.params = params
        self.N = build_frontier_counts(params.h, params.L)
        self.D = build_decoration_counts(params.w, params.L, params.S)
        self.M = self.N[params.h][params.L] * self.D[params.L][params.S]

    def metrics(self) -> StandardMetrics:
        """Return family, size, key-generation, and verification metrics."""

        p = self.params
        leaf_derivations = 1 << p.h
        keygen_chains = ((1 << (p.h + 1)) - 2) * (p.w - 1)
        keygen_merges = (1 << p.h) - 1
        revealed_chains = p.L * (p.w - 1) - p.S
        verification_merges = p.L - 1
        reconstructed_chains = (p.L - 2) * (p.w - 1)

        return StandardMetrics(
            family_size=self.M,
            selector_entropy=math.log2(self.M),
            signature_bytes=(p.L + 1) * p.n,
            public_key_bytes=p.n,
            secret_key_bytes=2 * p.n,
            keygen_leaf_derivations=leaf_derivations,
            keygen_chain_hashes=keygen_chains,
            keygen_merge_hashes=keygen_merges,
            keygen_total=leaf_derivations + keygen_chains + keygen_merges,
            verification_revealed_chain_hashes=revealed_chains,
            verification_merge_hashes=verification_merges,
            verification_reconstructed_chain_hashes=reconstructed_chains,
            verification_total=(p.L - 1) * (2 * p.w - 1) - p.S,
        )

    def encoding(self, rank: int) -> Encoding:
        """Return the decorated frontier with canonical rank."""

        p = self.params
        return unrank_encoding(rank, p.h, p.w, p.L, p.S, self.N, self.D)

    def knowledge_closure(
        self,
        observed: Iterable[Encoding],
    ) -> dict[int, int]:
        """Return the earliest known position at each forward-derivable node."""

        earliest: dict[int, int] = {}

        for frontier, decoration in observed:
            if len(frontier) != self.params.L or len(decoration) != self.params.L:
                raise ValueError("each encoding must contain exactly L pairs")
            for node, position in zip(frontier, decoration):
                earliest[node] = min(position, earliest.get(node, position))

        for depth in range(self.params.h, 0, -1):
            parents = {
                node // 2
                for node in tuple(earliest)
                if node.bit_length() - 1 == depth
                and node % 2 == 0
                and node + 1 in earliest
            }
            for parent in parents:
                earliest[parent] = 0

        return earliest

    def is_forgeable(
        self,
        observed: Iterable[Encoding],
        target: Encoding,
    ) -> bool:
        """Return whether the observations reveal every target value."""

        earliest = self.knowledge_closure(observed)
        frontier, decoration = target
        return all(
            node in earliest and earliest[node] <= position
            for node, position in zip(frontier, decoration)
        )

    def forgeable_target_count(self, observed: Iterable[Encoding]) -> int:
        """Count all family encodings forgeable from the observations.

        The dynamic program simultaneously enforces the frontier structure,
        frontier size L, decoration bounds, and total decoration sum S.
        """

        earliest = self.knowledge_closure(observed)
        active = {1}
        for known_node in earliest:
            node = known_node
            while node:
                active.add(node)
                node //= 2

        p = self.params

        def count_at(node: int, depth: int) -> dict[tuple[int, int], int]:
            if node not in active:
                return {}

            counts: dict[tuple[int, int], int] = {}

            # Stop at this non-root node and select one reachable chain value.
            if node != 1 and node in earliest:
                last_position = min(p.w - 1, p.S)
                for position in range(earliest[node], last_position + 1):
                    counts[(1, position)] = 1

            # Or split into a frontier in each child subtree.
            if depth < p.h:
                left = count_at(2 * node, depth + 1)
                right = count_at(2 * node + 1, depth + 1)

                for (left_size, left_sum), left_count in left.items():
                    for (right_size, right_sum), right_count in right.items():
                        size = left_size + right_size
                        decoration_sum = left_sum + right_sum
                        if size <= p.L and decoration_sum <= p.S:
                            key = (size, decoration_sum)
                            counts[key] = counts.get(key, 0) + left_count * right_count

            return counts

        return count_at(1, 0).get((p.L, p.S), 0)

    def lambda2(
        self,
        *,
        method: str = "auto",
        samples: int = 10_000,
        seed: int | None = None,
        max_pairs: int = 100_000,
    ) -> Lambda2Result:
        """Compute or estimate lambda_2 = -log2(p_2).

        Exact analysis averages the forgeable-target count over all M^2 ordered
        observation pairs. Conditional Monte Carlo samples observation pairs
        but counts all compatible targets for each sampled pair.
        """

        self._positive_integer(max_pairs, "max_pairs")
        if method == "auto":
            method = "exact" if self.M * self.M <= max_pairs else "mc"
        if method == "exact":
            return self._lambda2_exact(max_pairs)
        if method in {"mc", "monte_carlo"}:
            self._positive_integer(samples, "samples")
            return self._lambda2_monte_carlo(samples, seed)
        raise ValueError("method must be 'auto', 'exact', 'mc', or 'monte_carlo'")

    def _lambda2_exact(self, max_pairs: int) -> Lambda2Result:
        pair_count = self.M * self.M
        if pair_count > max_pairs:
            raise ValueError(
                f"exact lambda_2 requires {pair_count:,} observation pairs; "
                f"max_pairs is {max_pairs:,}"
            )

        family = tuple(self.encoding(rank) for rank in range(self.M))
        total_forgeable = 0
        # Knowledge union is symmetric in the two observations, so evaluate
        # each unordered pair once and give distinct pairs weight two.
        for first_rank, first in enumerate(family):
            for second_rank in range(first_rank, self.M):
                weight = 1 if first_rank == second_rank else 2
                total_forgeable += weight * self.forgeable_target_count(
                    (first, family[second_rank])
                )

        denominator = self.M**3
        lambda_bits = self._negative_log2_ratio(total_forgeable, denominator)
        return Lambda2Result(
            method="exact",
            lambda_bits=lambda_bits,
            probability_numerator=total_forgeable,
            probability_denominator=denominator,
            observation_pairs=pair_count,
        )

    def _lambda2_monte_carlo(
        self,
        samples: int,
        seed: int | None,
    ) -> Lambda2Result:
        rng = random.Random(seed)
        total_forgeable = 0
        squared_forgeable = 0

        for _ in range(samples):
            observed = (
                self.encoding(rng.randrange(self.M)),
                self.encoding(rng.randrange(self.M)),
            )
            count = self.forgeable_target_count(observed)
            total_forgeable += count
            squared_forgeable += count * count

        denominator = samples * self.M
        lambda_bits = self._negative_log2_ratio(total_forgeable, denominator)

        # Delta-method uncertainty of -log2(sample mean). This is an
        # approximation, but the conditional estimator is always positive.
        if samples == 1:
            standard_error_bits = None
        else:
            moment_ratio = self._positive_ratio(
                samples * squared_forgeable,
                total_forgeable * total_forgeable,
            )
            relative_variance = max(0.0, (moment_ratio - 1.0) / (samples - 1))
            standard_error_bits = math.sqrt(relative_variance) / math.log(2)

        return Lambda2Result(
            method="conditional_monte_carlo",
            lambda_bits=lambda_bits,
            probability_numerator=total_forgeable,
            probability_denominator=denominator,
            observation_pairs=samples,
            standard_error_bits=standard_error_bits,
        )

    @staticmethod
    def _negative_log2_ratio(numerator: int, denominator: int) -> float:
        if numerator <= 0:
            return math.inf
        return math.log2(denominator) - math.log2(numerator)

    @staticmethod
    def _positive_ratio(numerator: int, denominator: int) -> float:
        if numerator <= 0 or denominator <= 0:
            raise ValueError("ratio inputs must be positive")
        return 2.0 ** (math.log2(numerator) - math.log2(denominator))

    @staticmethod
    def _positive_integer(value: int, name: str) -> None:
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
