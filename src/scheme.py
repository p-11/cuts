"""CUTS: Chained Uniform-sum Tree Signatures.

A reference implementation of the CUTS few-time hash-based signature scheme.

Two ways of mapping a message to an encoding are supported:

- CUTS-cs (``counter_bits=0``, the specification as written): the message hash
  is unranked directly into the encoding family.
- CUTS+C (``counter_bits > 0``), following the CUTS+C specification: the
  message hash, with a counter, is read as an ``index_bits``-bit integer, and
  the signer grinds the counter until that integer is below M. It is then
  unranked as in CUTS-cs. The counter is sent with the signature.

Usage::

    from cuts import CUTS, CUTSParams

    scheme = CUTS(CUTSParams(n=16, h=14, w=2, L=49, S=24))
    sk, pk = scheme.keygen()
    sig = scheme.sign(sk, b"message")
    assert scheme.verify(pk, b"message", sig)

The only dependency is the Python standard library (``hashlib`` for SHAKE256).

Research code. Not audited and not constant-time. Do not use it to protect
anything.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import secrets
from dataclasses import dataclass
from typing import NamedTuple, overload

__all__ = ["CUTS", "CUTSParams", "Encoding", "SecretKey"]

# Domain separation tags
DST_LEAF = b"CUTS/LEAF/v0"
DST_MERGE = b"CUTS/MERGE/v0"
DST_CHAIN = b"CUTS/CHAIN/v0"
DST_RAND = b"CUTS/RAND/v0"
DST_HMSG = b"CUTS/HMSG/v0"

# Tags of the hashes that process the message rather than the tree. The
# verification counts (V) exclude these.
MESSAGE_DSTS = (DST_HMSG,)

# CUTS+C parameter sets must make it this unlikely (log2) that a signer runs
# through every counter value without landing in the family.
GRIND_FAILURE_LOG2_MAX = -128

_U64_MAX = (1 << 64) - 1

Frontier = tuple[int, ...]
Decoration = tuple[int, ...]


def enc(x: int) -> bytes:
    """Encode an integer as exactly 8 bytes, unsigned big-endian."""
    if not 0 <= x <= _U64_MAX:
        raise ValueError(f"enc() needs 0 <= x < 2^64, got {x}")
    return x.to_bytes(8, "big")


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CUTSParams:
    """One CUTS instance: (n, h, w, L, S), plus (r, s) for CUTS+C.

    n: hash output length in bytes. Also the length of SK, PK, rho and each
       revealed value.
    h: tree height. Leaves sit at depth h, the root at depth 0.
    w: positions per chain, so each chain has w - 1 hash steps.
    L: frontier size, the number of values revealed per signature.
    S: required sum of the L revealed chain positions.
    index_bits: CUTS+C only (r in the specification). The message hash is
       read as an r-bit integer, so 2^r must be at least M.
    counter_bits: CUTS+C only (s in the specification). Width of the grinding
       counter sent in the signature (SHRINCS uses 16). 0 means CUTS-cs.
    """

    n: int
    h: int
    w: int
    L: int
    S: int
    index_bits: int = 0
    counter_bits: int = 0

    def __post_init__(self) -> None:
        # Validity conditions from the head of the specification.
        if not 1 <= self.n <= _U64_MAX:
            raise ValueError("need 1 <= n < 2^64")
        if not 1 <= self.w <= _U64_MAX:
            raise ValueError("need 1 <= w < 2^64")
        if not 1 <= self.h <= 63:
            raise ValueError("need 1 <= h <= 63")
        if not 2 <= self.L <= 1 << self.h:
            raise ValueError("need 2 <= L <= 2^h")
        if not 0 <= self.S <= min(self.L * (self.w - 1), _U64_MAX):
            raise ValueError("need 0 <= S <= L(w - 1) and S < 2^64")
        if not 0 <= self.counter_bits <= 64:
            raise ValueError("need 0 <= counter_bits <= 64")
        if (self.counter_bits == 0) != (self.index_bits == 0):
            raise ValueError("need both index_bits and counter_bits for CUTS+C")
        if not 0 <= self.index_bits <= _U64_MAX:
            raise ValueError("need 0 <= index_bits < 2^64")

    @property
    def pid(self) -> bytes:
        """Parameter identifier mixed into every hash.

        Always seven fields, with r = s = 0 for CUTS-cs. The fixed length keeps
        every hash input unambiguous across the two modes, and a CUTS-cs key
        and a CUTS+C key with the same (n, h, w, L, S) never share a tree.
        """
        fields = (self.n, self.h, self.w, self.L, self.S)
        fields += (self.index_bits, self.counter_bits)
        return b"".join(enc(x) for x in fields)

    @property
    def counter_bytes(self) -> int:
        """Bytes the grinding counter takes in a signature (0 for CUTS-cs)."""
        return (self.counter_bits + 7) // 8

    # Metrics. These are exact counts, derived from the parameters alone.

    @property
    def signature_bytes(self) -> int:
        """|sigma| = (L + 1) n, plus the counter for CUTS+C."""
        return (self.L + 1) * self.n + self.counter_bytes

    @property
    def verification_hashes(self) -> int:
        """V = (L - 1)(2w - 1) - S, excluding message hashing.

        Message hashing adds one more hash, for CUTS-cs and CUTS+C alike (the
        CUTS+C specification counts it, giving V + 1).
        """
        return (self.L - 1) * (2 * self.w - 1) - self.S

    @property
    def keygen_hashes(self) -> int:
        """G = (2^(h+1) - 2) w + 1: leaves, chain steps and merges."""
        return ((1 << (self.h + 1)) - 2) * self.w + 1


@dataclass(frozen=True)
class SecretKey:
    """Stored secret key sk = (SK, PK).

    PK is kept alongside SK so signing does not have to rebuild the root.
    """

    seed: bytes  # SK, n uniformly random bytes
    public_key: bytes  # PK, the n-byte root value

    def to_bytes(self) -> bytes:
        return self.seed + self.public_key

    @classmethod
    def from_bytes(cls, data: bytes, n: int) -> SecretKey:
        if len(data) != 2 * n:
            raise ValueError(f"secret key must be {2 * n} bytes")
        return cls(seed=data[:n], public_key=data[n:])


class Encoding(NamedTuple):
    """The decorated frontier a message selects, and the counter that found it."""

    frontier: Frontier
    positions: Decoration
    counter: int | None = None  # CUTS+C only


# --------------------------------------------------------------------------
# Counting tables
# --------------------------------------------------------------------------


def frontier_counts(h: int, L: int) -> list[list[int]]:
    """N[t][size]: number of size-node frontiers of a complete height-t binary tree.

    A frontier either selects the subtree root (size 1) or splits `size` between
    the left and right subtrees. A height-(t-1) subtree has 2^(t-1) leaves,
    so each side can hold between 1 and 2^(t-1) nodes. Only size <= L is needed.
    """
    N = [[0] * (L + 1) for _ in range(h + 1)]
    N[0][1] = 1
    for t in range(1, h + 1):
        half = 1 << (t - 1)
        N[t][1] = 1
        for size in range(2, L + 1):
            lo, hi = max(1, size - half), min(half, size - 1)
            N[t][size] = sum(
                N[t - 1][a] * N[t - 1][size - a] for a in range(lo, hi + 1)
            )
    return N


def decoration_counts(w: int, L: int, S: int) -> list[list[int]]:
    """D[k][s]: number of k-tuples over {0, ..., w-1} that sum to s.

    D[k][s] is the sum of D[k-1][s-a] for a = 0 .. min(w-1, s). A running
    prefix sum over D[k-1] turns each entry into one subtraction.
    """
    D = [[0] * (S + 1) for _ in range(L + 1)]
    D[0][0] = 1
    for k in range(1, L + 1):
        prefix = [0]
        for value in D[k - 1]:
            prefix.append(prefix[-1] + value)
        for s in range(S + 1):
            D[k][s] = prefix[s + 1] - prefix[max(0, s - w + 1)]
    return D


def acceptance_probability(M: int, index_bits: int) -> float:
    """Chance that one CUTS+C counter lands in the family: M / 2^r."""
    return M / (1 << index_bits)


def grind_failure_log2(p_accept: float, counter_bits: int) -> float:
    """log2 of the chance that all 2^counter_bits counters are rejected."""
    if p_accept >= 1:
        return -math.inf
    return (1 << counter_bits) * math.log2(1 - p_accept)


def widest_index_bits(M: int, counter_bits: int) -> int:
    """Largest r for which grinding still fails with probability <= 2^-128.

    Every security figure gains r - log2 M bits, so this is the r to choose
    unless signing time matters more. A counter of s bits allows about
    s - 6.5 bits of grinding.

    Raises ValueError if no r works, i.e. the counter is too short even
    for the smallest r with 2^r >= M.
    """
    r = max(1, (M - 1).bit_length())  # smallest r with 2^r >= M
    if (
        grind_failure_log2(acceptance_probability(M, r), counter_bits)
        > GRIND_FAILURE_LOG2_MAX
    ):
        raise ValueError(
            f"a {counter_bits}-bit counter is too short for M = {M} at any index_bits"
        )
    while (
        grind_failure_log2(acceptance_probability(M, r + 1), counter_bits)
        <= GRIND_FAILURE_LOG2_MAX
    ):
        r += 1
    return r


# --------------------------------------------------------------------------
# The scheme
# --------------------------------------------------------------------------


class CUTS:
    """A CUTS instance. Build one per parameter set and reuse it.

    Construction precomputes the counting tables and the message-hash
    constants, so the per-call methods only do hashing and table lookups.
    """

    def __init__(self, params: CUTSParams) -> None:
        self.params = params
        p = params
        self._pid = p.pid
        self._N = frontier_counts(p.h, p.L)
        self._D = decoration_counts(p.w, p.L, p.S)

        # Size of the encoding family.
        self.M = self._N[p.h][p.L] * self._D[p.L][p.S]

        if p.counter_bits:
            if 1 << p.index_bits < self.M:
                raise ValueError(
                    f"need 2^index_bits >= M, i.e. index_bits >= "
                    f"{(self.M - 1).bit_length()}"
                )
            self._index_bytes = (p.index_bits + 7) // 8
            self.p_accept = acceptance_probability(self.M, p.index_bits)
            if (
                grind_failure_log2(self.p_accept, p.counter_bits)
                > GRIND_FAILURE_LOG2_MAX
            ):
                try:
                    limit = f"at most {widest_index_bits(self.M, p.counter_bits)}"
                except ValueError:
                    limit = "no index_bits works"
                raise ValueError(
                    f"{p.counter_bits}-bit counter is too short for index_bits = "
                    f"{p.index_bits}: {limit}"
                )
        else:
            # Message-hash output length and rejection bound.
            # (M - 1).bit_length() is ceil(log2 M) for every M >= 1.
            self._beta = ((self.M - 1).bit_length() + 256 + 7) // 8
            self._Q = ((1 << (8 * self._beta)) // self.M) * self.M
            self.p_accept = 1.0

    def __repr__(self) -> str:
        return f"CUTS({self.params!r})"

    @property
    def selector_entropy_bits(self) -> float:
        """log2 M, the entropy of the encoding family itself."""
        return math.log2(self.M)

    @property
    def grinding_bits(self) -> float:
        """g = r - log2 M: extra work per forgery attempt under CUTS+C.

        A forger has to grind too, so each attempt lands in any fixed set of
        encodings with probability 2^-g times what it would under CUTS-cs.
        Every q-use security figure computed over the family gains g bits;
        the specification writes lambda_2 over 2^r for this. Zero for CUTS-cs.
        """
        return math.log2(1 / self.p_accept)

    @property
    def one_use_security_bits(self) -> float:
        """min(8n, log2 M) for CUTS-cs, min(8n, r) for CUTS+C.

        After one signature the only forgeable encoding is the signed one.
        """
        return min(8 * self.params.n, self.selector_entropy_bits + self.grinding_bits)

    # ---- public API -------------------------------------------------------

    def keygen(self, seed: bytes | None = None) -> tuple[SecretKey, bytes]:
        """Generate a key pair. Returns (sk, pk).

        Pass `seed` (n bytes) for reproducible test vectors. Otherwise SK is
        drawn from the OS random number generator.

        Cost: G = params.keygen_hashes hash calls, since the whole tree is
        built to reach the root.
        """
        n = self.params.n
        if seed is None:
            seed = secrets.token_bytes(n)
        elif len(seed) != n:
            raise ValueError(f"seed must be {n} bytes")
        pk = self._merge(1, self._chain_end(seed, 2), self._chain_end(seed, 3))
        return SecretKey(seed=bytes(seed), public_key=pk), pk

    def sign(self, sk: SecretKey, message: bytes) -> bytes:
        """Sign `message`. Returns rho || counter || Z_0 || ... || Z_{L-1}.

        The counter is present only under CUTS+C.

        Signing is deterministic: the same key and message always give the
        same signature. Signing *different* messages under one key reveals
        more of the tree each time; security after q signatures is what the
        parameter set was sized for.

        Cost: the key stores only (SK, PK), so each Z_j = X_{v_j, a_j} is
        recomputed from the leaves of v_j's subtree. The frontier's subtrees
        cover every leaf, so one signature costs close to one keygen. CUTS+C
        adds 2^g message hashes on average, one per counter tried.
        """
        message = bytes(message)
        rho = self._shake(DST_RAND, self._pid, sk.seed, sk.public_key, message)
        encoding = self.select(sk.public_key, rho, message)
        values = [
            self._advance(v, self._chain_start(sk.seed, v), 0, a)
            for v, a in zip(encoding.frontier, encoding.positions, strict=True)
        ]
        counter = b""
        if encoding.counter is not None:
            counter = encoding.counter.to_bytes(self.params.counter_bytes, "big")
        return rho + counter + b"".join(values)

    def verify(self, pk: bytes, message: bytes, signature: bytes) -> bool:
        """Return True exactly when `signature` is valid for `message` under `pk`.

        Cost: V = params.verification_hashes hash calls, plus message hashing.
        """
        p = self.params
        n, w, L = p.n, p.w, p.L
        if len(pk) != n or len(signature) != p.signature_bytes:
            return False

        rho = signature[:n]
        counter = None
        if p.counter_bits:
            counter = int.from_bytes(signature[n : n + p.counter_bytes], "big")
        body = signature[n + p.counter_bytes :]
        values = [body[j * n : (j + 1) * n] for j in range(L)]

        encoding = self.select(pk, rho, bytes(message), counter)
        if encoding is None:  # CUTS+C: the counter does not land in the family
            return False

        # Hash each revealed value to the end of its chain, giving X_{v, w-1}.
        chain_ends = {
            v: self._advance(v, z, a, w - 1)
            for v, a, z in zip(
                encoding.frontier, encoding.positions, values, strict=True
            )
        }
        return hmac.compare_digest(self._recover(1, chain_ends), pk)

    # ---- message selection --------------------------------------

    @overload
    def select(
        self, pk: bytes, rho: bytes, message: bytes, counter: None = None
    ) -> Encoding: ...

    @overload
    def select(
        self, pk: bytes, rho: bytes, message: bytes, counter: int
    ) -> Encoding | None: ...

    def select(
        self, pk: bytes, rho: bytes, message: bytes, counter: int | None = None
    ) -> Encoding | None:
        """Map (PK, rho, m) to the encoding both sides use.

        CUTS-cs: `counter` must be None.
        CUTS+C, signer: leave `counter` as None to grind for the first
        counter that lands in the family.
        CUTS+C, verifier: pass the signature's counter. Returns None if it
        does not land in the family. That is the only case that returns None,
        so leaving `counter` as None always gives an encoding.
        """
        transcript = pk + rho + message
        if not self.params.counter_bits:
            if counter is not None:
                raise ValueError("CUTS-cs signatures carry no counter")
            return Encoding(*self.unrank(self._hash_to_rank(transcript)))

        if counter is not None:
            return self._grind(transcript, counter)
        for c in range(1 << self.params.counter_bits):
            encoding = self._grind(transcript, c)
            if encoding is not None:
                return encoding
        # Parameter validation makes this less likely than 2^-128.
        raise RuntimeError("no counter value lands in the encoding family")

    def unrank(self, r: int) -> tuple[Frontier, Decoration]:
        """The canonical bijection from [0, M) onto the encoding family."""
        if not 0 <= r < self.M:
            raise ValueError(f"rank must be in [0, {self.M})")
        p = self.params
        r_frontier, r_decoration = divmod(r, self._D[p.L][p.S])
        return (
            self._unrank_frontier(1, p.h, p.L, r_frontier),
            self._unrank_decoration(p.L, p.S, r_decoration),
        )

    def _hash_to_rank(self, transcript: bytes) -> int:
        """CUTS-cs: hash PID || enc(c) || T, T = PK || rho || m, to [0, M).

        Candidates u_c are beta-byte integers. Accepting only u_c < Q, a
        multiple of M, makes u_c mod M exactly uniform. Q covers all but a
        fraction below 2^-256 of the range, so c = 0 is accepted in practice.
        """
        for c in range(_U64_MAX + 1):
            digest = self._shake(
                DST_HMSG, self._pid, enc(c), transcript, length=self._beta
            )
            u = int.from_bytes(digest, "big")
            if u < self._Q:
                return u % self.M
        raise AssertionError("unreachable")

    def _grind(self, transcript: bytes, counter: int) -> Encoding | None:
        """CUTS+C: the encoding counter `counter` gives, or None if it misses.

        u = H_r(DST_hmsg || PID || enc_s(counter) || T): the first r bits of the
        SHAKE256 output, read as an unsigned integer. The counter is accepted
        only if u < M, so an accepted u is uniform on [0, M) and is unranked
        as in CUTS-cs. enc_s is the counter in exactly ceil(s / 8) bytes.
        """
        p = self.params
        if not 0 <= counter < 1 << p.counter_bits:
            return None
        counter_bytes = counter.to_bytes(p.counter_bytes, "big")
        hashed = self._shake(
            DST_HMSG, self._pid, counter_bytes, transcript, length=self._index_bytes
        )
        u = int.from_bytes(hashed, "big") >> (8 * self._index_bytes - p.index_bits)
        if u >= self.M:
            return None
        return Encoding(*self.unrank(u), counter)

    def _unrank_frontier(self, v: int, t: int, size: int, r: int) -> Frontier:
        """Frontier of rank r among the size-node frontiers under node v, height t.

        Frontiers are ordered first by how many nodes a go to the left
        subtree, then by left rank, then by right rank.
        """
        if size == 1:
            return (v,)
        N = self._N[t - 1]
        half = 1 << (t - 1)
        for a in range(max(1, size - half), min(half, size - 1) + 1):
            block = N[a] * N[size - a]
            if r < block:
                r_left, r_right = divmod(r, N[size - a])
                return self._unrank_frontier(
                    2 * v, t - 1, a, r_left
                ) + self._unrank_frontier(2 * v + 1, t - 1, size - a, r_right)
            r -= block
        raise AssertionError("unreachable for valid inputs")

    def _unrank_decoration(self, k: int, s: int, r: int) -> Decoration:
        """Position vector of rank r among k-tuples over [0, w) summing to s.

        Written as a loop: at each step pick the first entry a, whose block
        of tails has size D[k-1][s-a], then continue with the remaining k-1.
        """
        w = self.params.w
        out = []
        while k > 0:
            for a in range(max(0, s - (k - 1) * (w - 1)), min(w - 1, s) + 1):
                block = self._D[k - 1][s - a]
                if r < block:
                    break
                r -= block
            else:
                raise AssertionError("unreachable for valid inputs")
            out.append(a)
            k, s = k - 1, s - a
        return tuple(out)

    # ---- the hash tree ----------------------------------------------

    def _shake(self, *parts: bytes, length: int | None = None) -> bytes:
        """SHAKE256 over the concatenation of `parts`, `length` bytes out (default n)."""
        return hashlib.shake_256(b"".join(parts)).digest(length or self.params.n)

    def _leaf(self, seed: bytes, v: int) -> bytes:
        """F_leaf(SK, v): start of a leaf's chain."""
        return self._shake(DST_LEAF, self._pid, enc(v), seed)

    def _merge(self, v: int, left: bytes, right: bytes) -> bytes:
        """H_merge(v, x_L, x_R): start of an internal node's chain, or PK at v = 1."""
        return self._shake(DST_MERGE, self._pid, enc(v), left, right)

    def _advance(self, v: int, x: bytes, start: int, stop: int) -> bytes:
        """Apply H_chain(v, j, .) for j = start .. stop-1: X_{v,start} -> X_{v,stop}.

        With stop = w - 1 this is Advance(v, start, x).
        """
        for j in range(start, stop):
            x = self._shake(DST_CHAIN, self._pid, enc(v), enc(j), x)
        return x

    def _chain_start(self, seed: bytes, v: int) -> bytes:
        """X_{v,0}, computed from the leaves of v's subtree."""
        if v.bit_length() - 1 == self.params.h:  # depth of v is h: a leaf
            return self._leaf(seed, v)
        return self._merge(
            v, self._chain_end(seed, 2 * v), self._chain_end(seed, 2 * v + 1)
        )

    def _chain_end(self, seed: bytes, v: int) -> bytes:
        """X_{v,w-1}, the value v contributes to its parent's merge."""
        return self._advance(v, self._chain_start(seed, v), 0, self.params.w - 1)

    def _recover(self, v: int, chain_ends: dict[int, bytes]) -> bytes:
        """Recover(v): rebuild v's chain end from the revealed frontier.

        Frontier nodes return their advanced value. Any other node merges its
        two children and, unless it is the root, walks its own chain to the
        end. A frontier covers every root-to-leaf path, so the recursion
        always stops at a frontier node before running past a leaf.
        """
        if v in chain_ends:
            return chain_ends[v]
        y = self._merge(
            v, self._recover(2 * v, chain_ends), self._recover(2 * v + 1, chain_ends)
        )
        return y if v == 1 else self._advance(v, y, 0, self.params.w - 1)
