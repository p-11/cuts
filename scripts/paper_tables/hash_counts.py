"""Hash-function costs of the paper's parameter sets.

Counts, for key generation, signing and verification, the Keccak-f[1600]
permutations of SHAKE256 and the compression-function calls of SHA-256,
instead of timings.

The byte length of every hash input comes from the reference implementation:
`measure` wraps CUTS._shake, signs and verifies one message, and records each
call. Signing and verification are counted call by call. Key generation is
counted from the cost theorem, per primitive, using the lengths signing
measured (`keygen_calls`; tests check it against an instrumented KeyGen).
CUTS+C signing is charged the expected number of message hashes,
(1 - (1 - p)^(2^s)) / p, not the number one message happened to need.

SHAKE256 absorbs 136-byte blocks: an n-byte input with output length d costs
floor(n / 136) + 1 permutations, plus ceil(d / 136) - 1 more to squeeze.

The reference implementation does not instantiate SHA-256. The SHA-256
figures hash the same serialized inputs: a message of n bytes costs
ceil((n + 9) / 64) compressions, and an output longer than 32 bytes is
expanded with MGF1-SHA-256 over the 32-byte digest, one compression per
32-byte block. The table caches no midstate. Every tree input starts with a
domain tag and the parameter identifier, which depend on the parameter set
but not on the key; `Cost.sha256_cached` also counts SHA-256 with that prefix
precomputed, saving one compression per whole 64-byte block it fills.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import cache

from cuts.scheme import (
    CUTS,
    DST_CHAIN,
    DST_HMSG,
    DST_LEAF,
    DST_MERGE,
    DST_RAND,
    CUTSParams,
    SecretKey,
)

MESSAGE = hashlib.sha256(b"CUTS hash-count message").digest()  # a 32-byte digest
TREE = (DST_LEAF, DST_MERGE, DST_CHAIN)
SHAKE256_RATE = 136
SHA256_BLOCK, SHA256_DIGEST = 64, 32


def shake256_permutations(in_bytes: int, out_bytes: int) -> int:
    return in_bytes // SHAKE256_RATE + 1 + max(0, -(-out_bytes // SHAKE256_RATE) - 1)


def sha256_compressions(in_bytes: int, out_bytes: int) -> int:
    inner = -(-(in_bytes + 9) // SHA256_BLOCK)
    if out_bytes <= SHA256_DIGEST:
        return inner
    # MGF1 over the digest: SHA-256(digest || 4-byte counter), 36 bytes each.
    return inner + -(-out_bytes // SHA256_DIGEST)


@dataclass(frozen=True)
class Call:
    dst: bytes
    in_bytes: int
    out_bytes: int
    prefix_bytes: int  # leading bytes fixed by the parameter set: tag (+ PID)


def sha256_cached(in_bytes: int, out_bytes: int, prefix_bytes: int) -> int:
    """Compressions when the midstate after the fixed prefix is precomputed."""
    return sha256_compressions(in_bytes, out_bytes) - prefix_bytes // SHA256_BLOCK


@dataclass
class Cost:
    shake256: float = 0.0
    sha256: float = 0.0
    sha256_cached: float = 0.0

    def add(self, call: Call, times: float = 1) -> None:
        self.shake256 += times * shake256_permutations(call.in_bytes, call.out_bytes)
        self.sha256 += times * sha256_compressions(call.in_bytes, call.out_bytes)
        self.sha256_cached += times * sha256_cached(
            call.in_bytes, call.out_bytes, call.prefix_bytes
        )


def record(scheme: CUTS, run) -> list[Call]:
    """Run `run()` and return every _shake call it made, with its byte lengths."""
    calls: list[Call] = []
    original = scheme._shake

    def recording(*parts: bytes, length: int | None = None) -> bytes:
        prefix = len(parts[0])
        if len(parts) > 1 and parts[1] == scheme._pid:
            prefix += len(parts[1])
        calls.append(
            Call(parts[0], sum(map(len, parts)), length or scheme.params.n, prefix)
        )
        return original(*parts, length=length)

    scheme._shake = recording
    try:
        run()
    finally:
        del scheme._shake
    return calls


def expected_message_hashes(scheme: CUTS) -> float:
    """Message hashes the signer makes on average.

    CUTS+C: counters tried until one is accepted (CUTS+C selection theorem).
    CUTS: the first candidate is rejected with probability below 2^-256, so 1.
    """
    p = scheme.params
    if not p.counter_bits:
        return 1.0
    q = scheme.p_accept
    return (1 - (1 - q) ** (1 << p.counter_bits)) / q


def keygen_calls(p: CUTSParams) -> dict[bytes, int]:
    """Tree primitive calls of KeyGen, per primitive (the cost theorem's G)."""
    return {
        DST_LEAF: 1 << p.h,
        DST_MERGE: (1 << p.h) - 1,
        DST_CHAIN: ((1 << (p.h + 1)) - 2) * (p.w - 1),
    }


@dataclass(frozen=True)
class HashCosts:
    keygen: Cost
    sign: Cost  # expected
    verify: Cost


@cache
def measure(params: CUTSParams) -> HashCosts:
    scheme = CUTS(params)
    n = params.n
    sk = SecretKey(seed=bytes(n), public_key=bytes(n))  # values do not matter
    signature = scheme.sign(sk, MESSAGE)
    sign_calls = record(scheme, lambda: scheme.sign(sk, MESSAGE))
    verify_calls = record(
        scheme, lambda: scheme.verify(sk.public_key, MESSAGE, signature)
    )

    # One call of each kind fixes its lengths; every call of a kind is alike.
    kind = {}
    for call in sign_calls + verify_calls:
        assert kind.setdefault(call.dst, call) == call, call
    chains = {DST_CHAIN} if params.w > 1 else set()  # w = 1 has no chain steps
    assert set(kind) == {DST_LEAF, DST_MERGE, DST_RAND, DST_HMSG} | chains, set(kind)

    keygen = Cost()
    for dst, count in keygen_calls(params).items():
        if count:
            keygen.add(kind[dst], count)

    sign = Cost()
    for call in sign_calls:
        if call.dst != DST_HMSG:
            sign.add(call)
    sign.add(kind[DST_HMSG], expected_message_hashes(scheme))

    verify = Cost()
    for call in verify_calls:
        verify.add(call)
    return HashCosts(keygen, sign, verify)


def rounded(x: float) -> int:
    return math.floor(x + 0.5)


def tree_prefix_bytes(params: CUTSParams) -> list[int]:
    """The parameter-fixed prefix of each tree primitive's input, in bytes."""
    scheme = CUTS(params)
    sk = SecretKey(seed=bytes(params.n), public_key=bytes(params.n))
    calls = record(scheme, lambda: scheme.sign(sk, MESSAGE))
    return sorted({c.prefix_bytes for c in calls if c.dst in TREE})
