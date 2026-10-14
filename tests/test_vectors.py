"""The test vectors in tests/vectors/vectors.json against the reference implementation.

Each vector is re-derived from its seed and message: the public key and the
signature must come out byte for byte, the signature must verify, and a
changed signature must not. A change to the scheme that alters any byte fails
here until the vectors are regenerated with scripts/test_vectors/make_vectors.py.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from cuts import CUTS, CUTSParams

VECTORS: list[dict[str, Any]] = json.loads(
    (Path(__file__).parent / "vectors" / "vectors.json").read_text()
)["vectors"]
NAMES = [v["name"] for v in VECTORS]


def params_of(v: dict[str, Any]) -> CUTSParams:
    x = v["params"]
    return CUTSParams(x["n"], x["h"], x["w"], x["L"], x["S"], x["r"], x["s"])


@pytest.mark.parametrize("v", VECTORS, ids=NAMES)
def test_vector_matches_the_reference_implementation(v: dict[str, Any]):
    scheme = CUTS(params_of(v))
    message = bytes.fromhex(v["message"])
    sk, pk = scheme.keygen(bytes.fromhex(v["seed"]))
    sig = scheme.sign(sk, message)
    assert pk.hex() == v["public_key"]
    assert sig.hex() == v["signature"]
    assert scheme.verify(pk, message, sig)


@pytest.mark.parametrize("v", VECTORS, ids=NAMES)
def test_a_changed_signature_does_not_verify(v: dict[str, Any]):
    scheme = CUTS(params_of(v))
    sig = bytearray.fromhex(v["signature"])
    sig[-1] ^= 1
    pk = bytes.fromhex(v["public_key"])
    assert not scheme.verify(pk, bytes.fromhex(v["message"]), bytes(sig))
