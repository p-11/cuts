"""Generate the test vectors from the reference implementation.

Writes tests/vectors/vectors.json, with one vector for each mode on the small
family (h, w, L, S) = (2, 3, 3, 2) and one for each set in parameter_sets.py.
Every vector is checked by verifying it before it is written. The file must
not be edited by hand.

Each vector's seed is the first n bytes of SHAKE256 over the ASCII string
"CUTS test vector seed: " followed by the vector's name. The message is the
ASCII string "CUTS test vector".

Usage, from the repository root:
    PYTHONPATH=scripts python scripts/test_vectors/make_vectors.py
    PYTHONPATH=scripts python scripts/test_vectors/make_vectors.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import TypedDict

from paper_tables.parameter_sets import COUNTER_BITS, PARAMETER_SETS

from cuts import CUTS, CUTSParams
from cuts.scheme import widest_index_bits

ROOT = Path(__file__).parents[2]
JSON_OUT = ROOT / "tests" / "vectors" / "vectors.json"
MESSAGE = b"CUTS test vector"
TOY = (16, 2, 3, 3, 2)  # (n, h, w, L, S), the family of the toy example


class VectorParams(TypedDict):
    n: int
    h: int
    w: int
    L: int
    S: int
    r: int  # 0 for CUTS
    s: int  # 0 for CUTS


class Vector(TypedDict):
    """One test vector, as stored in vectors.json. Byte strings are hex."""

    name: str
    mode: str  # "CUTS" or "CUTS+C"
    params: VectorParams
    seed: str
    message: str
    public_key: str
    rho: str
    counter: int | None  # CUTS+C only
    frontier: list[int]
    positions: list[int]
    values: list[str]  # Z_0 .. Z_{L-1}
    signature: str


def seed_for(name: str, n: int) -> bytes:
    return hashlib.shake_256(b"CUTS test vector seed: " + name.encode()).digest(n)


def toy_params() -> dict[str, CUTSParams]:
    cuts = CUTSParams(*TOY)
    r = widest_index_bits(CUTS(cuts).M, COUNTER_BITS)
    return {"toy-cuts": cuts, "toy-cuts-c": CUTSParams(*TOY, r, COUNTER_BITS)}


def make_vector(name: str, params: CUTSParams) -> Vector:
    """One vector: key pair, signature and the encoding it opens, verified."""
    scheme = CUTS(params)
    seed = seed_for(name, params.n)
    sk, pk = scheme.keygen(seed)
    sig = scheme.sign(sk, MESSAGE)
    if not scheme.verify(pk, MESSAGE, sig):
        raise AssertionError(f"{name}: the reference implementation rejects it")
    n, c = params.n, params.counter_bytes
    encoding = scheme.select(pk, sig[:n], MESSAGE, encoding_counter(params, sig))
    assert encoding is not None
    return {
        "name": name,
        "mode": "CUTS+C" if params.counter_bits else "CUTS",
        "params": {
            "n": n,
            "h": params.h,
            "w": params.w,
            "L": params.L,
            "S": params.S,
            "r": params.index_bits,
            "s": params.counter_bits,
        },
        "seed": seed.hex(),
        "message": MESSAGE.hex(),
        "public_key": pk.hex(),
        "rho": sig[:n].hex(),
        "counter": encoding.counter,
        "frontier": list(encoding.frontier),
        "positions": list(encoding.positions),
        "values": [
            sig[n + c + i * n : n + c + (i + 1) * n].hex() for i in range(params.L)
        ],
        "signature": sig.hex(),
    }


def encoding_counter(params: CUTSParams, sig: bytes) -> int | None:
    """The counter carried in a CUTS+C signature; None for CUTS."""
    if not params.counter_bits:
        return None
    n = params.n
    return int.from_bytes(sig[n : n + params.counter_bytes], "big")


def all_vectors() -> list[Vector]:
    named = list(toy_params().items()) + [(p.name, p.params) for p in PARAMETER_SETS]
    return [make_vector(name, params) for name, params in named]


def render_json(vectors: list[Vector]) -> str:
    doc: dict[str, object] = {
        "generator": "scripts/test_vectors/make_vectors.py",
        "message_ascii": MESSAGE.decode(),
        "vectors": vectors,
    }
    return json.dumps(doc, indent=2) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument(
        "--check", action="store_true", help="fail if the written files are out of date"
    )
    args = p.parse_args(argv)
    vectors = all_vectors()
    files = {JSON_OUT: render_json(vectors)}
    stale: list[str] = []
    for path, text in files.items():
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(str(path.relative_to(ROOT)))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"Wrote {path.relative_to(ROOT)}")
    if stale:
        sys.exit(f"Out of date: {', '.join(stale)}. Run make_vectors.py.")


if __name__ == "__main__":
    main()
