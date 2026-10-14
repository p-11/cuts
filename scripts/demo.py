"""How to use the CUTS scheme"""

import time

from cuts import CUTS, CUTSParams, SecretKey
from cuts.scheme import widest_index_bits

# ---- 1. Pick a parameter set and build an instance -----------------------
# A parameter set meant for a single signature per key.
params = CUTSParams(n=16, h=14, w=2, L=49, S=24)
scheme = CUTS(params)

print(scheme)
print(f"  signature      {params.signature_bytes} B")
print(f"  verification   {params.verification_hashes} hashes")
print(f"  keygen         {params.keygen_hashes} hashes")
print(f"  log2 M         {scheme.selector_entropy_bits:.1f} bits")

# ---- 2. keygen / sign / verify -------------------------------------------
t = time.perf_counter()
sk, pk = scheme.keygen()
print(f"\nkeygen  {time.perf_counter() - t:.2f}s   pk = {pk.hex()}")

msg = b"spend outpoint 7f3a...:0 to bc1q..."
t = time.perf_counter()
sig = scheme.sign(sk, msg)
print(f"sign    {time.perf_counter() - t:.2f}s   {len(sig)} bytes")

t = time.perf_counter()
ok = scheme.verify(pk, msg, sig)
print(f"verify  {time.perf_counter() - t:.4f}s  -> {ok}")
print(f"verify(other message) -> {scheme.verify(pk, b'something else', sig)}")

# Which frontier and positions this signature opened.
F, A, _ = scheme.select(pk, sig[: params.n], msg)
print(f"\nfrontier  {F[:6]} ... ({len(F)} nodes)")
print(f"positions {A[:6]} ... (sum {sum(A)})")

# ---- 3. Storing keys ------------------------------------------------------
blob = sk.to_bytes()  # SK || PK, 2n bytes
sk_again = SecretKey.from_bytes(blob, params.n)
assert scheme.verify(pk, msg, scheme.sign(sk_again, msg))

# ---- 4. Several instances side by side -----------------------------------
# Each application holds its own CUTS object. Keys and signatures from one
# instance never verify under another, because the parameter identifier is
# hashed into every tree node.
other = CUTS(CUTSParams(n=16, h=15, w=5, L=81, S=162))
print(
    f"\n{other}: {other.params.signature_bytes} B, "
    f"{other.params.verification_hashes} verification hashes"
)
print(f"cross-instance verify -> {other.verify(pk, msg, sig)}")

# ---- 5. CUTS+C -------------------------------------------------------------
# Same tree and keys; the signer grinds a counter until the r-bit message hash
# falls below M, and sends it. r is the widest a 16-bit counter allows.
family = CUTSParams(n=16, h=11, w=5, L=32, S=78)
r = widest_index_bits(CUTS(family).M, 16)
plus_c = CUTS(CUTSParams(16, 11, 5, 32, 78, index_bits=r, counter_bits=16))
sk_c, pk_c = plus_c.keygen()
sig_c = plus_c.sign(sk_c, msg)
counter = plus_c.select(pk_c, sig_c[:16], msg).counter
print(f"\n{plus_c}")
print(f"  signature      {plus_c.params.signature_bytes} B (counter {counter})")
print(f"  verification   {plus_c.params.verification_hashes} hashes")
print(
    f"  grinding       +{plus_c.grinding_bits:.1f} bits, one-use "
    f"{plus_c.one_use_security_bits:.0f} bits"
)
print(f"  verify -> {plus_c.verify(pk_c, msg, sig_c)}")
