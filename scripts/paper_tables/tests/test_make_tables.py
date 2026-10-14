from paper_tables import make_tables as mt
from paper_tables.parameter_sets import PARAMETER_SETS


def test_writes_every_table(tmp_path):
    mt.main(["--out", str(tmp_path)])
    for name, text in mt.outputs().items():
        assert (tmp_path / name).read_text() == text


def test_every_set_appears_once_with_its_numbers():
    for p in PARAMETER_SETS:
        rows = [
            line
            for line in mt.render(p.scheme).splitlines()
            if line.startswith(mt.cells(p)[0] + " &")
        ]
        assert len(rows) == 1, p.name
        x = p.params
        for value in (x.h, x.w, x.L, x.S, x.signature_bytes, x.verification_hashes):
            assert f" {mt.grouped(value)} " in rows[0], (p.name, value)
        assert mt.grouped(x.keygen_hashes) in rows[0]


def test_lambda2_column_shows_the_proven_bound():
    for p in PARAMETER_SETS:
        assert mt.cells(p)[-1] == f"{mt.proven_lambda2(p.name):.1f}"
