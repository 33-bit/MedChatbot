from __future__ import annotations

from src.server.source_documents import resolve_bachmai_source_pdf


def test_resolves_split_bachmai_pdf_by_disease_slug(tmp_path):
    pdf = tmp_path / "chuong_11_di_ung" / "may_day" / "source.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4\n")

    assert resolve_bachmai_source_pdf("may_day", raw_dir=tmp_path) == pdf


def test_rejects_unsafe_source_slug(tmp_path):
    assert resolve_bachmai_source_pdf("../may_day", raw_dir=tmp_path) is None
    assert resolve_bachmai_source_pdf("may day", raw_dir=tmp_path) is None


def test_returns_none_when_pdf_is_missing_or_ambiguous(tmp_path):
    assert resolve_bachmai_source_pdf("may_day", raw_dir=tmp_path) is None

    for chapter in ("chuong_11_a", "chuong_11_b"):
        pdf = tmp_path / chapter / "may_day" / "source.pdf"
        pdf.parent.mkdir(parents=True)
        pdf.write_bytes(b"%PDF-1.4\n")

    assert resolve_bachmai_source_pdf("may_day", raw_dir=tmp_path) is None
