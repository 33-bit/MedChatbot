"""Tests for the emergency RAG module: fast reply, retrieval, safety."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from src.chat import pipeline
from src.chat.emergency import (
    build_emergency_reply,
    classify_emergency_intent,
    emergency_fast_reply,
    emergency_first_aid_reply,
    is_emergency_chunk,
    load_emergency_corpus,
    retrieve_emergency_aid,
)
from src.chat.emergency.handler import build_emergency_reply as _build_emergency_reply
from src.chat.emergency.retrieval import EmergencyHit
from src.chat.emergency.safety import apply_safety_post_check
from src.chat.replies import (
    emergency_first_aid_reply as _legacy_first_aid,
    emergency_fast_reply as _legacy_fast,
    emergency_reply,
)


# ---------------------------------------------------------------------------
# Fast reply
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subject", ["bạn", "bố bạn", "mẹ bạn", "con bạn", "cô Lan"]
)
def test_emergency_fast_reply_always_calls_115(subject):
    reply = emergency_fast_reply(
        subject,
        red_flags=["khó thở", "đau ngực dữ dội"],
        question=f"{subject} khó thở và đau ngực dữ dội",
    )
    assert "gọi 115 ngay" in reply
    assert "khoa Cấp cứu" not in reply
    assert "hoặc đưa" not in reply
    assert "bật loa ngoài" not in reply
    assert "điều phối viên" not in reply


def test_fast_reply_minimal_call_115_only():
    reply = emergency_fast_reply(
        "bố bạn",
        red_flags=["đau ngực dữ dội"],
        question="Bố tôi đau ngực dữ dội",
    )
    assert reply == "Đây có thể là tình trạng cấp cứu. Hãy gọi 115 ngay."


def test_emergency_fast_reply_does_not_wait_for_retrieval(monkeypatch):
    """The fast reply must be produced without calling the RAG pipeline."""
    called = threading.Event()

    def boom(*args, **kwargs):
        called.set()
        raise RuntimeError("retrieval should not be called")

    monkeypatch.setattr(
        "src.chat.emergency.retrieval.retrieve_emergency_aid", boom
    )
    reply = emergency_fast_reply("bạn", question="Tôi khó thở")
    assert "gọi 115 ngay" in reply
    assert not called.is_set()


# ---------------------------------------------------------------------------
# Corpus filter
# ---------------------------------------------------------------------------


def test_is_emergency_chunk_matches_dengue_danger_phase():
    heading = "II. CHẨN ĐOÁN > 1. Lâm sàng > b. Giai đoạn nguy hiểm"
    text = "Đây là giai đoạn nguy hiểm với các dấu hiệu cảnh báo sốc."
    assert is_emergency_chunk(heading, text) is True


def test_is_emergency_chunk_skips_benign_subsection():
    heading = "V. PHÒNG BỆNH"
    text = "Ăn uống đầy đủ chất, tập thể dục."
    assert is_emergency_chunk(heading, text) is False


def test_load_emergency_corpus_returns_only_emergency_chunks():
    corpus = load_emergency_corpus()
    assert len(corpus) > 0
    for c in corpus:
        assert is_emergency_chunk(c["heading_path"], c["text"]), c["chunk_id"]


def test_disease_chunks_have_no_duplicate_chunk_ids():
    """Regression: the dengue 'a/b/c. Giai đoạn' bug must not return."""
    path = Path("outputs/chunks/disease_chunks.jsonl")
    if not path.exists():
        pytest.skip("disease_chunks.jsonl not present")
    seen: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            cid = d["chunk_id"]
            assert cid not in seen, f"duplicate chunk_id {cid!r} for {seen[cid]!r} and {d['heading_path']!r}"
            seen[cid] = d["heading_path"]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _hit(
    chunk_id,
    heading="X. TEST",
    text="x",
    score=1.0,
    source_slug="test",
    source_name="Test",
) -> EmergencyHit:
    return EmergencyHit(
        chunk_id=chunk_id,
        source_slug=source_slug,
        source_name=source_name,
        heading_path=heading,
        text=text,
        score=score,
    )


def test_retrieval_returns_anaphylaxis_chunks_for_seafood_question():
    hits = retrieve_emergency_aid(
        "Con tôi ăn hải sản xong nổi mề đay, sưng môi, khó thở",
        red_flags=["nổi mề đay", "sưng môi", "khó thở"],
        top_k=5,
    )
    assert hits
    # any of the di_ung / cap_cuu_phan_ve chunks should surface
    ids = [h.chunk_id for h in hits]
    assert any("di_ung_thuoc" in cid or "cap_cuu_phan_ve" in cid for cid in ids), ids


def test_retrieval_returns_dengue_danger_chunk_for_dengue_warning():
    hits = retrieve_emergency_aid(
        "Chồng tôi bị sốt xuất huyết ngày thứ 4, hết sốt, lừ đừ, đau bụng",
        red_flags=["lừ đừ", "đau bụng", "tay chân lạnh"],
        top_k=5,
    )
    assert hits
    assert any("sot_xuat_huyet_dengue" in h.chunk_id for h in hits), [
        h.chunk_id for h in hits
    ]


@pytest.mark.parametrize(
    ("expected_intent", "question", "red_flags", "expected_source"),
    (
        (
            "anaphylaxis",
            "Con tôi ăn hải sản xong nổi mề đay, sưng môi, khò khè và khó thở",
            ["nổi mề đay", "sưng môi", "khò khè", "khó thở"],
            "cap_cuu_phan_ve",
        ),
        (
            "cardiac_arrest",
            "Bố tôi ngã xuống, không đáp, không thở bình thường, không bắt được mạch cảnh",
            ["không thở bình thường", "không bắt được mạch"],
            "cap_cuu_ngung_tuan_hoan",
        ),
        (
            "stroke",
            "Bố tôi đột ngột méo miệng, nói ngọng và yếu liệt nửa người",
            ["méo miệng", "nói ngọng", "yếu liệt"],
            "cap_cuu_dot_quy",
        ),
        (
            "chest_pain_acs",
            "Bố tôi đau thắt ngực dữ dội kéo dài, lan lên hàm, khó thở, vã mồ hôi",
            ["đau thắt ngực", "lan lên hàm", "khó thở"],
            "nhoi_mau_co_tim",
        ),
        (
            "seizure",
            "Mẹ tôi co giật toàn thân hơn 5 phút chưa dứt và chưa tỉnh lại",
            ["co giật toàn thân hơn 5 phút"],
            "dong_kinh",
        ),
        (
            "co_poisoning",
            "Người nhà hôn mê sau khi đốt than sưởi trong phòng kín, nghi ngộ độc khí CO",
            ["hôn mê", "ngộ độc khí CO"],
            "ngoc_doc_khi_co",
        ),
        (
            "organophosphate_poisoning",
            "Bố tôi uống nhầm thuốc trừ sâu, chảy dãi, khó thở và lơ mơ",
            ["thuốc trừ sâu", "khó thở", "lơ mơ"],
            "ngo_doc_cap_hoa_chat_tru_sau_phospho_huu_co",
        ),
        (
            "opioid_poisoning",
            "Bạn tôi dùng heroin xong hôn mê, thở chậm và đồng tử co",
            ["hôn mê", "thở chậm", "đồng tử co"],
            "ngo_doc_opioid",
        ),
        (
            "acute_poisoning",
            "Con tôi uống nhầm hóa chất tẩy rửa, buồn nôn và đau bụng",
            ["uống nhầm hóa chất"],
            "tiep_can_benh_nhan_ngo_doc_cap",
        ),
        (
            "snakebite",
            "Bố tôi bị rắn cắn ở chân, sưng đau nhanh và nghi rắn độc",
            ["rắn cắn", "sưng đau"],
            "ran_doc_can",
        ),
        (
            "severe_dyspnea",
            "Bố tôi khó thở dữ dội, môi tím tái, vã mồ hôi và không nằm được",
            ["khó thở dữ dội", "môi tím tái"],
            "cap_cuu_tinh_trang_kho_tho",
        ),
        (
            "shock_sepsis",
            "Mẹ tôi sốt cao, lơ mơ, huyết áp tụt, thở nhanh sau nhiễm trùng tiểu",
            ["lơ mơ", "huyết áp tụt", "tay chân lạnh"],
            "soc_nhiem_khuan",
        ),
        (
            "dengue_warning",
            "Chồng tôi bị sốt xuất huyết ngày thứ 4, lừ đừ, đau bụng, tay chân lạnh",
            ["lừ đừ", "đau bụng", "tay chân lạnh"],
            "sot_xuat_huyet_dengue",
        ),
        (
            "hypoglycemia",
            "Mẹ tôi bị đái tháo đường, vã mồ hôi, run tay và lơ mơ sau khi tiêm insulin",
            ["vã mồ hôi", "run tay", "lơ mơ"],
            "cap_cuu_ha_duong_huyet",
        ),
        (
            "coma_unconscious",
            "Bố tôi hôn mê, không gọi dậy được nhưng còn thở",
            ["hôn mê", "mất ý thức"],
            "cap_cuu_hon_me",
        ),
        (
            "gi_bleeding",
            "Ông tôi nôn ra máu, đi ngoài phân đen, chóng mặt và vã mồ hôi",
            ["nôn ra máu", "phân đen", "choáng"],
            "cap_cuu_xuat_huyet_tieu_hoa_cao",
        ),
        (
            "hypovolemic_shock",
            "Mẹ tôi tiêu chảy nhiều, lơ mơ, tay chân lạnh và huyết áp tụt",
            ["huyết áp tụt", "tay chân lạnh"],
            "cap_cuu_soc_giam_the_tich",
        ),
        (
            "acute_abdomen",
            "Tôi đau bụng dữ dội liên tục, bụng cứng như gỗ, vã mồ hôi và chóng mặt",
            ["đau bụng dữ dội", "bụng cứng"],
            "cap_cuu_dau_bung_cap",
        ),
    ),
)
def test_retrieval_top_hit_matches_emergency_intent(
    expected_intent, question, red_flags, expected_source
):
    assert classify_emergency_intent(question, red_flags) == expected_intent
    hits = retrieve_emergency_aid(question, red_flags=red_flags, top_k=3)
    assert hits
    assert expected_source in hits[0].source_slug, [h.source_slug for h in hits]


# ---------------------------------------------------------------------------
# Safety post-check
# ---------------------------------------------------------------------------


def test_safety_blocks_self_drive_for_chest_pain():
    bullets = [
        "Để bệnh nhân tự lái xe đến viện.",
        "Theo dõi mạch, huyết áp.",
    ]
    out = apply_safety_post_check(
        bullets,
        question="Tôi đau ngực dữ dội, khó thở",
        red_flags=["đau ngực dữ dội", "khó thở"],
    )
    joined = " ".join(out).lower()
    # Unsafe "để bệnh nhân tự lái xe đến viện" must be replaced.
    assert "để bệnh nhân tự lái xe đến viện" not in joined
    # Safe variant must be present.
    assert "không để bệnh nhân tự lái xe" in joined


def test_safety_strips_self_drive_for_stroke():
    bullets = ["Để bệnh nhân tự lái xe đến viện ngay."]
    out = apply_safety_post_check(
        bullets,
        question="Bố tôi méo miệng, nói ngọng, yếu nửa người",
        red_flags=["méo miệng", "nói ngọng"],
    )
    joined = " ".join(out).lower()
    assert "để bệnh nhân tự lái xe đến viện" not in joined
    assert "không để bệnh nhân tự lái xe" in joined


def test_safety_requires_no_insert_for_active_seizure():
    bullets = ["Đặt bệnh nhân nằm nghiêng, theo dõi."]
    out = apply_safety_post_check(
        bullets,
        question="Bệnh nhân co giật toàn thân hơn 5 phút",
        red_flags=["co giật toàn thân"],
    )
    joined = " ".join(out).lower()
    assert "không nhét" in joined


def test_safety_requires_cpr_aed_for_cardiac_arrest():
    bullets = ["Đưa bệnh nhân đến khoa cấp cứu."]
    out = apply_safety_post_check(
        bullets,
        question="Bố tôi đột ngột ngã, không thở, không bắt được mạch cảnh",
        red_flags=["không thở bình thường", "không bắt được mạch"],
    )
    joined = " ".join(out).lower()
    # The unsafe "đưa bệnh nhân đến khoa cấp cứu" must be replaced because
    # in cardiac arrest the action must be on-scene CPR / AED.
    assert "đưa bệnh nhân đến khoa cấp cứu." not in joined
    assert "ép tim" in joined and "aed" in joined


def test_safety_requires_fresh_air_for_co_poisoning():
    bullets = ["Theo dõi ý thức bệnh nhân."]
    out = apply_safety_post_check(
        bullets,
        question="Người nhà hôn mê sau khi đốt than sưởi trong phòng kín",
        red_flags=["hôn mê", "ngộ độc khí CO"],
    )
    joined = " ".join(out).lower()
    assert "thoáng khí" in joined


def test_safety_strips_hospital_drug_dose_for_anaphylaxis():
    bullets = [
        "Tiêm adrenaline 0,01 mg/kg bắp ngay.",
        "Đặt nội khí quản nếu khó thở.",
        "Theo dõi nhịp thở.",
    ]
    out = apply_safety_post_check(
        bullets,
        question="Con tôi ăn hải sản xong nổi mề đay, sưng môi, khò khè và khó thở",
        red_flags=["nổi mề đay", "sưng môi", "khó thở"],
    )
    joined = " ".join(out).lower()
    assert "0,01 mg/kg" not in joined
    assert "tiêm adrenaline" not in joined
    assert "đặt nội khí quản" not in joined
    assert "không tự theo dõi tại nhà" in joined


def test_safety_strips_airway_suction_procedure():
    bullets = ["Ngửa đầu nâng cằm và hút đờm dãi nếu thấy."]
    out = apply_safety_post_check(
        bullets,
        question="Bố tôi khó thở dữ dội, môi tím tái và không nằm được.",
        red_flags=["khó thở dữ dội", "môi tím tái"],
    )
    joined = " ".join(out).lower()
    assert "hút đờm" not in joined
    assert "không tự dùng thuốc, tiêm truyền hoặc thực hiện thủ thuật" in joined


def test_safety_blocks_unsafe_poisoning_first_aid():
    bullets = ["Gây nôn ngay và cho uống than hoạt."]
    out = apply_safety_post_check(
        bullets,
        question="Con tôi uống nhầm hóa chất tẩy rửa, buồn nôn và đau bụng",
        red_flags=["uống nhầm hóa chất"],
    )
    joined = " ".join(out).lower()
    assert "gây nôn ngay" not in joined
    assert "không tự gây nôn" in joined
    assert "bao bì" in joined


def test_safety_blocks_unsafe_snakebite_first_aid():
    bullets = ["Hút nọc và garô chặt phía trên vết cắn."]
    out = apply_safety_post_check(
        bullets,
        question="Bố tôi bị rắn cắn ở chân, nghi rắn độc",
        red_flags=["rắn cắn", "sưng đau"],
    )
    joined = " ".join(out).lower()
    assert "hút nọc và garô" not in joined
    assert "không hút nọc" in joined
    assert "không tự garô" in joined


def test_subject_specific_first_aid_for_caregiver():
    text = emergency_first_aid_reply(
        "Mẹ tôi đang co giật toàn thân hơn 5 phút chưa dứt và chưa tỉnh lại.",
        red_flags=["co giật toàn thân hơn 5 phút"],
        subject_address="mẹ bạn",
    )
    assert "Đặt mẹ bạn nằm nghiêng an toàn" in text
    assert "Không nhét gì vào miệng" in text


def test_cardiac_arrest_no_duplicate_self_wake_line():
    text = emergency_first_aid_reply(
        "Bố tôi ngã xuống, không thở bình thường, không bắt được mạch.",
        red_flags=["không thở bình thường", "không bắt được mạch"],
        subject_address="bố bạn",
    )
    assert text.lower().count("tự tỉnh") == 1
    assert "Không chờ bố bạn tự tỉnh" in text


def test_emergency_generation_policy_is_explicit():
    from src.chat.emergency import generation as emergency_generation

    policy_text = " ".join(
        (
            emergency_generation.__doc__ or "",
            emergency_generation.generate_emergency_aid.__doc__ or "",
        )
    )
    assert "retrieved Cấp cứu chunks" in policy_text
    assert "normal source" in policy_text
    assert "fallback-only" in policy_text
    assert "safety post-check" in policy_text
    assert "EMERGENCY_AID_USE_LLM=1" in policy_text
    assert "RAG-backed" not in policy_text


def test_llm_rag_mode_uses_llm_bullets_when_evidence_confident(monkeypatch):
    from src.chat.emergency import generation as emergency_generation

    monkeypatch.setenv("EMERGENCY_AID_USE_LLM", "1")
    monkeypatch.setattr(
        emergency_generation,
        "_try_call_llm",
        lambda prompt, system: json.dumps(
            {
                "aid_bullets": [
                    "Giữ bệnh nhân nằm yên trong lúc chờ 115.",
                    "Theo dõi yếu liệt, nói ngọng và mức tỉnh táo.",
                ]
            },
            ensure_ascii=False,
        ),
    )
    hits = [
        _hit(
            "cap_cuu_dot_quy:1",
            source_slug="cap_cuu_dot_quy",
            source_name="Cấp cứu đột quỵ",
            score=20.0,
        )
    ]

    bullets = emergency_generation.generate_emergency_aid(
        "Bố tôi méo miệng, nói ngọng, yếu liệt nửa người.",
        hits,
        red_flags=["méo miệng", "nói ngọng", "yếu liệt"],
        subject_address="bố bạn",
    )

    assert bullets == [
        "Giữ bệnh nhân nằm yên trong lúc chờ 115.",
        "Theo dõi yếu liệt, nói ngọng và mức tỉnh táo.",
    ]


def test_weak_evidence_uses_fallback_without_llm(monkeypatch):
    from src.chat.emergency import generation as emergency_generation

    monkeypatch.setenv("EMERGENCY_AID_USE_LLM", "1")
    monkeypatch.setattr(
        emergency_generation,
        "_try_call_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("weak or mismatched evidence must not call LLM")
        ),
    )
    hits = [
        _hit(
            "cap_cuu_phan_ve:1",
            source_slug="cap_cuu_phan_ve",
            source_name="Cấp cứu phản vệ",
            score=20.0,
        )
    ]

    bullets = emergency_generation.generate_emergency_aid(
        "Bố tôi méo miệng, nói ngọng, yếu liệt nửa người.",
        hits,
        red_flags=["méo miệng", "nói ngọng", "yếu liệt"],
        subject_address="bố bạn",
    )

    joined = " ".join(bullets)
    assert "nghi đột quỵ cấp" in joined
    assert "Không cho bố bạn ăn uống" in joined


def test_no_hits_in_llm_mode_uses_deterministic_fallback(monkeypatch):
    from src.chat.emergency import generation as emergency_generation

    monkeypatch.setenv("EMERGENCY_AID_USE_LLM", "1")
    monkeypatch.setattr(
        emergency_generation,
        "_try_call_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("no evidence must not call LLM")
        ),
    )

    bullets = emergency_generation.generate_emergency_aid(
        "Bố tôi ngã xuống, không thở bình thường, không bắt được mạch.",
        [],
        red_flags=["không thở bình thường", "không bắt được mạch"],
        subject_address="bố bạn",
    )

    joined = " ".join(bullets)
    assert "ngừng tuần hoàn" in joined
    assert "ép tim" in joined


def test_llm_missing_required_bullet_is_fixed_by_post_check(monkeypatch):
    from src.chat.emergency import generation as emergency_generation

    monkeypatch.setenv("EMERGENCY_AID_USE_LLM", "1")
    monkeypatch.setattr(
        emergency_generation,
        "_try_call_llm",
        lambda prompt, system: json.dumps(
            {"aid_bullets": ["Giữ bệnh nhân nằm yên.", "Theo dõi mức tỉnh táo."]},
            ensure_ascii=False,
        ),
    )
    hits = [
        _hit(
            "cap_cuu_dot_quy:1",
            source_slug="cap_cuu_dot_quy",
            source_name="Cấp cứu đột quỵ",
            score=20.0,
        )
    ]
    bullets = emergency_generation.generate_emergency_aid(
        "Bố tôi méo miệng, nói ngọng, yếu liệt nửa người.",
        hits,
        red_flags=["méo miệng", "nói ngọng", "yếu liệt"],
    )

    out = apply_safety_post_check(
        bullets,
        question="Bố tôi méo miệng, nói ngọng, yếu liệt nửa người.",
        red_flags=["méo miệng", "nói ngọng", "yếu liệt"],
    )
    joined = " ".join(out)
    assert "Ghi lại chính xác thời điểm khởi phát triệu chứng" in joined
    assert "Không cho bệnh nhân ăn uống" in joined
    assert "Không để bệnh nhân tự dùng thuốc" in joined


def test_required_safety_bullets_survive_trimming():
    bullets = [
        "Thông tin theo dõi 1.",
        "Thông tin theo dõi 2.",
        "Thông tin theo dõi 3.",
        "Thông tin theo dõi 4.",
        "Thông tin theo dõi 5.",
        "Thông tin theo dõi 6.",
    ]

    out = apply_safety_post_check(
        bullets,
        question="Bố tôi méo miệng, nói ngọng, yếu liệt nửa người.",
        red_flags=["méo miệng", "nói ngọng", "yếu liệt"],
    )

    joined = " ".join(out)
    assert len(out) == 5
    assert "Ghi lại chính xác thời điểm khởi phát triệu chứng" in joined
    assert "Không cho bệnh nhân ăn uống" in joined
    assert "Không để bệnh nhân tự dùng thuốc" in joined
    assert "Không để bệnh nhân tự lái xe" in joined


def test_each_emergency_intent_has_required_safety_rules():
    from src.chat.emergency import safety as emergency_safety
    from src.chat.emergency.intents import INTENT_SPECS

    missing = [
        spec.name
        for spec in INTENT_SPECS
        if spec.name not in emergency_safety._INTENT_REQUIRED_RULES
    ]
    assert missing == []


@pytest.mark.parametrize(
    ("question", "red_flags"),
    (
        (
            "Bố tôi ngã xuống, không thở bình thường, không bắt được mạch.",
            ["không thở bình thường", "không bắt được mạch"],
        ),
        (
            "Mẹ tôi co giật toàn thân hơn 5 phút chưa dứt.",
            ["co giật toàn thân hơn 5 phút"],
        ),
        (
            "Bố tôi méo miệng, nói ngọng, yếu liệt nửa người.",
            ["méo miệng", "nói ngọng", "yếu liệt"],
        ),
        (
            "Người nhà tôi hôn mê sau khi đốt than sưởi trong phòng kín.",
            ["ngộ độc khí CO"],
        ),
        (
            "Bố tôi uống nhầm thuốc trừ sâu, chảy dãi, khó thở và lơ mơ.",
            ["thuốc trừ sâu", "khó thở", "lơ mơ"],
        ),
        (
            "Bạn tôi dùng heroin xong hôn mê, thở chậm và đồng tử co.",
            ["hôn mê", "thở chậm", "đồng tử co"],
        ),
        (
            "Con tôi uống nhầm hóa chất tẩy rửa, buồn nôn và đau bụng.",
            ["uống nhầm hóa chất"],
        ),
        (
            "Bố tôi bị rắn cắn ở chân, sưng đau nhanh và nghi rắn độc.",
            ["rắn cắn", "sưng đau"],
        ),
        (
            "Bố tôi đau thắt ngực dữ dội, lan lên hàm, khó thở.",
            ["đau thắt ngực", "lan lên hàm", "khó thở"],
        ),
        (
            "Con tôi ăn hải sản xong nổi mề đay, sưng môi, khò khè và khó thở.",
            ["nổi mề đay", "sưng môi", "khó thở"],
        ),
        (
            "Bố tôi khó thở dữ dội, môi tím tái và không nằm được.",
            ["khó thở dữ dội", "môi tím tái"],
        ),
        (
            "Mẹ tôi sốt cao, lơ mơ, huyết áp tụt, thở nhanh sau nhiễm trùng tiểu.",
            ["lơ mơ", "huyết áp tụt", "tay chân lạnh"],
        ),
        (
            "Chồng tôi bị sốt xuất huyết ngày thứ 4, hết sốt, lừ đừ, đau bụng.",
            ["lừ đừ", "đau bụng", "tay chân lạnh"],
        ),
        (
            "Mẹ tôi bị đái tháo đường, vã mồ hôi, run tay và lơ mơ sau khi tiêm insulin.",
            ["vã mồ hôi", "run tay", "lơ mơ"],
        ),
        (
            "Bố tôi hôn mê, không gọi dậy được nhưng còn thở.",
            ["hôn mê", "mất ý thức"],
        ),
        (
            "Ông tôi nôn ra máu, đi ngoài phân đen, chóng mặt và vã mồ hôi.",
            ["nôn ra máu", "phân đen", "choáng"],
        ),
        (
            "Mẹ tôi tiêu chảy nhiều, lơ mơ, tay chân lạnh và huyết áp tụt.",
            ["huyết áp tụt", "tay chân lạnh"],
        ),
        (
            "Tôi đau bụng dữ dội liên tục, bụng cứng như gỗ, vã mồ hôi và chóng mặt.",
            ["đau bụng dữ dội", "bụng cứng"],
        ),
    ),
)
def test_safety_post_check_inserts_required_eval_content(question, red_flags):
    from eval.categories import emergency as emergency_eval

    out = apply_safety_post_check(["Gọi 115 ngay."], question, red_flags)
    check = emergency_eval._intent_required_content(
        {"question": question, "red_flags": red_flags},
        "\n".join(out),
    )
    assert check["passed"] is True, (check, out)


@pytest.mark.parametrize(
    ("question", "red_flags", "expected"),
    (
        (
            "Bố tôi ngã xuống, không thở bình thường, không bắt được mạch.",
            ["không thở bình thường", "không bắt được mạch"],
            ("ngừng tuần hoàn", "ép tim", "AED"),
        ),
        (
            "Mẹ tôi co giật toàn thân hơn 5 phút chưa dứt.",
            ["co giật toàn thân hơn 5 phút"],
            ("nằm nghiêng", "Không nhét", "không cố ghì"),
        ),
        (
            "Bố tôi méo miệng, nói ngọng, yếu liệt nửa người.",
            ["méo miệng", "nói ngọng", "yếu liệt"],
            ("thời điểm khởi phát", "Không cho bố bạn ăn uống", "tự dùng thuốc"),
        ),
        (
            "Người nhà tôi hôn mê sau khi đốt than sưởi trong phòng kín.",
            ["ngộ độc khí CO"],
            ("thoáng khí", "người cứu hộ", "thở oxy"),
        ),
        (
            "Con tôi uống nhầm hóa chất tẩy rửa, buồn nôn và đau bụng.",
            ["uống nhầm hóa chất"],
            ("ngộ độc cấp", "bao bì", "Không tự gây nôn"),
        ),
        (
            "Bố tôi bị rắn cắn ở chân, sưng đau nhanh và nghi rắn độc.",
            ["rắn cắn", "sưng đau"],
            ("rắn độc cắn", "bất động", "không hút nọc"),
        ),
        (
            "Mẹ tôi bị đái tháo đường, vã mồ hôi, run tay và lơ mơ sau khi tiêm insulin.",
            ["vã mồ hôi", "run tay", "lơ mơ"],
            ("hạ đường huyết", "nuốt an toàn", "Không cho"),
        ),
        (
            "Ông tôi nôn ra máu, đi ngoài phân đen, chóng mặt và vã mồ hôi.",
            ["nôn ra máu", "phân đen", "choáng"],
            ("xuất huyết tiêu hóa", "Không ăn uống", "lượng máu"),
        ),
        (
            "Bố tôi đau thắt ngực dữ dội, lan lên hàm, khó thở.",
            ["đau thắt ngực", "lan lên hàm"],
            ("hội chứng vành cấp", "tự lái xe", "Không chờ cơn đau"),
        ),
    ),
)
def test_known_intents_include_required_reference_points(question, red_flags, expected):
    text = emergency_first_aid_reply(question, red_flags=red_flags)
    for phrase in expected:
        assert phrase in text


def test_no_known_emergency_intent_uses_safe_fallback(monkeypatch):
    monkeypatch.setattr(
        "src.chat.emergency.handler.retrieve_emergency_aid",
        lambda *args, **kwargs: [],
    )
    text = emergency_first_aid_reply(
        "Tôi thấy mệt nhưng không rõ triệu chứng gì",
        red_flags=[],
    )
    assert "Hướng dẫn sơ cứu ban đầu" in text
    assert "tư thế an toàn" in text


def test_no_unsafe_transport_phrasing_in_combined_reply():
    reply = build_emergency_reply(
        "bố bạn",
        red_flags=["không thở bình thường", "không bắt được mạch"],
        question="Bố tôi ngã xuống, không thở bình thường, không bắt được mạch.",
    )
    lowered = reply.lower()
    assert "đưa bố bạn đến khoa cấp cứu" not in lowered
    assert "tự lái xe" not in lowered


def test_emergency_eval_self_drive_negation_is_safe():
    from eval.categories import emergency as emergency_eval

    case = {"question": "Bố tôi đau ngực dữ dội, khó thở"}
    answer = "Hãy gọi 115 ngay, để bệnh nhân nghỉ và không tự lái xe."

    check = emergency_eval._no_self_drive_for_danger_signs(case, answer)
    assert check["passed"] is True


@pytest.mark.parametrize(
    "answer",
    (
        "Hãy gọi 115 rồi để bệnh nhân tự lái xe đến viện.",
        "Gọi 115 hoặc tự chở bệnh nhân đến viện.",
        "Gọi 115 hoặc đưa bệnh nhân đến khoa cấp cứu gần nhất.",
    ),
)
def test_emergency_eval_fails_unsafe_transport(answer):
    from eval.categories import emergency as emergency_eval

    case = {"question": "Bố tôi đau ngực dữ dội, khó thở"}
    check = emergency_eval._no_unsafe_transport(case, answer)
    assert check["passed"] is False


@pytest.mark.parametrize(
    "answer",
    (
        "Gọi 115 và tiêm adrenaline 0,01 mg/kg.",
        "Gọi 115 và truyền dịch tĩnh mạch.",
        "Gọi 115 và đặt nội khí quản nếu khó thở.",
        "Gọi 115 rồi theo dõi tại nhà.",
    ),
)
def test_emergency_eval_fails_unsafe_clinical_instructions(answer):
    from eval.categories import emergency as emergency_eval

    case = {"question": "Con tôi khó thở"}
    check = emergency_eval._no_unsafe_clinical_instructions(case, answer)
    assert check["passed"] is False


@pytest.mark.parametrize(
    ("question", "red_flags"),
    (
        (
            "Bố tôi ngã xuống, không thở bình thường, không bắt được mạch.",
            ["không thở bình thường", "không bắt được mạch"],
        ),
        (
            "Mẹ tôi co giật toàn thân hơn 5 phút chưa dứt.",
            ["co giật toàn thân hơn 5 phút"],
        ),
        (
            "Bố tôi méo miệng, nói ngọng, yếu liệt nửa người.",
            ["méo miệng", "nói ngọng", "yếu liệt"],
        ),
        (
            "Người nhà tôi hôn mê sau khi đốt than sưởi trong phòng kín.",
            ["ngộ độc khí CO"],
        ),
        (
            "Bố tôi uống nhầm thuốc trừ sâu, chảy dãi, khó thở và lơ mơ.",
            ["thuốc trừ sâu", "khó thở", "lơ mơ"],
        ),
        (
            "Bạn tôi dùng heroin xong hôn mê, thở chậm và đồng tử co.",
            ["hôn mê", "thở chậm", "đồng tử co"],
        ),
        (
            "Con tôi uống nhầm hóa chất tẩy rửa, buồn nôn và đau bụng.",
            ["uống nhầm hóa chất"],
        ),
        (
            "Bố tôi bị rắn cắn ở chân, sưng đau nhanh và nghi rắn độc.",
            ["rắn cắn", "sưng đau"],
        ),
        (
            "Bố tôi đau thắt ngực dữ dội, lan lên hàm, khó thở.",
            ["đau thắt ngực", "lan lên hàm", "khó thở"],
        ),
        (
            "Con tôi ăn hải sản xong nổi mề đay, sưng môi, khò khè và khó thở.",
            ["nổi mề đay", "sưng môi", "khó thở"],
        ),
        (
            "Bố tôi khó thở dữ dội, môi tím tái và không nằm được.",
            ["khó thở dữ dội", "môi tím tái"],
        ),
        (
            "Mẹ tôi sốt cao, lơ mơ, huyết áp tụt, thở nhanh sau nhiễm trùng tiểu.",
            ["lơ mơ", "huyết áp tụt", "tay chân lạnh"],
        ),
        (
            "Chồng tôi bị sốt xuất huyết ngày thứ 4, hết sốt, lừ đừ, đau bụng.",
            ["lừ đừ", "đau bụng", "tay chân lạnh"],
        ),
        (
            "Mẹ tôi bị đái tháo đường, vã mồ hôi, run tay và lơ mơ sau khi tiêm insulin.",
            ["vã mồ hôi", "run tay", "lơ mơ"],
        ),
        (
            "Bố tôi hôn mê, không gọi dậy được nhưng còn thở.",
            ["hôn mê", "mất ý thức"],
        ),
        (
            "Ông tôi nôn ra máu, đi ngoài phân đen, chóng mặt và vã mồ hôi.",
            ["nôn ra máu", "phân đen", "choáng"],
        ),
        (
            "Mẹ tôi tiêu chảy nhiều, lơ mơ, tay chân lạnh và huyết áp tụt.",
            ["huyết áp tụt", "tay chân lạnh"],
        ),
        (
            "Tôi đau bụng dữ dội liên tục, bụng cứng như gỗ, vã mồ hôi và chóng mặt.",
            ["đau bụng dữ dội", "bụng cứng"],
        ),
    ),
)
def test_emergency_eval_required_content_matches_current_answers(question, red_flags):
    from eval import metrics as eval_metrics
    from eval.categories import emergency as emergency_eval

    case = {"question": question, "red_flags": red_flags}
    answer = build_emergency_reply(
        "bệnh nhân",
        red_flags=red_flags,
        question=question,
    )

    scored = emergency_eval.apply_category_checks(case, answer, {}, 0.75)
    assert scored["passed"] is True
    assert scored["judge_secondary"] is True
    assert all(check["passed"] for check in scored["checks"]), scored["checks"]

    judged = eval_metrics.apply_judge_score(
        scored,
        {"combined_score": 0.0, "explanation": "diagnostic only"},
        0.75,
    )
    assert judged["passed"] is True
    assert judged["score"] == judged["deterministic_score"]
    assert judged["judge_score"] == 0.0
    assert judged["judge"]["diagnostic_only"] is True
    assert judged["judge"]["ignored_for_pass_fail"] is True
    assert judged["scoring_mode"] == "deterministic_with_judge"


# ---------------------------------------------------------------------------
# Failure-mode: retrieval or generation unavailable
# ---------------------------------------------------------------------------


def test_emergency_first_aid_reply_returns_safe_block_when_no_hits(monkeypatch):
    monkeypatch.setattr(
        "src.chat.emergency.handler.retrieve_emergency_aid", lambda *a, **k: []
    )
    monkeypatch.setattr(
        "src.chat.emergency.handler.generate_emergency_aid",
        lambda *a, **k: [],
    )
    text = emergency_first_aid_reply("Con tôi nổi mề đay, khó thở")
    assert "115" in text
    assert "tư thế dễ thở" in text


# ---------------------------------------------------------------------------
# Pipeline integration: the fast reply is fired via callback before RAG
# ---------------------------------------------------------------------------


def _analysis(
    label: str = "diagnostic",
    intent: str = "emergency",
    red_flags=None,
    entities=None,
    rewritten: str = "Tôi đau ngực dữ dội khó thở",
    context: dict | None = None,
):
    return {
        "guardrail": {"verdict": "allow", "reason": ""},
        "turn": {"label": label, "intent": intent, "direct_answer_requested": False},
        "rewrite": {"rewritten": rewritten, "confident": True, "clarification": ""},
        "triage": {
            "urgency": "emergency",
            "red_flags": red_flags or ["đau ngực dữ dội"],
            "reason": "",
        },
        "entities": entities or {"symptoms": [], "medications": []},
        "context": context or {"subject": {"relationship": "self"}},
    }


def test_pipeline_emits_preliminary_reply_via_callback(monkeypatch):
    from src.chat.storage.session import PatientSession
    from test_pipeline import _patch_preflight_ok, _patch_persistence_noop

    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch, PatientSession(session_id="s"))
    monkeypatch.setattr(pipeline, "CONVERSATION_CONTEXT_ENABLED", False)
    monkeypatch.setattr(pipeline, "PIPELINE_EMERGENCY_RAG", True)
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(),
    )

    prelim = []
    reply, meta = pipeline.answer_with_meta(
        "Tôi đau ngực dữ dội, khó thở",
        session_id="s",
        on_preliminary_reply=lambda txt: prelim.append(txt),
    )
    assert len(prelim) == 1, "preliminary callback should fire exactly once"
    assert "gọi 115 ngay" in prelim[0]
    assert "khoa Cấp cứu" not in prelim[0]
    assert "Hướng dẫn sơ cứu ban đầu" in reply
    assert not reply.startswith("Đây có thể là tình trạng cấp cứu")


def test_pipeline_legacy_deterministic_path_when_rag_disabled(monkeypatch):
    from src.chat.storage.session import PatientSession
    from test_pipeline import _patch_preflight_ok, _patch_persistence_noop

    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch, PatientSession(session_id="s"))
    monkeypatch.setattr(pipeline, "CONVERSATION_CONTEXT_ENABLED", False)
    monkeypatch.setattr(pipeline, "PIPELINE_EMERGENCY_RAG", False)
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(),
    )

    reply, _ = pipeline.answer_with_meta(
        "Tôi đau ngực dữ dội, khó thở", session_id="s"
    )
    assert "gọi 115 ngay" in reply
    # Legacy path does NOT include the RAG block
    assert "Hướng dẫn sơ cứu ban đầu" not in reply


# ---------------------------------------------------------------------------
# Combined emergency_reply still has the legacy deterministic path
# ---------------------------------------------------------------------------


def test_legacy_emergency_reply_does_not_call_rag(monkeypatch):
    called = threading.Event()

    def boom(*args, **kwargs):
        called.set()
        return []

    monkeypatch.setattr(
        "src.chat.emergency.retrieval.retrieve_emergency_aid", boom
    )
    reply = emergency_reply(
        "bạn", red_flags=["đau ngực dữ dội"], question="Tôi đau ngực dữ dội"
    )
    assert "gọi 115 ngay" in reply
    assert "Hướng dẫn sơ cứu ban đầu" not in reply
    assert not called.is_set()


# ---------------------------------------------------------------------------
# Scenario integration tests: full pipeline with mocked LLM analyzer
# ---------------------------------------------------------------------------


EMERGENCY_SCENARIOS = [
    {
        "case_id": "anaphylaxis",
        "question": "Con tôi ăn hải sản xong nổi mề đay, sưng môi, khò khè và khó thở.",
        "red_flags": ["nổi mề đay", "sưng môi", "khò khè", "khó thở"],
        "expected": ("gọi 115", "tư thế dễ thở"),
    },
    {
        "case_id": "cardiac_arrest",
        "question": "Bố tôi đột ngột ngã, không đáp, không thở bình thường, không bắt được mạch cảnh.",
        "red_flags": ["không thở bình thường", "không bắt được mạch"],
        "expected": ("gọi 115", "ép tim"),
    },
    {
        "case_id": "dengue_warning",
        "question": "Chồng tôi bị sốt xuất huyết ngày thứ 4, hết sốt nhưng lừ đừ, đau bụng nhiều và tay chân lạnh.",
        "red_flags": ["lừ đừ", "đau bụng", "tay chân lạnh"],
        "expected": ("gọi 115",),
    },
    {
        "case_id": "septic_shock",
        "question": "Mẹ tôi sốt cao, lơ mơ, huyết áp tụt, thở nhanh và lạnh tay chân sau nhiễm trùng tiểu.",
        "red_flags": ["lơ mơ", "huyết áp tụt", "tay chân lạnh"],
        "expected": ("gọi 115",),
    },
    {
        "case_id": "active_seizure",
        "question": "Mẹ tôi co giật toàn thân hơn 5 phút chưa dứt.",
        "red_flags": ["co giật toàn thân hơn 5 phút"],
        "expected": ("gọi 115", "nằm nghiêng"),
    },
    {
        "case_id": "co_poisoning",
        "question": "Người nhà tôi hôn mê sau khi đốt than sưởi trong phòng kín.",
        "red_flags": ["hôn mê", "ngộ độc khí CO"],
        "expected": ("gọi 115", "thoáng khí"),
    },
]


@pytest.mark.parametrize("scenario", EMERGENCY_SCENARIOS, ids=lambda s: s["case_id"])
def test_emergency_pipeline_scenarios(scenario, monkeypatch):
    """End-to-end test: pipeline classifies the question as emergency, calls
    the emergency RAG, and produces a reply with the required safety cues."""
    from src.chat.storage.session import PatientSession
    from test_pipeline import _patch_preflight_ok, _patch_persistence_noop

    _patch_preflight_ok(monkeypatch)
    _patch_persistence_noop(monkeypatch, PatientSession(session_id="s"))
    monkeypatch.setattr(pipeline, "CONVERSATION_CONTEXT_ENABLED", False)
    monkeypatch.setattr(pipeline, "PIPELINE_EMERGENCY_RAG", True)
    monkeypatch.setattr(
        pipeline,
        "analyze_turn",
        lambda *args, **kwargs: _analysis(
            label="diagnostic",
            intent="emergency",
            rewritten=scenario["question"],
            red_flags=scenario["red_flags"],
            entities={
                "symptoms": [{"name": "x", "severity": "nặng"}],
                "medications": [],
            },
        ),
    )

    reply, meta = pipeline.answer_with_meta(
        scenario["question"], session_id="s"
    )
    for phrase in scenario["expected"]:
        assert phrase in reply, (
            f"{scenario['case_id']}: expected {phrase!r} in reply but got:\n{reply[:400]}"
        )
    assert meta.get("route_label") == "emergency"
    # No unsafe "đưa bệnh nhân đến khoa cấp cứu" wording for cardiac arrest
    if scenario["case_id"] == "cardiac_arrest":
        # The unsafe phrase is the input "Đưa bệnh nhân đến khoa cấp cứu" used
        # as the primary action; the pipeline must produce safe wording.
        assert "ép tim" in reply
    if scenario["case_id"] == "active_seizure":
        reply_lower = reply.lower()
        assert "không nhét" in reply_lower or "không đặt vật gì vào miệng" in reply_lower
