"""Emergency aid generation.

Default policy: deterministic fallback aid.

When EMERGENCY_AID_USE_LLM=1, confident retrieved Cấp cứu chunks become the
normal source of first-aid detail. Deterministic templates are fallback-only
for missing evidence, LLM failure, or unusable LLM output. The safety post-check
is the invariant layer that sanitizes and injects mandatory emergency actions.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Iterable

from src.chat.emergency.intents import (
    classify_emergency_intent,
    get_intent_spec,
    normalize_emergency_text,
)
from src.chat.emergency.retrieval import EmergencyHit

log = logging.getLogger(__name__)

_DEFAULT_MIN_LLM_EVIDENCE_SCORE = 5.0

EMERGENCY_AID_SYSTEM_PROMPT = (
    "Bạn là trợ lý y khoa. Nhiệm vụ: dựa TRÊN CÁC ĐOẠN TÀI LIỆU CẤP CỨU được "
    "cung cấp bên dưới, viết phần SƠ CỨU BAN ĐẦU để người hỏi làm theo trong lúc "
    "chờ xe cấp cứu 115.\n\n"
    "QUY TẮC BẮT BUỘC:\n"
    "1. CHỈ dùng thông tin từ các đoạn tài liệu được cung cấp. KHÔNG bịa thuốc, "
    "liều thuốc, thủ thuật hay phác đồ ngoài tài liệu.\n"
    "2. Nếu tài liệu được cung cấp mỏng hoặc không khớp câu hỏi, nói rõ "
    "\"Theo các đoạn tài liệu được cung cấp chưa đủ căn cứ, hãy làm theo hướng "
    "dẫn trực tiếp của điều phối viên 115\" thay vì tự suy đoán.\n"
    "3. Tối đa 5 gạch đầu dòng ngắn (mỗi gạch ≤ 25 từ), hành động cụ thể, "
    "không giải thích dài dòng.\n"
    "4. LUÔN nhắc lại hành động an toàn phù hợp với tình huống: ví dụ "
    "ngừng tuần hoàn → ép tim ngoài lồng ngực + AED nếu có; co giật → không "
    "nhét gì vào miệng; ngộ độc khí CO → đưa ra nơi thoáng khí nếu an toàn; "
    "đột quỵ/đau ngực → không tự lái xe.\n"
    "5. Chỉ viết hành động người thường có thể làm trước khi 115 tới. CẤM liều "
    "thuốc, tiêm/truyền tĩnh mạch, đặt nội khí quản, thở máy, thủ thuật bệnh viện, "
    "tự lái xe, tự chở người bệnh, hoặc theo dõi tại nhà.\n"
    "6. Chỉ được nhắc bút tiêm epinephrine/adrenaline nếu ghi rõ là bút đã được "
    "kê sẵn và dùng theo hướng dẫn đi kèm; không nêu liều tiêm.\n"
    "7. Trả về JSON: {\"aid_bullets\": [\"...\", \"...\", ...]}"
)


def build_emergency_aid_prompt(
    question: str,
    hits: Iterable[EmergencyHit],
    red_flags: Iterable[str] | str | None = None,
    required_points: Iterable[str] | None = None,
) -> str:
    flags_text = ""
    if red_flags:
        if isinstance(red_flags, str):
            flags_text = red_flags
        else:
            flags_text = " ".join(str(f) for f in red_flags)
    hit_list = list(hits)
    if hit_list:
        chunks_text = []
        for i, h in enumerate(hit_list, 1):
            chunks_text.append(
                f"[ĐOẠN {i}] nguồn: {h.source_name} | mục: {h.heading_path}\n"
                f"{h.text}"
            )
        chunks_block = "\n\n".join(chunks_text)
    else:
        chunks_block = (
            "(Không tìm được đoạn tài liệu cấp cứu phù hợp trong kho dữ liệu. "
            "Hãy trả lời bằng câu fallback an toàn.)"
        )
    required = "\n".join(f"- {point}" for point in (required_points or ()))
    return (
        f"CÂU HỎI: {question}\n"
        f"DẤU HIỆU CẢNH BÁO: {flags_text or '(không có)'}\n\n"
        f"CÁC ĐOẠN TÀI LIỆU CẤP CỨU:\n{chunks_block}\n\n"
        f"Ý BẮT BUỘC CẦN BAO PHỦ:\n{required or '- Làm theo hướng dẫn 115 và chỉ nêu sơ cứu an toàn.'}\n\n"
        "Trả về JSON {\"aid_bullets\": [...]} theo đúng quy tắc đã nêu."
    )


def _try_call_llm(prompt: str, system: str) -> str:
    """Best-effort LLM call. We try the same client as the chat pipeline but
    fall back gracefully if the API key is missing or the call fails — the
    emergency route must still produce a safe answer."""
    try:
        from src.config import FAST_MODEL, LLM_API_KEY, BASE_URL
        if not LLM_API_KEY:
            return ""
        from openai import OpenAI

        client = OpenAI(api_key=LLM_API_KEY, base_url=BASE_URL or None)
        resp = client.chat.completions.create(
            model=os.getenv("EMERGENCY_AID_MODEL", FAST_MODEL),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=600,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:  # pragma: no cover - depends on env
        log.warning("Emergency aid LLM call failed: %s", exc)
        return ""


def _parse_bullets(text: str) -> list[str]:
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # try to extract a JSON object inside the response
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return []
        else:
            return []
    bullets = data.get("aid_bullets") or data.get("bullets") or []
    if not isinstance(bullets, list):
        return []
    out: list[str] = []
    for b in bullets:
        if not isinstance(b, str):
            continue
        b = b.strip()
        if b:
            out.append(b)
    return out


def _min_llm_evidence_score() -> float:
    try:
        return float(
            os.getenv(
                "EMERGENCY_AID_MIN_LLM_EVIDENCE_SCORE",
                str(_DEFAULT_MIN_LLM_EVIDENCE_SCORE),
            )
        )
    except ValueError:
        return _DEFAULT_MIN_LLM_EVIDENCE_SCORE


def _hit_matches_intent_source(hit: EmergencyHit, intent: str | None) -> bool:
    spec = get_intent_spec(intent)
    if spec is None:
        return True
    source_norm = normalize_emergency_text(f"{hit.source_slug} {hit.source_name}")
    allowed_sources = spec.primary_sources + spec.secondary_sources
    return any(source in source_norm for source in allowed_sources)


def _has_confident_evidence(intent: str | None, hits: list[EmergencyHit]) -> bool:
    min_score = _min_llm_evidence_score()
    return any(
        hit.score >= min_score and _hit_matches_intent_source(hit, intent)
        for hit in hits
    )


def _fallback_bullets(hits: list[EmergencyHit]) -> list[str]:
    """Generic fallback when no deterministic intent template can be selected."""
    return [
        "Theo các đoạn tài liệu cấp cứu hiện có, hãy làm theo hướng dẫn trực tiếp của điều phối viên 115.",
        "Trong lúc chờ xe cấp cứu, để bệnh nhân ở tư thế an toàn, dễ thở và có người theo dõi liên tục.",
    ]


def _subject_label(question: str, subject_address: str = "bạn") -> str:
    subject = (subject_address or "").strip()
    if subject and subject != "bạn":
        return subject
    q_norm = normalize_emergency_text(question)
    if "con toi" in q_norm or "tre" in q_norm:
        return "con bạn"
    if "bo toi" in q_norm or "cha toi" in q_norm:
        return "bố bạn"
    if "me toi" in q_norm:
        return "mẹ bạn"
    if "chong toi" in q_norm:
        return "chồng bạn"
    if "vo toi" in q_norm:
        return "vợ bạn"
    if "nguoi nha" in q_norm:
        return "người nhà bạn"
    if q_norm.startswith("toi ") or " toi " in f" {q_norm} ":
        return "bạn"
    return "bệnh nhân"


def _intent_required_points(intent: str | None, subject: str) -> list[str]:
    if intent == "anaphylaxis":
        return [
            "nghi phản vệ nặng, nguy cơ đe dọa tính mạng",
            "gọi 115 ngay",
            "không tự theo dõi tại nhà",
            f"để {subject} ở tư thế dễ thở",
            "chuẩn bị thông tin thức ăn/thuốc và thời điểm xuất hiện triệu chứng",
        ]
    if intent == "cardiac_arrest":
        return [
            "nghi ngừng tuần hoàn",
            "ép tim ngoài lồng ngực ngay nếu không thở bình thường",
            "nhờ người lấy AED nếu có",
            "không chờ tự tỉnh",
        ]
    if intent == "stroke":
        return [
            "nghi đột quỵ cấp",
            "ghi lại thời điểm khởi phát",
            "không cho ăn uống",
            "không tự dùng thuốc",
            "bệnh viện có khả năng cấp cứu đột quỵ qua 115",
        ]
    if intent == "chest_pain_acs":
        return [
            "nghi hội chứng vành cấp",
            "nghỉ, tránh gắng sức",
            "không tự lái xe hoặc tự chở",
            "không chờ cơn đau tự hết",
        ]
    if intent == "seizure":
        return [
            "co giật kéo dài hoặc trạng thái động kinh",
            f"đặt {subject} nằm nghiêng an toàn",
            "tránh va đập",
            "không nhét gì vào miệng",
            "không cố ghì",
        ]
    if intent == "co_poisoning":
        return [
            "nghi ngộ độc khí CO",
            f"đưa {subject} ra nơi thoáng khí nếu an toàn",
            "tránh người cứu hộ hít khí độc",
            "không trì hoãn thở oxy và hồi sức tại bệnh viện",
        ]
    if intent == "organophosphate_poisoning":
        return [
            "nghi ngộ độc hóa chất trừ sâu phospho hữu cơ",
            "gọi 115 ngay",
            "rời khỏi khu vực nhiễm độc nếu an toàn",
            "cởi bỏ quần áo nhiễm hóa chất và rửa vùng tiếp xúc bằng nước sạch",
            "không tự gây nôn hoặc cho uống thuốc giải độc tại nhà",
        ]
    if intent == "opioid_poisoning":
        return [
            "nghi ngộ độc opioid, nguy cơ thở chậm hoặc ngừng thở",
            "gọi 115 ngay",
            "theo dõi nhịp thở và ý thức",
            "đặt nằm nghiêng an toàn nếu còn thở",
            "ép tim ngoài lồng ngực nếu không thở bình thường",
        ]
    if intent == "acute_poisoning":
        return [
            "nghi ngộ độc cấp",
            "gọi 115 ngay",
            "đưa ra khỏi nguồn độc nếu an toàn",
            "giữ lại thuốc, hóa chất, thức ăn hoặc bao bì nghi gây độc",
            "không tự gây nôn, cho uống thuốc, than hoạt hoặc mẹo dân gian",
        ]
    if intent == "snakebite":
        return [
            "nghi rắn độc cắn",
            "gọi 115 ngay",
            f"cho {subject} nằm yên và bất động chi bị cắn",
            "tháo nhẫn, vòng hoặc đồ bó chặt gần vùng bị cắn",
            "không rạch vết cắn, không hút nọc, không chườm đá, không tự garô",
        ]
    if intent == "severe_dyspnea":
        return [
            "khó thở nặng, nguy cơ suy hô hấp",
            f"để {subject} ở tư thế dễ thở",
            "nới lỏng quần áo",
            "không tự đi lại hoặc tự lái xe",
            "theo dõi ý thức, nhịp thở",
        ]
    if intent == "shock_sepsis":
        return [
            "nghi sốc nhiễm khuẩn hoặc suy tuần hoàn",
            "gọi 115 ngay",
            "nằm an toàn, giữ ấm nhẹ",
            "không tự dùng thuốc hoặc truyền dịch tại nhà",
        ]
    if intent == "dengue_warning":
        return [
            "dấu hiệu cảnh báo nguy hiểm trong sốt xuất huyết Dengue",
            "giai đoạn hết sốt có thể là giai đoạn nguy hiểm",
            "gọi 115 ngay",
            "không trì hoãn gọi 115",
        ]
    if intent == "hypoglycemia":
        return [
            "nghi hạ đường huyết cấp, có thể diễn biến nhanh đến hôn mê",
            "gọi 115 ngay nếu lơ mơ, co giật, hôn mê hoặc không cải thiện nhanh",
            f"nếu {subject} còn tỉnh và nuốt an toàn, cho dùng đồ uống hoặc thức ăn có đường",
            f"không cho {subject} ăn uống nếu lơ mơ, co giật hoặc hôn mê",
            "chuẩn bị thông tin insulin, thuốc hạ đường huyết, bữa ăn gần nhất",
        ]
    if intent == "coma_unconscious":
        return [
            "hôn mê hoặc mất ý thức là tình trạng cấp cứu",
            "gọi 115 ngay",
            f"đặt {subject} nằm nghiêng an toàn nếu còn thở",
            "theo dõi nhịp thở; ép tim nếu không thở bình thường",
            "không cho ăn uống hoặc tự dùng thuốc",
        ]
    if intent == "gi_bleeding":
        return [
            "nghi xuất huyết tiêu hóa, có thể mất máu nặng",
            "gọi 115 ngay",
            f"để {subject} nằm nghỉ, nghiêng đầu sang bên nếu buồn nôn hoặc nôn máu",
            "không ăn uống hoặc tự dùng thuốc cầm máu trong lúc chờ cấp cứu",
            "chuẩn bị thông tin lượng máu nôn/đi ngoài, thuốc đang dùng và bệnh nền",
        ]
    if intent == "hypovolemic_shock":
        return [
            "nghi sốc giảm thể tích hoặc mất máu nặng",
            "gọi 115 ngay",
            f"để {subject} nằm an toàn, giữ ấm nhẹ và theo dõi ý thức",
            "nếu có chảy máu ngoài, ép trực tiếp lên vết thương bằng gạc hoặc khăn sạch",
            "không tự truyền dịch hoặc cho ăn uống nếu lơ mơ, nôn nhiều",
        ]
    if intent == "acute_abdomen":
        return [
            "đau bụng dữ dội kèm bụng cứng có thể là bệnh lý ổ bụng cấp",
            "gọi 115 ngay",
            "không tự dùng thuốc giảm đau",
            "không ăn uống trong lúc chờ đánh giá cấp cứu",
        ]
    return []


def _deterministic_bullets(
    question: str,
    red_flags: Iterable[str] | str | None = None,
    subject_address: str = "bạn",
) -> list[str]:
    """Fallback-only deterministic aid for retrieval/LLM failure."""
    intent = classify_emergency_intent(question, red_flags)
    subject = _subject_label(question, subject_address)
    if intent == "anaphylaxis":
        return [
            "Đây là dấu hiệu nghi phản vệ nặng, có thể đe dọa tính mạng.",
            "Gọi 115 ngay; không tự theo dõi tại nhà.",
            f"Để {subject} ở tư thế dễ thở và theo dõi nhịp thở, ý thức.",
            "Tránh tiếp xúc thêm với dị nguyên nghi ngờ nếu làm được an toàn.",
            "Chuẩn bị thông tin về thức ăn, thuốc và thời điểm xuất hiện triệu chứng.",
        ]
    if intent == "cardiac_arrest":
        return [
            "Đây là tình huống nghi ngừng tuần hoàn.",
            f"Nếu {subject} không thở bình thường, ép tim ngoài lồng ngực ngay trong lúc chờ 115.",
            "Nhờ người xung quanh lấy AED nếu có sẵn và làm theo hướng dẫn của máy.",
            f"Không chờ {subject} tự tỉnh; tiếp tục ép tim cho đến khi nhân viên y tế tiếp nhận.",
        ]
    if intent == "stroke":
        return [
            "Đây là dấu hiệu nghi đột quỵ cấp.",
            "Ghi lại chính xác thời điểm khởi phát triệu chứng.",
            f"Không cho {subject} ăn uống hoặc tự dùng thuốc trong lúc chờ 115.",
            f"Không để {subject} tự lái xe; chờ 115 đưa đến bệnh viện có khả năng cấp cứu đột quỵ.",
        ]
    if intent == "chest_pain_acs":
        return [
            "Đây là dấu hiệu nghi hội chứng vành cấp.",
            f"Để {subject} nghỉ, tránh gắng sức trong lúc chờ 115.",
            "Nới lỏng quần áo chật và theo dõi khó thở, vã mồ hôi, ý thức.",
            f"Không để {subject} tự lái xe hoặc tự chở đi cấp cứu.",
            "Không chờ cơn đau tự hết nếu đau ngực dữ dội hoặc kéo dài.",
        ]
    if intent == "seizure":
        return [
            "Đây là cấp cứu co giật kéo dài hoặc nghi trạng thái động kinh.",
            f"Đặt {subject} nằm nghiêng an toàn và bảo vệ khỏi va đập.",
            f"Không nhét gì vào miệng và không cố ghì {subject} trong cơn co giật.",
            "Theo dõi thời gian co giật; nếu kéo dài trên 5 phút hoặc chưa tỉnh lại, duy trì liên hệ 115.",
        ]
    if intent == "co_poisoning":
        return [
            "Đây là tình huống nghi ngộ độc khí CO.",
            f"Nếu an toàn, đưa {subject} ra nơi thoáng khí; không quay lại phòng kín nhiễm khí.",
            "Tránh để người cứu hộ tiếp tục hít khí độc.",
            "Không trì hoãn để bệnh nhân được thở oxy và hồi sức tại bệnh viện.",
            f"Nếu {subject} không thở bình thường, bắt đầu ép tim ngoài lồng ngực trong lúc chờ 115.",
        ]
    if intent == "organophosphate_poisoning":
        return [
            "Đây là tình huống nghi ngộ độc hóa chất trừ sâu phospho hữu cơ.",
            "Gọi 115 ngay; rời khỏi khu vực nhiễm độc nếu làm được an toàn.",
            "Cởi bỏ quần áo nhiễm hóa chất và rửa vùng da tiếp xúc bằng nước sạch.",
            "Không tự gây nôn hoặc cho uống thuốc giải độc tại nhà.",
            "Giữ lại chai, bao bì hoặc tên hóa chất để giao cho nhân viên y tế.",
        ]
    if intent == "opioid_poisoning":
        return [
            "Đây là tình huống nghi ngộ độc opioid, có nguy cơ thở chậm hoặc ngừng thở.",
            "Gọi 115 ngay và theo dõi nhịp thở, ý thức liên tục.",
            f"Nếu {subject} còn thở, đặt nằm nghiêng an toàn để giảm nguy cơ sặc.",
            f"Nếu {subject} không thở bình thường, bắt đầu ép tim ngoài lồng ngực trong lúc chờ 115.",
            "Giữ lại thông tin thuốc/chất đã dùng và thời điểm sử dụng nếu biết.",
        ]
    if intent == "acute_poisoning":
        return [
            "Đây là tình huống nghi ngộ độc cấp.",
            "Gọi 115 ngay; đưa bệnh nhân ra khỏi nguồn độc nếu làm được an toàn.",
            "Giữ lại thuốc, hóa chất, thức ăn hoặc bao bì nghi gây độc để giao cho nhân viên y tế.",
            "Không tự gây nôn, cho uống thuốc, than hoạt hoặc mẹo dân gian nếu chưa được 115 hướng dẫn.",
            f"Nếu {subject} lơ mơ, đặt nằm nghiêng an toàn và theo dõi nhịp thở.",
        ]
    if intent == "snakebite":
        return [
            "Đây là tình huống nghi rắn độc cắn.",
            f"Cho {subject} nằm yên, hạn chế cử động và bất động chi bị cắn trong lúc chờ 115.",
            "Tháo nhẫn, vòng hoặc đồ bó chặt gần vùng bị cắn trước khi sưng nề tăng.",
            "Không rạch vết cắn, không hút nọc, không chườm đá, không tự garô.",
            "Nếu nhận diện được rắn thì ghi nhớ đặc điểm hoặc chụp ảnh từ xa an toàn, không cố bắt rắn.",
        ]
    if intent == "severe_dyspnea":
        return [
            "Đây là khó thở nặng, có nguy cơ suy hô hấp.",
            f"Để {subject} ở tư thế dễ thở, nới lỏng quần áo trong lúc chờ 115.",
            f"Không để {subject} tự đi lại hoặc tự lái xe.",
            f"Theo dõi nhịp thở, ý thức; báo ngay khi gọi 115 nếu {subject} ngừng thở.",
        ]
    if intent == "shock_sepsis":
        return [
            "Đây là dấu hiệu nghi sốc nhiễm khuẩn hoặc suy tuần hoàn.",
            f"Để {subject} nằm an toàn, giữ ấm nhẹ và theo dõi ý thức trong lúc chờ 115.",
            "Gọi 115 ngay; nếu được nhân viên y tế hướng dẫn hoặc không thể gọi 115 thì đến cơ sở cấp cứu phù hợp.",
            "Không tự dùng thuốc hoặc truyền dịch tại nhà.",
            "Báo với điều phối viên 115 về sốt cao, lơ mơ, tụt huyết áp, thở nhanh.",
        ]
    if intent == "dengue_warning":
        return [
            "Đây là dấu hiệu cảnh báo nguy hiểm trong sốt xuất huyết Dengue.",
            "Giai đoạn hết sốt có thể là giai đoạn nguy hiểm.",
            "Gọi 115 ngay; nếu được nhân viên y tế hướng dẫn hoặc không thể gọi 115 thì đến cơ sở cấp cứu phù hợp.",
            f"Để {subject} nghỉ, theo dõi ý thức và chuẩn bị thông tin ngày bệnh, thuốc đã dùng.",
            "Không tự dùng thuốc giảm đau kháng viêm; không trì hoãn gọi 115.",
        ]
    if intent == "hypoglycemia":
        return [
            "Đây là tình huống nghi hạ đường huyết cấp, có thể diễn biến nhanh đến hôn mê.",
            "Gọi 115 ngay nếu có lơ mơ, co giật, hôn mê hoặc không cải thiện nhanh.",
            f"Nếu {subject} còn tỉnh và nuốt an toàn, cho dùng đồ uống hoặc thức ăn có đường.",
            f"Không cho {subject} ăn uống nếu lơ mơ, co giật hoặc hôn mê.",
            "Chuẩn bị thông tin insulin, thuốc hạ đường huyết và bữa ăn gần nhất.",
        ]
    if intent == "coma_unconscious":
        return [
            "Hôn mê hoặc mất ý thức là tình trạng cấp cứu.",
            "Gọi 115 ngay và theo dõi nhịp thở trong lúc chờ hỗ trợ.",
            f"Nếu {subject} còn thở, đặt nằm nghiêng an toàn để giảm nguy cơ tụt lưỡi, sặc.",
            f"Nếu {subject} không thở bình thường, bắt đầu ép tim ngoài lồng ngực trong lúc chờ 115.",
            "Không cho ăn uống hoặc tự dùng thuốc.",
        ]
    if intent == "gi_bleeding":
        return [
            "Đây là tình huống nghi xuất huyết tiêu hóa, có thể mất máu nặng.",
            "Gọi 115 ngay; không chờ tự hết nếu nôn máu, đi ngoài phân đen/ra máu hoặc choáng.",
            f"Để {subject} nằm nghỉ, nghiêng đầu sang bên nếu buồn nôn hoặc nôn máu.",
            "Không ăn uống hoặc tự dùng thuốc cầm máu trong lúc chờ cấp cứu.",
            "Chuẩn bị thông tin lượng máu nôn/đi ngoài, thuốc đang dùng và bệnh nền.",
        ]
    if intent == "hypovolemic_shock":
        return [
            "Đây là tình huống nghi sốc giảm thể tích hoặc mất máu nặng.",
            "Gọi 115 ngay; theo dõi ý thức, nhịp thở và tay chân lạnh trong lúc chờ hỗ trợ.",
            f"Để {subject} nằm an toàn, giữ ấm nhẹ.",
            "Nếu có chảy máu ngoài, ép trực tiếp lên vết thương bằng gạc hoặc khăn sạch.",
            "Không tự truyền dịch hoặc cho ăn uống nếu lơ mơ, nôn nhiều.",
        ]
    if intent == "acute_abdomen":
        return [
            "Đau bụng dữ dội kèm bụng cứng, vã mồ hôi hoặc chóng mặt có thể là bệnh lý ổ bụng cấp.",
            "Gọi 115 ngay; nếu được nhân viên y tế hướng dẫn hoặc không thể gọi 115 thì đến cơ sở cấp cứu phù hợp.",
            "Không tự dùng thuốc giảm đau, không ăn uống trong lúc chờ được đánh giá cấp cứu.",
            "Nằm yên ở tư thế đỡ đau và theo dõi chóng mặt, vã mồ hôi, ngất.",
        ]
    return []


def generate_emergency_aid(
    question: str,
    hits: list[EmergencyHit],
    red_flags: Iterable[str] | str | None = None,
    subject_address: str = "bạn",
) -> list[str]:
    """Return 3-5 first-aid bullet strings.

    With EMERGENCY_AID_USE_LLM=1 and confident retrieved evidence, LLM/RAG is
    primary. Deterministic templates are used only as fallback; mandatory safety
    actions are enforced later by ``apply_safety_post_check``.
    """
    intent = classify_emergency_intent(question, red_flags)
    subject = _subject_label(question, subject_address)
    if os.getenv("EMERGENCY_AID_USE_LLM", "0") != "1":
        return _deterministic_bullets(question, red_flags, subject) or _fallback_bullets(hits)
    if hits and _has_confident_evidence(intent, hits):
        prompt = build_emergency_aid_prompt(
            question,
            hits[:3],
            red_flags,
            required_points=_intent_required_points(intent, subject),
        )
        raw = _try_call_llm(prompt, EMERGENCY_AID_SYSTEM_PROMPT)
        bullets = _parse_bullets(raw)
        if bullets:
            return bullets
    return _deterministic_bullets(question, red_flags, subject) or _fallback_bullets(hits)
