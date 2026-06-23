from __future__ import annotations

from src.chat.context.resolver import format_subject_address
from src.chat.storage.session import PatientSession

_DETAIL_LABELS = {
    "onset": "khởi phát",
    "severity": "mức độ",
    "pattern": "diễn tiến",
    "associated": "triệu chứng kèm",
}


def symptom_names(session: PatientSession) -> list[str]:
    names = []
    for symptom in session.symptoms:
        name = symptom.get("name") or symptom.get("symptom_id", "")
        if name:
            details = [
                f"{label}: {_detail_value(symptom[key])}"
                for key, label in _DETAIL_LABELS.items()
                if symptom.get(key)
            ]
            if details:
                name = f"{name} ({'; '.join(details)})"
            names.append(name)
    return names


def _detail_value(value) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value if item)
    return str(value)


def candidate_names(session: PatientSession, limit: int = 5) -> list[str]:
    return [c.get("name", "") for c in session.candidate_diseases[:limit] if c.get("name")]


def build_general_triage_prompt(
    session: PatientSession,
    subject: dict | None = None,
) -> str:
    subject_label = format_subject_address(subject)
    symptoms = ", ".join(symptom_names(session)) or "triệu chứng này"
    symptoms_lower = symptoms.lower()
    if "đau bụng" in symptoms_lower:
        possible_causes = (
            "Với mô tả hiện tại, tôi chưa thể kết luận chính xác nguyên nhân. "
            "Tình trạng này có thể liên quan đến rối loạn tiêu hóa, đầy hơi/khó tiêu, "
            "viêm dạ dày-ruột, nhưng cũng cần theo dõi để loại trừ viêm ruột thừa "
            "hoặc bệnh lý bụng cấp nếu đau tăng dần hoặc khu trú rõ."
        )
        interim_care = (
            f"Trong lúc theo dõi, {subject_label} nên nghỉ ngơi, uống từng ngụm "
            "nước nhỏ nếu buồn nôn, "
            "ăn nhẹ thức ăn mềm nếu thấy đói và tránh đồ nhiều dầu mỡ/rượu bia. "
            "Không nên tự uống thuốc giảm đau mạnh, thuốc cầm tiêu chảy hoặc kháng sinh "
            "khi chưa rõ nguyên nhân."
        )
        red_flags = (
            "Nếu đau dữ dội tăng nhanh, bụng cứng, ngất/choáng, nôn liên tục, "
            "đi ngoài ra máu, sốt cao, khó thở, đau ngực, lơ mơ, hoặc có khả năng "
            "đang mang thai, hãy đi cấp cứu ngay."
        )
    else:
        possible_causes = (
            "Với thông tin hiện tại, tôi chưa thể kết luận chính xác nguyên nhân. "
            f"{symptoms.capitalize()} có thể liên quan đến tình trạng nhẹ thường gặp, "
            "nhưng cũng cần theo dõi để loại trừ các nguyên nhân cần khám sớm nếu "
            "triệu chứng nặng lên hoặc kéo dài."
        )
        interim_care = (
            f"Trong lúc theo dõi, {subject_label} nên nghỉ ngơi, uống đủ nước, "
            "ăn uống nhẹ nếu phù hợp "
            "và tránh tự dùng kháng sinh hoặc thuốc mạnh khi chưa rõ nguyên nhân."
        )
        red_flags = (
            "Nếu có khó thở, đau ngực dữ dội, lơ mơ, co giật, yếu liệt, chảy máu nhiều, "
            "ngất/choáng hoặc triệu chứng nặng lên nhanh, hãy gọi cấp cứu 115 hoặc đến "
            "cơ sở y tế gần nhất ngay."
        )

    return (
        f"Tôi hiểu {subject_label} đang bị {symptoms}. Triệu chứng này có nhiều nguyên nhân, "
        "nên trước hết tôi cần vài thông tin chung để xem có dấu hiệu cần đi khám gấp không.\n\n"
        f"{possible_causes}\n\n"
        f"{interim_care}\n\n"
        f"{red_flags}\n\n"
        "Để tôi định hướng tốt hơn, bạn cho tôi hỏi thêm một vài câu hỏi nhé."
    )


def direct_diagnostic_prompt(session: PatientSession, user_request: str) -> tuple[str, str]:
    symptoms = ", ".join(symptom_names(session)) or "chưa rõ"
    candidates = ", ".join(candidate_names(session)) or "chưa có danh sách bệnh nghi ngờ"
    retrieval_query = f"{symptoms} {candidates}".strip()
    prompt = (
        "Người dùng đã mô tả triệu chứng và không thể hoặc không muốn trả lời thêm "
        "câu hỏi làm rõ. Hãy trả lời trực tiếp dựa trên tài liệu.\n\n"
        f"Triệu chứng đã biết: {symptoms}.\n"
        f"Các bệnh hệ thống đang cân nhắc: {candidates}.\n"
        f"Yêu cầu mới nhất của người dùng: {user_request}.\n\n"
        "Yêu cầu trả lời theo template tư vấn:\n"
        "- Trình bày tối đa 4 phần, mỗi phần có tiêu đề ngắn và 1-3 ý ngắn.\n"
        "- Không dùng dòng phân cách như \"---\" và không tạo quá nhiều mục nhỏ.\n"
        "- Ghi nhận ngắn gọn điều người dùng đang lo.\n"
        "- Nhận định sơ bộ và nói rõ dữ kiện hiện tại chưa đủ để chẩn đoán chính xác, "
        "không chẩn đoán chắc chắn qua chat.\n"
        "- Nêu 2-3 khả năng có thể liên quan trong danh sách đang cân nhắc, "
        "mỗi khả năng 1-2 ý ngắn về dấu hiệu/lý do liên quan theo tài liệu.\n"
        "- Nêu chăm sóc tạm thời an toàn nếu phù hợp và cảnh báo không tự dùng thuốc nguy cơ cao.\n"
        "- Nêu dấu hiệu nguy hiểm cần đi khám sớm hoặc gọi cấp cứu 115.\n"
        "- Khuyên người dùng đi khám bác sĩ để được chẩn đoán và xử trí phù hợp.\n"
        "- Không hỏi thêm câu hỏi làm rõ trong câu trả lời này."
    )
    return prompt, retrieval_query
