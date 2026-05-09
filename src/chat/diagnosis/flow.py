from __future__ import annotations

from src.chat.storage.session import PatientSession


def symptom_names(session: PatientSession) -> list[str]:
    names = []
    for symptom in session.symptoms:
        name = symptom.get("name") or symptom.get("symptom_id", "")
        if name:
            names.append(name)
    return names


def candidate_names(session: PatientSession, limit: int = 5) -> list[str]:
    return [c.get("name", "") for c in session.candidate_diseases[:limit] if c.get("name")]


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
        "Yêu cầu trả lời:\n"
        "- Nói rõ rằng dữ kiện hiện tại chưa đủ để chẩn đoán chính xác.\n"
        "- Liệt kê một số bệnh có thể liên quan trong danh sách đang cân nhắc, "
        "mỗi bệnh 1-2 ý ngắn về dấu hiệu/lý do liên quan theo tài liệu.\n"
        "- Nêu dấu hiệu cần đi khám sớm hoặc cấp cứu nếu có.\n"
        "- Khuyên người dùng đi khám bác sĩ để được chẩn đoán và xử trí phù hợp.\n"
        "- Không hỏi thêm câu hỏi làm rõ trong câu trả lời này."
    )
    return prompt, retrieval_query
