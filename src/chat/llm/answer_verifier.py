"""Evidence-grounded answer repair for factual disease/drug answers."""

from __future__ import annotations

import json
import logging

from src.chat.llm.mini import call_mini
from src.config import FAST_MODEL_MAX_TOKENS

log = logging.getLogger(__name__)

_SUPPORTED_DOMAINS = {"disease_info", "drug_info"}

ANSWER_EVIDENCE_VERIFIER_SYSTEM = """
Bạn là bộ kiểm chứng câu trả lời y khoa theo bằng chứng đã truy xuất.
Chỉ sửa câu trả lời dựa trên tài liệu được cung cấp trong prompt.

Nhiệm vụ:
- Giữ lại các ý đúng và có căn cứ trong tài liệu.
- Bổ sung ý bắt buộc nếu tài liệu được chọn có nêu rõ.
- Nếu kế hoạch bằng chứng không nêu ý bắt buộc, hãy tự xác định các ý cốt lõi
  trong tài liệu trực tiếp trả lời câu hỏi.
- Xóa hoặc làm rõ các khẳng định không có trong tài liệu được chọn.
- Xóa các phần ngoài phạm vi câu hỏi, kể cả khi chúng có trong tài liệu, nếu
  chúng làm câu trả lời lan sang chẩn đoán/điều trị/biến chứng/cảnh báo không
  được hỏi trực tiếp.
- Không thêm kiến thức y khoa từ trí nhớ ngoài tài liệu.
- Giữ văn phong tiếng Việt tự nhiên, ngắn gọn, phù hợp người bệnh.
- Giữ citation dạng [1], [2] cho các ý được giữ hoặc bổ sung.
- Với câu hỏi tổng quan bệnh, ưu tiên định nghĩa, bản chất bệnh, tần suất/đối
  tượng thường gặp và ảnh hưởng chính nếu tài liệu nêu rõ; tránh liệt kê quy
  trình chẩn đoán hoặc điều trị khi người dùng không hỏi.
- Với câu hỏi thuốc, chỉ trả lời đúng loại thông tin người dùng hỏi; không thêm
  cảnh báo chung hoặc liều/cách dùng nếu tài liệu được chọn không nêu.

Trả về JSON hợp lệ với đúng các trường:
{
  "needs_rewrite": true | false,
  "missing_required_facts": ["..."],
  "unsupported_claims": ["..."],
  "repaired_answer": "..."
}
Nếu không cần sửa, đặt needs_rewrite=false và repaired_answer bằng câu trả lời ban đầu.
""".strip()


def repair_answer_with_evidence(
    *,
    question: str,
    answer: str,
    evidence_text: str,
    evidence_plan: dict | None,
    answer_domain: str,
) -> str:
    """Ask a small LLM to remove unsupported claims and fill supported facts."""
    if answer_domain not in _SUPPORTED_DOMAINS:
        return answer
    if not answer.strip() or not evidence_text.strip():
        return answer

    user_prompt = "\n\n".join(
        (
            f"Câu hỏi:\n{question}",
            "Kế hoạch bằng chứng dạng JSON:\n"
            + json.dumps(evidence_plan or {}, ensure_ascii=False, indent=2),
            f"Tài liệu được chọn:\n{evidence_text}",
            f"Câu trả lời cần kiểm chứng:\n{answer}",
        )
    )
    result = call_mini(
        ANSWER_EVIDENCE_VERIFIER_SYSTEM,
        user_prompt,
        max_tokens=max(FAST_MODEL_MAX_TOKENS, 1536),
        stage="answer_evidence_verifier",
    )
    if not isinstance(result, dict):
        return answer

    repaired = result.get("repaired_answer")
    if not isinstance(repaired, str) or not repaired.strip():
        return answer
    if not bool(result.get("needs_rewrite")):
        return repaired.strip() or answer

    log.info(
        "answer verifier rewrote domain=%s missing=%s unsupported=%s",
        answer_domain,
        len(result.get("missing_required_facts") or []),
        len(result.get("unsupported_claims") or []),
    )
    return repaired.strip()
