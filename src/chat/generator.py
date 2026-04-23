"""
generator.py
------------
Build prompt từ context + gọi Grok (xai_sdk) để sinh câu trả lời.
"""

from __future__ import annotations

from xai_sdk.chat import system, user

from src.chat.retriever import Hit
from src.config import MODEL, make_xai_client

SYSTEM_PROMPT = """Bạn là trợ lý y tế ảo, trả lời bằng tiếng Việt dựa trên tài liệu được cung cấp.

Nguyên tắc:
- CHỈ dựa vào phần "Tài liệu tham khảo" bên dưới. Nếu không đủ thông tin, nói thẳng "Tôi không có đủ thông tin trong tài liệu".
- Không tự chẩn đoán thay bác sĩ. Với triệu chứng nghiêm trọng, luôn khuyên người dùng đi khám/gọi cấp cứu.
- Với câu hỏi về thuốc OTC: nêu chỉ định, liều dùng, chống chỉ định, lưu ý — KHÔNG kê đơn thuốc kê toa.
- Trình bày gọn, có thể dùng gạch đầu dòng. Trích dẫn nguồn cuối câu trả lời dạng [1], [2]... khớp với danh sách nguồn.
- Nhiều đoạn tài liệu có thể chia sẻ cùng một chỉ số nguồn (ví dụ [1]); điều đó là cố ý và chính xác.
"""

DRUG_URL_TEMPLATE = "https://trungtamthuoc.com/hoat-chat/{slug}"


def _source_key(h: Hit) -> tuple:
    return (h.source_type, h.source_slug)


def _dedupe(hits: list[Hit]) -> tuple[list[Hit], list[int]]:
    seen: dict[tuple, int] = {}
    unique: list[Hit] = []
    cite_idx: list[int] = []
    for h in hits:
        k = _source_key(h)
        if k not in seen:
            unique.append(h)
            seen[k] = len(unique)
        cite_idx.append(seen[k])
    return unique, cite_idx


def _drug_label(h: Hit) -> str:
    url = DRUG_URL_TEMPLATE.format(slug=h.source_slug)
    return f"Dược thư Quốc gia 2022 - [{h.source_name}]({url})"


def _disease_label(h: Hit) -> str:
    chapter = (h.metadata or {}).get("chapter", "")
    return (
        "Hướng dẫn chẩn đoán và điều trị - Bệnh viện Bạch Mai - "
        f"{chapter} - {h.source_name}".rstrip(" -")
    )


def _format_context(hits: list[Hit], cite_idx: list[int]) -> str:
    blocks = []
    for i, h in enumerate(hits):
        header = f"[{cite_idx[i]}] {h.source_name}"
        if h.heading_path:
            header += f" — {h.heading_path}"
        blocks.append(f"{header}\n{h.text}")
    return "\n\n---\n\n".join(blocks)


def _format_sources(unique: list[Hit]) -> str:
    lines = []
    for i, h in enumerate(unique, 1):
        label = _drug_label(h) if h.source_type == "drug" else _disease_label(h)
        lines.append(f"[{i}] {label}")
    return "\n".join(lines)


def generate(question: str, hits: list[Hit], kg_text: str = "") -> str:
    if not hits and not kg_text:
        return ("Tôi không tìm thấy thông tin phù hợp trong tài liệu. "
                "Bạn vui lòng hỏi cụ thể hơn hoặc tham khảo ý kiến bác sĩ.")

    unique, cite_idx = _dedupe(hits)

    client = make_xai_client()
    context = _format_context(hits, cite_idx)

    prompt_parts = [f"Câu hỏi: {question}\n"]
    if kg_text:
        prompt_parts.append(f"Thông tin từ Knowledge Graph:\n{kg_text}\n")
    if context:
        prompt_parts.append(f"Tài liệu tham khảo:\n{context}\n")
    prompt_parts.append("Hãy trả lời câu hỏi trên dựa vào tài liệu.")
    prompt_user = "\n".join(prompt_parts)

    chat = client.chat.create(model=MODEL)
    chat.append(system(SYSTEM_PROMPT))
    chat.append(user(prompt_user))
    response = chat.sample()

    answer = (response.content or "").strip()
    sources = _format_sources(unique)
    return f"{answer}\n\nNguồn:\n{sources}"
