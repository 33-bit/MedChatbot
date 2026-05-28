"""
clarification_options.py
------------------------
Button option helpers for symptom clarification questions.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from typing import Any

CONTROL_OPTIONS = ("Không rõ", "Trả lời luôn")
PRESENCE_OPTIONS = ("Có", "Không", *CONTROL_OPTIONS)


def choice_key(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.strip().casefold())
    no_marks = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    no_marks = no_marks.replace("đ", "d")
    for char in ".!?":
        no_marks = no_marks.replace(char, " ")
    return " ".join(no_marks.split())


def normalize_options(options: Iterable[Any] | None) -> tuple[str, ...]:
    if not options:
        return ()
    normalized: list[str] = []
    seen: set[str] = set()
    for option in options:
        label = str(option).strip()
        key = choice_key(label)
        if not label or key in seen:
            continue
        seen.add(key)
        normalized.append(label)
    for control in CONTROL_OPTIONS:
        key = choice_key(control)
        if key not in seen:
            seen.add(key)
            normalized.append(control)
    return tuple(normalized)


def presence_options_from_catalog(
    question: str,
    catalog: dict[str, dict],
) -> tuple[str, ...]:
    subject = _presence_subject(question)
    if not subject:
        return ()
    subject_key = choice_key(subject)
    for entry in catalog.values():
        if choice_key(str(entry.get("name_vi", ""))) != subject_key:
            continue
        options = entry.get("clarification_options", {})
        if isinstance(options, dict):
            custom = normalize_options(options.get("presence"))
            if custom:
                return custom
        break
    return PRESENCE_OPTIONS


def detail_options_from_catalog(
    question: str,
    catalog: dict[str, dict],
) -> tuple[str, ...]:
    question_key = choice_key(question)
    for entry in catalog.values():
        questions = entry.get("clarification_questions", {}) or {}
        if not isinstance(questions, dict):
            continue
        for slot, stored_question in questions.items():
            if choice_key(str(stored_question)) != question_key:
                continue
            options = entry.get("clarification_options", {})
            if isinstance(options, dict):
                custom = normalize_options(options.get(slot))
                if custom:
                    return custom
            return ()
    return ()


def fallback_detail_options(question: str) -> tuple[str, ...]:
    key = choice_key(question)
    fallback = CONTROL_OPTIONS
    if "bat dau" in key or "xuat hien tu khi nao" in key or key.endswith("tu khi nao"):
        return ("Hôm nay", "Hôm qua", "2-3 ngày", "Trên 3 ngày", *fallback)
    if "tan suat" in key:
        if "di tieu" in key or "dai" in key or "tieu tien" in key:
            return (
                "Vài lần/ngày",
                "Mỗi 1-2 giờ",
                "30 phút/lần",
                "10 phút/lần",
                *fallback,
            )
        if "non" in key:
            return ("1-2 lần/ngày", "3-5 lần/ngày", "> 5 lần/ngày", *fallback)
        if "tieu chay" in key or "ia chay" in key:
            return ("1-2 lần/ngày", "3-5 lần/ngày", "> 5 lần/ngày", *fallback)
    if "lien tuc hay ngat quang" in key:
        return ("Liên tục", "Ngắt quãng", *fallback)
    if "lien tuc hay theo tung dot" in key:
        return ("Liên tục", "Từng đợt", *fallback)
    if "lien tuc hay tung con" in key:
        return ("Liên tục", "Từng cơn", *fallback)
    if "sot cao" in key or "nhiet do" in key:
        return ("< 38.5 độ", "38.5-39 độ", "> 39 độ", "Không đo", *fallback)
    if "0-10" in key or "muc do dau" in key:
        return ("Nhẹ", "Vừa", "Nặng", "Dữ dội", *fallback)
    if "muc do" in key:
        return ("Nhẹ", "Vừa", "Nặng", *fallback)
    associated_choices = _associated_detail_choices(question, fallback)
    if associated_choices:
        return associated_choices
    if key.startswith("co ") and key.endswith(" khong"):
        return ("Có", "Không", *fallback)
    return fallback


def validate_clarification_options(entry: dict) -> list[str]:
    errors: list[str] = []
    options = entry.get("clarification_options")
    if not isinstance(options, dict):
        return ["clarification_options must be an object"]
    if not normalize_options(options.get("presence")):
        errors.append("presence options are missing")

    questions = entry.get("clarification_questions", {}) or {}
    if not isinstance(questions, dict):
        questions = {}
    for slot, question in questions.items():
        slot_options = normalize_options(options.get(slot))
        if not question:
            continue
        if len(slot_options) < 2:
            errors.append(f"{slot} options are missing or too short")
        for control in CONTROL_OPTIONS:
            if choice_key(control) not in {choice_key(option) for option in slot_options}:
                errors.append(f"{slot} options must include {control}")
    return errors


def _presence_subject(question: str) -> str:
    text = question.strip().rstrip("?").strip()
    prefix = "Bạn có bị "
    suffix = " không"
    if not text.startswith(prefix) or not text.endswith(suffix):
        return ""
    return text[len(prefix):-len(suffix)].strip()


def _associated_detail_choices(
    question: str,
    fallback: tuple[str, ...],
) -> tuple[str, ...]:
    text = question.strip().rstrip("?").strip()
    prefixes = ("Có kèm theo ", "Có kèm ", "Có liên quan đến ")
    subject = ""
    for prefix in prefixes:
        if text.startswith(prefix):
            subject = text[len(prefix):]
            break
    if not subject:
        return ()
    if subject.casefold().endswith(" không"):
        subject = subject[:-len(" không")].strip()
    parts = [
        part.strip(" .")
        for part in subject.replace(" hoặc ", ",").replace(" hay ", ",").split(",")
        if part.strip(" .")
    ]
    if len(parts) < 2:
        return ("Có", "Không", *fallback)
    positive_choices = tuple(f"Có {part}" for part in parts[:4])
    combined_choice = ("Cả hai",) if len(parts) == 2 else ("Nhiều triệu chứng",)
    return (*positive_choices, *combined_choice, "Không", *fallback)
