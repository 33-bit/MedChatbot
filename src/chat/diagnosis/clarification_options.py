"""
clarification_options.py
------------------------
Button option helpers for symptom clarification questions.
"""

from __future__ import annotations

import ast
import unicodedata
from collections.abc import Iterable
from typing import Any

CONTROL_OPTIONS = ("Không rõ", "Trả lời luôn")
PRESENCE_OPTIONS = ("Có", "Không", *CONTROL_OPTIONS)
SINGLE_SELECT = "single"
MULTI_SELECT = "multi"
SELECTION_MODES = {SINGLE_SELECT, MULTI_SELECT}
_NEGATIVE_OPTION_KEYS = {
    "khong",
    "khong co",
    "khong lien quan",
}
_GENERIC_OPTION_KEYS = {
    "khac",
    "ca hai",
    "nhieu trieu chung",
}
_COURSE_PATTERN_KEYS = {
    "lien tuc",
    "tung con",
    "tung dot",
    "ngat quang",
}
_LOCATION_MARKERS = (
    "thuong vi",
    "quanh ron",
    "ha vi",
    "ho chau",
    "toan bung",
    "vung bung",
    "man suon",
)
_COMBINED_OPTION_PREFIXES = ("Cả ", "Nhiều triệu chứng", "Chỉ 1-2")


def _has_combined_options(options: tuple[str, ...]) -> bool:
    """A 'Cả X + Y' / 'Nhiều triệu chứng' / 'Chỉ 1-2' option only makes sense
    when the author intended multi-select; its presence is a hard signal."""
    return any(option.startswith(_COMBINED_OPTION_PREFIXES) for option in options)


def choice_key(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.strip().casefold())
    no_marks = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    no_marks = no_marks.replace("đ", "d")
    for char in ".!?":
        no_marks = no_marks.replace(char, " ")
    return " ".join(no_marks.split())


def _option_labels(options: Any) -> list[str]:
    if options is None:
        return []
    if isinstance(options, str):
        label = options.strip()
        if label.startswith("[") and label.endswith("]"):
            try:
                parsed = ast.literal_eval(label)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, (list, tuple, set)):
                return _option_labels(parsed)
        return [label] if label else []
    if isinstance(options, dict):
        return []
    if isinstance(options, Iterable):
        labels: list[str] = []
        for option in options:
            labels.extend(_option_labels(option))
        return labels
    label = str(options).strip()
    return [label] if label else []


def normalize_options(options: Iterable[Any] | None) -> tuple[str, ...]:
    if not options:
        return ()
    normalized: list[str] = []
    seen: set[str] = set()
    control_keys = {choice_key(control) for control in CONTROL_OPTIONS}
    for label in _option_labels(options):
        key = choice_key(label)
        if not label or key in seen or key in control_keys:
            continue
        seen.add(key)
        normalized.append(label)
    for control in CONTROL_OPTIONS:
        key = choice_key(control)
        if key not in seen:
            seen.add(key)
            normalized.append(control)
    return tuple(normalized)


def _question_variants(value: Any) -> list[str]:
    if not value:
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _slot_options(options: Any, index: int) -> tuple[str, ...]:
    if not isinstance(options, list):
        return ()
    if options and all(isinstance(item, list) for item in options):
        if 0 <= index < len(options):
            return normalize_options(options[index])
        return ()
    return normalize_options(options)


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
        for slot, stored_questions in questions.items():
            variants = _question_variants(stored_questions)
            matched_index = next(
                (
                    index
                    for index, stored_question in enumerate(variants)
                    if choice_key(str(stored_question)) == question_key
                ),
                None,
            )
            if matched_index is None:
                continue
            options = entry.get("clarification_options", {})
            if isinstance(options, dict):
                custom = _slot_options(options.get(slot), matched_index)
                if custom:
                    return custom
            return ()
    return ()


def _normalize_selection_mode(value: Any, default: str = SINGLE_SELECT) -> str:
    mode = str(value or "").strip().casefold()
    return mode if mode in SELECTION_MODES else default


def _slot_mode(
    modes: Any,
    slot: str,
    index: int,
    options: tuple[str, ...],
    question: str = "",
) -> str:
    inferred = infer_selection_mode(slot, options, question)
    if isinstance(modes, dict):
        slot_modes = modes.get(slot)
        if isinstance(slot_modes, list):
            if 0 <= index < len(slot_modes):
                mode = _normalize_selection_mode(slot_modes[index])
                return MULTI_SELECT if inferred == MULTI_SELECT else mode
        else:
            mode = _normalize_selection_mode(slot_modes, "")
            if mode:
                return MULTI_SELECT if inferred == MULTI_SELECT else mode
    return inferred


def detail_selection_mode_from_catalog(
    question: str,
    catalog: dict[str, dict],
) -> str:
    question_key = choice_key(question)
    for entry in catalog.values():
        questions = entry.get("clarification_questions", {}) or {}
        if not isinstance(questions, dict):
            continue
        for slot, stored_questions in questions.items():
            variants = _question_variants(stored_questions)
            matched_index = next(
                (
                    index
                    for index, stored_question in enumerate(variants)
                    if choice_key(str(stored_question)) == question_key
                ),
                None,
            )
            if matched_index is None:
                continue
            options = entry.get("clarification_options", {})
            slot_options = ()
            if isinstance(options, dict):
                slot_options = _slot_options(options.get(slot), matched_index)
            modes = entry.get("clarification_selection_modes", {})
            return _slot_mode(
                modes,
                str(slot),
                matched_index,
                slot_options,
                str(variants[matched_index]),
            )
    return ""


def infer_selection_mode(
    slot: str,
    options: tuple[str, ...] = (),
    question: str = "",
) -> str:
    if _has_combined_options(options):
        return MULTI_SELECT
    if slot != "associated":
        return MULTI_SELECT if _has_independent_option_dimensions(options, question) else SINGLE_SELECT
    positive = [
        option
        for option in options
        if choice_key(option).startswith("co ")
        and choice_key(option) not in {"co", "co khong"}
    ]
    return MULTI_SELECT if len(positive) > 1 else SINGLE_SELECT


def fallback_selection_mode(question: str) -> str:
    options = fallback_detail_options(question)
    key = choice_key(question)
    if (
        key.startswith("co kem")
        or key.startswith("co lien quan")
        or key.startswith("co kem theo")
    ):
        return infer_selection_mode("associated", options, question)
    if _has_independent_option_dimensions(options, question):
        return MULTI_SELECT
    return SINGLE_SELECT


def _positive_detail_options(options: tuple[str, ...]) -> list[str]:
    positive: list[str] = []
    control_keys = {choice_key(control) for control in CONTROL_OPTIONS}
    for option in options:
        key = choice_key(option)
        if (
            not key
            or key in control_keys
            or key in _NEGATIVE_OPTION_KEYS
            or key in _GENERIC_OPTION_KEYS
            or key.startswith("ca ")
        ):
            continue
        positive.append(option)
    return positive


def _option_dimension(option: str) -> str:
    key = choice_key(option)
    if key in _COURSE_PATTERN_KEYS:
        return "course"
    if key.startswith("co lan") or " lan ra " in f" {key} ":
        return "radiation"
    if "tu the" in key:
        return "posture"
    if any(marker in key for marker in _LOCATION_MARKERS):
        return "location"
    if key.startswith("co "):
        return "associated"
    return ""


def _has_independent_option_dimensions(
    options: tuple[str, ...],
    question: str = "",
) -> bool:
    positive = _positive_detail_options(options)
    if len(positive) < 2:
        return False
    dimensions = {
        dimension
        for option in positive
        if (dimension := _option_dimension(option))
    }
    key = choice_key(question)
    has_separate_prompts = question.count("?") > 1
    if len(dimensions) > 1:
        return has_separate_prompts
    return (
        has_separate_prompts
        and any(term in key for term in ("vi tri", "o dau", "co lan", "tu the"))
        and bool(dimensions)
    )


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
    for slot, question_value in questions.items():
        variants = _question_variants(question_value)
        if not variants:
            continue
        for index, _question in enumerate(variants):
            slot_options = _slot_options(options.get(slot), index)
            if len(slot_options) < 2:
                errors.append(f"{slot}[{index}] options are missing or too short")
            for control in CONTROL_OPTIONS:
                if choice_key(control) not in {choice_key(option) for option in slot_options}:
                    errors.append(f"{slot}[{index}] options must include {control}")
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
