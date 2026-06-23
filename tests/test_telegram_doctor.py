from __future__ import annotations

import asyncio
import base64
import time

import pytest

from src.chat.storage import doctors
from src.chat.storage.session import PatientSession
from src.server.channels import telegram_doctor


@pytest.fixture
def profile_identity(monkeypatch):
    from src import config

    encoded = base64.b64encode(b"doctor-profile-test-key-32-bytes!").decode()
    monkeypatch.setattr(config, "PROFILE_IDENTITY_ACTIVE_VERSION", "v1")
    monkeypatch.setattr(config, "PROFILE_IDENTITY_HMAC_KEY", encoded)
    monkeypatch.delenv("PROFILE_IDENTITY_KEY_V1", raising=False)

    def owner_key(user_id: int | str) -> str:
        from src.chat.security.identity import derive_owner_key
        return derive_owner_key("telegram", user_id)

    return owner_key


def test_doctor_command_shows_tier_menu(monkeypatch):
    sent: list[dict] = []

    async def fake_send_text(chat_id, text, *args, inline_keyboard=None, **kwargs):
        sent.append({"chat_id": chat_id, "text": text, "inline_keyboard": inline_keyboard})

    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fake_send_text)

    assert asyncio.run(telegram_doctor.handle_doctor_command(123)) is True

    assert sent[0]["chat_id"] == 123
    assert "Kết nối bác sĩ" in sent[0]["text"]
    # The tier menu explains each tier (price lives here now, not on buttons).
    text = sent[0]["text"]
    assert "kết nối trực tiếp với bác sĩ theo chuyên khoa phù hợp" in text
    assert "Có 2 hình thức tư vấn" in text
    assert "phiên tư vấn ngắn, phù hợp để trao đổi ban đầu" in text.casefold()
    assert "thời lượng dài hơn và có thể gia hạn khi cần" in text.casefold()
    assert "Miễn phí" in text and "Trả phí" in text
    assert "5 phút" in text          # free session length
    assert "2.000" in text or "2,000" in text  # paid per-minute rate
    assert "15 phút" in text         # paid block length
    callbacks = [btn["callback_data"] for row in sent[0]["inline_keyboard"]["inline_keyboard"] for btn in row]
    assert callbacks == ["doctor:tier:free", "doctor:tier:paid", "doctor:cancel"]


def test_handoff_keyboard_shows_suggested_specialty_as_option():
    telegram_doctor._HANDOFF_CONTEXTS["123"] = telegram_doctor.DoctorHandoffContext(
        question="Tôi ho khó thở",
        bot_answer="Không đủ thông tin.",
        summary="Tóm tắt từ bot",
        specialty_hint="Hô hấp",
        created_at=time.time(),
    )

    keyboard = telegram_doctor.handoff_keyboard(123)

    labels = [btn["text"] for row in keyboard["inline_keyboard"] for btn in row]
    callbacks = [btn["callback_data"] for row in keyboard["inline_keyboard"] for btn in row]
    assert any("Hô hấp" in label for label in labels)
    assert any("Chuyên khoa khác" in label for label in labels)
    assert callbacks == [
        telegram_doctor.HANDOFF_ACCEPT,
        telegram_doctor.HANDOFF_OTHER,
        telegram_doctor.HANDOFF_DECLINE,
    ]


def test_handoff_context_prefers_model_specialty(monkeypatch):
    monkeypatch.setattr(
        telegram_doctor,
        "load_session",
        lambda session_id: PatientSession(session_id=session_id),
    )

    telegram_doctor.register_handoff_context(
        127,
        "Tôi đang bị ho",
        "Bạn nên đi khám.",
        "Tiêu hóa",
    )

    assert telegram_doctor._HANDOFF_CONTEXTS["127"].specialty_hint == "Tiêu hóa"


def test_handoff_accept_shows_tier_menu_with_cancel(monkeypatch):
    edits: list[dict] = []
    answers: list[tuple[str, str]] = []
    telegram_doctor._HANDOFF_CONTEXTS["123"] = telegram_doctor.DoctorHandoffContext(
        question="Tôi ho khó thở",
        bot_answer="Không đủ thông tin.",
        summary="Tóm tắt từ bot",
        specialty_hint="Hô hấp",
        created_at=time.time(),
    )

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None):
        edits.append({"text": text, "inline_keyboard": inline_keyboard})

    async def fake_answer_callback_query(callback_query_id, text):
        answers.append((callback_query_id, text))

    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)

    handled = asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "cbhandoff",
        "data": telegram_doctor.HANDOFF_ACCEPT,
        "message": {"message_id": 9, "chat": {"id": 123}},
    }))

    assert handled is True
    assert "Gợi ý chuyên khoa" not in edits[0]["text"]
    callbacks = [btn["callback_data"] for row in edits[0]["inline_keyboard"]["inline_keyboard"] for btn in row]
    assert callbacks == [
        "doctor:tier:free:ho_hap",
        "doctor:tier:paid:ho_hap",
        "doctor:cancel",
    ]


def test_handoff_other_leads_to_other_specialty_folders(monkeypatch):
    doctors.create_doctor("BS Phổi", "Hô hấp", "free", 0, 1511)
    doctors.create_doctor("BS Tiêu hóa", "Tiêu hóa", "free", 0, 1512)
    telegram_doctor._HANDOFF_CONTEXTS["125"] = telegram_doctor.DoctorHandoffContext(
        question="Tôi ho khó thở",
        bot_answer="Không đủ thông tin.",
        summary="Tóm tắt từ bot",
        specialty_hint="Hô hấp",
        created_at=time.time(),
    )
    edits: list[dict] = []

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None):
        edits.append({"text": text, "inline_keyboard": inline_keyboard})

    async def fake_answer_callback_query(callback_query_id, text):
        pass

    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)

    asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "cbother",
        "data": telegram_doctor.HANDOFF_OTHER,
        "message": {"message_id": 9, "chat": {"id": 125}},
    }))
    tier_callbacks = [
        btn["callback_data"]
        for row in edits[-1]["inline_keyboard"]["inline_keyboard"]
        for btn in row
    ]
    assert tier_callbacks == ["doctor:tier:free", "doctor:tier:paid", "doctor:cancel"]

    asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "cbothertier",
        "data": "doctor:tier:free",
        "message": {"message_id": 9, "chat": {"id": 125}},
    }))
    labels = [
        btn["text"]
        for row in edits[-1]["inline_keyboard"]["inline_keyboard"]
        for btn in row
    ]
    assert any("Tiêu hóa" in label for label in labels)
    assert not any("Hô hấp" in label for label in labels)


def test_doctor_cancel_clears_handoff_context(monkeypatch):
    telegram_doctor._HANDOFF_CONTEXTS["126"] = telegram_doctor.DoctorHandoffContext(
        question="Tôi đau bụng",
        bot_answer="Bạn nên đi khám.",
        summary="Tóm tắt từ bot",
        specialty_hint="Tiêu hóa",
        created_at=time.time(),
    )
    edits: list[str] = []

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None):
        edits.append(text)

    async def fake_answer_callback_query(callback_query_id, text):
        pass

    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)

    handled = asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "cbcancel",
        "data": "doctor:cancel",
        "message": {"message_id": 9, "chat": {"id": 126}},
    }))

    assert handled is True
    assert "126" not in telegram_doctor._HANDOFF_CONTEXTS
    assert "hủy" in edits[0].lower()


def test_suggested_specialty_tier_shows_only_matching_doctors(monkeypatch):
    skin_id = doctors.create_doctor("BS Da", "Da liễu", "free", 0, 1501)
    lung_id = doctors.create_doctor("BS Phổi", "Hô hấp", "free", 0, 1502)
    telegram_doctor._HANDOFF_CONTEXTS["124"] = telegram_doctor.DoctorHandoffContext(
        question="Tôi ho khó thở",
        bot_answer="Không đủ thông tin.",
        summary="Tóm tắt từ bot",
        specialty_hint="Hô hấp",
        created_at=time.time(),
    )
    edits: list[dict] = []

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None):
        edits.append({"text": text, "inline_keyboard": inline_keyboard})

    async def fake_answer_callback_query(callback_query_id, text):
        pass

    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)

    asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "cbsuggested",
        "data": "doctor:tier:free:ho_hap",
        "message": {"message_id": 10, "chat": {"id": 124}},
    }))

    callbacks = [
        btn["callback_data"]
        for row in edits[0]["inline_keyboard"]["inline_keyboard"]
        for btn in row
    ]
    assert f"doctor:profile:{lung_id}:free:ho_hap" in callbacks
    assert f"doctor:profile:{skin_id}:free:da_lieu" not in callbacks


def test_tier_callback_shows_specialty_folders(monkeypatch):
    telegram_doctor._HANDOFF_CONTEXTS.pop("123", None)
    doctor_id = doctors.create_doctor("BS An", "Nội tổng quát", "free", 0, 501)
    lung_id = doctors.create_doctor("BS Phổi", "Hô hấp", "free", 0, 502)
    sent_edits: list[dict] = []
    answers: list[tuple[str, str]] = []

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None):
        sent_edits.append({"chat_id": chat_id, "message_id": message_id, "text": text, "inline_keyboard": inline_keyboard})

    async def fake_answer_callback_query(callback_query_id, text):
        answers.append((callback_query_id, text))

    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)

    handled = asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "cb1",
        "data": "doctor:tier:free",
        "message": {"message_id": 10, "chat": {"id": 123}},
    }))

    assert handled is True
    labels = [btn["text"] for row in sent_edits[0]["inline_keyboard"]["inline_keyboard"] for btn in row]
    assert any("Nội tổng quát" in label for label in labels)
    assert any("Hô hấp" in label for label in labels)
    assert not any("BS An" in label or "BS Phổi" in label for label in labels)
    callbacks = [btn["callback_data"] for row in sent_edits[0]["inline_keyboard"]["inline_keyboard"] for btn in row]
    assert "doctor:specialty:free:noi_tong_quat" in callbacks
    assert "doctor:specialty:free:ho_hap" in callbacks
    assert f"doctor:pick:{doctor_id}" not in callbacks
    assert f"doctor:pick:{lung_id}" not in callbacks


def test_specialty_callback_shows_available_doctors(monkeypatch):
    doctor_id = doctors.create_doctor("BS An", "Nội tổng quát", "free", 0, 503)
    doctors.create_doctor("BS Phổi", "Hô hấp", "free", 0, 504)
    sent_edits: list[dict] = []

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None):
        sent_edits.append({"text": text, "inline_keyboard": inline_keyboard})

    async def fake_answer_callback_query(callback_query_id, text):
        pass

    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)

    handled = asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "cbspecialty",
        "data": "doctor:specialty:free:noi_tong_quat",
        "message": {"message_id": 11, "chat": {"id": 123}},
    }))

    assert handled is True
    labels = [btn["text"] for row in sent_edits[0]["inline_keyboard"]["inline_keyboard"] for btn in row]
    assert any(label.startswith("🟢") and "BS An" in label for label in labels)
    assert not any("BS Phổi" in label for label in labels)
    callbacks = [btn["callback_data"] for row in sent_edits[0]["inline_keyboard"]["inline_keyboard"] for btn in row]
    assert f"doctor:profile:{doctor_id}:free:noi_tong_quat" in callbacks
    assert "doctor:refresh:free:noi_tong_quat" in callbacks


def test_doctor_profile_callback_shows_profile_with_connect_and_back(monkeypatch):
    doctor_id = doctors.create_doctor(
        "BS Profile",
        "Nội tổng quát",
        "free",
        0,
        505,
        degree="Thạc sĩ, Bác sĩ",
        experience_years=9,
        hospital="Bệnh viện Bạch Mai",
        bio="Tư vấn các vấn đề nội khoa thường gặp.",
    )
    sent_edits: list[dict] = []

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None):
        sent_edits.append({"text": text, "inline_keyboard": inline_keyboard})

    async def fake_answer_callback_query(callback_query_id, text):
        pass

    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)

    handled = asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "cbprofile",
        "data": f"doctor:profile:{doctor_id}:free:noi_tong_quat",
        "message": {"message_id": 13, "chat": {"id": 123}},
    }))

    assert handled is True
    text = sent_edits[0]["text"]
    assert "BS Profile" in text
    assert "Chuyên khoa: Nội tổng quát" in text
    assert "Học vị: Thạc sĩ, Bác sĩ" in text
    assert "Kinh nghiệm: 9 năm" in text
    assert "Công tác: Bệnh viện Bạch Mai" in text
    assert "Tư vấn các vấn đề nội khoa thường gặp." in text
    callbacks = [btn["callback_data"] for row in sent_edits[0]["inline_keyboard"]["inline_keyboard"] for btn in row]
    assert f"doctor:pick:{doctor_id}" in callbacks
    assert "doctor:specialty:free:noi_tong_quat" in callbacks
    assert "doctor:cancel" in callbacks


def test_doctor_button_no_price_on_paid(monkeypatch):
    doctor_id = doctors.create_doctor("BS Paid Lbl", "Tim mạch", "paid", 2000, 521)
    sent_edits: list[dict] = []

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None):
        sent_edits.append({"text": text, "inline_keyboard": inline_keyboard})

    async def fake_answer_callback_query(callback_query_id, text):
        pass

    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)

    asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "cbpaidlbl",
        "data": "doctor:specialty:paid:tim_mach",
        "message": {"message_id": 14, "chat": {"id": 125}},
    }))

    labels = [btn["text"] for row in sent_edits[0]["inline_keyboard"]["inline_keyboard"] for btn in row]
    # Available paid doctor still gets the green (available) marker, no price.
    assert any(label.startswith("🟢") and "BS Paid Lbl" in label for label in labels)
    assert not any("đ/phút" in label or "VND" in label for label in labels)


def test_busy_doctor_button_orange_with_waitlist_count(monkeypatch):
    # Busy doctor with two people waiting.
    doctor_id = doctors.create_doctor("BS Queue", "Da liễu", "free", 0, 531)
    occupy = doctors.create_consultation(99531, doctors.get_doctor(doctor_id), "free")
    doctors.accept_consultation(occupy)
    doctors.join_waitlist(doctor_id, 95001, "free")
    doctors.join_waitlist(doctor_id, 95002, "free")

    sent_edits: list[dict] = []

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None):
        sent_edits.append({"inline_keyboard": inline_keyboard})

    async def fake_answer_callback_query(callback_query_id, text):
        pass

    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)

    asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "cbq",
        "data": "doctor:specialty:free:da_lieu",
        "message": {"message_id": 15, "chat": {"id": 126}},
    }))

    labels = [btn["text"] for row in sent_edits[0]["inline_keyboard"]["inline_keyboard"] for btn in row]
    queue_label = next(label for label in labels if "BS Queue" in label)
    # Busy doctors get the orange marker and show how many are waiting.
    assert queue_label.startswith("🟠")
    assert "đang bận" in queue_label
    assert "2" in queue_label  # number of people waiting shown




def test_refresh_callback_rerenders_doctor_list(monkeypatch):
    doctor_id = doctors.create_doctor("BS Refresh", "Tim mạch", "free", 0, 511)
    sent_edits: list[dict] = []

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None):
        sent_edits.append({"text": text, "inline_keyboard": inline_keyboard})

    async def fake_answer_callback_query(callback_query_id, text):
        pass

    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)

    handled = asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "cbref",
        "data": "doctor:refresh:free:tim_mach",
        "message": {"message_id": 12, "chat": {"id": 124}},
    }))

    assert handled is True
    labels = [btn["text"] for row in sent_edits[0]["inline_keyboard"]["inline_keyboard"] for btn in row]
    assert any("BS Refresh" in label for label in labels)
    callbacks = [btn["callback_data"] for row in sent_edits[0]["inline_keyboard"]["inline_keyboard"] for btn in row]
    assert f"doctor:profile:{doctor_id}:free:tim_mach" in callbacks
    assert "doctor:refresh:free:tim_mach" in callbacks


def test_pick_with_user_identity_requires_profile_confirmation(
    monkeypatch, profile_identity,
):
    from src.chat.profile import ensure_subject

    patient_id = 6901
    owner_key = profile_identity(patient_id)
    ensure_subject(owner_key, "self", relationship="self", display_name="Lan")
    doctor_id = doctors.create_doctor("BS Consent", "Nội", "free", 0, 16901)
    edits: list[dict] = []
    sent: list[dict] = []

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None):
        edits.append({"text": text, "inline_keyboard": inline_keyboard})

    async def fake_send_text(chat_id, text, *args, **kwargs):
        sent.append({"chat_id": chat_id, "text": text})

    async def fake_answer_callback_query(callback_query_id, text):
        return None

    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)

    handled = asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "profile-consent",
        "data": f"doctor:pick:{doctor_id}",
        "from": {"id": patient_id},
        "message": {"message_id": 41, "chat": {"id": patient_id}},
    }))

    assert handled is True
    assert doctors.open_consultation_for_patient(patient_id) is None
    assert sent == []
    assert "Xác nhận chia sẻ hồ sơ" in edits[-1]["text"]
    assert "Lan" in edits[-1]["text"]
    callbacks = [
        button["callback_data"]
        for row in edits[-1]["inline_keyboard"]["inline_keyboard"]
        for button in row
    ]
    assert f"doctor:share:confirm:{doctor_id}:self" in callbacks
    assert f"doctor:share:choose:{doctor_id}" in callbacks


def test_doctor_share_defaults_to_active_conversation_subject(
    monkeypatch, profile_identity,
):
    from src.chat.context.domain import SessionState
    from src.chat.profile import ensure_subject
    from src.chat.security.identity import derive_request_identity

    patient_id = 6910
    owner_key = profile_identity(patient_id)
    ensure_subject(owner_key, "self", relationship="self")
    ensure_subject(
        owner_key,
        "person_mother",
        relationship="mother",
        display_name="Mai",
    )
    identity = derive_request_identity("telegram", patient_id, str(patient_id))
    state = SessionState(
        session_id=identity.session_key,
        owner_id=owner_key,
        active_subject_id="mother",
    )
    monkeypatch.setattr(
        "src.chat.context.context_store.load_conversation_context",
        lambda session_id, owner_id: (state, {}, True),
    )

    assert telegram_doctor._default_profile_subject_id(
        patient_id,
        patient_id,
        owner_key,
    ) == "person_mother"


def test_patient_can_select_relative_and_share_only_that_profile(
    monkeypatch, profile_identity,
):
    from src.chat.profile import ensure_subject, write_profile_fact

    patient_id = 6902
    owner_key = profile_identity(patient_id)
    ensure_subject(owner_key, "self", relationship="self", display_name="Huy")
    ensure_subject(
        owner_key,
        "person_mother",
        relationship="mother",
        display_name="Mai",
        birth_date="1958-03-10",
        gender="female",
    )
    write_profile_fact(
        owner_id=owner_key,
        subject_id="person_mother",
        fact_type="allergy",
        section="allergies",
        value={"name": "Penicillin"},
        entity_id="Penicillin",
    )
    doctor_id = doctors.create_doctor("BS Relative", "Nội", "free", 0, 16902)
    edits: list[dict] = []
    sent: list[dict] = []

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None):
        edits.append({"text": text, "inline_keyboard": inline_keyboard})

    async def fake_send_text(chat_id, text, *args, inline_keyboard=None, **kwargs):
        sent.append({"chat_id": chat_id, "text": text, "inline_keyboard": inline_keyboard})

    async def fake_answer_callback_query(callback_query_id, text):
        return None

    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)

    asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "choose-profile",
        "data": f"doctor:share:choose:{doctor_id}",
        "from": {"id": patient_id},
        "message": {"message_id": 42, "chat": {"id": patient_id}},
    }))
    subject_callbacks = [
        button["callback_data"]
        for row in edits[-1]["inline_keyboard"]["inline_keyboard"]
        for button in row
    ]
    assert f"doctor:share:subject:{doctor_id}:person_mother" in subject_callbacks

    asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "preview-relative",
        "data": f"doctor:share:subject:{doctor_id}:person_mother",
        "from": {"id": patient_id},
        "message": {"message_id": 42, "chat": {"id": patient_id}},
    }))
    assert "Mai (Mẹ)" in edits[-1]["text"]
    assert "Penicillin" in edits[-1]["text"]
    assert doctors.open_consultation_for_patient(patient_id) is None

    asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "confirm-relative",
        "data": f"doctor:share:confirm:{doctor_id}:person_mother",
        "from": {"id": patient_id},
        "message": {"message_id": 42, "chat": {"id": patient_id}},
    }))

    pending = doctors.open_consultation_for_patient(patient_id)
    assert pending is not None and pending["doctor_id"] == doctor_id
    doctor_message = next(message for message in sent if message["chat_id"] == 16902)
    assert "đã xem và đồng ý chia sẻ" in doctor_message["text"]
    assert "Mai (Mẹ)" in doctor_message["text"]
    assert "Penicillin" in doctor_message["text"]
    assert "Huy (Bạn)" not in doctor_message["text"]
    assert owner_key not in doctor_message["text"]
    assert "person_mother" not in doctor_message["text"]


def test_profile_share_rejects_subject_owned_by_another_user(
    monkeypatch, profile_identity,
):
    from src.chat.profile import ensure_subject

    patient_id = 6903
    other_id = 6904
    owner_key = profile_identity(patient_id)
    other_owner_key = profile_identity(other_id)
    ensure_subject(owner_key, "self", relationship="self")
    ensure_subject(
        other_owner_key,
        "private_relative",
        relationship="relative",
        display_name="Không được chia sẻ",
    )
    doctor_id = doctors.create_doctor("BS Isolation", "Nội", "free", 0, 16903)
    edits: list[str] = []
    sent: list[str] = []

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None):
        edits.append(text)

    async def fake_send_text(chat_id, text, *args, **kwargs):
        sent.append(text)

    async def fake_answer_callback_query(callback_query_id, text):
        return None

    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)

    asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "forged-profile",
        "data": f"doctor:share:confirm:{doctor_id}:private_relative",
        "from": {"id": patient_id},
        "message": {"message_id": 43, "chat": {"id": patient_id}},
    }))

    assert doctors.open_consultation_for_patient(patient_id) is None
    assert sent == []
    assert "không còn tồn tại" in edits[-1].lower()



def test_pick_doctor_creates_pending_and_messages_doctor(monkeypatch):
    doctor_id = doctors.create_doctor("BS Pick", "Da liễu", "free", 0, 601)
    sent: list[dict] = []
    edits: list[dict] = []

    async def fake_send_text(chat_id, text, *args, inline_keyboard=None, **kwargs):
        sent.append({"chat_id": chat_id, "text": text, "inline_keyboard": inline_keyboard})

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None):
        edits.append({"chat_id": chat_id, "message_id": message_id, "text": text})

    async def fake_answer_callback_query(callback_query_id, text):
        pass

    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)

    handled = asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "cb2",
        "data": f"doctor:pick:{doctor_id}",
        "message": {"message_id": 20, "chat": {"id": 7001}},
    }))

    assert handled is True
    pending = doctors.open_consultation_for_patient(7001)
    assert pending["doctor_id"] == doctor_id
    assert pending["status"] == "pending"
    assert sent[0]["chat_id"] == 601
    callbacks = [btn["callback_data"] for row in sent[0]["inline_keyboard"]["inline_keyboard"] for btn in row]
    assert f"doctor:accept:{pending['id']}" in callbacks
    assert f"doctor:decline:{pending['id']}" in callbacks
    assert "chờ" in edits[0]["text"].lower()


def test_pick_doctor_refuses_when_patient_already_open(monkeypatch):
    doctor_id = doctors.create_doctor("BS One", "Nội", "free", 0, 602)
    doctors.create_consultation(7002, doctors.get_doctor(doctor_id), "free")
    sent: list[str] = []

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None):
        sent.append(text)

    async def fake_answer_callback_query(callback_query_id, text):
        pass

    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)

    handled = asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "cb3",
        "data": f"doctor:pick:{doctor_id}",
        "message": {"message_id": 21, "chat": {"id": 7002}},
    }))

    assert handled is True
    assert "đang có phiên" in sent[0].lower()


def test_pick_free_doctor_blocked_during_cooldown(monkeypatch):
    import time as _t
    from src.chat.clients import get_sqlite

    # Patient just finished a free session → cooldown active.
    prev_doctor = doctors.create_doctor("BS Prev", "Nội", "free", 0, 651)
    prev = doctors.create_consultation(7050, doctors.get_doctor(prev_doctor), "free")
    doctors.accept_consultation(prev)
    doctors.end_consultation(prev)

    target = doctors.create_doctor("BS Target", "Nhi", "free", 0, 652)
    edits: list[str] = []

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None):
        edits.append(text)

    async def fake_answer_callback_query(callback_query_id, text):
        pass

    async def fail_send_text(chat_id, text, *args, **kwargs):
        raise AssertionError("doctor should not be contacted during cooldown")

    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)
    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fail_send_text)

    handled = asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "cbcool",
        "data": f"doctor:pick:{target}",
        "message": {"message_id": 30, "chat": {"id": 7050}},
    }))

    assert handled is True
    assert doctors.open_consultation_for_patient(7050) is None
    assert "miễn phí" in edits[0].lower()



def test_doctor_accept_opens_relay_and_notifies_both(monkeypatch):
    doctor_id = doctors.create_doctor("BS Accept", "Nội", "free", 0, 701)
    consult_id = doctors.create_consultation(8001, doctors.get_doctor(doctor_id), "free")
    sent: list[tuple[int | str, str]] = []
    edits: list[str] = []

    async def fake_send_text(chat_id, text, *args, **kwargs):
        sent.append((chat_id, text))

    async def fake_edit_message_reply_markup(chat_id, message_id, reply_markup):
        edits.append("edited")

    async def fake_answer_callback_query(callback_query_id, text):
        pass

    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_reply_markup", fake_edit_message_reply_markup)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)

    handled = asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "cb4",
        "data": f"doctor:accept:{consult_id}",
        "message": {"message_id": 33, "chat": {"id": 701}},
    }))

    assert handled is True
    assert doctors.get_consultation(consult_id)["status"] == "active"
    patient_messages = [text for cid, text in sent if cid == 8001]
    assert any("đã kết nối" in text.lower() for text in patient_messages)
    assert any("tuổi" in text.lower() and "giới tính" in text.lower() and "triệu chứng" in text.lower() for text in patient_messages)
    assert any(cid == 701 and "đã kết nối" in text.lower() for cid, text in sent)


def test_pick_after_handoff_sends_summary_to_doctor(monkeypatch):
    doctor_id = doctors.create_doctor("BS Summary", "Hô hấp", "free", 0, 1701)
    sent: list[dict] = []
    edits: list[str] = []
    telegram_doctor._HANDOFF_CONTEXTS["8800"] = telegram_doctor.DoctorHandoffContext(
        question="Tôi ho khó thở",
        bot_answer="Không đủ thông tin.",
        summary="Tóm tắt từ bot:\n- Câu hỏi gần nhất: Tôi ho khó thở",
        specialty_hint="Hô hấp",
        created_at=time.time(),
    )

    async def fake_send_text(chat_id, text, *args, inline_keyboard=None, **kwargs):
        sent.append({"chat_id": chat_id, "text": text, "inline_keyboard": inline_keyboard})

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None):
        edits.append(text)

    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_text", fake_edit_message_text)

    asyncio.run(telegram_doctor._handle_pick(8800, 77, doctor_id))

    doctor_messages = [m for m in sent if m["chat_id"] == 1701]
    assert doctor_messages
    assert "Tóm tắt từ bot" in doctor_messages[0]["text"]
    assert "Gợi ý chuyên khoa: Hô hấp" in doctor_messages[0]["text"]
    assert edits and "Đã gửi yêu cầu" in edits[0]



def test_doctor_decline_notifies_patient(monkeypatch):
    doctor_id = doctors.create_doctor("BS Decliner", "Nội", "free", 0, 702)
    consult_id = doctors.create_consultation(8002, doctors.get_doctor(doctor_id), "free")
    sent: list[tuple[int | str, str]] = []

    async def fake_send_text(chat_id, text, *args, **kwargs):
        sent.append((chat_id, text))

    async def fake_edit_message_reply_markup(chat_id, message_id, reply_markup):
        pass

    async def fake_answer_callback_query(callback_query_id, text):
        pass

    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_reply_markup", fake_edit_message_reply_markup)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)

    handled = asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "cb5",
        "data": f"doctor:decline:{consult_id}",
        "message": {"message_id": 34, "chat": {"id": 702}},
    }))

    assert handled is True
    assert doctors.get_consultation(consult_id)["status"] == "declined"
    assert any(cid == 8002 and "từ chối" in text.lower() for cid, text in sent)


def test_relay_text_patient_to_doctor(monkeypatch):
    doctor_id = doctors.create_doctor("BS Text", "Nội", "free", 0, 801)
    consult_id = doctors.create_consultation(8101, doctors.get_doctor(doctor_id), "free")
    doctors.accept_consultation(consult_id)
    sent: list[tuple[int | str, str]] = []

    async def fake_send_text(chat_id, text, *args, **kwargs):
        sent.append((chat_id, text))

    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fake_send_text)

    handled = asyncio.run(telegram_doctor.relay_message(8101, {"text": "Tôi bị đau đầu"}))

    assert handled is True
    assert sent == [(801, "👤 Bệnh nhân: Tôi bị đau đầu")]


def test_relay_text_doctor_to_patient(monkeypatch):
    doctor_id = doctors.create_doctor("BS Text2", "Nội", "free", 0, 802)
    consult_id = doctors.create_consultation(8102, doctors.get_doctor(doctor_id), "free")
    doctors.accept_consultation(consult_id)
    sent: list[tuple[int | str, str]] = []

    async def fake_send_text(chat_id, text, *args, **kwargs):
        sent.append((chat_id, text))

    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fake_send_text)

    handled = asyncio.run(telegram_doctor.relay_message(802, {"text": "Bạn nghỉ ngơi nhé"}))

    assert handled is True
    assert sent == [(8102, "👨‍⚕️ BS Text2: Bạn nghỉ ngơi nhé")]


def test_relay_returns_false_when_no_active_consultation():
    assert asyncio.run(telegram_doctor.relay_message(9999, {"text": "hello"})) is False


def test_relay_photo_uses_largest_photo_file_id(monkeypatch):
    doctor_id = doctors.create_doctor("BS Photo", "Da", "free", 0, 901)
    consult_id = doctors.create_consultation(8201, doctors.get_doctor(doctor_id), "free")
    doctors.accept_consultation(consult_id)
    photos: list[tuple[int | str, str, str]] = []

    async def fake_send_photo(chat_id, file_id, caption=""):
        photos.append((chat_id, file_id, caption))

    monkeypatch.setattr(telegram_doctor.telegram, "send_photo", fake_send_photo)

    handled = asyncio.run(telegram_doctor.relay_message(8201, {
        "photo": [{"file_id": "small"}, {"file_id": "large"}],
        "caption": "ảnh triệu chứng",
    }))

    assert handled is True
    assert photos == [(901, "large", "👤 Bệnh nhân: ảnh triệu chứng")]


def test_relay_voice_sends_label_then_voice(monkeypatch):
    doctor_id = doctors.create_doctor("BS Voice", "Tai mũi họng", "free", 0, 902)
    consult_id = doctors.create_consultation(8202, doctors.get_doctor(doctor_id), "free")
    doctors.accept_consultation(consult_id)
    texts: list[tuple[int | str, str]] = []
    voices: list[tuple[int | str, str]] = []

    async def fake_send_text(chat_id, text, *args, **kwargs):
        texts.append((chat_id, text))

    async def fake_send_voice(chat_id, file_id):
        voices.append((chat_id, file_id))

    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram_doctor.telegram, "send_voice", fake_send_voice)

    handled = asyncio.run(telegram_doctor.relay_message(8202, {"voice": {"file_id": "voice-1"}}))

    assert handled is True
    assert texts == [(902, "👤 Bệnh nhân gửi tin nhắn thoại:")]
    assert voices == [(902, "voice-1")]


@pytest.mark.parametrize(
    "payload",
    [
        {"message_id": 901, "video": {"file_id": "video-1"}},
        {"message_id": 902, "sticker": {"file_id": "sticker-1"}},
        {"message_id": 903, "location": {"latitude": 21.0, "longitude": 105.8}},
        {"message_id": 904, "document": {"file_id": "doc-1"}},
    ],
)
def test_relay_other_message_types_use_copy_message(monkeypatch, payload):
    doctor_id = doctors.create_doctor("BS Generic", "Nội", "free", 0, 9030)
    consult_id = doctors.create_consultation(8204, doctors.get_doctor(doctor_id), "free")
    doctors.accept_consultation(consult_id)
    texts: list[tuple[int | str, str]] = []
    copies: list[tuple[int | str, int | str, int]] = []

    async def fake_send_text(chat_id, text, *args, **kwargs):
        texts.append((chat_id, text))

    async def fake_copy_message(chat_id, from_chat_id, message_id):
        copies.append((chat_id, from_chat_id, message_id))

    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram_doctor.telegram, "copy_message", fake_copy_message)

    handled = asyncio.run(telegram_doctor.relay_message(8204, payload))

    assert handled is True
    assert texts == [(9030, "👤 Bệnh nhân gửi tin nhắn:")]
    assert copies == [(9030, 8204, payload["message_id"])]


def test_relay_unsupported_type_notifies_sender(monkeypatch):
    doctor_id = doctors.create_doctor("BS Unsupported", "Nội", "free", 0, 903)
    consult_id = doctors.create_consultation(8203, doctors.get_doctor(doctor_id), "free")
    doctors.accept_consultation(consult_id)
    texts: list[tuple[int | str, str]] = []
    copies: list[tuple[int | str, int | str, int]] = []

    async def fake_send_text(chat_id, text, *args, **kwargs):
        texts.append((chat_id, text))

    async def fail_copy_message(chat_id, from_chat_id, message_id):
        copies.append((chat_id, from_chat_id, message_id))
        raise RuntimeError("copy failed")

    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram_doctor.telegram, "copy_message", fail_copy_message)

    handled = asyncio.run(
        telegram_doctor.relay_message(8203, {"message_id": 905, "new_chat_members": [{"id": 1}]})
    )

    assert handled is True
    assert copies == [(903, 8203, 905)]
    assert texts == [
        (903, "👤 Bệnh nhân gửi tin nhắn:"),
        (8203, "Loại tin nhắn này chưa được hỗ trợ trong phiên tư vấn."),
    ]


def test_handle_end_from_patient_closes_and_notifies_both(monkeypatch):
    doctor_id = doctors.create_doctor("BS End", "Nội", "free", 0, 1001)
    consult_id = doctors.create_consultation(8301, doctors.get_doctor(doctor_id), "free")
    doctors.accept_consultation(consult_id)
    sent: list[tuple[int | str, str]] = []

    async def fake_send_text(chat_id, text, *args, **kwargs):
        sent.append((chat_id, text))

    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fake_send_text)

    handled = asyncio.run(telegram_doctor.handle_end(8301))

    assert handled is True
    assert doctors.get_consultation(consult_id)["status"] == "ended"
    assert any(cid == 8301 and "kết thúc" in text.lower() for cid, text in sent)
    assert any(cid == 1001 and "kết thúc" in text.lower() for cid, text in sent)


def test_handle_end_returns_false_when_no_consultation():
    assert asyncio.run(telegram_doctor.handle_end(99991)) is False


def _accept_with_expiry(patient_chat_id, doctor_tg, expires_in, tier="free"):
    import time as _t
    from src.chat.clients import get_sqlite

    doctor_id = doctors.create_doctor("BS Tick", "Nội", tier, 0, doctor_tg)
    consult_id = doctors.create_consultation(patient_chat_id, doctors.get_doctor(doctor_id), tier)
    doctors.accept_consultation(consult_id)
    conn = get_sqlite()
    conn.execute(
        "UPDATE doctor_consultation SET expires_at = ? WHERE id = ?",
        (_t.time() + expires_in, consult_id),
    )
    conn.commit()
    return consult_id


def test_run_session_tick_warns_both_sides_near_expiry(monkeypatch):
    consult_id = _accept_with_expiry(8401, 1101, expires_in=30)
    sent: list[tuple[int | str, str]] = []

    async def fake_send_text(chat_id, text, *args, **kwargs):
        sent.append((chat_id, text))

    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fake_send_text)

    asyncio.run(telegram_doctor.run_session_tick())

    recipients = {cid for cid, _ in sent}
    assert 8401 in recipients
    assert 1101 in recipients


def test_run_session_tick_ends_expired_and_notifies(monkeypatch):
    consult_id = _accept_with_expiry(8402, 1102, expires_in=-5)
    sent: list[tuple[int | str, str]] = []

    async def fake_send_text(chat_id, text, *args, **kwargs):
        sent.append((chat_id, text))

    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fake_send_text)

    asyncio.run(telegram_doctor.run_session_tick())

    assert doctors.get_consultation(consult_id)["status"] == "ended"
    assert any(cid == 8402 and "kết thúc" in text.lower() for cid, text in sent)
    assert any(cid == 1102 and "kết thúc" in text.lower() for cid, text in sent)


def _accept_paid(patient_chat_id, doctor_tg, rate=2000):
    doctor_id = doctors.create_doctor("BS PaidCh", "Nội", "paid", rate, doctor_tg)
    consult_id = doctors.create_consultation(patient_chat_id, doctors.get_doctor(doctor_id), "paid")
    doctors.accept_consultation(consult_id)
    return consult_id


def test_run_session_tick_bills_patient_wallet(monkeypatch):
    from src.chat.clients import get_sqlite
    from src.chat.storage import wallet

    consult_id = _accept_paid(8501, 1201, rate=2000)
    row = doctors.get_consultation(consult_id)
    # Pretend 3 minutes have elapsed in the block.
    conn = get_sqlite()
    conn.execute(
        "UPDATE doctor_consultation SET block_started_at = ? WHERE id = ?",
        (row["block_started_at"] - 200, consult_id),
    )
    conn.commit()

    debits: list[tuple[str, int]] = []
    monkeypatch.setattr(telegram_doctor.wallet, "debit", lambda acc, amt: debits.append((acc, amt)) or -amt)

    async def fake_send_text(chat_id, text, *args, **kwargs):
        pass

    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fake_send_text)

    asyncio.run(telegram_doctor.run_session_tick())

    assert ("tg:8501", 3 * 2000) in debits


def test_run_session_tick_paid_offers_extend_with_button(monkeypatch):
    consult_id = _accept_paid(8502, 1202, rate=2000)
    row = doctors.get_consultation(consult_id)
    from src.chat.clients import get_sqlite
    conn = get_sqlite()
    # Move expiry into the warn window.
    conn.execute(
        "UPDATE doctor_consultation SET expires_at = ?, block_started_at = ? WHERE id = ?",
        (time.time() + 30, time.time(), consult_id),
    )
    conn.commit()

    sent: list[dict] = []

    async def fake_send_text(chat_id, text, *args, inline_keyboard=None, **kwargs):
        sent.append({"chat_id": chat_id, "text": text, "inline_keyboard": inline_keyboard})

    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram_doctor.wallet, "debit", lambda acc, amt: 0)

    asyncio.run(telegram_doctor.run_session_tick())

    patient_msgs = [m for m in sent if m["chat_id"] == 8502 and m["inline_keyboard"]]
    assert patient_msgs
    callbacks = [
        btn["callback_data"]
        for row in patient_msgs[0]["inline_keyboard"]["inline_keyboard"]
        for btn in row
    ]
    assert f"doctor:extend:{consult_id}" in callbacks
    assert "3,000" in patient_msgs[0]["text"]


def test_extend_callback_renews_block(monkeypatch):
    consult_id = _accept_paid(8503, 1203, rate=2000)
    sent: list[tuple[int | str, str]] = []

    async def fake_send_text(chat_id, text, *args, **kwargs):
        sent.append((chat_id, text))

    async def fake_answer_callback_query(callback_query_id, text):
        pass

    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)

    handled = asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "cbext",
        "data": f"doctor:extend:{consult_id}",
        "message": {"message_id": 50, "chat": {"id": 8503}},
    }))

    assert handled is True
    renewed = doctors.get_consultation(consult_id)
    assert renewed["block_index"] == 1
    assert renewed["rate_per_min"] == 3000


def test_pick_paid_doctor_blocked_during_pair_cooldown(monkeypatch):
    consult_id = _accept_paid(8504, 1204, rate=2000)
    doctor_id = doctors.get_consultation(consult_id)["doctor_id"]
    doctors.end_consultation(consult_id)

    edits: list[str] = []

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None):
        edits.append(text)

    async def fake_answer_callback_query(callback_query_id, text):
        pass

    async def fail_send_text(chat_id, text, *args, **kwargs):
        raise AssertionError("doctor should not be contacted during pair cooldown")

    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)
    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fail_send_text)

    handled = asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "cbpc",
        "data": f"doctor:pick:{doctor_id}",
        "message": {"message_id": 51, "chat": {"id": 8504}},
    }))

    assert handled is True
    assert doctors.open_consultation_for_patient(8504) is None
    assert "chờ" in edits[0].lower()


def test_pick_doctor_blocked_when_in_debt(monkeypatch):
    from src.chat.storage import wallet

    wallet.debit("tg:8601", 5_000)  # patient now in debt
    doctor_id = doctors.create_doctor("BS DebtGate", "Nội", "free", 0, 1301)
    edits: list[str] = []

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None):
        edits.append(text)

    async def fake_answer_callback_query(callback_query_id, text):
        pass

    async def fail_send_text(chat_id, text, *args, **kwargs):
        raise AssertionError("doctor should not be contacted while patient in debt")

    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)
    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fail_send_text)

    handled = asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "cbdebt",
        "data": f"doctor:pick:{doctor_id}",
        "message": {"message_id": 60, "chat": {"id": 8601}},
    }))

    assert handled is True
    assert doctors.open_consultation_for_patient(8601) is None
    assert "nợ" in edits[0].lower()


def _busy_doctor_ch(doctor_tg, tier="free", rate=0):
    doctor_id = doctors.create_doctor("BS BusyCh", "Nội", tier, rate, doctor_tg)
    consult_id = doctors.create_consultation(60000 + doctor_tg, doctors.get_doctor(doctor_id), tier)
    doctors.accept_consultation(consult_id)
    return doctor_id


def test_pick_busy_doctor_offers_waitlist_join(monkeypatch):
    doctor_id = _busy_doctor_ch(1401)
    edits: list[dict] = []

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None):
        edits.append({"text": text, "inline_keyboard": inline_keyboard})

    async def fake_answer_callback_query(callback_query_id, text):
        pass

    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)

    handled = asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "cbwl",
        "data": f"doctor:pick:{doctor_id}",
        "message": {"message_id": 70, "chat": {"id": 8701}},
    }))

    assert handled is True
    callbacks = [
        btn["callback_data"]
        for row in edits[0]["inline_keyboard"]["inline_keyboard"]
        for btn in row
    ]
    assert f"doctor:wait:{doctor_id}" in callbacks


def test_join_waitlist_callback_shows_position_with_refresh_and_out(monkeypatch):
    doctor_id = _busy_doctor_ch(1402)
    edits: list[dict] = []

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None):
        edits.append({"text": text, "inline_keyboard": inline_keyboard})

    async def fake_answer_callback_query(callback_query_id, text):
        pass

    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)

    handled = asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "cbwl2",
        "data": f"doctor:wait:{doctor_id}",
        "message": {"message_id": 71, "chat": {"id": 8702}},
    }))

    assert handled is True
    assert doctors.waitlist_status(doctor_id, 8702)["position"] == 1
    text = edits[0]["text"].lower()
    assert "hàng đợi" in text or "chờ" in text
    callbacks = [
        btn["callback_data"]
        for row in edits[0]["inline_keyboard"]["inline_keyboard"]
        for btn in row
    ]
    assert f"doctor:wlrefresh:{doctor_id}" in callbacks
    assert f"doctor:wlleave:{doctor_id}" in callbacks


def test_waitlist_leave_callback_removes_patient(monkeypatch):
    doctor_id = _busy_doctor_ch(1403)
    doctors.join_waitlist(doctor_id, 8703, "free")
    edits: list[str] = []

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None):
        edits.append(text)

    async def fake_answer_callback_query(callback_query_id, text):
        pass

    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)

    handled = asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "cbwl3",
        "data": f"doctor:wlleave:{doctor_id}",
        "message": {"message_id": 72, "chat": {"id": 8703}},
    }))

    assert handled is True
    assert doctors.waitlist_status(doctor_id, 8703) is None


def test_run_session_tick_dispatches_waitlist_offer(monkeypatch):
    doctor_id = _busy_doctor_ch(1404)
    doctors.join_waitlist(doctor_id, 8704, "free")
    doctors.join_waitlist(doctor_id, 8705, "free")
    offered = doctors.promote_waitlist(doctor_id)

    from src.chat.clients import get_sqlite
    conn = get_sqlite()
    conn.execute(
        "UPDATE doctor_waitlist SET notified_at = ? WHERE id = ?",
        (time.time() - doctors.WAITLIST_CLAIM_SECONDS - 5, offered["id"]),
    )
    conn.commit()

    sent: list[dict] = []

    async def fake_send_text(chat_id, text, *args, inline_keyboard=None, **kwargs):
        sent.append({"chat_id": chat_id, "text": text, "inline_keyboard": inline_keyboard})

    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram_doctor.wallet, "debit", lambda acc, amt: 0)

    asyncio.run(telegram_doctor.run_session_tick())

    # Second patient gets an offer with an Accept (pick) button.
    offers = [m for m in sent if m["chat_id"] == 8705 and m["inline_keyboard"]]
    assert offers
    callbacks = [
        btn["callback_data"]
        for row in offers[0]["inline_keyboard"]["inline_keyboard"]
        for btn in row
    ]
    assert f"doctor:pick:{doctor_id}" in callbacks


def test_run_session_tick_dispatches_waitlist_offer_reminder(monkeypatch):
    doctor_id = doctors.create_doctor("BS ReminderCh", "Nội", "free", 0, 1405)
    doctors.join_waitlist(doctor_id, 8706, "free")
    offered = doctors.promote_waitlist(doctor_id)

    from src.chat.clients import get_sqlite
    conn = get_sqlite()
    conn.execute(
        "UPDATE doctor_waitlist SET last_reminded_at = ? WHERE id = ?",
        (time.time() - doctors.WAITLIST_REMINDER_SECONDS - 1, offered["id"]),
    )
    conn.commit()

    sent: list[dict] = []

    async def fake_send_text(chat_id, text, *args, inline_keyboard=None, **kwargs):
        sent.append({"chat_id": chat_id, "text": text, "inline_keyboard": inline_keyboard})

    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram_doctor.wallet, "debit", lambda acc, amt: 0)

    asyncio.run(telegram_doctor.run_session_tick())

    reminders = [m for m in sent if m["chat_id"] == 8706 and m["inline_keyboard"]]
    assert reminders
    assert "nhắc" in reminders[0]["text"].lower()
    callbacks = [
        btn["callback_data"]
        for row in reminders[0]["inline_keyboard"]["inline_keyboard"]
        for btn in row
    ]
    assert f"doctor:pick:{doctor_id}" in callbacks



def test_pick_after_offer_clears_waitlist_entry(monkeypatch):
    from src.chat.clients import get_sqlite

    # Free doctor, patient on the waitlist who has been offered the slot.
    doctor_id = doctors.create_doctor("BS Claim", "Nội", "free", 0, 1450)
    doctors.join_waitlist(doctor_id, 8801, "free")
    doctors.promote_waitlist(doctor_id)

    def _entry_status():
        conn = get_sqlite()
        row = conn.execute(
            "SELECT status FROM doctor_waitlist WHERE doctor_id = ? AND patient_chat_id = ?",
            (doctor_id, 8801),
        ).fetchone()
        return row[0] if row else None

    assert _entry_status() == "offered"

    async def fake_send_text(chat_id, text, *args, inline_keyboard=None, **kwargs):
        pass

    async def fake_edit_message_text(chat_id, message_id, text, inline_keyboard=None):
        pass

    async def fake_answer_callback_query(callback_query_id, text):
        pass

    monkeypatch.setattr(telegram_doctor.telegram, "send_text", fake_send_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_edit_message_text", fake_edit_message_text)
    monkeypatch.setattr(telegram_doctor.telegram, "_answer_callback_query", fake_answer_callback_query)

    handled = asyncio.run(telegram_doctor.handle_doctor_callback({
        "id": "cbclaim",
        "data": f"doctor:pick:{doctor_id}",
        "message": {"message_id": 80, "chat": {"id": 8801}},
    }))

    assert handled is True
    # Consultation was created and the offered entry no longer blocks the queue.
    assert doctors.open_consultation_for_patient(8801) is not None
    assert _entry_status() not in ("offered", "waiting")
