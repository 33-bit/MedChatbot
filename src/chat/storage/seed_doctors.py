"""Seed/update doctor registry rows.

Run manually from Python or import in scripts. Does not run automatically
at import time; call seed_default_doctors() from a script or admin path.
"""

from __future__ import annotations

from src.chat.clients import get_sqlite

# Base per-minute rate (VND) for every paid-tier doctor. Feature 3 (paid-tier
# billing) reads `doctor.price` as the VND/minute rate for the first 15-minute
# block; renewed blocks escalate from here.
PAID_RATE_PER_MIN = 2000

# Specialties covered by the default registry (Vietnamese).
SPECIALTIES = [
    "Nội tổng quát",
    "Nhi khoa",
    "Da liễu",
    "Tim mạch",
    "Tai mũi họng",
    "Tiêu hóa",
    "Hô hấp",
    "Cơ xương khớp",
    "Thần kinh",
    "Nội tiết",
    "Sản phụ khoa",
    "Mắt",
    "Răng hàm mặt",
    "Tiết niệu",
    "Tâm thần",
]

# Display names paired with each specialty: (free doctor, paid doctor).
_DOCTOR_NAMES = [
    ("BS Nguyễn Văn An", "BS Trần Thị Bình"),
    ("BS Lê Hoàng Cường", "BS Phạm Thị Dung"),
    ("BS Vũ Minh Đức", "BS Đỗ Thị Em"),
    ("BS Hoàng Văn Phú", "BS Bùi Thị Giang"),
    ("BS Đặng Quốc Hùng", "BS Ngô Thị Hoa"),
    ("BS Dương Văn Khoa", "BS Lý Thị Lan"),
    ("BS Trịnh Văn Long", "BS Mai Thị Mến"),
    ("BS Phan Thanh Nam", "BS Tô Thị Nga"),
    ("BS Lương Văn Oanh", "BS Hồ Thị Phương"),
    ("BS Tạ Văn Quang", "BS Đinh Thị Quỳnh"),
    ("BS Nguyễn Văn Sơn", "BS Trần Thị Thu"),
    ("BS Lê Văn Tài", "BS Phạm Thị Uyên"),
    ("BS Vũ Văn Vinh", "BS Đỗ Thị Xuân"),
    ("BS Hoàng Văn Yên", "BS Bùi Thị Ánh"),
    ("BS Đặng Văn Bảo", "BS Ngô Thị Cẩm"),
]

_DEGREES = [
    "Bác sĩ chuyên khoa I",
    "Thạc sĩ, Bác sĩ",
    "Bác sĩ chuyên khoa II",
    "Tiến sĩ, Bác sĩ",
]

_HOSPITALS = [
    "Bệnh viện Bạch Mai",
    "Bệnh viện Đại học Y Hà Nội",
    "Bệnh viện Trung ương Quân đội 108",
    "Bệnh viện E",
    "Bệnh viện Hữu nghị Việt Đức",
]

# Reserved placeholder Telegram user ids. An admin re-points these to real
# Telegram accounts later via seed_doctors() (upsert by telegram_user_id).
_FREE_ID_BASE = 900_000_001
_PAID_ID_BASE = 900_000_101


def _mock_profile(specialty: str, idx: int, paid: bool) -> dict:
    base_experience = 7 + idx % 6
    experience_years = base_experience + (5 if paid else 0)
    degree = _DEGREES[(idx + (1 if paid else 0)) % len(_DEGREES)]
    hospital = _HOSPITALS[(idx + (2 if paid else 0)) % len(_HOSPITALS)]
    focus = specialty.casefold()
    return {
        "degree": degree,
        "experience_years": experience_years,
        "hospital": hospital,
        "bio": (
            f"Có kinh nghiệm tư vấn và theo dõi các vấn đề {focus}; "
            "ưu tiên giải thích dễ hiểu và hướng dẫn người bệnh đi khám đúng lúc."
        ),
    }


def _build_default_doctors() -> list[dict]:
    rows: list[dict] = []
    for idx, specialty in enumerate(SPECIALTIES):
        free_name, paid_name = _DOCTOR_NAMES[idx]
        rows.append({
            "name": free_name,
            "specialty": specialty,
            "tier": "free",
            "price": 0,
            "telegram_user_id": _FREE_ID_BASE + idx,
            **_mock_profile(specialty, idx, paid=False),
        })
        rows.append({
            "name": paid_name,
            "specialty": specialty,
            "tier": "paid",
            "price": PAID_RATE_PER_MIN,
            "telegram_user_id": _PAID_ID_BASE + idx,
            **_mock_profile(specialty, idx, paid=True),
        })
    return rows


DEFAULT_DOCTORS = _build_default_doctors()


def seed_doctors(rows: list[dict]) -> None:
    conn = get_sqlite()
    for row in rows:
        conn.execute(
            "INSERT INTO doctor ("
            "name, specialty, tier, price, telegram_user_id, active, "
            "degree, experience_years, hospital, bio"
            ") VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?) "
            "ON CONFLICT(telegram_user_id) DO UPDATE SET "
            "name = excluded.name, specialty = excluded.specialty, tier = excluded.tier, "
            "price = excluded.price, active = 1, degree = excluded.degree, "
            "experience_years = excluded.experience_years, hospital = excluded.hospital, "
            "bio = excluded.bio",
            (
                row["name"],
                row.get("specialty"),
                row["tier"],
                int(row.get("price") or 0),
                int(row["telegram_user_id"]),
                row.get("degree"),
                row.get("experience_years"),
                row.get("hospital"),
                row.get("bio"),
            ),
        )
    conn.commit()


def seed_default_doctors() -> None:
    """Upsert the canonical default doctor registry. Idempotent."""
    seed_doctors(DEFAULT_DOCTORS)
