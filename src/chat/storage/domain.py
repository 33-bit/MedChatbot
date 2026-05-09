from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class PatientSession:
    session_id: str
    symptoms: list[dict] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    conversation: list[dict] = field(default_factory=list)
    candidate_diseases: list[dict] = field(default_factory=list)
    answered_questions: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> "PatientSession":
        return cls(**json.loads(data))

    def add_message(self, role: str, content: str, max_history: int = 20) -> None:
        self.conversation.append({"role": role, "content": content})
        if len(self.conversation) > max_history:
            self.conversation = self.conversation[-max_history:]

    def upsert_symptom(self, entry: dict) -> None:
        sid = entry.get("symptom_id")
        if not sid:
            return
        for i, s in enumerate(self.symptoms):
            if s.get("symptom_id") == sid:
                self.symptoms[i] = {**s, **{k: v for k, v in entry.items() if v}}
                return
        self.symptoms.append(entry)

    def add_medication(self, drug_id: str) -> None:
        if drug_id and drug_id not in self.medications:
            self.medications.append(drug_id)
