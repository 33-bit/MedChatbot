# Diagrams (Mermaid)

Mermaid versions of the report's TikZ figures (Chapter 4) plus code-derived
flows. Source-of-truth mapping:

| Mermaid below | Report figure |
|---|---|
| 1. Layered architecture | `fig:packages` |
| 2. Request pipeline (informational turn) | `fig:pipeline` |
| 3. Diagnostic narrowing | `alg:narrowing` |
| 4. Knowledge-graph schema | `fig:kg_schema` |
| 5. Doctor consultation lifecycle | code-derived (`doctors.py`, `telegram_doctor.py`) |
| 6. Top-up payment | code-derived (`payments/`, `wallet.py`) |

> Note: Mermaid has no native UML use-case notation, and `fig:kg_schema` is a
> relational graph, not a flowchart — rendered here as a labeled `flowchart`.
> Appendix A/B images (`IoT.png`, `Bia.PNG`) are template boilerplate, not project diagrams — excluded.

---

## 1. Layered architecture (`fig:packages`)

```mermaid
flowchart TD
    PRES["<b>Presentation layer</b><br/>server.app · server.channels (telegram, zalo, messenger, telegram_doctor) · server.payments"]
    APP["<b>Application layer</b><br/>chat.pipeline — preflight → analyze → route → retrieve → generate → persist"]
    DOM["<b>Domain layer</b><br/>chat.llm (analyzer, generator) · chat.diagnosis · chat.retrieval (dense, sparse, fusion, rerank, kg) · chat.guards"]
    DATA["<b>Data layer</b><br/>chat.storage (session, profile, wallet, doctors, feedback) → Redis, SQLite, Qdrant, Neo4j, LLM, PayOS"]
    OFF["<b>Offline build</b><br/>processing (bachmai, drugs, symptom_canon) · rag (chunker, build_qdrant, kg_builder) · eval"]
    PRES --> APP
    APP --> DOM
    DOM --> DATA
    OFF -. builds ahead of time .-> DATA
```

---

## 2. Request pipeline — informational turn (`fig:pipeline`)

Retrieval runs in parallel (two-worker pool); either source failing aborts the
turn with a safe technical-error reply (fail-closed). Guardrail/quota
short-circuits before any retrieval or generation.

```mermaid
flowchart TD
    IN(["User question + session"])
    PRE["Preflight: rate limit, regex guardrail, LLM quota"]
    AN["One-shot analyzer: guardrail, label, rewrite, entities"]
    ROUTE["Mode policy and routing; ingest entities into session"]
    RAG["Hybrid RAG:<br/>dense + sparse → RRF → rerank"]
    KG["KG retrieval:<br/>entity link → traverse relations"]
    GEN["Generator: answer from evidence + citations"]
    PERSIST["Persist: session, profile, consultation log"]
    OUT(["Cited answer"])
    CANNED(["Canned safe reply<br/>(refuse / emergency / technical-error)"])

    IN --> PRE
    PRE -->|blocked| CANNED
    PRE -->|ok| AN
    AN -->|guardrail / off-topic| CANNED
    AN -->|ok| ROUTE
    ROUTE --> RAG
    ROUTE --> KG
    RAG -->|fail| CANNED
    KG -->|fail| CANNED
    RAG --> GEN
    KG --> GEN
    GEN --> PERSIST
    PERSIST --> OUT
```

---

## 3. Diagnostic narrowing (`alg:narrowing`)

```mermaid
flowchart TD
    START(["User symptoms S + session state"])
    Q1{"First diagnostic turn<br/>and not force-answer?"}
    ACK["Return general triage acknowledgement<br/>+ red-flag warnings"]
    Q2{"Clarification queue non-empty<br/>and not force-answer?"}
    NEXTQ["Return next queued question"]
    RANK["rank_candidates(S):<br/>count symptom overlap in Neo4j → C"]
    Q3{"not force-answer<br/>and |C| > MIN_CANDIDATES_TO_STOP?"}
    DISC["discriminative_symptoms(C, known, BATCH_SIZE)"]
    Q4{"D non-empty?"}
    ENQ["Enqueue D;<br/>return first clarification question"]
    FINAL["Build preliminary guidance from S + shortlist C<br/>via constrained generation"]
    OUT(["Cited preliminary guidance<br/>— no certain diagnosis over chat"])

    START --> Q1
    Q1 -->|yes| ACK
    Q1 -->|no| Q2
    Q2 -->|yes| NEXTQ
    Q2 -->|no| RANK
    RANK --> Q3
    Q3 -->|no| FINAL
    Q3 -->|yes| DISC
    DISC --> Q4
    Q4 -->|yes| ENQ
    Q4 -->|no| FINAL
    FINAL --> OUT
```

---

## 4. Knowledge-graph schema (`fig:kg_schema`)

```mermaid
flowchart LR
    Disease["Disease"]
    Symptom["Symptom"]
    Drug["Drug"]
    Chapter["Chapter"]

    Disease -->|"HAS_SYMPTOM / RED_FLAG_FOR"| Symptom
    Drug -->|"TREATS"| Disease
    Drug -->|"RELIEVES"| Symptom
    Drug -->|"CONTRAINDICATED_FOR"| Disease
    Disease -->|"BELONGS_TO_CHAPTER"| Chapter
    Drug -->|"MAY_CAUSE_ADR"| Symptom
    Drug -->|"INTERACTS_WITH"| Drug
    Disease -->|"COMORBID_RISK"| Disease
```

---

## 5. Doctor consultation lifecycle (code-derived)

From `doctors.py` (status `pending → active → ended/declined`, waitlist,
per-minute billing) and `telegram_doctor.py`.

```mermaid
flowchart TD
    A(["Handoff offered after uncertain triage, or /doctor"])
    B{"Doctor available?"}
    W["Join waitlist<br/>(promoted when a slot frees)"]
    C["create_consultation → pending"]
    D{"Doctor accepts?"}
    DEC(["declined → pick another doctor"])
    E["active session — relayed live chat"]
    T{"tier?"}
    FREE["free: 5-min session, then cooldown"]
    PAID["paid: per-minute wallet debit<br/>15-min blocks, rate escalates on renew"]
    END(["ended — via /end, idle, or timeout"])

    A --> B
    B -->|no| W
    W --> C
    B -->|yes| C
    C --> D
    D -->|no| DEC
    D -->|yes| E
    E --> T
    T -->|free| FREE
    T -->|paid| PAID
    FREE --> END
    PAID --> END
```

---

## 6. Top-up payment (code-derived)

From `payments/payos.py`, `payments/router.py`, `wallet.py`.

```mermaid
sequenceDiagram
    actor Patient
    participant Bot as Telegram bot
    participant PayOS as Payment Gateway PayOS
    participant DB as Wallet SQLite

    Patient->>Bot: /topup amount
    Bot->>PayOS: create_payment(order_code, amount)
    PayOS-->>Bot: VietQR + checkout link
    Bot-->>Patient: show QR
    Patient->>PayOS: pay via bank app
    PayOS->>Bot: POST /webhook/payos (signed)
    Bot->>Bot: verify signature, dedupe, mark_order_paid
    Bot->>DB: apply_payment (credit or settle debt)
    Bot-->>Patient: balance updated
```
