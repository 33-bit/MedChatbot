#!/usr/bin/env python3
"""Shared evaluation core for category-specific eval scripts.

Category files under eval/categories are the executable entrypoints. This
module holds the shared mechanics for their subcommands:
  run-direct      Call src.chat.answer_with_meta in-process (full pipeline).
  run-api         Call a running /chat?include_meta=1 server.
  run-retrieval   Retrieval-only: recall@k, MRR (doc-level + chunk-level), context precision.
  score-file      Score answers produced by another bot from a JSONL file.
  compare         Aggregate result JSONL files into a side-by-side summary.

Generation scoring runs four metric families:
  - Deterministic checks (predefined guardrail replies / citation presence)
  - Retrieval metrics (when meta is available)
  - LLM judges (faithfulness, answer correctness, answer relevance) — opt-in via --use-judge
  - Operational (latency p50/p95, token usage, cost, error rate)

Cost is computed from token usage when available. Edit MODEL_PRICING to match
your billing.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval import metrics as eval_metrics
from eval.categories.registry import apply_category_checks as apply_registered_category_checks

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - only needed for API mode
    httpx = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------


DEFAULT_DATASET = Path("eval/datasets/medical_qa_benchmark.jsonl")
DEFAULT_OUT_DIR = Path("eval/results")
DEFAULT_PASS_THRESHOLD = 0.75
EVAL_CHAT_MODE = "information"
CITATION_RE = re.compile(r"\[(\d+)\]")

# Weights for the deterministic scorer.
WEIGHT_PREDEFINED_ANSWER = 1.0
WEIGHT_CITATION = 1.5
PREDEFINED_ANSWER_CATEGORIES = {
    "safety_prompt_injection",
    "safety_off_topic",
}

# Per-1k-token USD pricing. Edit to match the deployed model's invoicing.
# Entries default to 0 so missing models don't crash; cost just reads as 0.
MODEL_PRICING: dict[str, dict[str, float]] = {
    # "claude-sonnet-4-6": {"prompt": 0.003, "completion": 0.015},
}

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic IO helpers
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    text = (text or "").casefold().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def iter_cases(
    cases: list[dict[str, Any]],
    ids: set[str] | None,
    limit: int | None,
    exclude_categories: set[str] | None = None,
    include_categories: set[str] | None = None,
):
    selected = [
        case for case in cases
        if (not ids or case["id"] in ids)
        and (not include_categories or case.get("category") in include_categories)
        and (not exclude_categories or case.get("category") not in exclude_categories)
    ]
    if limit is not None:
        selected = selected[:limit]
    return selected


CLARIFICATION_QUESTION_RE = re.compile(r"\?\s*$")
FORCE_ANSWER_PROMPT = "tôi không rõ, cứ trả lời thẳng giúp tôi đi"


def is_still_clarifying(answer: str) -> bool:
    """A bot reply is treated as 'still clarifying' when it contains no
    citation marker. The generator emits a final-answer reply only when it
    cites at least one source via [n], so the absence of `[n]` is a reliable
    signal that the bot is asking follow-up questions instead of committing.
    """
    if not answer:
        return False
    return not CITATION_RE.search(answer)


def result_path(out_dir: Path, bot_name: str, suffix: str = "results") -> Path:
    safe_bot = re.sub(r"[^A-Za-z0-9_.-]+", "_", bot_name).strip("_") or "bot"
    return out_dir / f"{safe_bot}-{suffix}-{timestamp()}.jsonl"


def _include_categories_from_args(args: argparse.Namespace) -> set[str] | None:
    categories = set(getattr(args, "include_categories", None) or [])
    category = getattr(args, "category", None)
    if category:
        categories.add(category)
    return categories or None


def _result_path_for_args(args: argparse.Namespace, suffix: str) -> Path:
    category = getattr(args, "category", None)
    category_suffix = f"{category}-{suffix}" if category else suffix
    return result_path(args.out_dir, args.bot_name, suffix=category_suffix)


# ---------------------------------------------------------------------------
# Deterministic scoring
# ---------------------------------------------------------------------------


def _uses_predefined_answer(case: dict[str, Any]) -> bool:
    return case.get("category") in PREDEFINED_ANSWER_CATEGORIES


def _reference_answer_matches(case: dict[str, Any], answer: str) -> bool:
    reference = case.get("reference_answer") or ""
    return bool(reference) and normalize(answer) == normalize(reference)


def score_answer(
    case: dict[str, Any],
    answer: str,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    if _uses_predefined_answer(case):
        checks.append({
            "type": "predefined_answer",
            "target": case.get("reference_answer", ""),
            "passed": _reference_answer_matches(case, answer),
            "weight": WEIGHT_PREDEFINED_ANSWER,
        })

    if case.get("requires_citation", False):
        citation_indices = [int(m.group(1)) for m in CITATION_RE.finditer(answer)]
        checks.append({
            "type": "requires_citation", "target": "[n]",
            "matched": citation_indices,
            "max_index": max(citation_indices) if citation_indices else 0,
            "passed": bool(citation_indices), "weight": WEIGHT_CITATION,
        })

    weighted_total = sum(c.get("weight", 1.0) for c in checks)
    weighted_passed = sum(c.get("weight", 1.0) for c in checks if c["passed"])
    score = weighted_passed / weighted_total if weighted_total else 0.0
    return {
        "score": round(score, 4),
        "passed": score >= pass_threshold if weighted_total else False,
        "hard_fail": False,
        "checks": checks,
        "scoring_mode": "deterministic",
    }


# ---------------------------------------------------------------------------
# Retrieval metrics (doc-level + chunk-level + context precision)
# ---------------------------------------------------------------------------


def _slug_from_path(path: str) -> str:
    if not path:
        return ""
    name = path.rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0]


def _gold_slugs(case: dict[str, Any]) -> list[str]:
    return [_slug_from_path(d.get("path", "")) for d in case.get("source_docs", []) if d.get("path")]


def _gold_chunks(case: dict[str, Any]) -> list[str]:
    return list(case.get("gold_chunks") or [])


def retrieval_metrics(
    gold_slugs: list[str],
    retrieved_slugs: list[str],
    *,
    gold_chunks: list[str] | None = None,
    retrieved_chunks: list[str] | None = None,
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, Any]:
    """Compute doc-level recall@k / MRR; chunk-level recall@k / MRR when
    gold_chunks is provided; context precision @ top-k retrieved chunks.
    """
    metrics: dict[str, Any] = {}

    # Doc-level
    doc_hit_ranks = [i + 1 for i, slug in enumerate(retrieved_slugs) if slug in gold_slugs]
    for k in ks:
        metrics[f"recall@{k}"] = 1.0 if any(r <= k for r in doc_hit_ranks) else 0.0
    metrics["mrr"] = round(1.0 / doc_hit_ranks[0], 4) if doc_hit_ranks else 0.0
    metrics["first_hit_rank"] = doc_hit_ranks[0] if doc_hit_ranks else None

    # Chunk-level (only when both sides are available)
    if gold_chunks and retrieved_chunks is not None:
        chunk_set = set(gold_chunks)
        chunk_hit_ranks = [i + 1 for i, cid in enumerate(retrieved_chunks) if cid in chunk_set]
        for k in ks:
            metrics[f"chunk_recall@{k}"] = 1.0 if any(r <= k for r in chunk_hit_ranks) else 0.0
        metrics["chunk_mrr"] = round(1.0 / chunk_hit_ranks[0], 4) if chunk_hit_ranks else 0.0
        metrics["chunk_first_hit_rank"] = chunk_hit_ranks[0] if chunk_hit_ranks else None
        # Context precision @ top-k: how much of the retrieved context belongs
        # to a gold chunk. Conservative — relevant if chunk_id matches gold.
        for k in ks:
            top = retrieved_chunks[:k] or []
            metrics[f"context_precision@{k}"] = (
                round(sum(1 for cid in top if cid in chunk_set) / len(top), 4)
                if top else 0.0
            )
            coverage = eval_metrics.gold_chunk_coverage_at_k(gold_chunks, retrieved_chunks, k)
            if coverage is not None:
                metrics[f"gold_chunk_coverage@{k}"] = coverage
            for source_type, value in eval_metrics.source_type_coverage_at_k(
                gold_chunks,
                retrieved_chunks,
                k,
            ).items():
                metrics[f"{source_type}_source_coverage@{k}"] = value
    return metrics


# ---------------------------------------------------------------------------
# LLM judges (faithfulness, answer correctness, answer relevance)
# ---------------------------------------------------------------------------


FAITHFULNESS_PROMPT = """Bạn là chuyên gia kiểm thử y tế. Đánh giá xem CÂU TRẢ LỜI CỦA CHATBOT có hoàn toàn được hỗ trợ bởi NỘI DUNG NGUỒN không.

Câu hỏi gốc: {question}

Nội dung nguồn (được dùng để xây câu trả lời):
{reference}

Câu trả lời của chatbot:
{answer}

Đánh giá:
- "faithful": chatbot CÓ phát biểu nào MÂU THUẪN với nội dung nguồn hoặc BỊA RA sự kiện y tế (tên bệnh, liều thuốc, triệu chứng) không có trong nguồn không? Diễn đạt khác là OK.
- "unsupported_claims": tối đa 3 phát biểu cụ thể không được nguồn ủng hộ. Để rỗng nếu không có.
- "score": 1.0 nếu hoàn toàn đúng nguồn, 0.5 nếu có chi tiết nhỏ không được ủng hộ nhưng không gây hại, 0.0 nếu mâu thuẫn rõ ràng hoặc bịa nguy hiểm.

Trả về JSON: faithful (bool), unsupported_claims (list[str]), score (float)."""

CORRECTNESS_PROMPT = """Bạn là chuyên gia kiểm thử y tế. So khớp ngữ nghĩa giữa CÂU TRẢ LỜI CỦA CHATBOT và CÂU TRẢ LỜI CHUẨN.

Câu hỏi gốc: {question}

Câu trả lời chuẩn:
{reference}

Câu trả lời của chatbot:
{answer}

Đánh giá:
- "correct": câu trả lời chatbot có chứa ĐẦY ĐỦ các ý chính của câu trả lời chuẩn và KHÔNG mâu thuẫn với nó không? Thông tin ĐÚNG và LIÊN QUAN nằm NGOÀI câu trả lời chuẩn KHÔNG bị tính là lỗi.
- "score": 1.0 nếu có đủ ý chính và không mâu thuẫn (kể cả khi dài/chi tiết hơn câu chuẩn), 0.5 nếu thiếu một ý phụ hoặc có sai sót nhỏ, 0.0 nếu thiếu hoặc mâu thuẫn ý trọng tâm.
- "missing_or_wrong": tối đa 3 ý CÒN THIẾU so với câu chuẩn hoặc thông tin MÂU THUẪN/SAI. KHÔNG liệt kê thông tin thừa nhưng đúng.

Trả về JSON: correct (bool), score (float), missing_or_wrong (list[str])."""

RELEVANCE_PROMPT = """Bạn là chuyên gia kiểm thử y tế. Đánh giá xem CÂU TRẢ LỜI CỦA CHATBOT có thực sự trả lời CÂU HỎI hay không.

Câu hỏi gốc: {question}

Câu trả lời của chatbot:
{answer}

Đánh giá:
- "relevant": câu trả lời có địa chỉ trực tiếp câu hỏi không? Lan man / né tránh khi không cần né = KHÔNG relevant.
- "score": 1.0 đúng trọng tâm, 0.5 trả lời một phần, 0.0 lạc đề.
- "reason": một câu giải thích.

Trả về JSON: relevant (bool), score (float), reason (str)."""


@dataclass
class JudgeResult:
    faithful: bool | None = None
    faithful_score: float | None = None
    unsupported_claims: list[str] = field(default_factory=list)
    correct: bool | None = None
    correctness_score: float | None = None
    missing_or_wrong: list[str] = field(default_factory=list)
    relevant: bool | None = None
    relevant_score: float | None = None
    relevance_reason: str | None = None
    error: str | None = None

    @property
    def combined_score(self) -> float | None:
        scores = [s for s in (self.faithful_score, self.correctness_score, self.relevant_score) if s is not None]
        return round(sum(scores) / len(scores), 4) if scores else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "faithful": self.faithful,
            "faithful_score": self.faithful_score,
            "unsupported_claims": self.unsupported_claims,
            "correct": self.correct,
            "correctness_score": self.correctness_score,
            "missing_or_wrong": self.missing_or_wrong,
            "relevant": self.relevant,
            "relevant_score": self.relevant_score,
            "relevance_reason": self.relevance_reason,
            "combined_score": self.combined_score,
            "error": self.error,
        }


def _coerce_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return round(min(1.0, max(0.0, score)), 4)


def _ask_llm(client, model: str, prompt: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(response.choices[0].message.content or "{}")


def judge(case: dict[str, Any], answer: str, *, client=None, model: str | None = None) -> JudgeResult:
    """Run faithfulness + correctness + relevance judges. Errors captured, never raised."""
    if client is None:
        from src.chat.clients import get_openai
        client = get_openai()
    if model is None:
        from src.config import MODEL
        model = MODEL

    question = case.get("question") or " | ".join(case.get("turns") or [])
    reference = case.get("reference_answer", "")
    faith_reference = case.get("_retrieved_source_text") or reference
    result = JudgeResult()

    if not answer:
        result.error = "empty_answer"
        return result

    errors: list[str] = []

    try:
        faith = _ask_llm(client, model, FAITHFULNESS_PROMPT.format(
            question=question, reference=faith_reference, answer=answer,
        ))
        result.faithful = bool(faith.get("faithful"))
        result.faithful_score = _coerce_score(faith.get("score"))
        result.unsupported_claims = [str(c) for c in (faith.get("unsupported_claims") or [])][:3]
    except Exception as exc:
        log.warning("faithfulness judge failed: %s", exc)
        errors.append(f"faithfulness:{exc!r}")

    try:
        corr = _ask_llm(client, model, CORRECTNESS_PROMPT.format(
            question=question, reference=reference, answer=answer,
        ))
        result.correct = bool(corr.get("correct"))
        result.correctness_score = _coerce_score(corr.get("score"))
        result.missing_or_wrong = [str(c) for c in (corr.get("missing_or_wrong") or [])][:3]
    except Exception as exc:
        log.warning("correctness judge failed: %s", exc)
        errors.append(f"correctness:{exc!r}")

    try:
        rel = _ask_llm(client, model, RELEVANCE_PROMPT.format(
            question=question, answer=answer,
        ))
        result.relevant = bool(rel.get("relevant"))
        result.relevant_score = _coerce_score(rel.get("score"))
        if rel.get("reason") is not None:
            result.relevance_reason = str(rel["reason"])
    except Exception as exc:
        log.warning("relevance judge failed: %s", exc)
        errors.append(f"relevance:{exc!r}")

    if errors:
        result.error = "; ".join(errors)
    return result


def maybe_judge(case: dict[str, Any], answer: str, *, enabled: bool) -> dict[str, Any] | None:
    if not enabled or not answer or _uses_predefined_answer(case):
        return None
    try:
        return judge(case, answer).to_dict()
    except Exception as exc:
        return {"error": f"judge_setup:{exc!r}"}


def evaluate_answer(
    case: dict[str, Any],
    answer: str,
    *,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    use_judge: bool = False,
) -> dict[str, Any]:
    scored = score_answer(case, answer, pass_threshold)
    scored = apply_registered_category_checks(case, answer, scored, pass_threshold)
    judge_result = maybe_judge(case, answer, enabled=use_judge)
    if judge_result is None:
        return scored
    return eval_metrics.apply_judge_score(scored, judge_result, pass_threshold)


# ---------------------------------------------------------------------------
# Token usage / cost helpers
# ---------------------------------------------------------------------------


def usage_summary(usage_entries: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Sum prompt/completion/total tokens across all stages and price them.
    Accepts the list of {stage, prompt_tokens, completion_tokens, total_tokens, model}
    produced by `answer_with_meta`.
    """
    if not usage_entries:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}
    pt = sum(int(u.get("prompt_tokens") or 0) for u in usage_entries)
    ct = sum(int(u.get("completion_tokens") or 0) for u in usage_entries)
    tt = sum(int(u.get("total_tokens") or 0) for u in usage_entries) or (pt + ct)
    cost = 0.0
    for u in usage_entries:
        price = MODEL_PRICING.get(u.get("model") or "", {})
        cost += (int(u.get("prompt_tokens") or 0) / 1000) * price.get("prompt", 0.0)
        cost += (int(u.get("completion_tokens") or 0) / 1000) * price.get("completion", 0.0)
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt,
        "cost_usd": round(cost, 6),
    }


def _retrieved_slugs_from_meta(meta: dict[str, Any]) -> list[str]:
    return [r.get("source_slug", "") for r in (meta or {}).get("retrieved", []) if r]


def _retrieved_chunks_from_meta(meta: dict[str, Any]) -> list[str]:
    return [r.get("chunk_id", "") for r in (meta or {}).get("retrieved", []) if r]


def _retrieved_source_text_from_meta(meta: dict[str, Any]) -> str:
    """Join the actual retrieved chunk text the bot saw, for faithfulness judging.

    Dedupes identical chunk text and skips entries without text (e.g. when the
    pipeline did not record source text). Returns "" when no text is available,
    so the judge falls back to reference_answer.
    """
    blocks: list[str] = []
    seen: set[str] = set()
    for r in (meta or {}).get("retrieved", []):
        text = (r or {}).get("text")
        if not text or text in seen:
            continue
        seen.add(text)
        blocks.append(text)
    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Aggregation, percentiles, bootstrap CI, summary printing
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _bootstrap_pass_ci(items: list[dict[str, Any]], samples: int = 1000) -> tuple[float, float] | None:
    if len(items) < 2:
        return None
    flags = [1 if item.get("passed") else 0 for item in items]
    rng = random.Random(0)
    n = len(flags)
    rates: list[float] = []
    for _ in range(samples):
        resample = [flags[rng.randrange(n)] for _ in range(n)]
        rates.append(sum(resample) / n)
    rates.sort()
    lo = rates[int(0.025 * samples)]
    hi = rates[int(0.975 * samples) - 1]
    return round(lo, 4), round(hi, 4)


def _aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(items)
    passed = sum(1 for item in items if item.get("passed"))
    avg_score = sum(float(item.get("score", 0.0)) for item in items) / total if total else 0.0
    latencies = [float(item["latency_ms"]) for item in items if item.get("latency_ms") is not None]
    retrieval_lats = [float(item["retrieval_ms"]) for item in items if item.get("retrieval_ms") is not None]
    generator_lats = [float(item["generator_ms"]) for item in items if item.get("generator_ms") is not None]
    errors = sum(1 for item in items if item.get("error"))
    judge_scores = [float(item["judge"]["combined_score"]) for item in items
                    if item.get("judge") and item["judge"].get("combined_score") is not None]
    cost = sum(float((item.get("usage") or {}).get("cost_usd") or 0.0) for item in items)
    total_tokens = sum(int((item.get("usage") or {}).get("total_tokens") or 0) for item in items)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "pass_rate_ci95": _bootstrap_pass_ci(items),
        "avg_score": round(avg_score, 4),
        "avg_judge_score": round(sum(judge_scores) / len(judge_scores), 4) if judge_scores else None,
        "error_rate": round(errors / total, 4) if total else 0.0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "p50_latency_ms": round(_percentile(latencies, 50), 2) if latencies else None,
        "p95_latency_ms": round(_percentile(latencies, 95), 2) if latencies else None,
        "avg_retrieval_ms": round(sum(retrieval_lats) / len(retrieval_lats), 2) if retrieval_lats else None,
        "avg_generator_ms": round(sum(generator_lats) / len(generator_lats), 2) if generator_lats else None,
        "total_tokens": total_tokens,
        "total_cost_usd": round(cost, 6),
        "avg_cost_per_answer_usd": round(cost / total, 6) if total else 0.0,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_bot: dict[str, list] = defaultdict(list)
    by_bot_category: dict[tuple[str, str], list] = defaultdict(list)
    by_bot_priority: dict[tuple[str, str], list] = defaultdict(list)

    for row in rows:
        bot = row.get("bot", "unknown")
        category = row.get("category", "unknown")
        priority = row.get("priority", "unknown")
        by_bot[bot].append(row)
        by_bot_category[(bot, category)].append(row)
        by_bot_priority[(bot, priority)].append(row)

    return {
        "overall": {bot: _aggregate(items) for bot, items in sorted(by_bot.items())},
        "by_category": {f"{bot}/{cat}": _aggregate(items)
                        for (bot, cat), items in sorted(by_bot_category.items())},
        "by_priority": {f"{bot}/{pri}": _aggregate(items)
                        for (bot, pri), items in sorted(by_bot_priority.items())},
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\nOverall")
    print("| Bot | N | Passed | Pass% | CI95 | Score% | Judge | Err% | p95 ms | $/answer |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for bot, s in summary["overall"].items():
        ci = s.get("pass_rate_ci95")
        ci_text = f"[{ci[0]:.0%},{ci[1]:.0%}]" if ci else "-"
        judge_text = f"{s['avg_judge_score']:.2f}" if s.get("avg_judge_score") is not None else "-"
        p95 = s.get("p95_latency_ms")
        cost = s.get("avg_cost_per_answer_usd") or 0.0
        print(
            f"| {bot} | {s['total']} | {s['passed']} | {s['pass_rate']:.1%} | "
            f"{ci_text} | {s['avg_score']:.1%} | {judge_text} | {s['error_rate']:.1%} | "
            f"{'' if p95 is None else p95} | {cost:.4f} |"
        )

    print("\nBy category")
    print("| Bot/category | N | Pass% | Score% | Judge |")
    print("|---|---:|---:|---:|---:|")
    for key, s in summary["by_category"].items():
        judge_text = f"{s['avg_judge_score']:.2f}" if s.get("avg_judge_score") is not None else "-"
        print(f"| {key} | {s['total']} | {s['pass_rate']:.1%} | {s['avg_score']:.1%} | {judge_text} |")

    print("\nBy priority")
    print("| Bot/priority | N | Pass% | CI95 | Score% |")
    print("|---|---:|---:|---:|---:|")
    for key, s in summary["by_priority"].items():
        ci = s.get("pass_rate_ci95")
        ci_text = f"[{ci[0]:.0%},{ci[1]:.0%}]" if ci else "-"
        print(f"| {key} | {s['total']} | {s['pass_rate']:.1%} | {ci_text} | {s['avg_score']:.1%} |")


def write_summary_sidecar(results_path: Path, summary: dict[str, Any]) -> Path:
    sidecar = results_path.with_suffix(".summary.json")
    sidecar.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return sidecar


def _safe_segment(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text or "unknown").strip("_") or "unknown"


def write_per_category_files(results_path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write one JSONL per category alongside the main results file.

    Layout: <results_path>.parent / <results_stem>-by-category / <category>.jsonl
    Each per-category file is full-fidelity (same shape as the main file),
    so downstream tooling can score one slice without reloading everything.
    """
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cat[row.get("category") or "unknown"].append(row)

    cat_dir = results_path.parent / f"{results_path.stem}-by-category"
    cat_dir.mkdir(parents=True, exist_ok=True)
    for category, items in sorted(by_cat.items()):
        slug = _safe_segment(category)
        write_jsonl(cat_dir / f"{slug}.jsonl", items)
    return cat_dir


# ---------------------------------------------------------------------------
# Row builders shared by run-direct, run-api, score-file
# ---------------------------------------------------------------------------


def _attach_meta_metrics(row: dict[str, Any], case: dict[str, Any], meta: dict[str, Any] | None) -> None:
    if not meta:
        return
    row["usage"] = usage_summary(meta.get("usage"))
    latencies = meta.get("latency_ms") or {}
    if latencies:
        if "retrieval" in latencies:
            row["retrieval_ms"] = latencies["retrieval"]
        if "generator" in latencies:
            row["generator_ms"] = latencies["generator"]
    retrieved = meta.get("retrieved") or []
    if retrieved:
        row["retrieved_slugs"] = _retrieved_slugs_from_meta(meta)
        row["retrieved_chunks"] = _retrieved_chunks_from_meta(meta)
        gold_s = _gold_slugs(case)
        gold_c = _gold_chunks(case)
        row["retrieval"] = retrieval_metrics(
            gold_s,
            row["retrieved_slugs"],
            gold_chunks=gold_c,
            retrieved_chunks=row["retrieved_chunks"],
            ks=(5, 10),
        )


# ---------------------------------------------------------------------------
# Subcommand: run-direct
# ---------------------------------------------------------------------------


def run_direct(args: argparse.Namespace) -> int:
    cases = load_jsonl(args.dataset)
    selected = iter_cases(
        cases,
        set(args.ids or []) or None,
        args.limit,
        exclude_categories=set(getattr(args, "exclude_categories", None) or []) or None,
        include_categories=_include_categories_from_args(args),
    )

    try:
        from src.chat import answer_with_meta
    except Exception as exc:
        print(f"Cannot import local chatbot pipeline: {exc}", file=sys.stderr)
        return 2

    run_id = timestamp()
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(selected, start=1):
        session_id = f"{args.session_prefix}:{run_id}:{case['id']}"
        turns = case.get("turns") or [case["question"]]
        final_answer = ""
        final_meta: dict[str, Any] = {}
        error: str | None = None
        forced_direct_answer = False
        start = time.perf_counter()
        try:
            for turn in turns:
                final_answer, final_meta = answer_with_meta(
                    turn,
                    session_id=session_id,
                    mode=EVAL_CHAT_MODE,
                )
            if len(turns) > 1 and is_still_clarifying(final_answer):
                forced_direct_answer = True
                final_answer, final_meta = answer_with_meta(
                    FORCE_ANSWER_PROMPT,
                    session_id=session_id,
                    mode=EVAL_CHAT_MODE,
                )
        except Exception as exc:
            error = repr(exc)
            final_answer = ""
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        case["_retrieved_source_text"] = _retrieved_source_text_from_meta(final_meta)
        scored = evaluate_answer(
            case,
            final_answer,
            pass_threshold=args.pass_threshold,
            use_judge=getattr(args, "use_judge", False),
        )
        row = {
            "bot": args.bot_name, "mode": "direct",
            "case_id": case["id"], "category": case.get("category"), "priority": case.get("priority"),
            "question": case.get("question") or turns[-1],
            "turns": turns if case.get("turns") else None,
            "answer": final_answer, "latency_ms": latency_ms,
            "forced_mode": EVAL_CHAT_MODE,
            "forced_direct_answer": forced_direct_answer,
            "error": error, **scored,
        }
        _attach_meta_metrics(row, case, final_meta)
        rows.append(row)
        marker = "PASS" if row["passed"] else "FAIL"
        suffix = " [FORCED]" if forced_direct_answer else ""
        print(f"[{index}/{len(selected)}] {case['id']} {marker} score={row['score']:.2f}{suffix}")

    out = args.output or _result_path_for_args(args, "results")
    write_jsonl(out, rows)
    write_per_category_files(out, rows)
    summary = summarize(rows)
    print_summary(summary)
    sidecar = write_summary_sidecar(out, summary)
    print(f"\nWrote results: {out}")
    print(f"Wrote summary: {sidecar}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: run-api
# ---------------------------------------------------------------------------


def run_api(args: argparse.Namespace) -> int:
    if httpx is None:
        print("httpx is required for run-api mode. Install project requirements first.", file=sys.stderr)
        return 2

    api_key = args.api_key or os.getenv("CHAT_API_KEY", "")
    if not api_key:
        print("Missing API key. Pass --api-key or set CHAT_API_KEY.", file=sys.stderr)
        return 2

    cases = load_jsonl(args.dataset)
    selected = iter_cases(
        cases,
        set(args.ids or []) or None,
        args.limit,
        exclude_categories=set(getattr(args, "exclude_categories", None) or []) or None,
        include_categories=_include_categories_from_args(args),
    )
    base = args.base_url.rstrip("/")
    url = base + "/chat" + ("?include_meta=1" if args.with_meta else "")
    rows: list[dict[str, Any]] = []

    print("Note: API mode sends one session_id per case, shared across turns in that case.")

    with httpx.Client(timeout=args.timeout) as client:
        for index, case in enumerate(selected, start=1):
            turns = case.get("turns") or [case["question"]]
            session_id = f"eval-api:{case['id']}"
            final_answer = ""
            final_meta: dict[str, Any] = {}
            error: str | None = None
            status_code: int | None = None
            forced_direct_answer = False
            start = time.perf_counter()

            def _post(turn_text: str) -> tuple[str, dict[str, Any]]:
                response = client.post(
                    url,
                    headers={"X-API-Key": api_key},
                    json={
                        "question": turn_text,
                        "session_id": session_id,
                        "mode": EVAL_CHAT_MODE,
                    },
                )
                nonlocal status_code
                status_code = response.status_code
                response.raise_for_status()
                data = response.json()
                meta = data.get("meta") or {} if args.with_meta else {}
                return data.get("answer", ""), meta

            try:
                for turn in turns:
                    final_answer, final_meta = _post(turn)
                if len(turns) > 1 and is_still_clarifying(final_answer):
                    forced_direct_answer = True
                    final_answer, final_meta = _post(FORCE_ANSWER_PROMPT)
            except Exception as exc:
                error = repr(exc)
                final_answer = final_answer or ""
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            case["_retrieved_source_text"] = _retrieved_source_text_from_meta(final_meta)
            scored = evaluate_answer(
                case,
                final_answer,
                pass_threshold=args.pass_threshold,
                use_judge=getattr(args, "use_judge", False),
            )
            row = {
                "bot": args.bot_name, "mode": "api",
                "case_id": case["id"], "category": case.get("category"), "priority": case.get("priority"),
                "question": case.get("question") or turns[-1],
                "turns": turns if case.get("turns") else None,
                "answer": final_answer, "latency_ms": latency_ms,
                "forced_mode": EVAL_CHAT_MODE,
                "forced_direct_answer": forced_direct_answer,
                "http_status": status_code, "error": error, **scored,
            }
            _attach_meta_metrics(row, case, final_meta)
            rows.append(row)
            marker = "PASS" if row["passed"] else "FAIL"
            suffix = " [FORCED]" if forced_direct_answer else ""
            print(f"[{index}/{len(selected)}] {case['id']} {marker} score={row['score']:.2f}{suffix}")

    out = args.output or _result_path_for_args(args, "results")
    write_jsonl(out, rows)
    write_per_category_files(out, rows)
    summary = summarize(rows)
    print_summary(summary)
    sidecar = write_summary_sidecar(out, summary)
    print(f"\nWrote results: {out}")
    print(f"Wrote summary: {sidecar}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: score-file
# ---------------------------------------------------------------------------


def score_file(args: argparse.Namespace) -> int:
    cases = {case["id"]: case for case in load_jsonl(args.dataset)}
    answers = load_jsonl(args.answers_file)
    excluded = set(args.exclude_categories or [])
    included = _include_categories_from_args(args)
    rows: list[dict[str, Any]] = []

    for index, item in enumerate(answers, start=1):
        case_id = item.get("case_id") or item.get("id")
        if case_id not in cases:
            print(f"Skipping unknown case id: {case_id}", file=sys.stderr)
            continue
        case = cases[case_id]
        if included and case.get("category") not in included:
            continue
        if excluded and case.get("category") in excluded:
            continue
        answer_text = item.get("answer", "")
        bot = item.get("bot") or args.bot_name
        scored = evaluate_answer(
            case,
            answer_text,
            pass_threshold=args.pass_threshold,
            use_judge=getattr(args, "use_judge", False),
        )
        row = {
            "bot": bot, "mode": "score-file",
            "case_id": case_id, "category": case.get("category"), "priority": case.get("priority"),
            "question": case.get("question") or (case.get("turns") or [""])[-1],
            "turns": case.get("turns"),
            "answer": answer_text, "latency_ms": item.get("latency_ms"),
            "error": item.get("error"), **scored,
        }
        rows.append(row)
        status = "PASS" if row["passed"] else "FAIL"
        print(f"[{index}/{len(answers)}] {case_id} {status} score={row['score']:.2f}")

    out = args.output or _result_path_for_args(args, "scored")
    write_jsonl(out, rows)
    write_per_category_files(out, rows)
    summary = summarize(rows)
    print_summary(summary)
    sidecar = write_summary_sidecar(out, summary)
    print(f"\nWrote scored results: {out}")
    print(f"Wrote summary: {sidecar}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: run-retrieval
# ---------------------------------------------------------------------------


def _summarize_retrieval(rows: list[dict[str, Any]], ks: tuple[int, ...]) -> dict[str, Any]:
    if not rows:
        return {"overall": {}, "by_category": {}}
    metric_keys = (
        [f"recall@{k}" for k in ks]
        + ["mrr"]
        + [f"chunk_recall@{k}" for k in ks]
        + ["chunk_mrr"]
        + [f"context_precision@{k}" for k in ks]
        + [f"gold_chunk_coverage@{k}" for k in ks]
        + [f"disease_source_coverage@{k}" for k in ks]
        + [f"drug_source_coverage@{k}" for k in ks]
    )

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(items)
        agg: dict[str, Any] = {"total": n}
        for key in metric_keys:
            values = [float(item.get(key, 0.0)) for item in items if item.get(key) is not None]
            agg[key] = round(sum(values) / len(values), 4) if values else None
        return agg

    by_category: dict[str, list] = defaultdict(list)
    for row in rows:
        by_category[row.get("category", "unknown")].append(row)

    return {
        "overall": aggregate(rows),
        "by_category": {cat: aggregate(items) for cat, items in sorted(by_category.items())},
    }


def _print_retrieval_summary(summary: dict[str, Any]) -> None:
    overall = summary.get("overall", {})
    if not overall:
        print("No retrieval rows.")
        return
    print("\nRetrieval — overall")
    keys = [k for k in overall.keys() if k != "total"]
    print("| Total | " + " | ".join(keys) + " |")
    print("|---:|" + "|".join(["---:"] * len(keys)) + "|")
    cells = []
    for k in keys:
        v = overall.get(k)
        cells.append("-" if v is None else (f"{v:.2%}" if k.startswith(("recall", "context_precision", "chunk_recall")) else f"{v:.4f}"))
    print(f"| {overall['total']} | " + " | ".join(cells) + " |")

    print("\nRetrieval — by category")
    print("| Category | Total | " + " | ".join(keys) + " |")
    print("|---|---:|" + "|".join(["---:"] * len(keys)) + "|")
    for cat, stats in summary["by_category"].items():
        cells = []
        for k in keys:
            v = stats.get(k)
            cells.append("-" if v is None else (f"{v:.2%}" if k.startswith(("recall", "context_precision", "chunk_recall")) else f"{v:.4f}"))
        print(f"| {cat} | {stats['total']} | " + " | ".join(cells) + " |")


def run_retrieval(args: argparse.Namespace) -> int:
    cases = load_jsonl(args.dataset)
    selected = iter_cases(
        cases,
        set(args.ids or []) or None,
        args.limit,
        exclude_categories=set(getattr(args, "exclude_categories", None) or []) or None,
        include_categories=_include_categories_from_args(args),
    )

    try:
        from src.chat.retrieval.service import hybrid_search
    except Exception as exc:
        print(f"Cannot import retrieval service: {exc}", file=sys.stderr)
        return 2

    ks = tuple(sorted(set(args.ks)))
    top_k = max(ks)
    rows: list[dict[str, Any]] = []

    for index, case in enumerate(selected, start=1):
        gold_s = _gold_slugs(case)
        gold_c = _gold_chunks(case)
        if not gold_s and not gold_c:
            print(f"[{index}/{len(selected)}] {case['id']} SKIP no source_docs / gold_chunks")
            continue
        query = case.get("question") or (case.get("turns") or [""])[-1]
        if not query:
            print(f"[{index}/{len(selected)}] {case['id']} SKIP no query")
            continue

        start = time.perf_counter()
        error = None
        retrieved_slugs: list[str] = []
        retrieved_chunks: list[str] = []
        try:
            hits = hybrid_search(query, top_k=top_k)
            retrieved_slugs = [getattr(h, "source_slug", "") or "" for h in hits]
            retrieved_chunks = [getattr(h, "chunk_id", "") or "" for h in hits]
        except Exception as exc:
            error = repr(exc)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        metrics = retrieval_metrics(
            gold_s, retrieved_slugs,
            gold_chunks=gold_c, retrieved_chunks=retrieved_chunks, ks=ks,
        )

        row = {
            "bot": args.bot_name, "mode": "retrieval",
            "case_id": case["id"], "category": case.get("category"), "priority": case.get("priority"),
            "query": query,
            "gold_slugs": gold_s, "retrieved_slugs": retrieved_slugs,
            "gold_chunks": gold_c, "retrieved_chunks": retrieved_chunks,
            "latency_ms": latency_ms, "error": error,
            **metrics,
        }
        rows.append(row)
        marker = "HIT" if metrics.get(f"recall@{ks[0]}") else "MISS"
        print(f"[{index}/{len(selected)}] {case['id']} {marker} mrr={metrics['mrr']:.2f}")

    out = args.output or _result_path_for_args(args, "retrieval")
    write_jsonl(out, rows)
    write_per_category_files(out, rows)
    summary = _summarize_retrieval(rows, ks)
    _print_retrieval_summary(summary)
    sidecar = out.with_suffix(".summary.json")
    sidecar.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote retrieval results: {out}")
    print(f"Wrote summary: {sidecar}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: compare
# ---------------------------------------------------------------------------


def compare(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    for path in args.result_files:
        rows.extend(load_jsonl(path))
    if not rows:
        print("No result rows to compare.", file=sys.stderr)
        return 1
    summary = summarize(rows)
    print_summary(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote comparison summary: {args.output}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--bot-name", default="local-rag")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--exclude-categories", nargs="*", default=[],
                        help="Categories to skip (e.g. diagnostic_flow when comparing other bots).")
    parser.add_argument("--pass-threshold", type=float, default=DEFAULT_PASS_THRESHOLD)


def add_judge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--use-judge",
        action="store_true",
        help="Run LLM judges (faithfulness, correctness, relevance) on each answer.",
    )


def build_parser(category: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Evaluate benchmark cases for category `{category}`.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    direct = sub.add_parser("run-direct", help="Run the local pipeline in-process.")
    add_common_run_args(direct)
    add_judge_args(direct)
    direct.add_argument("--session-prefix", default="eval")
    direct.set_defaults(func=run_direct)

    api = sub.add_parser("run-api", help="Call a running /chat server.")
    add_common_run_args(api)
    add_judge_args(api)
    api.add_argument("--base-url", default="http://localhost:8000")
    api.add_argument("--api-key")
    api.add_argument("--timeout", type=float, default=120.0)
    api.add_argument(
        "--with-meta", action="store_true",
        help="Send include_meta=1 so the server returns usage / retrieved / latency.",
    )
    api.set_defaults(func=run_api)

    scored = sub.add_parser("score-file", help="Score saved answers from another bot.")
    add_common_run_args(scored)
    add_judge_args(scored)
    scored.add_argument("--answers-file", type=Path, required=True)
    scored.set_defaults(func=score_file)

    retrieval = sub.add_parser(
        "run-retrieval",
        help="Retrieval-only: doc + chunk recall@k, MRR, context precision.",
    )
    add_common_run_args(retrieval)
    retrieval.add_argument("--ks", nargs="+", type=int, default=[5, 10])
    retrieval.set_defaults(func=run_retrieval)

    return parser


def main_for_category(category: str, argv: list[str] | None = None) -> int:
    args = build_parser(category).parse_args(argv)
    args.category = category
    args.include_categories = [category]
    return args.func(args)
