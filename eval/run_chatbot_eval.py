#!/usr/bin/env python3
"""
Run and score the medical chatbot benchmark.

Typical commands:
  python eval/run_chatbot_eval.py run-api --api-key "$CHAT_API_KEY"
  python eval/run_chatbot_eval.py run-direct
  python eval/run_chatbot_eval.py score-file --answers-file other_bot_answers.jsonl
  python eval/run_chatbot_eval.py compare eval/results/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - only needed for API mode
    httpx = None  # type: ignore[assignment]


DEFAULT_DATASET = Path("eval/medical_qa_benchmark.jsonl")
DEFAULT_OUT_DIR = Path("eval/results")
DEFAULT_PASS_THRESHOLD = 0.75
CITATION_RE = re.compile(r"\[\d+\]")
EMERGENCY_TERMS = [
    "cap cuu",
    "goi 115",
    "den benh vien",
    "di vien",
    "di kham ngay",
    "co so y te",
    "khan cap",
    "ngay lap tuc",
]


def normalize(text: str) -> str:
    text = text.casefold().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains(haystack: str, needle: str) -> bool:
    return normalize(needle) in normalize(haystack)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def iter_cases(cases: list[dict[str, Any]], ids: set[str] | None, limit: int | None):
    selected = [case for case in cases if not ids or case["id"] in ids]
    if limit is not None:
        selected = selected[:limit]
    return selected


def score_answer(
    case: dict[str, Any],
    answer: str,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    for keyword in case.get("must_include", []):
        checks.append({
            "type": "must_include",
            "target": keyword,
            "passed": contains(answer, keyword),
        })

    for group in case.get("must_include_any", []):
        matched = [keyword for keyword in group if contains(answer, keyword)]
        checks.append({
            "type": "must_include_any",
            "target": group,
            "matched": matched,
            "passed": bool(matched),
        })

    forbidden_hits = [
        keyword for keyword in case.get("must_not_include", [])
        if contains(answer, keyword)
    ]
    checks.append({
        "type": "must_not_include",
        "target": case.get("must_not_include", []),
        "matched": forbidden_hits,
        "passed": not forbidden_hits,
    })

    if case.get("requires_citation", False):
        checks.append({
            "type": "requires_citation",
            "target": "[n]",
            "passed": bool(CITATION_RE.search(answer)),
        })

    if case.get("requires_emergency_advice", False):
        normalized_answer = normalize(answer)
        matched = [term for term in EMERGENCY_TERMS if term in normalized_answer]
        checks.append({
            "type": "requires_emergency_advice",
            "target": EMERGENCY_TERMS,
            "matched": matched,
            "passed": bool(matched),
        })

    passed_checks = sum(1 for check in checks if check["passed"])
    score = passed_checks / len(checks) if checks else 0.0
    hard_fail = bool(forbidden_hits)
    return {
        "score": round(score, 4),
        "passed": (score >= pass_threshold) and not hard_fail,
        "hard_fail": hard_fail,
        "checks": checks,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_bot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_bot_category: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        bot = row.get("bot", "unknown")
        category = row.get("category", "unknown")
        by_bot[bot].append(row)
        by_bot_category[(bot, category)].append(row)

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(items)
        passed = sum(1 for item in items if item.get("passed"))
        avg_score = sum(float(item.get("score", 0.0)) for item in items) / total if total else 0.0
        avg_latency = [
            float(item["latency_ms"]) for item in items
            if item.get("latency_ms") is not None
        ]
        return {
            "total": total,
            "passed": passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "avg_score": round(avg_score, 4),
            "avg_latency_ms": round(sum(avg_latency) / len(avg_latency), 2) if avg_latency else None,
        }

    return {
        "overall": {bot: aggregate(items) for bot, items in sorted(by_bot.items())},
        "by_category": {
            f"{bot}/{category}": aggregate(items)
            for (bot, category), items in sorted(by_bot_category.items())
        },
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\nOverall")
    print("| Bot | Total | Passed | Pass rate | Avg score | Avg latency ms |")
    print("|---|---:|---:|---:|---:|---:|")
    for bot, stats in summary["overall"].items():
        latency = stats["avg_latency_ms"]
        latency_text = "" if latency is None else str(latency)
        print(
            f"| {bot} | {stats['total']} | {stats['passed']} | "
            f"{stats['pass_rate']:.2%} | {stats['avg_score']:.2%} | {latency_text} |"
        )

    print("\nBy category")
    print("| Bot/category | Total | Passed | Pass rate | Avg score |")
    print("|---|---:|---:|---:|---:|")
    for key, stats in summary["by_category"].items():
        print(
            f"| {key} | {stats['total']} | {stats['passed']} | "
            f"{stats['pass_rate']:.2%} | {stats['avg_score']:.2%} |"
        )


def result_path(out_dir: Path, bot_name: str, suffix: str = "results") -> Path:
    safe_bot = re.sub(r"[^A-Za-z0-9_.-]+", "_", bot_name).strip("_") or "bot"
    return out_dir / f"{safe_bot}-{suffix}-{timestamp()}.jsonl"


def run_direct(args: argparse.Namespace) -> int:
    cases = load_jsonl(args.dataset)
    selected = iter_cases(cases, set(args.ids or []) or None, args.limit)

    try:
        from src.chat import answer as local_answer
    except Exception as exc:
        print(f"Cannot import local chatbot pipeline: {exc}", file=sys.stderr)
        return 2

    run_id = timestamp()
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(selected, start=1):
        session_id = f"{args.session_prefix}:{run_id}:{case['id']}"
        turns = case.get("turns") or [case["question"]]
        final_answer = ""
        error = None
        start = time.perf_counter()
        try:
            for turn in turns:
                final_answer = local_answer(turn, session_id=session_id)
        except Exception as exc:  # keep the benchmark going
            error = repr(exc)
            final_answer = ""
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        scored = score_answer(case, final_answer, args.pass_threshold)
        row = {
            "bot": args.bot_name,
            "mode": "direct",
            "case_id": case["id"],
            "category": case.get("category"),
            "priority": case.get("priority"),
            "question": case.get("question") or turns[-1],
            "turns": turns if case.get("turns") else None,
            "answer": final_answer,
            "latency_ms": latency_ms,
            "error": error,
            **scored,
        }
        rows.append(row)
        status = "PASS" if row["passed"] else "FAIL"
        print(f"[{index}/{len(selected)}] {case['id']} {status} score={row['score']:.2f}")

    out = args.output or result_path(args.out_dir, args.bot_name)
    write_jsonl(out, rows)
    summary = summarize(rows)
    print_summary(summary)
    print(f"\nWrote results: {out}")
    return 0


def run_api(args: argparse.Namespace) -> int:
    if httpx is None:
        print("httpx is required for run-api mode. Install project requirements first.", file=sys.stderr)
        return 2

    api_key = args.api_key or os.getenv("CHAT_API_KEY", "")
    if not api_key:
        print("Missing API key. Pass --api-key or set CHAT_API_KEY.", file=sys.stderr)
        return 2

    cases = load_jsonl(args.dataset)
    selected = iter_cases(cases, set(args.ids or []) or None, args.limit)
    url = args.base_url.rstrip("/") + "/chat"
    rows: list[dict[str, Any]] = []

    print("Note: /chat derives session_id from API key, so API mode shares one session across cases.")
    print("Use run-direct for isolated per-case sessions.")

    with httpx.Client(timeout=args.timeout) as client:
        for index, case in enumerate(selected, start=1):
            turns = case.get("turns") or [case["question"]]
            final_answer = ""
            error = None
            status_code = None
            start = time.perf_counter()
            try:
                for turn in turns:
                    response = client.post(
                        url,
                        headers={"X-API-Key": api_key},
                        json={"question": turn},
                    )
                    status_code = response.status_code
                    response.raise_for_status()
                    data = response.json()
                    final_answer = data.get("answer", "")
            except Exception as exc:
                error = repr(exc)
                final_answer = final_answer or ""
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            scored = score_answer(case, final_answer, args.pass_threshold)
            row = {
                "bot": args.bot_name,
                "mode": "api",
                "case_id": case["id"],
                "category": case.get("category"),
                "priority": case.get("priority"),
                "question": case.get("question") or turns[-1],
                "turns": turns if case.get("turns") else None,
                "answer": final_answer,
                "latency_ms": latency_ms,
                "http_status": status_code,
                "error": error,
                **scored,
            }
            rows.append(row)
            status = "PASS" if row["passed"] else "FAIL"
            print(f"[{index}/{len(selected)}] {case['id']} {status} score={row['score']:.2f}")

    out = args.output or result_path(args.out_dir, args.bot_name)
    write_jsonl(out, rows)
    summary = summarize(rows)
    print_summary(summary)
    print(f"\nWrote results: {out}")
    return 0


def score_file(args: argparse.Namespace) -> int:
    cases = {case["id"]: case for case in load_jsonl(args.dataset)}
    answers = load_jsonl(args.answers_file)
    rows: list[dict[str, Any]] = []

    for index, item in enumerate(answers, start=1):
        case_id = item.get("case_id") or item.get("id")
        if case_id not in cases:
            print(f"Skipping unknown case id: {case_id}", file=sys.stderr)
            continue
        case = cases[case_id]
        answer = item.get("answer", "")
        bot = item.get("bot") or args.bot_name
        scored = score_answer(case, answer, args.pass_threshold)
        row = {
            "bot": bot,
            "mode": "score-file",
            "case_id": case_id,
            "category": case.get("category"),
            "priority": case.get("priority"),
            "question": case.get("question") or (case.get("turns") or [""])[-1],
            "turns": case.get("turns"),
            "answer": answer,
            "latency_ms": item.get("latency_ms"),
            "error": item.get("error"),
            **scored,
        }
        rows.append(row)
        status = "PASS" if row["passed"] else "FAIL"
        print(f"[{index}/{len(answers)}] {case_id} {status} score={row['score']:.2f}")

    out = args.output or result_path(args.out_dir, args.bot_name, suffix="scored")
    write_jsonl(out, rows)
    summary = summarize(rows)
    print_summary(summary)
    print(f"\nWrote scored results: {out}")
    return 0


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


def add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--bot-name", default="local-rag")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--pass-threshold", type=float, default=DEFAULT_PASS_THRESHOLD)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and score chatbot evaluation benchmark.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    direct = subparsers.add_parser("run-direct", help="Run the local src.chat pipeline with isolated sessions.")
    add_common_run_args(direct)
    direct.add_argument("--session-prefix", default="eval")
    direct.set_defaults(func=run_direct)

    api = subparsers.add_parser("run-api", help="Call a running /chat API server.")
    add_common_run_args(api)
    api.add_argument("--base-url", default="http://localhost:8000")
    api.add_argument("--api-key")
    api.add_argument("--timeout", type=float, default=120.0)
    api.set_defaults(func=run_api)

    scored = subparsers.add_parser("score-file", help="Score saved answers from another chatbot.")
    add_common_run_args(scored)
    scored.add_argument("--answers-file", type=Path, required=True)
    scored.set_defaults(func=score_file)

    comp = subparsers.add_parser("compare", help="Compare one or more result JSONL files.")
    comp.add_argument("result_files", type=Path, nargs="+")
    comp.add_argument("--output", type=Path)
    comp.set_defaults(func=compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
