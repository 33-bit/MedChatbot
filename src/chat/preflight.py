from __future__ import annotations

RATE_LIMIT_MSG = "Bạn đang hỏi quá nhanh. Vui lòng chờ 1 phút rồi thử lại."


def preflight(
    question: str,
    session_id: str,
    guardrail_reply,
    check_rate_limit,
    regex_check,
    check_llm_quota,
) -> str | None:
    """Run non-LLM checks. Return a short-circuit reply, or None to continue."""
    if not check_rate_limit(session_id):
        return RATE_LIMIT_MSG

    regex_guard = regex_check(question)
    if regex_guard is not None:
        return guardrail_reply(session_id, question, regex_guard["verdict"])

    allowed, quota_msg = check_llm_quota(session_id)
    if not allowed:
        return quota_msg

    return None
