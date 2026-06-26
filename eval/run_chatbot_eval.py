"""Top-level CLI for the medical chatbot evaluation suite.

Usage:
    python eval/run_chatbot_eval.py disease_info run-direct --limit 5 --concurrency 4
    python eval/run_chatbot_eval.py drug_info run-direct --limit 5
    python eval/run_chatbot_eval.py health_insurance_info run-api --limit 5
    python eval/run_chatbot_eval.py emergency score-file --answers-file path.jsonl
    python eval/run_chatbot_eval.py drug_info run-retrieval --limit 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Make the project root importable when the script is run as a file.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _build_top_level_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_chatbot_eval",
        description="Medical chatbot evaluation entry point.",
    )
    sub = parser.add_subparsers(dest="category", required=True)
    sub.add_parser(
        "drug_info",
        add_help=False,
        help="Drug-information category evaluator.",
    )
    sub.add_parser(
        "disease_info",
        add_help=False,
        help="Disease-information category evaluator.",
    )
    sub.add_parser(
        "health_insurance_info",
        add_help=False,
        help="Health-insurance category evaluator.",
    )
    sub.add_parser(
        "emergency",
        add_help=False,
        help="Emergency-triage category evaluator.",
    )
    sub.add_parser("base", add_help=False, help="Default category evaluator.")
    return parser


def main(argv: list[str] | None = None) -> int:
    from eval import core

    top = _build_top_level_parser()
    top_args, remaining = top.parse_known_args(argv)
    return core.main_for_category(top_args.category, remaining)


if __name__ == "__main__":
    raise SystemExit(main())
