import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

from paper_agent.search import collect_candidates
from paper_agent.filter import filter_process_metasurface_must, pick_top_n
from paper_agent.summarize import enrich_with_summaries
from paper_agent.email_sender import send_digest_email


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


JST = timezone(timedelta(hours=9))


def job(days_back: int, n: int):
    logger.info("Agent started.")

    # 1) Collect candidates (arXiv + OpenAlex + Crossref)
    candidates = collect_candidates(days_back=days_back)
    logger.info(f"Collected {len(candidates)} candidates")

    # 2) MUST: metasurface + process-focused
    filtered = filter_process_metasurface_must(candidates)
    logger.info(f"Filtered to {len(filtered)} (metasurface MUST + process-focused MUST)")

    # 3) pick N
    selected = pick_top_n(filtered, n=n)
    logger.info(f"Selected {len(selected)} papers")

    if not selected:
        logger.warning("No papers selected. Sending a notice email.")
        send_digest_email(
            papers=[],
            subject=f"[Paper Digest] No hits today (metasurface MUST) {datetime.now(JST).strftime('%Y-%m-%d')}",
            notice="今日は条件（metasurface MUST + プロセス特化）に合う論文が見つかりませんでした。days_backを増やすとヒットしやすいです。",
        )
        return

    # 4) Summarize (OpenAI if available; fallback to abstract truncation)
    enriched = enrich_with_summaries(selected)

    # 5) Send email
    subject = f"[Paper Digest] metasurface x process (Top {len(enriched)}) {datetime.now(JST).strftime('%Y-%m-%d')}"
    send_digest_email(papers=enriched, subject=subject)
    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Daily Paper Digest (metasurface MUST, process-focused)")
    parser.add_argument("--now", action="store_true", help="Run immediately and exit")
    parser.add_argument("--days-back", type=int, default=365, help="Lookback window in days")
    parser.add_argument("--n", type=int, default=5, help="Number of papers to send")
    parser.add_argument("--test-email", action="store_true", help="Send a test email (no external search)")
    args = parser.parse_args()

    if args.test_email:
        logger.info("Sending test email...")
        dummy = [{
            "title": "Test: TiO2 metasurface fabrication via ALD and plasma etching",
            "authors": ["Taro Yamada", "Hanako Suzuki"],
            "published": datetime.now(JST).strftime("%Y-%m-%d"),
            "source": "TEST",
            "link": "https://example.com",
            "doi": None,
            "abstract": "This is a dummy abstract.",
            "summary_5_10_lines": "1) 背景: テスト\n2) 目的: テスト\n3) 方法: テスト\n4) 結果: テスト\n5) 意義: テスト\n6) 次: テスト",
        }]
        send_digest_email(papers=dummy, subject="[Paper Digest] TEST", notice="これはテストメールです。")
        return

    if args.now:
        job(days_back=args.days_back, n=args.n)
        return

    # If you want local scheduling, do it with OS scheduler / cron / Task Scheduler.
    # In GitHub Actions, schedule triggers the run, so this mode is rarely used.
    logger.error("No mode specified. Use --now (Actions) or --test-email.")
    sys.exit(2)


if __name__ == "__main__":
    main()


