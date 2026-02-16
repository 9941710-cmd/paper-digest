import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

from paper_agent.search import collect_candidates
from paper_agent.filtering import filter_process_metasurface_must, pick_top_n
from paper_agent.summarize import enrich_with_summaries
from paper_agent.email_sender import send_digest_email
from paper_agent.history import filter_new, update_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

DEFAULT_DAYS_BACK = 365   # ★ 1年以内


def job(days_back: int, n: int):
    logger.info("Agent started.")

    # 1) Collect candidates (arXiv + OpenAlex + Crossref) within days_back
    candidates = collect_candidates(days_back=days_back)
    logger.info(f"Collected {len(candidates)} candidates")

    # 2) MUST: metasurface + process-focused
    filtered = filter_process_metasurface_must(candidates)
    logger.info(f"Filtered to {len(filtered)} (metasurface MUST + process-focused MUST)")

    # 3) pick top N (by score)
    selected = pick_top_n(filtered, n=n)
    logger.info(f"Selected {len(selected)} papers (before history)")

    # 4) remove previously sent (persistent)
    selected = filter_new(selected)
    logger.info(f"After removing previously sent: {len(selected)}")

    if not selected:
        logger.warning("No NEW papers selected. Sending a notice email.")
        send_digest_email(
            papers=[],
            subject=f"[Paper Digest] No new hits (≤{days_back} days) {datetime.now(JST).strftime('%Y-%m-%d')}",
            notice="今日は新規の該当論文が見つかりませんでした（過去送付済みを除外）。条件緩和やdays_back拡大でヒットしやすくなります。",
        )
        return

    # 5) Summarize (OpenAI if available; fallback)
    enriched = enrich_with_summaries(selected)

    # 6) Send email
    subject = f"[Paper Digest] metasurface x process (Top {len(enriched)}) {datetime.now(JST).strftime('%Y-%m-%d')}"
    send_digest_email(papers=enriched, subject=subject)

    # 7) Update history AFTER successful send
    update_history(selected)
    logger.info("History updated. Done.")


def main():
    parser = argparse.ArgumentParser(description="Daily Paper Digest (metasurface MUST, process-focused, no-duplicates)")
    parser.add_argument("--now", action="store_true", help="Run immediately and exit")
    parser.add_argument("--days-back", type=int, default=DEFAULT_DAYS_BACK, help="Lookback window in days (default: 365)")
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

    logger.error("No mode specified. Use --now (Actions) or --test-email.")
    sys.exit(2)


if __name__ == "__main__":
    main()
