import schedule
import time
import logging
import argparse
import sys
from paper_agent import config, search, filter, summarize, email_sender

# Configure logging
config.configure_logging()
logger = logging.getLogger(__name__)

def job():
    logger.info("Starting scheduled job...")
    try:
        # 1. Search
        logger.info("Searching for papers...")
        papers = search.get_papers()
        
        if not papers:
            logger.info("No papers found.")
            return

        # 2. Filter & Score
        logger.info("Filtering and scoring papers...")
        top_papers = filter.filter_papers(papers, top_n=5)
        
        # 3. Summarize
        logger.info(f"Summarizing {len(top_papers)} papers...")
        summarized_papers = summarize.summarize_papers(top_papers)
        
        # 4. Email
        logger.info("Sending email...")
        email_sender.send_email(summarized_papers)
        
        logger.info("Job completed successfully.")
        
    except Exception as e:
        logger.error(f"Job failed: {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="Daily Paper Search & Email Agent")
    parser.add_argument("--now", action="store_true", help="Run the job immediately and exit")
    parser.add_argument("--test-email", action="store_true", help="Send a test email with dummy data")
    args = parser.parse_args()

    logger.info("Agent started (CLI Mode).")
    

    if args.test_email:
        logger.info("Sending test email...")
        dummy_papers = [{
            "title": "Test Paper Title: Advanced ALD Process for High-k Dielectrics",
            "title_jp": "高誘電率膜のための先進的ALDプロセス",
            "link": "https://example.com",
            "authors": ["Taro Yamada", "Hanako Suzuki"],
            "source": "Test Source",
            "published": "2024-01-01",
            "background": "次世代メモリデバイスでは、EOTスケーリングに伴い、高アスペクト比構造への均一なHigh-k膜形成が求められている。しかし、従来の高温プロセスでは下地へのダメージが課題であった。",
            "purpose": "低温（200℃以下）において、高い膜密度と低いリーク電流を実現する新規プラズマALDプロセスを開発する。",
            "conditions": "・装置: 容量結合型PEALD装置 (13.56 MHz)\n・前駆体: TDMAT (50℃), 酸化剤: O2プラズマ\n・基板温度: 150～250℃\n・圧力: 100 Pa\n・RFパワー: 50-200 W\n・サイクル: Pulse 2s / Purge 5s / RF 5s / Purge 5s",
            "methods": "XPS (組成分析), SE (膜厚・屈折率), XRR (密度), C-V測定 (電気的特性)",
            "results": "1. 150℃の低温において、GPC 1.2 A/cycleを達成。\n2. 屈折率は2.1 (at 633nm) とバルクに近い値を示した。\n3. アスペクト比 1:10 のトレンチにおいて、段差被覆率 95% 以上を確認。\n4. リーク電流は 1E-8 A/cm2 (@ 1MV/cm) まで低減。",
            "significance": "熱ダメージに弱い有機基板やバックエンドプロセスへの適用を可能にする低温高品質成膜技術を確立した。",
            "implications": "現状の装置構成で即座に適用可能であるが、スループット向上のためにパージ時間の短縮検討が必要である。3D NANDの絶縁膜としての応用が期待できる。"
        }]
        email_sender.send_email(dummy_papers)
        return

    job()




if __name__ == "__main__":
    main()
