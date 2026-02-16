import logging
import os
import time
from typing import Dict, List

from openai import OpenAI


logger = logging.getLogger(__name__)

SYSTEM = """You are a research assistant for semiconductor / nanofabrication processes.
Write a 5–10 line Japanese summary focused on PROCESS details.
Must include: (1) what was fabricated/processed, (2) key materials (e.g., TiO2), (3) key steps (ALD/etch/lithography), (4) key results, (5) why it matters.
Avoid generic fluff. If abstract is missing, infer only cautiously and say "要旨情報が少ないため推定".
"""

def _fallback_summary(p: Dict) -> str:
    abs_ = (p.get("abstract") or "").strip()
    if not abs_:
        return "要旨情報が少ないため、タイトルとメタデータからの推定のみです。プロセス詳細はリンク先をご確認ください。"
    # crude 5-10 lines: split by sentences
    s = abs_.replace("\n", " ")
    # keep short
    if len(s) > 900:
        s = s[:900] + "…"
    return s


def _openai_client():
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    return OpenAI(api_key=key)


def _summarize_one(client: OpenAI, p: Dict) -> str:
    title = p.get("title","")
    abstract = p.get("abstract","")
    source = p.get("source","")
    doi = p.get("doi")
    link = p.get("link")

    user = f"""タイトル: {title}
ソース: {source}
DOI: {doi}
リンク: {link}

要旨:
{abstract if abstract else "(要旨なし)"}"""

    # Try a cheap model name; you can change later
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def enrich_with_summaries(papers: List[Dict]) -> List[Dict]:
    client = _openai_client()
    out = []
    for i, p in enumerate(papers, start=1):
        logger.info(f"Summarizing {i}/{len(papers)}: {p.get('title','')[:80]}")
        p2 = dict(p)
        if client is None:
            p2["summary_5_10_lines"] = _fallback_summary(p2)
            out.append(p2)
            continue

        # simple retry
        for attempt in range(3):
            try:
                p2["summary_5_10_lines"] = _summarize_one(client, p2)
                break
            except Exception as ex:
                wait = 1.0 + attempt * 2.0
                logger.warning(f"OpenAI summarize failed (attempt {attempt+1}): {ex}. wait={wait}s")
                time.sleep(wait)
                p2["summary_5_10_lines"] = _fallback_summary(p2)
        out.append(p2)
    return out
