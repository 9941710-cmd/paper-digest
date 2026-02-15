import argparse
import datetime as dt
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

# あなたのGmail API送信（完成したやつ）
from paper_agent import email_sender

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# -----------------------------
# 設定
# -----------------------------
MAX_PAPERS_PER_DAY = 5

# 優先トピック（日本語/英語混在OK）
KEYWORDS = [
    # ALD / PEALD
    "ALD", "atomic layer deposition", "PEALD", "plasma-enhanced ALD",
    # NLD-RIE（ユーザーがよく使う表記を広めに）
    "NLD", "NLD-RIE", "neutral loop discharge", "neutral loop discharge RIE",
    "RIE", "reactive ion etching", "plasma etching",
    # metasurface process
    "metasurface", "metasurfaces", "meta-surface",
    "nanofabrication", "lithography", "nanoimprint", "NIL",
    # TiO2 & microstructure
    "TiO2", "titania", "titanium dioxide",
    "nanostructure", "nanostructured", "microstructure", "micro-structure",
    "high aspect ratio", "HAR", "conformal", "step coverage",
]

# arXivクエリ：材料/ナノ/応用物理あたりを広めに
ARXIV_CATEGORIES = [
    "cond-mat.mtrl-sci",
    "cond-mat.mes-hall",
    "physics.app-ph",
    "physics.ins-det",
    "eess.SP",
    "eess.SY",
    "cs.NI",
]

# -----------------------------
# データ構造
# -----------------------------
@dataclass
class Paper:
    title: str
    link: str
    authors: List[str]
    published: str  # ISO string
    abstract: str
    source: str = "arXiv"

    # 要約結果（入ればメールに載る）
    background: str = ""
    purpose: str = ""
    conditions: str = ""
    methods: str = ""
    results: str = ""
    significance: str = ""
    implications: str = ""


# -----------------------------
# Utility
# -----------------------------
def _normalize_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _keyword_score(text: str) -> int:
    t = text.lower()
    score = 0
    for kw in KEYWORDS:
        if kw.lower() in t:
            # 重要語を少し重め（ざっくり）
            if kw.lower() in ["ald", "peald", "tio2", "metasurface", "nld-rie", "nld"]:
                score += 5
            else:
                score += 1
    return score


def _parse_arxiv_atom(xml_bytes: bytes) -> List[Paper]:
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(xml_bytes)
    papers: List[Paper] = []
    for entry in root.findall("atom:entry", ns):
        title = _normalize_text(entry.findtext("atom:title", default="", namespaces=ns))
        summary = _normalize_text(entry.findtext("atom:summary", default="", namespaces=ns))

        link = ""
        for l in entry.findall("atom:link", ns):
            if l.attrib.get("rel") == "alternate":
                link = l.attrib.get("href", "")
                break
        if not link:
            link = entry.findtext("atom:id", default="", namespaces=ns)

        authors = []
        for a in entry.findall("atom:author", ns):
            name = a.findtext("atom:name", default="", namespaces=ns)
            if name:
                authors.append(name)

        published = entry.findtext("atom:published", default="", namespaces=ns)

        if title and link:
            papers.append(
                Paper(
                    title=title,
                    link=link,
                    authors=authors,
                    published=published,
                    abstract=summary,
                    source="arXiv",
                )
            )
    return papers


def fetch_arxiv_papers(days_back: int = 3, max_results: int = 100) -> List[Paper]:
    """
    直近days_back日くらいの範囲で、カテゴリ横断で拾う（arXiv APIは範囲指定が弱いので多めに取得→後で絞る）。
    """
    # キーワードは arXiv query にも入れて、ある程度絞る（完全一致じゃなくても拾える）
    # arXivクエリ例：all:"atomic layer deposition" OR all:ALD ...
    kw_query = " OR ".join([f'all:"{kw}"' if " " in kw else f"all:{kw}" for kw in KEYWORDS[:25]])
    cat_query = " OR ".join([f"cat:{c}" for c in ARXIV_CATEGORIES])
    query = f"({cat_query}) AND ({kw_query})"

    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    logger.info("Fetching arXiv: %s", url)

    req = urllib.request.Request(url, headers={"User-Agent": "paper-digest-agent/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml_bytes = resp.read()

    papers = _parse_arxiv_atom(xml_bytes)

    # days_backで絞る
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=days_back)
    filtered: List[Paper] = []
    for p in papers:
        try:
            # published: 2026-02-15T12:34:56Z
            pub_dt = dt.datetime.fromisoformat(p.published.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            pub_dt = None
        if pub_dt is None or pub_dt >= cutoff:
            filtered.append(p)

    logger.info("Fetched %d papers, filtered to %d (days_back=%d)", len(papers), len(filtered), days_back)
    return filtered


def select_top_papers(papers: List[Paper], n: int = MAX_PAPERS_PER_DAY) -> List[Paper]:
    scored = []

    for p in papers:
        text = f"{p.title}\n{p.abstract}"
        score = _keyword_score(text)
        pub_key = p.published or ""
        scored.append((score, pub_key, p))

    # スコア優先 → 新しい順
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    selected = []
    seen = set()

    # まずキーワードヒット優先
    for score, _, p in scored:
        if score > 0:
            if p.title not in seen:
                selected.append(p)
                seen.add(p.title)
            if len(selected) >= n:
                return selected

    # 足りなければ最新から埋める
    for _, _, p in scored:
        if p.title not in seen:
            selected.append(p)
            seen.add(p.title)
        if len(selected) >= n:
            break

    return selected



# -----------------------------
# 要約（OpenAIは任意）
# -----------------------------
def summarize_with_openai(p: Paper) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # ない場合はabstractから最低限の“整形”だけ
        p.background = ""
        p.purpose = ""
        p.conditions = ""
        p.methods = ""
        p.results = _normalize_text(p.abstract)[:1200]
        p.significance = ""
        p.implications = ""
        return

    # 遅延import（APIキーなしでも動くように）
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    system = (
        "あなたは半導体プロセス分野の研究アシスタントです。"
        "入力された論文のタイトルと要旨から、研究者向けに日本語で技術要約してください。"
        "推測は最小限にし、要旨にある事実を中心に、具体的に書いてください。"
        "必ず次の見出しで出力してください：\n"
        "背景:\n目的:\n実験条件:\n手法:\n結果:\n意義:\n今後の示唆:\n"
    )

    user = f"タイトル:\n{p.title}\n\n要旨:\n{p.abstract}\n\nリンク:\n{p.link}\n"

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    text = resp.choices[0].message.content or ""

    # 見出しでざっくり分解
    def grab(section: str) -> str:
        m = re.search(rf"{section}:\s*(.*?)(?=\n[A-Za-zぁ-んァ-ン一-龥]+?:|\Z)", text, re.S)
        return _normalize_text(m.group(1)) if m else ""

    p.background = grab("背景")
    p.purpose = grab("目的")
    p.conditions = grab("実験条件")
    p.methods = grab("手法")
    p.results = grab("結果")
    p.significance = grab("意義")
    p.implications = grab("今後の示唆")


# -----------------------------
# ジョブ本体
# -----------------------------
def job() -> None:
    papers = fetch_arxiv_papers(days_back=14, max_results=200)
    selected = select_top_papers(papers, n=MAX_PAPERS_PER_DAY)
  


    logger.info("Selected %d papers", len(selected))

    # 要約（OpenAIがあるなら生成、無ければabstract整形）
    for i, p in enumerate(selected, 1):
        logger.info("Summarizing %d/%d: %s", i, len(selected), p.title[:80])
        summarize_with_openai(p)

    # email_senderが期待するdict形式へ
    payload: List[Dict] = []
    for p in selected:
        payload.append({
            "title": p.title,
            "title_jp": "",  # 必要なら後で翻訳を追加
            "link": p.link,
            "authors": p.authors,
            "source": p.source,
            "published": p.published,
            "background": p.background,
            "purpose": p.purpose,
            "conditions": p.conditions,
            "methods": p.methods,
            "results": p.results,
            "significance": p.significance,
            "implications": p.implications,
        })

    subject = f"Daily Paper Digest ({dt.date.today().isoformat()})"
    email_sender.send_email(payload, subject=subject)
    logger.info("Email sent.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Paper Search & Email Agent (GitHub Actions Ready)")
    parser.add_argument("--test-email", action="store_true", help="Send a test email with dummy data")
    args = parser.parse_args()

    logger.info("Agent started (CLI Mode).")

    if args.test_email:
        logger.info("Sending test email...")
        dummy = [{
            "title": "Test Paper Title: Advanced ALD Process for High-k Dielectrics",
            "title_jp": "高誘電率膜のための先進的ALDプロセス",
            "link": "https://example.com",
            "authors": ["Taro Yamada", "Hanako Suzuki"],
            "source": "Test Source",
            "published": "2024-01-01",
            "background": "（テスト）背景",
            "purpose": "（テスト）目的",
            "conditions": "（テスト）条件",
            "methods": "（テスト）手法",
            "results": "（テスト）結果",
            "significance": "（テスト）意義",
            "implications": "（テスト）示唆",
        }]
        email_sender.send_email(dummy, subject="Daily Paper Digest (TEST)")
        return

    job()


if __name__ == "__main__":
    main()



