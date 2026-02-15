import argparse
import datetime as dt
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from paper_agent import email_sender

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# -----------------------------
# Settings
# -----------------------------
MAX_PAPERS_PER_DAY = 5

# プロセス寄りに絞ったカテゴリ（通信・制御・ネットワーク系は外す）
ARXIV_CATEGORIES = [
    "cond-mat.mtrl-sci",   # 材料
    "physics.app-ph",      # 応用物理（プロセス/デバイス）
    "cond-mat.other",      # その他物性（薄膜/材料が混ざる）
    "physics.ins-det",     # 装置/計測（プロセス装置/評価が混ざる）
]

# 「これが入ってない論文は基本的に要らない」必須語（ノイズ除去の要）
MUST_HAVE = [
    "atomic layer deposition", "ald", "peald",
    "etch", "etching", "plasma etching", "reactive ion etching", "rie",
    "nanoimprint", "nil", "lithography", "patterning",
    "tio2", "titania", "titanium dioxide",
    "thin film", "thin-film", "deposition",
    "metasurface",
]

# 明確に除外したいノイズ（通信/最適化/学習など）
# ※プロセス論文にも "optimization" が入ることはあるので、強すぎない語は入れない
EXCLUDE_IF_TITLE_MATCH = [
    "ris", "reconfigurable intelligent surface", "beamforming",
    "wireless", "mimo", "channel", "antenna", "trajectory-aware",
    "diffusion model", "denoising diffusion", "reinforcement learning",
]

# OpenAI要約（任意）
OPENAI_MODEL_DEFAULT = "gpt-4o-mini"

# -----------------------------
# Data Model
# -----------------------------
@dataclass
class Paper:
    title: str
    link: str
    authors: List[str]
    published: str  # ISO string
    abstract: str
    source: str = "arXiv"

    background: str = ""
    purpose: str = ""
    conditions: str = ""
    methods: str = ""
    results: str = ""
    significance: str = ""
    implications: str = ""


# -----------------------------
# Utils
# -----------------------------
def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _title_excluded(title: str) -> bool:
    t = (title or "").lower()
    return any(bad in t for bad in EXCLUDE_IF_TITLE_MATCH)


def _has_must_have(title: str, abstract: str) -> bool:
    t = (title + " " + abstract).lower()
    return any(m in t for m in MUST_HAVE)


def _keyword_score(text: str) -> int:
    """
    プロセス論文が上に来るように強く加点する。
    """
    t = text.lower()
    score = 0

    # 強い優先（プロセスど真ん中）
    strong = [
        "atomic layer deposition", "ald", "peald",
        "etching", "plasma etching", "reactive ion etching", "rie",
        "nanoimprint", "nil", "lithography",
        "tio2", "titania", "titanium dioxide",
        "conformal", "step coverage", "high aspect ratio",
        "thin film", "thin-film", "deposition",
        "mask", "selectivity", "anisotropic", "isotropic",
        "tdmat", "temat", "tma", "ozone", "o3", "o2 plasma",
    ]
    for s in strong:
        if s in t:
            score += 8

    # 中くらい（評価・周辺語）
    medium = [
        "xps", "ellipsometry", "xrr", "xrd",
        "sem", "tem", "afm", "profilometry",
        "gpc", "growth per cycle",
        "refractive index", "density",
        "etch rate", "etch profile",
        "nanofabrication", "patterning",
        "metasurface", "metasurfaces",
    ]
    for m in medium:
        if m in t:
            score += 2

    return score


# -----------------------------
# arXiv Fetch
# -----------------------------
def _parse_arxiv_atom(xml_bytes: bytes) -> List[Paper]:
    ns = {"atom": "http://www.w3.org/2005/Atom"}
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


def fetch_arxiv_papers(days_back: int = 21, max_results: int = 200) -> List[Paper]:
    """
    広めに取得 → must-have & スコアでプロセス寄りに絞る。
    """
    cat_query = " OR ".join([f"cat:{c}" for c in ARXIV_CATEGORIES])
    query = f"({cat_query})"

    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    logger.info("Fetching arXiv (process-focused): %s", url)

    req = urllib.request.Request(url, headers={"User-Agent": "paper-digest-agent/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml_bytes = resp.read()

    papers = _parse_arxiv_atom(xml_bytes)

    cutoff = dt.datetime.utcnow() - dt.timedelta(days=days_back)
    filtered: List[Paper] = []
    for p in papers:
        try:
            pub_dt = dt.datetime.fromisoformat(p.published.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            pub_dt = None
        if pub_dt is None or pub_dt >= cutoff:
            filtered.append(p)

    logger.info("Fetched %d papers, filtered to %d (days_back=%d)", len(papers), len(filtered), days_back)
    return filtered


def select_top_papers(papers: List[Paper], n: int = MAX_PAPERS_PER_DAY) -> List[Paper]:
    # まずタイトル除外（通信/最適化系が混ざるのを強く防ぐ）
    papers = [p for p in papers if not _title_excluded(p.title)]

    # must-haveで門前払い（これが効く）
    papers_must = [p for p in papers if _has_must_have(p.title, p.abstract)]
    if papers_must:
        papers = papers_must  # mustに引っかかったものがあるなら、それだけで勝負
    # mustが0件のときだけ、完全ゼロ回避のために元のpapersを使う

    scored: List[Tuple[int, str, Paper]] = []
    for p in papers:
        text = f"{p.title}\n{p.abstract}"
        score = _keyword_score(text)
        pub_key = p.published or ""
        scored.append((score, pub_key, p))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    selected: List[Paper] = []
    seen = set()

    # まずスコア>0を優先
    for score, _, p in scored:
        if score <= 0:
            continue
        if p.title in seen:
            continue
        selected.append(p)
        seen.add(p.title)
        if len(selected) >= n:
            return selected

    # 足りなければ最新から埋める（毎日届かないのを防ぐ）
    for _, _, p in scored:
        if p.title in seen:
            continue
        selected.append(p)
        seen.add(p.title)
        if len(selected) >= n:
            break

    return selected


# -----------------------------
# Summarization (Optional OpenAI)
# -----------------------------
def summarize_with_openai(p: Paper) -> None:
    """
    OPENAI_API_KEYが無ければabstract整形。
    429/クォータ不足でも落とさずabstractで続行。
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        p.results = _normalize_text(p.abstract)[:1500]
        return

    try:
        from openai import OpenAI
        from openai import RateLimitError
    except Exception:
        # ライブラリが入ってない等
        p.results = _normalize_text(p.abstract)[:1500]
        return

    client = OpenAI(api_key=api_key)

    system = (
        "あなたは半導体プロセス分野（成膜・エッチング・リソグラフィ・プロセス統合）の研究アシスタントです。"
        "タイトルと要旨から、研究者向けに日本語で技術要約してください。"
        "推測は最小限にし、要旨に基づく内容を具体的に。\n"
        "必ず次の見出しで出力:\n"
        "背景:\n目的:\n実験条件:\n手法:\n結果:\n意義:\n今後の示唆:\n"
        "特に『成膜条件/プラズマ条件/ガス/温度/圧力/パワー/評価手法/膜質/プロファイル』があれば優先して書く。"
    )
    user = f"タイトル:\n{p.title}\n\n要旨:\n{p.abstract}\n\nリンク:\n{p.link}\n"

    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", OPENAI_MODEL_DEFAULT),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        text = resp.choices[0].message.content or ""
    except RateLimitError:
        # クォータ不足でも落とさない
        p.results = _normalize_text(p.abstract)[:1500]
        return
    except Exception:
        p.results = _normalize_text(p.abstract)[:1500]
        return

    def grab(section: str) -> str:
        m = re.search(rf"{section}:\s*(.*?)(?=\n[^\n]+?:|\Z)", text, re.S)
        return _normalize_text(m.group(1)) if m else ""

    p.background = grab("背景")
    p.purpose = grab("目的")
    p.conditions = grab("実験条件")
    p.methods = grab("手法")
    p.results = grab("結果") or _normalize_text(p.abstract)[:1500]
    p.significance = grab("意義")
    p.implications = grab("今後の示唆")


# -----------------------------
# Job
# -----------------------------
def job() -> None:
    # 広めに取る（プロセス寄りに絞っても、母数が必要）
    papers = fetch_arxiv_papers(days_back=21, max_results=200)
    selected = select_top_papers(papers, n=MAX_PAPERS_PER_DAY)

    logger.info("Selected %d papers", len(selected))

    for i, p in enumerate(selected, 1):
        logger.info("Summarizing %d/%d: %s", i, len(selected), p.title[:90])
        summarize_with_openai(p)

    payload: List[Dict] = []
    for p in selected:
        payload.append({
            "title": p.title,
            "title_jp": "",
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

    # 0件でも空メールにならないように必ず何か入れる
    if not payload:
        payload = [{
            "title": "本日の条件に合う論文が見つかりませんでした（0件）",
            "link": "",
            "results": "カテゴリ/必須語の条件が厳しすぎる可能性があります。MUST_HAVE/カテゴリを調整してください。",
        }]

    subject = f"Daily Paper Digest (Process) {dt.date.today().isoformat()}"
    email_sender.send_email(payload, subject=subject)
    logger.info("Email sent.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Paper Digest (Process-focused, Gmail API)")
    parser.add_argument("--test-email", action="store_true", help="Send a test email with dummy data")
    args = parser.parse_args()

    logger.info("Agent started (CLI Mode).")

    if args.test_email:
        dummy = [{
            "title": "Test Paper: PEALD TiO2 on High-Aspect-Ratio Structures",
            "title_jp": "（テスト）高アスペクト比構造へのPEALD TiO2",
            "link": "https://example.com",
            "authors": ["Taro Yamada", "Hanako Suzuki"],
            "source": "Test Source",
            "published": "2026-02-15",
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
