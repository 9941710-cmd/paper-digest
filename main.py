# main.py
# Daily paper digest: arXiv検索 →（プロセス寄りにフィルタ）→ OpenAIで5–10行要約 → Gmail APIでメール送信
# GitHub Actions で 5分おき起動しても「JST 9:00〜9:04だけ送る」ようにしてあります

import argparse
import base64
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import requests

# ---- Optional (OpenAI / Gmail) ----
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

try:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
except Exception:
    Credentials = None  # type: ignore
    build = None  # type: ignore


# =========================
# Config
# =========================
JST = timezone(timedelta(hours=9))
ARXIV_BASE = "http://export.arxiv.org/api/query"

# ✅ 検索カテゴリを拡張（メタサーフェス母集団を増やす）
DEFAULT_CATEGORIES = [
    "physics.optics",        # metasurface本命
    "physics.app-ph",
    "cond-mat.mtrl-sci",
    "cond-mat.mes-hall",
    "cond-mat.other",
    "physics.ins-det",
    "eess.SP",               # 光/電波デバイス寄り metasurface が混ざる
]

# ✅ 広めに拾う OR キーワード（arXivの検索段階で“ゆるく”母集団を作る）
BROAD_TERMS = [
    # metasurface
    "metasurface", "metasurfaces", "metalens", "meta surface", "meta-surface",
    # process
    "atomic layer deposition", "ALD", "PEALD", "ALE",
    "etching", "RIE", "plasma etching", "plasma",
    "nanoimprint", "NIL", "lithography", "nanofabrication", "patterning",
    "thin film", "deposition", "fabrication",
    # materials
    "TiO2", "titania", "titanium dioxide",
]

# ✅ 通信/RIS/ネットワーク寄りのノイズ除外
EXCLUDE_KEYWORDS = [
    "ris", "reconfigurable intelligent surface",
    "beamforming", "mimo", "channel", "wireless",
    "6g", "5g", "mmwave", "mm-wave", "terahertz communication", "thz communication",
    "network", "multi-user", "base station", "ue", "bs",
    "trajectory", "localization", "positioning",
    "rate", "throughput", "spectral efficiency",
]

# 「プロセス寄り」の強いキーワード（タイトル/要旨にあると加点）
PROCESS_KEYWORDS_STRONG = [
    "atomic layer deposition", "ald", "peald", "plasma-enhanced ald",
    "atomic layer etching", "ale",
    "reactive ion etching", "rie", "plasma etching", "etching",
    "neutral loop discharge", "nld", "nld-rie",
    "nanoimprint", "nil", "lithography", "nanofabrication", "patterning",
    "thin film", "deposition", "process", "fabrication", "manufacturing",
    "dry etch", "dry-etch", "etch-back", "etchback",
    "tio2", "titania", "titanium dioxide",
    "metasurface", "metasurfaces", "meta-surface", "meta surface",
]

# メタサーフェス判定（最低1本は必ず入れたい）
METASURFACE_KEYWORDS = [
    "metasurface", "metasurfaces", "meta-surface", "meta surface",
    "metalens", "meta-lens", "metagrating",
]

# 「プロセス寄り」をさらに強化する語（あると追加加点）
PROCESS_KEYWORDS_EXTRA = [
    "etch selectivity", "anisotropic", "sidewall", "profile", "scallop",
    "aspect ratio", "conformality", "step coverage", "gpc", "precursor",
    "tdmat", "tma", "plasma", "rf power", "bias", "pressure", "flow rate",
    "recipe", "process window", "uniformity",
]

UA = {"User-Agent": "paper-digest/1.0 (+github-actions)"}


# =========================
# Data model
# =========================
@dataclass
class Paper:
    title: str
    summary: str
    link: str
    arxiv_id: str
    published: str
    updated: str
    authors: List[str]
    categories: List[str]

    score: float = 0.0
    is_metasurface: bool = False
    is_processy: bool = False

    # OpenAI output
    digest: Optional[str] = None
    keywords_hit: Optional[List[str]] = None


# =========================
# Utilities
# =========================
def now_jst() -> datetime:
    return datetime.now(JST)


def is_target_time_window(hour: int = 9, minute_window: int = 5) -> bool:
    """JSTで hour:00〜hour:(minute_window-1) の間だけ True"""
    n = now_jst()
    return n.hour == hour and 0 <= n.minute < minute_window


def normalize_text(s: str) -> str:
    s = s or ""
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_json_loads(s: str) -> Dict:
    try:
        return json.loads(s)
    except Exception:
        return {}


def env_get(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return v


# =========================
# arXiv fetch (Atom XML)
# =========================
def build_arxiv_query(
    categories: List[str],
    must_terms: Optional[List[str]] = None,
    max_results: int = 500,
) -> str:
    cat_q = " OR ".join([f"cat:{c}" for c in categories])
    q = f"({cat_q})"

    if must_terms:
        # ORで広く拾う（all:term OR all:"multi word"）
        parts = []
        for t in must_terms:
            t = t.strip()
            if " " in t:
                parts.append(f'all:"{t}"')
            else:
                parts.append(f"all:{t}")
        q += " AND (" + " OR ".join(parts) + ")"

    params = (
        f"search_query={quote_plus(q)}"
        f"&start=0&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )
    return f"{ARXIV_BASE}?{params}"


def parse_arxiv_atom(xml_text: str) -> List[Paper]:
    import xml.etree.ElementTree as ET

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(xml_text)

    papers: List[Paper] = []
    for entry in root.findall("atom:entry", ns):
        title = normalize_text(entry.findtext("atom:title", default="", namespaces=ns))
        summary = normalize_text(entry.findtext("atom:summary", default="", namespaces=ns))
        published = normalize_text(entry.findtext("atom:published", default="", namespaces=ns))
        updated = normalize_text(entry.findtext("atom:updated", default="", namespaces=ns))

        _id = normalize_text(entry.findtext("atom:id", default="", namespaces=ns))
        arxiv_id = _id.rsplit("/", 1)[-1] if _id else ""

        link = ""
        for l in entry.findall("atom:link", ns):
            if l.attrib.get("rel") == "alternate":
                link = l.attrib.get("href", "")
                break
        if not link and arxiv_id:
            link = f"https://arxiv.org/abs/{arxiv_id}"

        authors = []
        for a in entry.findall("atom:author", ns):
            name = normalize_text(a.findtext("atom:name", default="", namespaces=ns))
            if name:
                authors.append(name)

        categories = []
        for c in entry.findall("atom:category", ns):
            term = c.attrib.get("term", "")
            if term:
                categories.append(term)

        papers.append(
            Paper(
                title=title,
                summary=summary,
                link=link,
                arxiv_id=arxiv_id,
                published=published,
                updated=updated,
                authors=authors,
                categories=categories,
            )
        )
    return papers


def fetch_arxiv(
    categories: List[str],
    must_terms: Optional[List[str]],
    max_results: int = 500,
    timeout: int = 30,
) -> List[Paper]:
    url = build_arxiv_query(categories, must_terms=must_terms, max_results=max_results)
    print(f"[INFO] Fetching arXiv: {url}")
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return parse_arxiv_atom(r.text)


# =========================
# Filtering / scoring
# =========================
def within_days_back(iso_ts: str, days_back: int) -> bool:
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except Exception:
        return True
    return dt >= (datetime.now(timezone.utc) - timedelta(days=days_back))


def keyword_hits(text: str, keywords: List[str]) -> List[str]:
    t = (text or "").lower()
    hits = []
    for k in keywords:
        if k.lower() in t:
            hits.append(k)
    return hits


def should_exclude(p: Paper) -> bool:
    t = f"{p.title} {p.summary}".lower()
    return any(k in t for k in EXCLUDE_KEYWORDS)


def score_paper(p: Paper) -> Paper:
    text = f"{p.title} {p.summary}"
    hits_strong = keyword_hits(text, PROCESS_KEYWORDS_STRONG)
    hits_extra = keyword_hits(text, PROCESS_KEYWORDS_EXTRA)
    hits_meta = keyword_hits(text, METASURFACE_KEYWORDS)

    score = 0.0
    score += 3.0 * len(hits_strong)
    score += 1.5 * len(hits_extra)

    title_l = p.title.lower()
    if any(k in title_l for k in ["ald", "atomic layer deposition", "peald", "nld", "nld-rie", "ale", "nanoimprint", "nil", "metasurface", "metalens"]):
        score += 8.0

    is_meta = len(hits_meta) > 0
    if is_meta:
        score += 10.0  # metasurface優先

    # processy definition
    is_processy = (len(hits_strong) >= 2) or any(
        k in title_l for k in ["ald", "atomic layer deposition", "etch", "etching", "deposition", "fabrication", "nanoimprint", "lithography", "patterning"]
    )

    p.score = score
    p.is_metasurface = is_meta
    p.is_processy = is_processy
    p.keywords_hit = sorted(set(hits_strong + hits_extra + hits_meta))
    return p


def load_sent_db(path: str = "sent_db.json") -> Dict[str, Dict]:
    if not os.path.exists(path):
        return {"sent": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "sent" not in data or not isinstance(data["sent"], dict):
            return {"sent": {}}
        return data
    except Exception:
        return {"sent": {}}


def save_sent_db(data: Dict, path: str = "sent_db.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def already_sent(sent_db: Dict, p: Paper) -> bool:
    key = p.arxiv_id or p.link
    return key in sent_db.get("sent", {})


def mark_sent(sent_db: Dict, selected: List[Paper]) -> None:
    sent = sent_db.setdefault("sent", {})
    ts = now_jst().isoformat()
    for p in selected:
        key = p.arxiv_id or p.link
        sent[key] = {
            "title": p.title,
            "link": p.link,
            "published": p.published,
            "sent_at": ts,
        }


def pick_top_n(
    candidates: List[Paper],
    n: int,
    metasurface_at_least_one: bool = True,
) -> List[Paper]:
    candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
    if not candidates:
        return []

    selected: List[Paper] = []

    # できれば1本はメタサーフェスを確保
    if metasurface_at_least_one:
        meta = [p for p in candidates if p.is_metasurface]
        if meta:
            selected.append(meta[0])

    for p in candidates:
        if len(selected) >= n:
            break
        if p in selected:
            continue
        selected.append(p)

    return selected[:n]


# =========================
# OpenAI summarization
# =========================
def openai_client() -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("openai パッケージが入っていません（pip install openai）")
    key = env_get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY がありません（GitHub Secrets / 環境変数に設定してください）")
    return OpenAI(api_key=key)


def summarize_with_openai(p: Paper, model: str = "gpt-4o-mini") -> str:
    client = openai_client()

    prompt = f"""
あなたは半導体プロセス/ナノ加工の研究者向けに論文を要約するアシスタントです。
以下の論文のタイトルと要旨から、5〜10行で日本語要約してください。
必ず「プロセス条件/装置/材料/評価/結果」の観点があれば優先して触れてください。
推測はしない。要旨にない値は作らない。箇条書きでOK。

[Title]
{p.title}

[Abstract]
{p.summary}
""".strip()

    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You write concise, technical Japanese summaries for fabrication/process papers."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            txt = (resp.choices[0].message.content or "").strip()
            return txt
        except Exception as e:
            wait = 0.7 * (2 ** attempt)
            print(f"[WARN] OpenAI error: {e} (retry in {wait:.1f}s)")
            time.sleep(wait)

    raise RuntimeError("OpenAI要約が連続で失敗しました（APIキー/課金/レート制限を確認）")


# =========================
# Gmail send (API)
# =========================
def get_gmail_service():
    if Credentials is None or build is None:
        raise RuntimeError("google API パッケージが入っていません（pip install google-api-python-client google-auth）")

    token = env_get("GMAIL_TOKEN")
    if not token:
        raise RuntimeError("GMAIL_TOKEN is missing (GitHub Secrets に入れてください)")

    token = token.strip()
    if token.startswith("{"):
        token_info = safe_json_loads(token)
    else:
        try:
            decoded = base64.b64decode(token).decode("utf-8", errors="ignore").strip()
            token_info = safe_json_loads(decoded)
        except Exception:
            token_info = {}

    if not token_info:
        raise RuntimeError("GMAIL_TOKEN の形式が不正です（JSON か base64(JSON) である必要があります）")

    creds = Credentials.from_authorized_user_info(token_info, scopes=["https://www.googleapis.com/auth/gmail.send"])
    return build("gmail", "v1", credentials=creds)


def build_email_text(papers: List[Paper], header: str) -> str:
    lines: List[str] = []
    lines.append(header)
    lines.append("")
    for i, p in enumerate(papers, 1):
        lines.append(f"==== {i}. {p.title}")
        lines.append(f"Link: {p.link}")
        if p.published:
            lines.append(f"Published: {p.published}")
        if p.authors:
            lines.append("Authors: " + ", ".join(p.authors[:8]) + (" ..." if len(p.authors) > 8 else ""))
        if p.keywords_hit:
            lines.append("Hits: " + ", ".join(p.keywords_hit[:15]) + (" ..." if len(p.keywords_hit) > 15 else ""))
        if p.digest:
            lines.append("")
            lines.append(p.digest.strip())
        else:
            lines.append("")
            lines.append("(要約なし)")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def send_gmail(subject: str, body_text: str) -> None:
    recipient = env_get("RECIPIENT_EMAIL")
    sender = env_get("SENDER_EMAIL")
    if not recipient:
        raise RuntimeError("RECIPIENT_EMAIL is missing")
    if not sender:
        raise RuntimeError("SENDER_EMAIL is missing")

    service = get_gmail_service()

    msg = EmailMessage()
    msg["To"] = recipient
    msg["From"] = sender
    msg["Subject"] = subject
    msg.set_content(body_text)

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    service.users().messages().send(userId="me", body={"raw": raw}).execute()
    print("[INFO] Email sent.")


# =========================
# Main job
# =========================
def job(
    days_back: int,
    n: int,
    metasurface_must: bool,   # Trueなら“なるべく”確保（0ならフォールバック）
    process_strict: bool,
    max_results: int = 500,   # ✅ 増やした
) -> Tuple[List[Paper], List[Paper]]:
    sent_db = load_sent_db("sent_db.json")

    # ✅ 広めキーワードで母集団を増やす（ORで拾う）
    raw = fetch_arxiv(DEFAULT_CATEGORIES, must_terms=BROAD_TERMS, max_results=max_results)

    # filter by date
    raw = [p for p in raw if within_days_back(p.updated or p.published, days_back)]

    # ✅ ノイズ除外
    raw = [p for p in raw if not should_exclude(p)]

    print(f"[INFO] Fetched {len(raw)} papers within days_back={days_back} (after exclude)")

    scored: List[Paper] = []
    for p in raw:
        score_paper(p)
        if already_sent(sent_db, p):
            continue
        scored.append(p)

    print(f"[INFO] Candidates after de-dup: {len(scored)}")

    # process focus filter
    if process_strict:
        filtered = [p for p in scored if p.is_processy]
    else:
        # relaxed: keep if any strong keyword hits or decent score
        filtered = [p for p in scored if p.score >= 10.0 or p.is_processy]

    print(f"[INFO] Filtered candidates: {len(filtered)} (process_strict={process_strict})")

    # metasurface logic: MUST指定でも0件ならフォールバック（“0件地獄”を避ける）
    if metasurface_must:
        meta_only = [p for p in filtered if p.is_metasurface]
        if not meta_only:
            print("[WARN] metasurface requested, but none found. Falling back to best process papers.")
            metasurface_at_least_one = False
            pool = filtered
        else:
            metasurface_at_least_one = True
            pool = filtered
    else:
        # 既定：最低1本はメタサーフェス（可能な範囲で）
        metasurface_at_least_one = True
        pool = filtered

    selected = pick_top_n(pool, n=n, metasurface_at_least_one=metasurface_at_least_one)
    print(f"[INFO] Selected {len(selected)} papers")

    if not selected:
        subject = f"Paper Digest（候補なし） {now_jst().strftime('%Y-%m-%d')}"
        body = (
            "条件に合う新着が見つかりませんでした。\n\n"
            "緩和案:\n"
            "・--process-strict を外す\n"
            "・--days-back を増やす（今は1年以内）\n"
            "・EXCLUDE_KEYWORDS を減らす\n"
        )
        send_gmail(subject, body)
        return [], []

    # Summarize
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    for i, p in enumerate(selected, 1):
        print(f"[INFO] Summarizing {i}/{len(selected)}: {p.title[:80]}")
        p.digest = summarize_with_openai(p, model=model)

    # Send
    subject = f"Paper Digest {now_jst().strftime('%Y-%m-%d')}（{len(selected)}本）"
    header = (
        "本日の論文ダイジェストです（5–10行要約）。\n"
        f"条件: days_back={days_back}, process_strict={process_strict}, metasurface_pref={not metasurface_must or metasurface_must}\n"
        f"生成時刻(JST): {now_jst().strftime('%Y-%m-%d %H:%M')}\n"
        "※ 通信/RIS系は除外しています。\n"
    )
    body = build_email_text(selected, header)
    send_gmail(subject, body)

    # Mark sent
    mark_sent(sent_db, selected)
    save_sent_db(sent_db, "sent_db.json")

    return selected, filtered


def send_test_email() -> None:
    subject = f"Paper Digest TEST {now_jst().strftime('%Y-%m-%d %H:%M')}"
    body = (
        "これはテストメールです。\n"
        "Gmail API の送信が通るかだけ確認しています。\n"
    )
    send_gmail(subject, body)


def main():
    parser = argparse.ArgumentParser(description="Daily Paper Digest Agent (arXiv -> OpenAI -> Gmail)")
    parser.add_argument("--now", action="store_true", help="Run immediately (ignore JST time window)")
    parser.add_argument("--test-email", action="store_true", help="Send a simple test email and exit")
    parser.add_argument("--days-back", type=int, default=365, help="Look back N days (default: 365)")
    parser.add_argument("--n", type=int, default=5, help="Number of papers to send (default: 5)")
    parser.add_argument(
        "--metasurface-must",
        action="store_true",
        help="Try to enforce metasurface presence (fallback if none)",
    )
    parser.add_argument("--process-strict", action="store_true", help="Stricter process-only filtering")
    parser.add_argument("--max-results", type=int, default=500, help="arXiv max_results (default: 500)")
    parser.add_argument("--model", type=str, default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), help="OpenAI model")
    args = parser.parse_args()

    if args.model:
        os.environ["OPENAI_MODEL"] = args.model

    if args.test_email:
        print("[INFO] Sending test email...")
        send_test_email()
        return

    # Default behavior for GitHub Actions: run every 5 min but only send at 09:00-09:04 JST
    if not args.now:
        if not is_target_time_window(9, 5):
            print("[INFO] Not in JST 09:00-09:04 window. Skipping.")
            return

    try:
        selected, _filtered = job(
            days_back=args.days_back,
            n=args.n,
            metasurface_must=args.metasurface_must,
            process_strict=args.process_strict,
            max_results=args.max_results,
        )
        print(f"[INFO] Done. sent={len(selected)}")
    except Exception as e:
        print(f"[ERROR] {e}")
        try:
            subject = f"Paper Digest ERROR {now_jst().strftime('%Y-%m-%d %H:%M')}"
            body = f"実行中にエラー:\n\n{e}\n"
            send_gmail(subject, body)
        except Exception as e2:
            print(f"[ERROR] Failed to send error email: {e2}")
        raise


if __name__ == "__main__":
    main()
