import argparse
import base64
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Dict, List
from urllib.parse import quote_plus

import requests

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
except Exception:
    Credentials = None
    build = None


UTC = timezone.utc
JST = timezone(timedelta(hours=9))
ARXIV_BASE = "http://export.arxiv.org/api/query"

DEFAULT_CATEGORIES = [
    "cond-mat.mtrl-sci",
    "cond-mat.other",
    "physics.app-ph",
    "physics.ins-det",
    "physics.optics",
    "eess.SP",
    "eess.SY",
]

UA = {"User-Agent": "paper-digest-stable/2.1"}

KEYWORDS = [
    "metasurface", "metalens",
    "ALD", "atomic layer deposition",
    "etching", "RIE", "nanoimprint",
    "nanofabrication", "thin film",
]

SCORE_WORDS = [
    "ald", "atomic layer", "etch", "plasma",
    "nanoimprint", "lithography",
    "metasurface", "metalens",
]


@dataclass
class Paper:
    title: str
    summary: str
    link: str
    arxiv_id: str
    published: str
    updated: str
    score: float = 0.0
    is_meta: bool = False


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\n", " ")).strip()


def parse_iso_to_utc_aware(s: str) -> datetime:
    """
    arXivの '2026-02-16T01:29:12Z' / '+00:00' / 末尾なし などを安全にUTC awareへ。
    """
    s = (s or "").strip()
    if not s:
        return datetime(1970, 1, 1, tzinfo=UTC)

    # Zを+00:00へ
    s = s.replace("Z", "+00:00")

    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        # フォールバック（日時だけなど）
        try:
            # YYYY-MM-DD だけなら 00:00 UTC
            dt = datetime.fromisoformat(s.split("T")[0])
        except Exception:
            return datetime(1970, 1, 1, tzinfo=UTC)

    # naiveならUTCとみなす
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)

    # UTCへ
    return dt.astimezone(UTC)


def build_query(max_results=500) -> str:
    cat_q = " OR ".join([f"cat:{c}" for c in DEFAULT_CATEGORIES])
    key_q = " OR ".join([f'all:"{k}"' for k in KEYWORDS])
    query = f"({cat_q}) AND ({key_q})"

    return (
        f"{ARXIV_BASE}?search_query={quote_plus(query)}"
        f"&start=0&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )


def fetch_arxiv(max_results=500) -> List[Paper]:
    url = build_query(max_results=max_results)
    print("[INFO] Fetch:", url)
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()

    import xml.etree.ElementTree as ET
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.text)

    papers: List[Paper] = []
    for entry in root.findall("atom:entry", ns):
        title = normalize_text(entry.findtext("atom:title", default="", namespaces=ns))
        summary = normalize_text(entry.findtext("atom:summary", default="", namespaces=ns))
        published = normalize_text(entry.findtext("atom:published", default="", namespaces=ns))
        updated = normalize_text(entry.findtext("atom:updated", default="", namespaces=ns))
        _id = normalize_text(entry.findtext("atom:id", default="", namespaces=ns))
        arxiv_id = _id.rsplit("/", 1)[-1] if _id else ""
        link = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else _id

        papers.append(Paper(title, summary, link, arxiv_id, published, updated))

    return papers


def score_paper(p: Paper) -> None:
    t = (p.title + " " + p.summary).lower()
    score = 0
    for w in SCORE_WORDS:
        if w in t:
            score += 5
    if "metasurface" in t or "metalens" in t:
        p.is_meta = True
        score += 10
    p.score = float(score)


def load_db() -> Dict[str, Dict]:
    if not os.path.exists("sent_db.json"):
        return {}
    try:
        with open("sent_db.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_db(db: Dict[str, Dict]) -> None:
    with open("sent_db.json", "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def clean_old_entries(db: Dict[str, Dict], keep_days: int = 90) -> Dict[str, Dict]:
    cutoff = datetime.now(UTC) - timedelta(days=keep_days)
    new_db = {}
    for k, v in db.items():
        ts = v.get("sent_at", "")
        dt = parse_iso_to_utc_aware(ts)
        if dt > cutoff:
            new_db[k] = v
    return new_db


def openai_summarize(p: Paper) -> str:
    if OpenAI is None:
        return "OpenAIライブラリが無いため要約できません。"
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return "OPENAI_API_KEY が無いため要約できません。"

    client = OpenAI(api_key=api_key)
    prompt = f"""
以下の論文を5〜10行、日本語でプロセス寄りに要約してください。

Title:
{p.title}

Abstract:
{p.summary}
""".strip()

    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def parse_gmail_token() -> Dict:
    token = os.environ.get("GMAIL_TOKEN", "").strip()
    if not token:
        raise RuntimeError("GMAIL_TOKEN is missing")
    if token.startswith("{"):
        return json.loads(token)
    # base64(JSON) fallback
    decoded = base64.b64decode(token).decode("utf-8", errors="ignore").strip()
    return json.loads(decoded)


def send_email(body: str, subject: str) -> None:
    if Credentials is None or build is None:
        raise RuntimeError("google api libs missing")

    recipient = os.environ.get("RECIPIENT_EMAIL", "").strip()
    sender = os.environ.get("SENDER_EMAIL", "").strip()
    if not recipient:
        raise RuntimeError("RECIPIENT_EMAIL is missing")
    if not sender:
        raise RuntimeError("SENDER_EMAIL is missing")

    token_info = parse_gmail_token()
    creds = Credentials.from_authorized_user_info(
        token_info,
        scopes=["https://www.googleapis.com/auth/gmail.send"],
    )
    service = build("gmail", "v1", credentials=creds)

    msg = EmailMessage()
    msg["To"] = recipient
    msg["From"] = sender
    msg["Subject"] = subject
    msg.set_content(body)

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    service.users().messages().send(userId="me", body={"raw": raw}).execute()
    print("[INFO] Email sent.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days-back", type=int, default=365)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--max-results", type=int, default=500)
    args = parser.parse_args()

    papers = fetch_arxiv(max_results=args.max_results)

    cutoff = datetime.now(UTC) - timedelta(days=args.days_back)

    # ✅ UTC awareで比較（ここが修正点）
    filtered_by_date = []
    for p in papers:
        upd = parse_iso_to_utc_aware(p.updated)
        if upd > cutoff:
            filtered_by_date.append(p)

    db = clean_old_entries(load_db(), keep_days=90)

    candidates = []
    for p in filtered_by_date:
        if p.arxiv_id and p.arxiv_id in db:
            continue
        score_paper(p)
        candidates.append(p)

    candidates.sort(key=lambda x: x.score, reverse=True)
    selected = candidates[: args.n]

    if not selected:
        send_email("該当論文なし（条件緩和推奨）", "Paper Digest - No Matches")
        return

    body_lines = []
    for p in selected:
        summary = openai_summarize(p)
        body_lines.append(f"==== {p.title}")
        body_lines.append(p.link)
        body_lines.append("")
        body_lines.append(summary)
        body_lines.append("")

        if p.arxiv_id:
            db[p.arxiv_id] = {"sent_at": datetime.now(UTC).isoformat()}

    save_db(db)
    send_email("\n".join(body_lines), f"Paper Digest {datetime.now(JST).strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    main()
