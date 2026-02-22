# main.py
# Stable daily paper digest (metasurface/process oriented)

import argparse
import base64
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Dict, List, Optional
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

UA = {"User-Agent": "paper-digest-stable/2.0"}


# =====================
# Data Model
# =====================

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


# =====================
# Utilities
# =====================

def now_jst():
    return datetime.now(JST)


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\n", " ")).strip()


# =====================
# arXiv Fetch
# =====================

def build_query(max_results=500):
    cat_q = " OR ".join([f"cat:{c}" for c in DEFAULT_CATEGORIES])
    keywords = [
        "metasurface", "metalens",
        "ALD", "atomic layer deposition",
        "etching", "RIE", "nanoimprint",
        "nanofabrication", "thin film"
    ]

    key_q = " OR ".join([f'all:"{k}"' for k in keywords])

    query = f"({cat_q}) AND ({key_q})"

    return (
        f"{ARXIV_BASE}?search_query={quote_plus(query)}"
        f"&start=0&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )


def fetch_arxiv():
    url = build_query()
    print("[INFO] Fetch:", url)
    r = requests.get(url, headers=UA)
    r.raise_for_status()

    import xml.etree.ElementTree as ET
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.text)

    papers = []
    for entry in root.findall("atom:entry", ns):
        title = normalize_text(entry.findtext("atom:title", default="", namespaces=ns))
        summary = normalize_text(entry.findtext("atom:summary", default="", namespaces=ns))
        published = normalize_text(entry.findtext("atom:published", default="", namespaces=ns))
        updated = normalize_text(entry.findtext("atom:updated", default="", namespaces=ns))
        _id = normalize_text(entry.findtext("atom:id", default="", namespaces=ns))
        arxiv_id = _id.rsplit("/", 1)[-1]
        link = f"https://arxiv.org/abs/{arxiv_id}"

        papers.append(Paper(title, summary, link, arxiv_id, published, updated))

    return papers


# =====================
# Filtering
# =====================

def score_paper(p: Paper):
    t = (p.title + " " + p.summary).lower()

    score = 0
    keywords = [
        "ald", "atomic layer", "etch", "plasma",
        "nanoimprint", "lithography",
        "metasurface", "metalens"
    ]

    for k in keywords:
        if k in t:
            score += 5

    if "metasurface" in t or "metalens" in t:
        p.is_meta = True
        score += 10

    p.score = score
    return p


# =====================
# sent_db handling (90日保持)
# =====================

def load_db():
    if not os.path.exists("sent_db.json"):
        return {}
    return json.load(open("sent_db.json"))


def save_db(db):
    json.dump(db, open("sent_db.json", "w"), indent=2)


def clean_old_entries(db):
    cutoff = datetime.utcnow() - timedelta(days=90)
    new_db = {}
    for k, v in db.items():
        ts = datetime.fromisoformat(v["sent_at"])
        if ts > cutoff:
            new_db[k] = v
    return new_db


# =====================
# OpenAI Summarize
# =====================

def summarize(p: Paper):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = f"""
以下の論文を5〜10行、日本語でプロセス寄りに要約してください。

Title:
{p.title}

Abstract:
{p.summary}
"""
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return resp.choices[0].message.content.strip()


# =====================
# Gmail Send
# =====================

def send_email(body, subject):
    creds = Credentials.from_authorized_user_info(
        json.loads(os.environ["GMAIL_TOKEN"]),
        scopes=["https://www.googleapis.com/auth/gmail.send"],
    )
    service = build("gmail", "v1", credentials=creds)

    msg = EmailMessage()
    msg["To"] = os.environ["RECIPIENT_EMAIL"]
    msg["From"] = os.environ["SENDER_EMAIL"]
    msg["Subject"] = subject
    msg.set_content(body)

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    service.users().messages().send(userId="me", body={"raw": raw}).execute()


# =====================
# Main
# =====================

def main():
    papers = fetch_arxiv()

    # 1年以内
    cutoff = datetime.utcnow() - timedelta(days=365)
    papers = [
        p for p in papers
        if datetime.fromisoformat(p.updated.replace("Z","+00:00")) > cutoff
    ]

    db = load_db()
    db = clean_old_entries(db)

    filtered = []
    for p in papers:
        score_paper(p)
        if p.arxiv_id not in db:
            filtered.append(p)

    filtered.sort(key=lambda x: x.score, reverse=True)

    selected = filtered[:5]

    if not selected:
        send_email("該当論文なし（条件緩和推奨）", "Paper Digest - No Matches")
        return

    body_lines = []
    for p in selected:
        summary = summarize(p)
        body_lines.append(f"==== {p.title}")
        body_lines.append(p.link)
        body_lines.append("")
        body_lines.append(summary)
        body_lines.append("\n")

        db[p.arxiv_id] = {
            "sent_at": datetime.utcnow().isoformat()
        }

    save_db(db)

    send_email(
        "\n".join(body_lines),
        f"Paper Digest {datetime.utcnow().strftime('%Y-%m-%d')}"
    )


if __name__ == "__main__":
    main()
