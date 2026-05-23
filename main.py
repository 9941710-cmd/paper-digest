import os
import json
import time
import html
import re
import smtplib
import requests
import xml.etree.ElementTree as ET

from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from openai import OpenAI


ARXIV_URL = "http://export.arxiv.org/api/query"

# ==========================================
# SEARCH TARGETS
# ==========================================

KEYWORDS = [

    # ======================================
    # EBL / Overlay / Alignment
    # ======================================

    "electron beam lithography",
    "EBL",
    "overlay",
    "overlay accuracy",
    "overlay metrology",
    "alignment",
    "alignment mark",
    "alignment offset",
    "box in box",
    "registration error",
    "pattern placement error",
    "critical dimension",
    "CD uniformity",
    "stitching error",
    "field stitching",
    "beam drift",
    "stage error",
    "multilayer lithography",
    "double patterning",
    "proximity effect",
    "self-aligned",
    "self aligned",

    # ======================================
    # Nanofabrication
    # ======================================

    "nanofabrication",
    "dry etch",
    "ICP etch",
    "reactive ion etching",
    "RIE",
    "atomic layer deposition",
    "ALD",
    "atomic layer etching",
    "ALE",

    # ======================================
    # III-V / InP Photonics
    # ======================================

    "InP",
    "InGaAsP",
    "III-V",
    "laser",
    "DFB laser",
    "DBR laser",
    "ridge waveguide",
    "waveguide",
    "regrowth",
    "epitaxial regrowth",

    # ======================================
    # Integration
    # ======================================

    "heterogeneous integration",
    "wafer bonding",
    "alignment tolerance",
]

# ==========================================
# arXiv categories
# ==========================================

CATEGORIES = [
    "physics.app-ph",
    "physics.optics",
    "cond-mat.mtrl-sci",
]

OPENALEX_EMAIL = os.environ.get("OPENALEX_EMAIL", "")
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ==========================================
# sent database
# ==========================================

def load_db():

    if not os.path.exists("sent_db.json"):
        return {}

    with open("sent_db.json", "r", encoding="utf-8") as f:
        return json.load(f)


def save_db(db):

    with open("sent_db.json", "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


def clean_db(db):

    limit = datetime.now(timezone.utc) - timedelta(days=90)

    new = {}

    for k, v in db.items():

        try:

            t = datetime.fromisoformat(v["sent_at"])

            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)

            t = t.astimezone(timezone.utc)

            if t > limit:
                new[k] = v

        except Exception:
            continue

    return new


# ==========================================
# scoring
# ==========================================

def score_paper(text):

    t = (text or "").lower()

    score = 0

    # ======================================
    # EBL / Overlay
    # ======================================

    if "electron beam lithography" in t:
        score += 25

    if re.search(r"\bebl\b", t):
        score += 20

    if "overlay" in t:
        score += 22

    if "overlay accuracy" in t:
        score += 25

    if "overlay metrology" in t:
        score += 25

    if "alignment" in t:
        score += 18

    if "alignment mark" in t:
        score += 22

    if "alignment offset" in t:
        score += 20

    if "box in box" in t:
        score += 28

    if "registration error" in t:
        score += 22

    if "pattern placement error" in t:
        score += 25

    if "critical dimension" in t:
        score += 15

    if "cd uniformity" in t:
        score += 15

    if "stitching error" in t:
        score += 24

    if "field stitching" in t:
        score += 22

    if "beam drift" in t:
        score += 18

    if "stage error" in t:
        score += 18

    if "multilayer lithography" in t:
        score += 24

    if "double patterning" in t:
        score += 20

    if "proximity effect" in t:
        score += 20

    if "self-aligned" in t or "self aligned" in t:
        score += 20

    # ======================================
    # Process
    # ======================================

    if "dry etch" in t:
        score += 10

    if "icp etch" in t:
        score += 10

    if "rie" in t:
        score += 8

    if "ald" in t:
        score += 6

    if "ale" in t:
        score += 6

    # ======================================
    # III-V / InP photonics
    # ======================================

    if "inp" in t:
        score += 12

    if "ingaasp" in t:
        score += 10

    if "iii-v" in t:
        score += 8

    if "laser" in t:
        score += 8

    if "dfb laser" in t:
        score += 12

    if "dbr laser" in t:
        score += 10

    if "ridge waveguide" in t:
        score += 12

    if "waveguide" in t:
        score += 6

    if "regrowth" in t:
        score += 10

    if "epitaxial regrowth" in t:
        score += 12

    # ======================================
    # Integration
    # ======================================

    if "heterogeneous integration" in t:
        score += 10

    if "wafer bonding" in t:
        score += 10

    if "alignment tolerance" in t:
        score += 14

    # ======================================
    # de-prioritize
    # ======================================

    if "metalens" in t:
        score -= 5

    if "metasurface" in t:
        score -= 5

    if "biology" in t:
        score -= 10

    return score


# ==========================================
# utility
# ==========================================

def request_with_retry(
    url,
    *,
    params=None,
    headers=None,
    timeout=60,
    retries=3,
    sleep_sec=5
):

    last_error = None

    for i in range(retries):

        try:

            r = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout
            )

            r.raise_for_status()

            return r

        except Exception as e:

            last_error = e

            print(
                f"request failed ({i+1}/{retries}) "
                f"url={url} error={e}"
            )

            if i < retries - 1:
                time.sleep(sleep_sec)

    raise last_error


def normalize_whitespace(s):

    if not s:
        return ""

    return " ".join(str(s).split())


def build_abstract_from_inverted_index(inv):

    if not inv:
        return ""

    positions = {}

    for word, pos_list in inv.items():

        for pos in pos_list:
            positions[pos] = word

    return " ".join(
        positions[i] for i in sorted(positions)
    )


def strip_html_tags(text):

    if not text:
        return ""

    text = html.unescape(text)

    text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================
# arXiv
# ==========================================

def search_arxiv():

    query = " OR ".join([
        f'all:"{k}"' for k in KEYWORDS
    ])

    cat = " OR ".join([
        f"cat:{c}" for c in CATEGORIES
    ])

    url = (
        f"{ARXIV_URL}"
        f"?search_query=({cat}) AND ({query})"
        f"&start=0"
        f"&max_results=50"
        f"&sortBy=submittedDate"
        f"&sortOrder=descending"
    )

    print("fetching arXiv", url)

    r = request_with_retry(url, timeout=60)

    root = ET.fromstring(r.text)

    ns = {"atom": "http://www.w3.org/2005/Atom"}

    papers = []

    for e in root.findall("atom:entry", ns):

        title = normalize_whitespace(
            e.find("atom:title", ns).text
            if e.find("atom:title", ns) is not None
            else ""
        )

        abstract = normalize_whitespace(
            e.find("atom:summary", ns).text
            if e.find("atom:summary", ns) is not None
            else ""
        )

        link = normalize_whitespace(
            e.find("atom:id", ns).text
            if e.find("atom:id", ns) is not None
            else ""
        )

        if title and link:

            papers.append({
                "title": title,
                "abstract": abstract,
                "link": link,
                "source": "arxiv",
            })

    return papers


# ==========================================
# OpenAlex
# ==========================================

def search_openalex():

    query = " OR ".join(KEYWORDS)

    url = "https://api.openalex.org/works"

    params = {
        "search": query,
        "sort": "publication_date:desc",
        "per-page": 25,
    }

    headers = {}

    if OPENALEX_EMAIL:

        headers["User-Agent"] = (
            f"paper-digest/1.0 "
            f"(mailto:{OPENALEX_EMAIL})"
        )

    print("fetching OpenAlex", params)

    r = request_with_retry(
        url,
        params=params,
        headers=headers,
        timeout=60
    )

    data = r.json()

    papers = []

    for w in data.get("results", []):

        title = normalize_whitespace(
            w.get("display_name", "")
        )

        abstract = build_abstract_from_inverted_index(
            w.get("abstract_inverted_index")
        )

        link = normalize_whitespace(
            w.get("id", "")
        )

        if not link:

            primary_location = (
                w.get("primary_location") or {}
            )

            link = normalize_whitespace(
                primary_location.get(
                    "landing_page_url"
                )
                or primary_location.get(
                    "pdf_url"
                )
                or ""
            )

        if title and link:

            papers.append({
                "title": title,
                "abstract": normalize_whitespace(
                    abstract
                ),
                "link": link,
                "source": "openalex",
            })

    return papers


# ==========================================
# Semantic Scholar
# ==========================================

def search_semantic_scholar():

    query = " OR ".join(KEYWORDS[:10])

    url = (
        "https://api.semanticscholar.org/"
        "graph/v1/paper/search"
    )

    params = {
        "query": query,
        "limit": 10,
        "fields":
            "title,abstract,url,"
            "year,publicationDate",
    }

    headers = {}

    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = (
            SEMANTIC_SCHOLAR_API_KEY
        )

    print("fetching Semantic Scholar", params)

    try:

        r = request_with_retry(
            url,
            params=params,
            headers=headers,
            timeout=60,
            retries=2,
            sleep_sec=10
        )

        data = r.json()

    except Exception as e:

        print(
            f"search_semantic_scholar failed: {e}"
        )

        return []

    papers = []

    for p in data.get("data", []):

        title = normalize_whitespace(
            p.get("title", "")
        )

        abstract = normalize_whitespace(
            p.get("abstract", "")
        )

        link = normalize_whitespace(
            p.get("url", "")
        )

        if title and link:

            papers.append({
                "title": title,
                "abstract": abstract,
                "link": link,
                "source": "semantic_scholar",
            })

    return papers


# ==========================================
# Crossref
# ==========================================

def search_crossref():

    query = " ".join([
        "electron beam lithography",
        "overlay",
        "alignment",
        "box in box",
        "stitching",
        "InP",
        "waveguide",
    ])

    url = "https://api.crossref.org/works"

    params = {
        "query": query,
        "rows": 25,
        "sort": "published",
        "order": "desc",
        "select":
            "DOI,title,abstract,"
            "URL,published",
    }

    headers = {
        "User-Agent":
            (
                f"paper-digest/1.0 "
                f"(mailto:{OPENALEX_EMAIL})"
            )
            if OPENALEX_EMAIL
            else "paper-digest/1.0"
    }

    print("fetching Crossref", params)

    r = request_with_retry(
        url,
        params=params,
        headers=headers,
        timeout=60
    )

    data = r.json()

    papers = []

    for item in data.get(
        "message",
        {}
    ).get("items", []):

        title_list = item.get("title", [])

        title = normalize_whitespace(
            title_list[0]
            if title_list
            else ""
        )

        abstract = strip_html_tags(
            item.get("abstract", "")
        )

        link = normalize_whitespace(
            item.get("URL", "")
        )

        if title and link:

            papers.append({
                "title": title,
                "abstract": abstract,
                "link": link,
                "source": "crossref",
            })

    return papers


# ==========================================
# collect / dedup
# ==========================================

def collect_papers():

    all_papers = []

    search_functions = [
        search_arxiv,
        search_openalex,
        search_semantic_scholar,
        search_crossref,
    ]

    for fn in search_functions:

        try:

            papers = fn()

            print(
                f"{fn.__name__}: "
                f"{len(papers)} papers"
            )

            all_papers.extend(papers)

        except Exception as e:

            print(f"{fn.__name__} failed: {e}")

    uniq = {}

    for p in all_papers:

        key = normalize_whitespace(
            p["title"]
        ).lower()

        if key and key not in uniq:
            uniq[key] = p

    print(f"unique papers: {len(uniq)}")

    return list(uniq.values())


# ==========================================
# summarize
# ==========================================

def summarize(title, abstract):

    abstract = (
        abstract
        or "No abstract available."
    )

    prompt = f"""
以下の論文を日本語で5〜10行で要約してください。

特に以下を優先してください。

- EBL overlay
- 二層アライメント
- alignment mark
- stitching
- overlay metrology
- box in box
- pattern placement error
- waveguide alignment
- InP / III-V process
- dry etch / regrowth

研究者向けに、
加工観点・位置合わせ観点を
優先して要約してください。

title:
{title}

abstract:
{abstract}
"""

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
    )

    return (
        r.choices[0]
        .message.content
        .strip()
    )


# ==========================================
# email
# ==========================================

def send_email(body):

    sender = os.environ["SENDER_EMAIL"]

    recipient = os.environ["RECIPIENT_EMAIL"]

    password = os.environ["SMTP_PASSWORD"]

    msg = EmailMessage()

    msg["To"] = recipient

    msg["From"] = sender

    msg["Subject"] = (
        "EBL / Overlay / Alignment "
        "Paper Digest"
    )

    msg.set_content(body)

    with smtplib.SMTP_SSL(
        "smtp.gmail.com",
        465
    ) as smtp:

        smtp.login(sender, password)

        smtp.send_message(msg)


# ==========================================
# main
# ==========================================

def main():

    db = clean_db(load_db())

    papers = collect_papers()

    scored = []

    for p in papers:

        text = (
            f"{p.get('title', '')} "
            f"{p.get('abstract', '')}"
        )

        s = score_paper(text)

        if s <= 0:
            continue

        p["score"] = s

        scored.append(p)

    scored.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    selected = []

    for p in scored:

        if len(selected) == 5:
            break

        if p["link"] in db:
            continue

        selected.append(p)

    if len(selected) == 0:

        send_email("該当論文なし")

        return

    body_lines = []

    body_lines.append(
        "EBL / Overlay / Alignment "
        "Paper Digest"
    )

    body_lines.append("")

    for p in selected:

        summary = summarize(
            p["title"],
            p.get("abstract", "")
        )

        body_lines.append(
            f"Source: "
            f"{p.get('source', 'unknown')}"
        )

        body_lines.append(
            f"Score: {p['score']}"
        )

        body_lines.append(p["title"])

        body_lines.append(p["link"])

        body_lines.append("")

        body_lines.append(summary)

        body_lines.append("")

        body_lines.append(
            "--------------------------"
        )

        db[p["link"]] = {
            "sent_at":
                datetime.now(
                    timezone.utc
                ).isoformat()
        }

    save_db(db)

    send_email("\n".join(body_lines))


if __name__ == "__main__":
    main()
