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


# ============================================================
# Basic settings
# ============================================================

ARXIV_URL = "https://export.arxiv.org/api/query"

OPENALEX_EMAIL = os.environ.get("OPENALEX_EMAIL", "")
SEMANTIC_SCHOLAR_API_KEY = os.environ.get(
    "SEMANTIC_SCHOLAR_API_KEY", ""
)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ============================================================
# Search keywords
#
# Priority:
#   1. HOE / holographic exposure / inspection
#   2. HOE materials / optical properties
#   3. Micro-optics / fabrication
#   4. AR waveguide / diffractive optics
#   5. Future: CPO / optical interconnect / InP
# ============================================================

KEYWORDS = [

    # --------------------------------------------------------
    # HOE / holography
    # --------------------------------------------------------
    "holographic optical element",
    "HOE",
    "volume hologram",
    "volume holographic grating",
    "volume phase grating",
    "holographic grating",
    "holographic waveguide",
    "holographic recording",

    # --------------------------------------------------------
    # Exposure / recording
    # --------------------------------------------------------
    "holographic exposure",
    "interference exposure",
    "interference lithography",
    "two beam interference",
    "laser interference lithography",
    "holographic recording",
    "laser direct writing",
    "direct laser writing",
    "spatial light modulator",
    "SLM",
    "phase modulation",
    "exposure uniformity",
    "dose uniformity",

    # --------------------------------------------------------
    # HOE materials
    # --------------------------------------------------------
    "photopolymer",
    "photosensitive material",
    "photosensitive film",
    "holographic material",
    "refractive index modulation",
    "index modulation",
    "photoresist",
    "volume phase hologram",

    # --------------------------------------------------------
    # Inspection / metrology
    # --------------------------------------------------------
    "diffraction efficiency",
    "diffraction angle",
    "angular selectivity",
    "spectral selectivity",
    "wavefront measurement",
    "wavefront metrology",
    "interferometry",
    "phase measurement",
    "optical metrology",
    "grating uniformity",
    "diffraction uniformity",
    "scattering measurement",

    # --------------------------------------------------------
    # Diffractive optics / AR
    # --------------------------------------------------------
    "diffractive optical element",
    "DOE",
    "diffraction grating",
    "grating coupler",
    "waveguide display",
    "AR waveguide",
    "augmented reality waveguide",
    "near eye display",
    "near-eye display",
    "pupil expansion",
    "exit pupil expansion",

    # --------------------------------------------------------
    # Micro / nano fabrication
    # --------------------------------------------------------
    "micro optical element",
    "micro-optics",
    "nanofabrication",
    "nanoimprint",
    "nanoimprint lithography",
    "electron beam lithography",
    "EBL",
    "dry etch",
    "reactive ion etching",
    "RIE",
    "atomic layer deposition",
    "ALD",

    # --------------------------------------------------------
    # Future: optical communication / CPO
    # --------------------------------------------------------
    "co-packaged optics",
    "CPO",
    "optical interconnect",
    "data center optics",
    "short reach optical communication",
    "silicon photonics",
    "photonic integrated circuit",
    "PIC",
    "heterogeneous integration",
    "III-V integration",
    "InP photonics",
]


# ============================================================
# arXiv categories
# ============================================================

CATEGORIES = [
    "physics.optics",
    "physics.app-ph",
    "cond-mat.mtrl-sci",
]


# ============================================================
# Sent database
# ============================================================

def load_db():

    if not os.path.exists("sent_db.json"):
        return {}

    with open(
        "sent_db.json",
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def save_db(db):

    with open(
        "sent_db.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            db,
            f,
            indent=2,
            ensure_ascii=False
        )


def clean_db(db):

    limit = (
        datetime.now(timezone.utc)
        - timedelta(days=90)
    )

    new = {}

    for k, v in db.items():

        try:

            t = datetime.fromisoformat(
                v["sent_at"]
            )

            if t.tzinfo is None:
                t = t.replace(
                    tzinfo=timezone.utc
                )

            t = t.astimezone(
                timezone.utc
            )

            if t > limit:
                new[k] = v

        except Exception:
            continue

    return new


# ============================================================
# Utility
# ============================================================

def normalize_whitespace(s):

    if not s:
        return ""

    return " ".join(
        str(s).split()
    )


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
                f"request failed "
                f"({i + 1}/{retries}) "
                f"url={url} "
                f"error={e}"
            )

            if i < retries - 1:
                time.sleep(sleep_sec)

    raise last_error


def build_abstract_from_inverted_index(inv):

    if not inv:
        return ""

    positions = {}

    for word, pos_list in inv.items():

        for pos in pos_list:
            positions[pos] = word

    return " ".join(
        positions[i]
        for i in sorted(positions)
    )


def strip_html_tags(text):

    if not text:
        return ""

    text = html.unescape(text)

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# Scoring
# ============================================================

def score_paper(text):

    t = (text or "").lower()

    score = 0


    # ========================================================
    # TIER 1
    # HOE itself
    # ========================================================

    if "holographic optical element" in t:
        score += 40

    if re.search(r"\bhoe\b", t):
        score += 35

    if "volume hologram" in t:
        score += 30

    if "volume holographic grating" in t:
        score += 35

    if "volume phase grating" in t:
        score += 30

    if "holographic grating" in t:
        score += 28

    if "holographic waveguide" in t:
        score += 35

    if "holographic recording" in t:
        score += 25


    # ========================================================
    # TIER 1
    # Exposure / recording
    # ========================================================

    if "holographic exposure" in t:
        score += 35

    if "interference exposure" in t:
        score += 32

    if "interference lithography" in t:
        score += 28

    if "laser interference lithography" in t:
        score += 30

    if "two beam interference" in t:
        score += 25

    if "laser direct writing" in t:
        score += 18

    if "direct laser writing" in t:
        score += 18

    if "spatial light modulator" in t:
        score += 18

    if re.search(r"\bslm\b", t):
        score += 12

    if "phase modulation" in t:
        score += 15

    if "exposure uniformity" in t:
        score += 25

    if "dose uniformity" in t:
        score += 20


    # ========================================================
    # TIER 1
    # Inspection / metrology
    # ========================================================

    if "diffraction efficiency" in t:
        score += 30

    if "diffraction angle" in t:
        score += 20

    if "angular selectivity" in t:
        score += 28

    if "spectral selectivity" in t:
        score += 25

    if "wavefront measurement" in t:
        score += 25

    if "wavefront metrology" in t:
        score += 28

    if "interferometry" in t:
        score += 15

    if "phase measurement" in t:
        score += 15

    if "optical metrology" in t:
        score += 18

    if "grating uniformity" in t:
        score += 28

    if "diffraction uniformity" in t:
        score += 30

    if "scattering measurement" in t:
        score += 15


    # ========================================================
    # TIER 2
    # Materials / films
    # ========================================================

    if "photopolymer" in t:
        score += 28

    if "photosensitive material" in t:
        score += 20

    if "photosensitive film" in t:
        score += 22

    if "holographic material" in t:
        score += 25

    if "refractive index modulation" in t:
        score += 28

    if "index modulation" in t:
        score += 20

    if "volume phase hologram" in t:
        score += 30

    if "photoresist" in t:
        score += 8


    # ========================================================
    # TIER 2
    # AR / diffractive optics
    # ========================================================

    if "diffractive optical element" in t:
        score += 18

    if "diffraction grating" in t:
        score += 15

    if "waveguide display" in t:
        score += 25

    if "ar waveguide" in t:
        score += 25

    if "augmented reality waveguide" in t:
        score += 28

    if "near eye display" in t:
        score += 18

    if "near-eye display" in t:
        score += 18

    if "pupil expansion" in t:
        score += 20

    if "exit pupil expansion" in t:
        score += 22

    if "grating coupler" in t:
        score += 12


    # ========================================================
    # TIER 3
    # Micro / nano fabrication
    # ========================================================

    if "micro-optics" in t:
        score += 10

    if "micro optical element" in t:
        score += 10

    if "nanofabrication" in t:
        score += 8

    if "nanoimprint" in t:
        score += 8

    if "electron beam lithography" in t:
        score += 6

    if re.search(r"\bebl\b", t):
        score += 5

    if "dry etch" in t:
        score += 5

    if "reactive ion etching" in t:
        score += 5

    if "atomic layer deposition" in t:
        score += 4


    # ========================================================
    # TIER 4
    # Future optical communication / CPO
    # ========================================================

    if "co-packaged optics" in t:
        score += 14

    if re.search(r"\bcpo\b", t):
        score += 10

    if "optical interconnect" in t:
        score += 12

    if "data center optics" in t:
        score += 10

    if "short reach optical communication" in t:
        score += 10

    if "silicon photonics" in t:
        score += 8

    if "photonic integrated circuit" in t:
        score += 7

    if "heterogeneous integration" in t:
        score += 8

    if "iii-v integration" in t:
        score += 8

    if "inp photonics" in t:
        score += 8


    # ========================================================
    # Combination bonuses
    #
    # These are important.
    # A paper containing both HOE and measurement/exposure
    # should rank much higher than a generic holography paper.
    # ========================================================

    hoe_terms = [
        "holographic optical element",
        "volume hologram",
        "holographic grating",
        "holographic waveguide",
        "volume phase grating",
    ]

    exposure_terms = [
        "exposure",
        "recording",
        "interference",
        "laser",
        "spatial light modulator",
    ]

    measurement_terms = [
        "diffraction efficiency",
        "wavefront",
        "uniformity",
        "angular selectivity",
        "spectral selectivity",
        "metrology",
        "measurement",
    ]

    has_hoe = any(
        term in t
        for term in hoe_terms
    )

    has_exposure = any(
        term in t
        for term in exposure_terms
    )

    has_measurement = any(
        term in t
        for term in measurement_terms
    )

    if has_hoe and has_exposure:
        score += 25

    if has_hoe and has_measurement:
        score += 30

    if (
        has_hoe
        and has_exposure
        and has_measurement
    ):
        score += 30


    # ========================================================
    # Noise reduction
    # ========================================================

    if "digital holography" in t:
        score -= 12

    if "holographic microscopy" in t:
        score -= 20

    if "biomedical" in t:
        score -= 20

    if "biological" in t:
        score -= 15

    if "medical imaging" in t:
        score -= 20

    if "acoustic" in t:
        score -= 15

    return score


# ============================================================
# arXiv
# ============================================================

def search_arxiv():

    query = " OR ".join(
        [
            f'all:"{k}"'
            for k in KEYWORDS
        ]
    )

    cat = " OR ".join(
        [
            f"cat:{c}"
            for c in CATEGORIES
        ]
    )

    params = {
        "search_query":
            f"({cat}) AND ({query})",
        "start": 0,
        "max_results": 100,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    print("fetching arXiv")

    r = request_with_retry(
        ARXIV_URL,
        params=params,
        timeout=60
    )

    root = ET.fromstring(r.text)

    ns = {
        "atom":
            "http://www.w3.org/2005/Atom"
    }

    papers = []

    for e in root.findall(
        "atom:entry",
        ns
    ):

        title_node = e.find(
            "atom:title",
            ns
        )

        abstract_node = e.find(
            "atom:summary",
            ns
        )

        link_node = e.find(
            "atom:id",
            ns
        )

        title = normalize_whitespace(
            title_node.text
            if title_node is not None
            else ""
        )

        abstract = normalize_whitespace(
            abstract_node.text
            if abstract_node is not None
            else ""
        )

        link = normalize_whitespace(
            link_node.text
            if link_node is not None
            else ""
        )

        if title and link:

            papers.append({
                "title": title,
                "abstract": abstract,
                "link": link,
                "source": "arXiv",
            })

    return papers


# ============================================================
# OpenAlex
# ============================================================

def search_openalex():

    query = " OR ".join(KEYWORDS)

    url = (
        "https://api.openalex.org/works"
    )

    params = {
        "search": query,
        "sort": "publication_date:desc",
        "per-page": 50,
    }

    headers = {}

    if OPENALEX_EMAIL:

        headers["User-Agent"] = (
            "paper-digest/1.0 "
            f"(mailto:{OPENALEX_EMAIL})"
        )

    print("fetching OpenAlex")

    r = request_with_retry(
        url,
        params=params,
        headers=headers,
        timeout=60
    )

    data = r.json()

    papers = []

    for w in data.get(
        "results", []
    ):

        title = normalize_whitespace(
            w.get(
                "display_name", ""
            )
        )

        abstract = (
            build_abstract_from_inverted_index(
                w.get(
                    "abstract_inverted_index"
                )
            )
        )

        # Prefer DOI or landing page over
        # OpenAlex page when possible.
        doi = w.get("doi")

        primary_location = (
            w.get("primary_location")
            or {}
        )

        link = (
            doi
            or primary_location.get(
                "landing_page_url"
            )
            or w.get("id")
            or ""
        )

        link = normalize_whitespace(link)

        if title and link:

            papers.append({
                "title": title,
                "abstract":
                    normalize_whitespace(
                        abstract
                    ),
                "link": link,
                "source": "OpenAlex",
            })

    return papers


# ============================================================
# Semantic Scholar
# ============================================================

def search_semantic_scholar():

    # Keep this shorter because the public endpoint
    # is more sensitive to rate limits.
    query_terms = [
        "holographic optical element",
        "holographic exposure",
        "volume hologram",
        "diffraction efficiency",
        "waveguide display",
        "photopolymer",
    ]

    query = " OR ".join(query_terms)

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

    print(
        "fetching Semantic Scholar"
    )

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
            "Semantic Scholar failed:",
            e
        )

        return []

    papers = []

    for p in data.get(
        "data", []
    ):

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
                "source":
                    "Semantic Scholar",
            })

    return papers


# ============================================================
# Crossref
# ============================================================

def search_crossref():

    query = (
        "holographic optical element "
        "volume hologram "
        "holographic exposure "
        "diffraction efficiency "
        "waveguide display "
        "photopolymer"
    )

    url = (
        "https://api.crossref.org/works"
    )

    params = {
        "query": query,
        "rows": 30,
        "sort": "published",
        "order": "desc",
        "select":
            "DOI,title,abstract,"
            "URL,published",
    }

    if OPENALEX_EMAIL:

        user_agent = (
            "paper-digest/1.0 "
            f"(mailto:{OPENALEX_EMAIL})"
        )

    else:

        user_agent = (
            "paper-digest/1.0"
        )

    headers = {
        "User-Agent": user_agent
    }

    print("fetching Crossref")

    r = request_with_retry(
        url,
        params=params,
        headers=headers,
        timeout=60
    )

    data = r.json()

    papers = []

    items = (
        data.get("message", {})
        .get("items", [])
    )

    for item in items:

        title_list = item.get(
            "title", []
        )

        title = (
            normalize_whitespace(
                title_list[0]
            )
            if title_list
            else ""
        )

        abstract = strip_html_tags(
            item.get(
                "abstract", ""
            )
        )

        link = normalize_whitespace(
            item.get("URL", "")
        )

        if title and link:

            papers.append({
                "title": title,
                "abstract": abstract,
                "link": link,
                "source": "Crossref",
            })

    return papers


# ============================================================
# Collect / deduplicate
# ============================================================

def normalize_title_for_dedup(title):

    title = title.lower()

    title = re.sub(
        r"[^a-z0-9]+",
        "",
        title
    )

    return title


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

            all_papers.extend(
                papers
            )

        except Exception as e:

            print(
                f"{fn.__name__} "
                f"failed: {e}"
            )

    uniq = {}

    for p in all_papers:

        key = (
            normalize_title_for_dedup(
                p["title"]
            )
        )

        if (
            key
            and key not in uniq
        ):
            uniq[key] = p

    print(
        "unique papers:",
        len(uniq)
    )

    return list(
        uniq.values()
    )


# ============================================================
# GPT summarization
# ============================================================

def summarize(
    title,
    abstract,
    score
):

    abstract = (
        abstract
        or "No abstract available."
    )

    prompt = f"""
以下の論文を、日本語で研究者向けに要約してください。

このDigestでは特に
HOE（Holographic Optical Element）の
露光技術・膜材料・検査評価技術を重視しています。

以下の順で書いてください。

【概要】
論文が何をした研究なのかを2〜3行。

【HOE・露光との関連】
HOE、ホログラフィック露光、干渉露光、
レーザー露光、回折格子形成との関連を説明。
直接関係がなければ「直接的な関連は薄い」と明記。

【材料・プロセス】
材料、膜、露光条件、波長、プロセス、
屈折率変調など具体的な情報がabstractにあれば記載。
abstractに無い数値や条件は推測しないこと。

【検査・評価】
回折効率、角度選択性、波長選択性、
wavefront、uniformity、scatterなど、
評価方法や結果があれば記載。

【実務的に気になる点】
HOEの露光装置・検査装置・プロセス開発の観点から、
この論文の有用性を1〜3行で説明。

検索スコア:
{score}

Title:
{title}

Abstract:
{abstract}
"""

    r = (
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.15,
        )
    )

    return (
        r.choices[0]
        .message.content
        .strip()
    )


# ============================================================
# Email
# ============================================================

def send_email(body):

    sender = os.environ[
        "SENDER_EMAIL"
    ]

    recipient = os.environ[
        "RECIPIENT_EMAIL"
    ]

    password = os.environ[
        "SMTP_PASSWORD"
    ]

    msg = EmailMessage()

    msg["To"] = recipient

    msg["From"] = sender

    msg["Subject"] = (
        "HOE / Exposure / "
        "Metrology Paper Digest"
    )

    msg.set_content(body)

    with smtplib.SMTP_SSL(
        "smtp.gmail.com",
        465
    ) as smtp:

        smtp.login(
            sender,
            password
        )

        smtp.send_message(msg)


# ============================================================
# Main
# ============================================================

def main():

    db = clean_db(
        load_db()
    )

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

    print(
        "scored papers:",
        len(scored)
    )

    # Debug: show top candidates
    print("\nTOP CANDIDATES")

    for p in scored[:10]:

        print(
            p["score"],
            p["source"],
            p["title"]
        )

    selected = []

    for p in scored:

        if len(selected) >= 5:
            break

        if p["link"] in db:
            continue

        selected.append(p)

    if len(selected) == 0:

        send_email(
            "本日は該当する新着論文がありませんでした。"
        )

        return

    body_lines = []

    body_lines.append(
        "HOE / Exposure / "
        "Metrology Paper Digest"
    )

    body_lines.append("")

    body_lines.append(
        "HOE膜・露光・検査技術を"
        "優先して選定した論文です。"
    )

    body_lines.append("")

    body_lines.append(
        "================================"
    )

    for index, p in enumerate(
        selected,
        start=1
    ):

        summary = summarize(
            p["title"],
            p.get(
                "abstract", ""
            ),
            p["score"]
        )

        body_lines.append("")
        body_lines.append(
            f"【{index}】"
        )

        body_lines.append(
            f"Score: {p['score']}"
        )

        body_lines.append(
            f"Source: "
            f"{p.get('source', 'unknown')}"
        )

        body_lines.append("")

        body_lines.append(
            p["title"]
        )

        body_lines.append(
            p["link"]
        )

        body_lines.append("")

        body_lines.append(
            summary
        )

        body_lines.append("")

        body_lines.append(
            "================================"
        )

        db[p["link"]] = {
            "sent_at":
                datetime.now(
                    timezone.utc
                ).isoformat()
        }

    save_db(db)

    send_email(
        "\n".join(body_lines)
    )


if __name__ == "__main__":
    main()
