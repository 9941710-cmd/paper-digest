import os
import re
import json
import time
import html
import smtplib
import requests
import xml.etree.ElementTree as ET

from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from openai import OpenAI


# ============================================================
# SETTINGS
# ============================================================

ARXIV_URL = "https://export.arxiv.org/api/query"

OPENALEX_URL = "https://api.openalex.org/works"

CROSSREF_URL = "https://api.crossref.org/works"

SEMANTIC_URL = (
    "https://api.semanticscholar.org/"
    "graph/v1/paper/search"
)

OPENALEX_EMAIL = os.environ.get(
    "OPENALEX_EMAIL", ""
)

SEMANTIC_SCHOLAR_API_KEY = os.environ.get(
    "SEMANTIC_SCHOLAR_API_KEY", ""
)

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

MAX_PAPERS = 5

DB_FILE = "sent_db.json"


# ============================================================
# SEARCH QUERIES
#
# 複数の短い検索式で幅広く取得する
# ============================================================

SEARCH_GROUPS = {

    "PVG": [
        "polarization volume grating",
        "polarization volume hologram",
        "polarization selective grating",
        "liquid crystal polarization grating",
        "liquid crystal volume grating",
        "Bragg polarization grating",
        "cholesteric liquid crystal grating",
        "geometric phase grating",
        "Pancharatnam Berry phase grating",
    ],

    "MLA": [
        "microlens array",
        "micro lens array",
        "microlens array fabrication",
        "microlens array replication",
        "polymer microlens array",
        "wafer level optics microlens",
        "freeform microlens array",
        "microlens array molding",
    ],

    "AR_DISPLAY": [
        "augmented reality waveguide",
        "AR waveguide display",
        "near eye display",
        "near-eye display",
        "waveguide combiner",
        "diffractive waveguide display",
        "holographic waveguide display",
        "exit pupil expansion",
        "pupil replication display",
        "light field near eye display",
    ],

    "PVG_PROCESS": [
        "photoalignment liquid crystal",
        "liquid crystal alignment exposure",
        "reactive mesogen grating",
        "polarization holography",
        "polarization interference lithography",
        "liquid crystal photopolymerization",
        "polarization grating fabrication",
        "liquid crystal optical alignment",
    ],

    "MLA_PROCESS": [
        "microlens electroforming",
        "nickel electroforming microlens",
        "microlens nanoimprint",
        "UV imprint microlens",
        "microlens injection molding",
        "microlens hot embossing",
        "polycarbonate microlens",
        "microlens replication",
    ],

    "METROLOGY": [
        "microlens array optical characterization",
        "microlens array wavefront",
        "microlens array MTF",
        "polarization grating diffraction efficiency",
        "polarization grating angular bandwidth",
        "waveguide display uniformity",
        "near eye display optical metrology",
    ],
}


# ============================================================
# UTILITIES
# ============================================================

def normalize(text):

    return " ".join(
        str(text or "").split()
    )


def normalize_title(title):

    return re.sub(
        r"[^a-z0-9]+",
        "",
        normalize(title).lower()
    )


def strip_html(text):

    text = html.unescape(
        str(text or "")
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    return normalize(text)


def reconstruct_abstract(inv):

    if not inv:
        return ""

    positions = {}

    for word, indexes in inv.items():

        for index in indexes:
            positions[index] = word

    return " ".join(
        positions[i]
        for i in sorted(positions)
    )


def request_with_retry(
    url,
    params=None,
    headers=None,
    retries=3
):

    last_error = None

    for attempt in range(retries):

        try:

            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=60
            )

            response.raise_for_status()

            return response

        except requests.RequestException as e:

            last_error = e

            print(
                f"Request failed "
                f"{attempt + 1}/{retries}: {e}"
            )

            if attempt < retries - 1:
                time.sleep(5)

    raise last_error


# ============================================================
# DATABASE
# ============================================================

def load_db():

    if not os.path.exists(DB_FILE):
        return {}

    with open(
        DB_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_db(db):

    with open(
        DB_FILE,
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

    cutoff = (
        datetime.now(timezone.utc)
        - timedelta(days=90)
    )

    cleaned = {}

    for key, value in db.items():

        try:

            sent_at = datetime.fromisoformat(
                value["sent_at"]
            )

            if sent_at.tzinfo is None:

                sent_at = sent_at.replace(
                    tzinfo=timezone.utc
                )

            if sent_at > cutoff:
                cleaned[key] = value

        except Exception:
            continue

    return cleaned


# ============================================================
# SCORING
# ============================================================

def contains(text, terms):

    return any(
        term in text
        for term in terms
    )


def score_paper(title, abstract):

    title = normalize(title).lower()

    abstract = normalize(abstract).lower()

    text = title + " " + abstract

    score = 0

    tags = []

    # --------------------------------------------------------
    # PVG
    # --------------------------------------------------------

    pvg_terms = [
        "polarization volume grating",
        "polarization volume hologram",
        "polarization selective grating",
        "liquid crystal polarization grating",
        "liquid crystal volume grating",
        "bragg polarization grating",
        "cholesteric liquid crystal grating",
    ]

    if contains(text, pvg_terms):

        score += 65
        tags.append("PVG")

    elif (
        "polarization grating" in text
        or "geometric phase grating" in text
        or "pancharatnam" in text
    ):

        score += 35
        tags.append("Polarization Grating")

    # --------------------------------------------------------
    # MLA
    # --------------------------------------------------------

    mla_terms = [
        "microlens array",
        "micro lens array",
        "micro-lens array",
        "microlens arrays",
        "microlens-array",
    ]

    if contains(text, mla_terms):

        score += 60
        tags.append("MLA")

    elif "microlens" in text:

        score += 25
        tags.append("Microlens")

    # --------------------------------------------------------
    # AR / Near-eye
    # --------------------------------------------------------

    ar_terms = [
        "augmented reality",
        "near-eye display",
        "near eye display",
        "waveguide display",
        "ar waveguide",
        "optical see-through",
        "head mounted display",
        "head-mounted display",
    ]

    has_ar = contains(text, ar_terms)

    if has_ar:

        score += 35
        tags.append("AR Display")

    # --------------------------------------------------------
    # PVG PROCESS
    # --------------------------------------------------------

    pvg_process = [
        "photoalignment",
        "photo-alignment",
        "reactive mesogen",
        "polarization holography",
        "polarization interference",
        "liquid crystal alignment",
        "cholesteric liquid crystal",
    ]

    if contains(text, pvg_process):

        score += 25
        tags.append("PVG Process")

    # --------------------------------------------------------
    # MLA PROCESS
    # --------------------------------------------------------

    mla_process = [
        "electroforming",
        "nickel mold",
        "nickel mould",
        "uv imprint",
        "uv nanoimprint",
        "hot embossing",
        "injection molding",
        "injection moulding",
        "replication process",
        "polycarbonate",
    ]

    if contains(text, mla_process):

        score += 18
        tags.append("MLA Process")

    # --------------------------------------------------------
    # OPTICAL METROLOGY
    # --------------------------------------------------------

    measurement_terms = [
        "diffraction efficiency",
        "angular selectivity",
        "angular bandwidth",
        "polarization efficiency",
        "wavefront aberration",
        "modulation transfer function",
        "optical uniformity",
        "focal length measurement",
        "surface profile measurement",
    ]

    if contains(text, measurement_terms):

        score += 20
        tags.append("Metrology")

    # --------------------------------------------------------
    # OPTICAL DESIGN
    # --------------------------------------------------------

    optical_terms = [
        "pupil expansion",
        "exit pupil expansion",
        "pupil replication",
        "light field display",
        "light-field display",
        "waveguide combiner",
        "eyebox",
        "eye box",
    ]

    if contains(text, optical_terms):

        score += 20
        tags.append("Optical Design")

    # --------------------------------------------------------
    # COMBINATION BONUS
    # --------------------------------------------------------

    has_pvg = (
        "PVG" in tags
        or "Polarization Grating" in tags
    )

    has_mla = (
        "MLA" in tags
        or "Microlens" in tags
    )

    if has_pvg and has_ar:
        score += 45

    if has_mla and has_ar:
        score += 45

    if has_pvg and "PVG Process" in tags:
        score += 30

    if has_mla and "MLA Process" in tags:
        score += 30

    if has_pvg and "Metrology" in tags:
        score += 20

    if has_mla and "Metrology" in tags:
        score += 20

    # --------------------------------------------------------
    # TITLE BONUS
    # --------------------------------------------------------

    if contains(title, pvg_terms):
        score += 30

    if contains(title, mla_terms):
        score += 30

    if has_ar and contains(title, ar_terms):
        score += 15

    # --------------------------------------------------------
    # NOISE REDUCTION
    # --------------------------------------------------------

    negative_terms = [
        "biomedical",
        "cell imaging",
        "medical imaging",
        "acoustic",
        "ultrasound",
        "astronomy",
    ]

    for term in negative_terms:

        if term in text:
            score -= 20

    return score, tags


# ============================================================
# ARXIV
# ============================================================

def search_arxiv():

    papers = []

    queries = [
        'all:"polarization grating"',
        'all:"liquid crystal grating"',
        'all:"microlens array"',
        'all:"near-eye display"',
        'all:"waveguide display"',
        'all:"photoalignment"',
        'all:"light field display"',
    ]

    for query in queries:

        try:

            params = {
                "search_query": query,
                "start": 0,
                "max_results": 30,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }

            response = request_with_retry(
                ARXIV_URL,
                params=params
            )

            root = ET.fromstring(
                response.content
            )

            ns = {
                "atom":
                    "http://www.w3.org/2005/Atom"
            }

            for entry in root.findall(
                "atom:entry", ns
            ):

                title = normalize(
                    entry.findtext(
                        "atom:title",
                        default="",
                        namespaces=ns
                    )
                )

                abstract = normalize(
                    entry.findtext(
                        "atom:summary",
                        default="",
                        namespaces=ns
                    )
                )

                link = normalize(
                    entry.findtext(
                        "atom:id",
                        default="",
                        namespaces=ns
                    )
                )

                if title and link:

                    papers.append({
                        "title": title,
                        "abstract": abstract,
                        "link": link,
                        "source": "arXiv",
                    })

            time.sleep(1)

        except Exception as e:

            print(
                "arXiv error:",
                query,
                e
            )

    return papers


# ============================================================
# OPENALEX
# ============================================================

def search_openalex():

    papers = []

    queries = []

    for group in SEARCH_GROUPS.values():
        queries.extend(group)

    # 代表的な検索語を優先して使用
    queries = list(
        dict.fromkeys(queries)
    )

    headers = {}

    if OPENALEX_EMAIL:

        headers["User-Agent"] = (
            "paper-digest/1.0 "
            f"(mailto:{OPENALEX_EMAIL})"
        )

    for query in queries:

        try:

            params = {
                "search": query,
                "sort": "publication_date:desc",
                "per-page": 10,
            }

            response = request_with_retry(
                OPENALEX_URL,
                params=params,
                headers=headers
            )

            data = response.json()

            for work in data.get(
                "results", []
            ):

                title = normalize(
                    work.get("display_name", "")
                )

                abstract = reconstruct_abstract(
                    work.get(
                        "abstract_inverted_index"
                    )
                )

                location = (
                    work.get("primary_location")
                    or {}
                )

                link = (
                    work.get("doi")
                    or location.get(
                        "landing_page_url"
                    )
                    or work.get("id")
                    or ""
                )

                if title and link:

                    papers.append({
                        "title": title,
                        "abstract": abstract,
                        "link": link,
                        "source": "OpenAlex",
                    })

            time.sleep(0.2)

        except Exception as e:

            print(
                "OpenAlex error:",
                query,
                e
            )

    return papers


# ============================================================
# SEMANTIC SCHOLAR
# ============================================================

def search_semantic_scholar():

    papers = []

    queries = [
        "polarization volume grating",
        "microlens array",
        "near eye display",
    ]

    headers = {}

    if SEMANTIC_SCHOLAR_API_KEY:

        headers["x-api-key"] = (
            SEMANTIC_SCHOLAR_API_KEY
        )

    for query in queries:

        try:

            response = request_with_retry(
                SEMANTIC_URL,
                params={
                    "query": query,
                    "limit": 10,
                    "fields":
                        "title,abstract,url,year"
                },
                headers=headers,
                retries=2
            )

            for paper in response.json().get(
                "data", []
            ):

                title = normalize(
                    paper.get("title", "")
                )

                link = normalize(
                    paper.get("url", "")
                )

                if title and link:

                    papers.append({
                        "title": title,
                        "abstract": normalize(
                            paper.get(
                                "abstract", ""
                            )
                        ),
                        "link": link,
                        "source":
                            "Semantic Scholar",
                    })

            time.sleep(2)

        except Exception as e:

            print(
                "Semantic Scholar error:",
                query,
                e
            )

    return papers


# ============================================================
# CROSSREF
# ============================================================

def search_crossref():

    papers = []

    queries = [
        "polarization volume grating",
        "microlens array augmented reality",
        "liquid crystal grating display",
        "microlens array fabrication",
        "near eye display optics",
    ]

    headers = {
        "User-Agent": (
            "paper-digest/1.0"
        )
    }

    for query in queries:

        try:

            response = request_with_retry(
                CROSSREF_URL,
                params={
                    "query": query,
                    "rows": 15,
                    "sort": "published",
                    "order": "desc",
                    "select":
                        "DOI,title,abstract,"
                        "URL,published",
                },
                headers=headers
            )

            items = (
                response.json()
                .get("message", {})
                .get("items", [])
            )

            for item in items:

                titles = item.get(
                    "title", []
                )

                title = (
                    normalize(titles[0])
                    if titles
                    else ""
                )

                link = normalize(
                    item.get("URL", "")
                )

                if title and link:

                    papers.append({
                        "title": title,
                        "abstract": strip_html(
                            item.get(
                                "abstract", ""
                            )
                        ),
                        "link": link,
                        "source": "Crossref",
                    })

            time.sleep(0.5)

        except Exception as e:

            print(
                "Crossref error:",
                query,
                e
            )

    return papers


# ============================================================
# COLLECT
# ============================================================

def collect_papers():

    papers = []

    functions = [
        search_arxiv,
        search_openalex,
        search_semantic_scholar,
        search_crossref,
    ]

    for function in functions:

        try:

            result = function()

            print(
                function.__name__,
                len(result)
            )

            papers.extend(result)

        except Exception as e:

            print(
                "Search failed:",
                function.__name__,
                e
            )

    unique = {}

    for paper in papers:

        key = normalize_title(
            paper["title"]
        )

        if key and key not in unique:

            unique[key] = paper

    return list(unique.values())


# ============================================================
# SUMMARY
# ============================================================

def summarize(paper):

    prompt = f"""
以下の論文を日本語で要約してください。

対象分野：
・マイクロレンズアレイ（MLA）
・偏光体積格子（PVG）
・ARディスプレイ
・Near-eye Display
・光学素子の製造・検査技術

以下の形式で回答してください。

【研究概要】
研究の目的と内容を簡潔に説明。

【技術的なポイント】
新規性、光学設計、材料、構造、
製造技術などを説明。

【製造プロセス】
特に以下に注目すること。

MLA：
・マスター作製
・Ni電鋳
・金型
・UVインプリント
・ポリカーボネート
・射出成形
・形状転写精度

PVG：
・液晶材料
・Photoalignment
・偏光露光
・配向制御
・重合・硬化
・膜厚
・回折効率

【評価技術】
回折効率、偏光特性、MTF、
収差、均一性、光学性能など。

【ARディスプレイへの応用】
AR光学系への応用可能性を説明。
直接関係しない場合は、その旨を明記。

【実務への有用性】
MLA・PVGの製造装置、プロセス、
検査技術の開発に役立つ点を説明。

論文に記載されていない具体的数値や
実験結果は推測しないこと。

Title:
{paper["title"]}

Abstract:
{paper["abstract"]}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.15,
    )

    return (
        response.choices[0]
        .message.content.strip()
    )


# ============================================================
# EMAIL
# ============================================================

def send_email(body):

    sender = os.environ["SENDER_EMAIL"]

    recipient = os.environ["RECIPIENT_EMAIL"]

    password = os.environ["SMTP_PASSWORD"]

    msg = EmailMessage()

    msg["From"] = sender
    msg["To"] = recipient

    msg["Subject"] = (
        "AR Display / MLA / PVG Paper Digest"
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
# MAIN
# ============================================================

def main():

    db = clean_db(
        load_db()
    )

    papers = collect_papers()

    print(
        "Total unique papers:",
        len(papers)
    )

    scored = []

    for paper in papers:

        score, tags = score_paper(
            paper["title"],
            paper["abstract"]
        )

        if score <= 0:
            continue

        paper["score"] = score
        paper["tags"] = tags

        scored.append(paper)

    scored.sort(
        key=lambda p: p["score"],
        reverse=True
    )

    print("\nTOP 20 CANDIDATES")

    for paper in scored[:20]:

        print(
            paper["score"],
            paper["tags"],
            paper["title"]
        )

    selected = []

    for paper in scored:

        if len(selected) >= MAX_PAPERS:
            break

        if paper["link"] in db:
            continue

        selected.append(paper)

    if not selected:

        send_email(
            "本日はMLA・PVG・ARディスプレイに"
            "関連する未配信論文を取得できませんでした。"
        )

        return

    lines = [
        "AR Display / MLA / PVG Paper Digest",
        "",
        "マイクロレンズアレイ・偏光体積格子・"
        "ARディスプレイ関連論文",
        "",
        "================================",
    ]

    sent_at = datetime.now(
        timezone.utc
    ).isoformat()

    for index, paper in enumerate(
        selected,
        start=1
    ):

        summary = summarize(paper)

        lines.extend([
            "",
            f"【{index}】",
            f"Score: {paper['score']}",
            f"Category: {', '.join(paper['tags'])}",
            f"Source: {paper['source']}",
            "",
            paper["title"],
            paper["link"],
            "",
            summary,
            "",
            "================================",
        ])

        db[paper["link"]] = {
            "sent_at": sent_at
        }

    # メール送信成功後に履歴保存
    send_email(
        "\n".join(lines)
    )

    save_db(db)


if __name__ == "__main__":
    main()
