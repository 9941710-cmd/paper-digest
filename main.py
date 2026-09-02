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
# Search queries
#
# ポイント：
# 1個の巨大な検索式にせず、複数の比較的広い検索を実行する。
# その後score_paper()で順位付けする。
# ============================================================

SEARCH_QUERIES = [

    # HOE / holography
    "holographic optical element",
    "volume hologram",
    "volume holographic grating",
    "volume phase grating",
    "holographic grating",

    # recording / exposure
    "holographic recording",
    "interference exposure",
    "interference lithography",
    "laser interference lithography",
    "two beam interference",

    # materials
    "photopolymer hologram",
    "holographic photopolymer",
    "refractive index modulation",
    "photosensitive holographic material",

    # inspection / characterization
    "diffraction efficiency grating",
    "angular selectivity hologram",
    "spectral selectivity hologram",
    "wavefront grating",
    "grating uniformity",
    "diffractive optical metrology",

    # AR / waveguide
    "diffractive waveguide",
    "holographic waveguide",
    "waveguide display",
    "augmented reality waveguide",
    "near eye display grating",

    # exposure equipment related
    "spatial light modulator lithography",
    "spatial light modulator holography",
    "laser direct writing grating",
    "phase modulation holography",

    # broad fallback
    "diffraction grating fabrication",
    "diffractive optical element fabrication",
    "micro optical fabrication",

    # future themes
    "co-packaged optics",
    "optical interconnect",
    "silicon photonics",
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

    # 60日に短縮。
    # 古い論文が再候補になる余地を少し増やす。
    limit = (
        datetime.now(timezone.utc)
        - timedelta(days=60)
    )

    new_db = {}

    for key, value in db.items():

        try:

            sent_at = datetime.fromisoformat(
                value["sent_at"]
            )

            if sent_at.tzinfo is None:

                sent_at = sent_at.replace(
                    tzinfo=timezone.utc
                )

            if sent_at > limit:
                new_db[key] = value

        except Exception:
            continue

    return new_db


# ============================================================
# Utility
# ============================================================

def normalize_whitespace(text):

    if not text:
        return ""

    return " ".join(
        str(text).split()
    )


def normalize_title(title):

    title = title.lower()

    title = re.sub(
        r"[^a-z0-9]+",
        "",
        title
    )

    return title


def strip_html_tags(text):

    if not text:
        return ""

    text = html.unescape(text)

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def build_abstract_from_inverted_index(inv):

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

            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout
            )

            response.raise_for_status()

            return response

        except Exception as e:

            last_error = e

            print(
                f"request failed "
                f"{i + 1}/{retries}: {e}"
            )

            if i < retries - 1:
                time.sleep(sleep_sec)

    raise last_error


# ============================================================
# Scoring
# ============================================================

def score_paper(text):

    t = (text or "").lower()

    score = 0


    # ========================================================
    # HOE / volume hologram
    # ========================================================

    if "holographic optical element" in t:
        score += 50

    if re.search(r"\bhoe\b", t):
        score += 35

    if "volume hologram" in t:
        score += 35

    if "volume holographic" in t:
        score += 35

    if "volume phase grating" in t:
        score += 30

    if "holographic grating" in t:
        score += 28

    if "holographic waveguide" in t:
        score += 35


    # ========================================================
    # Recording / exposure
    # ========================================================

    if "holographic recording" in t:
        score += 30

    if "interference exposure" in t:
        score += 30

    if "interference lithography" in t:
        score += 25

    if "laser interference lithography" in t:
        score += 30

    if "two beam interference" in t:
        score += 22

    if "two-beam interference" in t:
        score += 22

    if "direct laser writing" in t:
        score += 15

    if "laser direct writing" in t:
        score += 15

    if "spatial light modulator" in t:
        score += 18

    if re.search(r"\bslm\b", t):
        score += 10

    if "phase modulation" in t:
        score += 12

    if "exposure uniformity" in t:
        score += 25

    if "dose uniformity" in t:
        score += 15


    # ========================================================
    # Materials
    # ========================================================

    if "photopolymer" in t:
        score += 25

    if "photosensitive material" in t:
        score += 15

    if "photosensitive film" in t:
        score += 15

    if "holographic material" in t:
        score += 22

    if "refractive index modulation" in t:
        score += 28

    if "index modulation" in t:
        score += 18

    if "photoresist" in t:
        score += 5


    # ========================================================
    # Evaluation / metrology
    # ========================================================

    if "diffraction efficiency" in t:
        score += 30

    if "angular selectivity" in t:
        score += 25

    if "spectral selectivity" in t:
        score += 22

    if "wavefront" in t:
        score += 15

    if "wavefront measurement" in t:
        score += 10

    if "wavefront metrology" in t:
        score += 15

    if "interferometry" in t:
        score += 10

    if "optical metrology" in t:
        score += 12

    if "uniformity" in t:
        score += 10

    if "grating uniformity" in t:
        score += 15

    if "scatter" in t:
        score += 6

    if "scattering" in t:
        score += 6


    # ========================================================
    # Diffractive optics / AR
    # ========================================================

    if "diffractive optical element" in t:
        score += 18

    if re.search(r"\bdoe\b", t):
        score += 8

    if "diffraction grating" in t:
        score += 15

    if "diffractive waveguide" in t:
        score += 25

    if "waveguide display" in t:
        score += 25

    if "augmented reality" in t:
        score += 12

    if "near-eye display" in t:
        score += 12

    if "near eye display" in t:
        score += 12

    if "pupil expansion" in t:
        score += 15

    if "exit pupil" in t:
        score += 15


    # ========================================================
    # Fabrication
    # ========================================================

    if "fabrication" in t:
        score += 5

    if "nanofabrication" in t:
        score += 8

    if "microfabrication" in t:
        score += 8

    if "nanoimprint" in t:
        score += 8

    if "electron beam lithography" in t:
        score += 5

    if "dry etch" in t:
        score += 4

    if "reactive ion etching" in t:
        score += 4


    # ========================================================
    # Future: optical communication / CPO
    # ========================================================

    if "co-packaged optics" in t:
        score += 12

    if re.search(r"\bcpo\b", t):
        score += 8

    if "optical interconnect" in t:
        score += 10

    if "data center" in t and "optical" in t:
        score += 8

    if "silicon photonics" in t:
        score += 7

    if "inp photonics" in t:
        score += 7

    if "heterogeneous integration" in t:
        score += 6


    # ========================================================
    # Combination bonuses
    # ========================================================

    holography_terms = [
        "hologram",
        "holographic",
        "volume grating",
        "volume phase",
    ]

    exposure_terms = [
        "exposure",
        "recording",
        "interference",
        "writing",
    ]

    measurement_terms = [
        "diffraction efficiency",
        "selectivity",
        "wavefront",
        "uniformity",
        "metrology",
        "measurement",
    ]

    material_terms = [
        "photopolymer",
        "photosensitive",
        "index modulation",
    ]

    has_holography = any(
        x in t for x in holography_terms
    )

    has_exposure = any(
        x in t for x in exposure_terms
    )

    has_measurement = any(
        x in t for x in measurement_terms
    )

    has_material = any(
        x in t for x in material_terms
    )

    if has_holography and has_exposure:
        score += 25

    if has_holography and has_measurement:
        score += 25

    if has_holography and has_material:
        score += 20

    if (
        has_holography
        and has_exposure
        and has_measurement
    ):
        score += 25


    # ========================================================
    # Noise reduction
    # ========================================================

    if "holographic microscopy" in t:
        score -= 40

    if "digital holographic microscopy" in t:
        score -= 40

    if "biomedical" in t:
        score -= 25

    if "biological" in t:
        score -= 20

    if "medical imaging" in t:
        score -= 25

    if "cell imaging" in t:
        score -= 25

    if "acoustic holography" in t:
        score -= 30

    return score


# ============================================================
# arXiv
# ============================================================

def search_arxiv():

    papers = []

    # arXivはクエリを分割して検索する
    # → HOEという単語がない関連論文も拾いやすい
    arxiv_queries = [

        (
            'all:"volume hologram" '
            'OR all:"holographic grating" '
            'OR all:"holographic recording"'
        ),

        (
            'all:"photopolymer" '
            'OR all:"refractive index modulation"'
        ),

        (
            'all:"diffraction efficiency" '
            'OR all:"angular selectivity"'
        ),

        (
            'all:"diffractive waveguide" '
            'OR all:"waveguide display"'
        ),

        (
            'all:"interference lithography" '
            'OR all:"laser interference"'
        ),

        (
            'all:"diffractive optical element" '
            'OR all:"diffraction grating"'
        ),
    ]

    categories = " OR ".join(
        f"cat:{cat}"
        for cat in CATEGORIES
    )

    for query in arxiv_queries:

        try:

            params = {
                "search_query":
                    f"({categories}) AND ({query})",
                "start": 0,
                "max_results": 30,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }

            response = request_with_retry(
                ARXIV_URL,
                params=params,
                timeout=60
            )

            root = ET.fromstring(
                response.text
            )

            ns = {
                "atom":
                    "http://www.w3.org/2005/Atom"
            }

            for entry in root.findall(
                "atom:entry",
                ns
            ):

                title_node = entry.find(
                    "atom:title",
                    ns
                )

                abstract_node = entry.find(
                    "atom:summary",
                    ns
                )

                link_node = entry.find(
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

            time.sleep(1)

        except Exception as e:

            print(
                "arXiv query failed:",
                query,
                e
            )

    return papers


# ============================================================
# OpenAlex
# ============================================================

def search_openalex():

    papers = []

    headers = {}

    if OPENALEX_EMAIL:

        headers["User-Agent"] = (
            "paper-digest/1.0 "
            f"(mailto:{OPENALEX_EMAIL})"
        )

    # OpenAlexも複数の短い検索にする
    queries = [

        "volume hologram",

        "holographic grating",

        "holographic recording",

        "photopolymer hologram",

        "diffraction efficiency hologram",

        "angular selectivity hologram",

        "diffractive waveguide",

        "waveguide display",

        "interference lithography",

        "diffraction grating fabrication",

        "spatial light modulator holography",

        "diffractive optical element",
    ]

    for query in queries:

        try:

            params = {
                "search": query,
                "sort":
                    "publication_date:desc",
                "per-page": 15,
            }

            response = request_with_retry(
                "https://api.openalex.org/works",
                params=params,
                headers=headers,
                timeout=60
            )

            data = response.json()

            for work in data.get(
                "results",
                []
            ):

                title = normalize_whitespace(
                    work.get(
                        "display_name",
                        ""
                    )
                )

                abstract = (
                    build_abstract_from_inverted_index(
                        work.get(
                            "abstract_inverted_index"
                        )
                    )
                )

                primary_location = (
                    work.get(
                        "primary_location"
                    )
                    or {}
                )

                link = (
                    work.get("doi")
                    or primary_location.get(
                        "landing_page_url"
                    )
                    or work.get("id")
                    or ""
                )

                link = normalize_whitespace(
                    link
                )

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

            time.sleep(0.5)

        except Exception as e:

            print(
                "OpenAlex query failed:",
                query,
                e
            )

    return papers


# ============================================================
# Semantic Scholar
# ============================================================

def search_semantic_scholar():

    papers = []

    queries = [
        "volume hologram",
        "holographic grating",
        "photopolymer hologram",
        "diffractive waveguide",
    ]

    headers = {}

    if SEMANTIC_SCHOLAR_API_KEY:

        headers["x-api-key"] = (
            SEMANTIC_SCHOLAR_API_KEY
        )

    for query in queries:

        try:

            params = {
                "query": query,
                "limit": 10,
                "fields":
                    "title,abstract,url,"
                    "year,publicationDate",
            }

            response = request_with_retry(
                (
                    "https://api.semanticscholar.org/"
                    "graph/v1/paper/search"
                ),
                params=params,
                headers=headers,
                timeout=60,
                retries=2,
                sleep_sec=10
            )

            data = response.json()

            for paper in data.get(
                "data",
                []
            ):

                title = normalize_whitespace(
                    paper.get(
                        "title",
                        ""
                    )
                )

                abstract = normalize_whitespace(
                    paper.get(
                        "abstract",
                        ""
                    )
                )

                link = normalize_whitespace(
                    paper.get(
                        "url",
                        ""
                    )
                )

                if title and link:

                    papers.append({
                        "title": title,
                        "abstract": abstract,
                        "link": link,
                        "source":
                            "Semantic Scholar",
                    })

            time.sleep(2)

        except Exception as e:

            # 429等でも全体を止めない
            print(
                "Semantic Scholar failed:",
                query,
                e
            )

    return papers


# ============================================================
# Crossref
# ============================================================

def search_crossref():

    papers = []

    queries = [
        "volume hologram",
        "holographic grating",
        "photopolymer holography",
        "diffractive waveguide",
        "diffraction grating fabrication",
    ]

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

    for query in queries:

        try:

            params = {
                "query": query,
                "rows": 15,
                "sort": "published",
                "order": "desc",
                "select":
                    "DOI,title,abstract,"
                    "URL,published",
            }

            response = request_with_retry(
                "https://api.crossref.org/works",
                params=params,
                headers=headers,
                timeout=60
            )

            data = response.json()

            items = (
                data.get(
                    "message",
                    {}
                )
                .get(
                    "items",
                    []
                )
            )

            for item in items:

                titles = item.get(
                    "title",
                    []
                )

                title = (
                    normalize_whitespace(
                        titles[0]
                    )
                    if titles
                    else ""
                )

                abstract = strip_html_tags(
                    item.get(
                        "abstract",
                        ""
                    )
                )

                link = normalize_whitespace(
                    item.get(
                        "URL",
                        ""
                    )
                )

                if title and link:

                    papers.append({
                        "title": title,
                        "abstract": abstract,
                        "link": link,
                        "source": "Crossref",
                    })

            time.sleep(0.5)

        except Exception as e:

            print(
                "Crossref query failed:",
                query,
                e
            )

    return papers


# ============================================================
# Collect / deduplicate
# ============================================================

def collect_papers():

    all_papers = []

    search_functions = [
        search_arxiv,
        search_openalex,
        search_semantic_scholar,
        search_crossref,
    ]

    for function in search_functions:

        try:

            result = function()

            print(
                f"{function.__name__}: "
                f"{len(result)} papers"
            )

            all_papers.extend(result)

        except Exception as e:

            print(
                f"{function.__name__} failed:",
                e
            )

    unique = {}

    for paper in all_papers:

        key = normalize_title(
            paper["title"]
        )

        if (
            key
            and key not in unique
        ):
            unique[key] = paper

    print(
        "unique papers:",
        len(unique)
    )

    return list(
        unique.values()
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
以下の論文を、日本語で研究開発者向けに要約してください。

主な関心分野は、
HOE（Holographic Optical Element）の
膜形成、露光、ホログラム記録、
回折格子形成、光学評価、検査装置です。

ただしHOEそのものの論文でなくても、
HOEの露光・検査・材料・装置開発に
応用できそうな技術であれば評価してください。

以下の形式で回答してください。

【概要】
研究内容を2〜3行。

【HOEとの関連】
HOE、volume hologram、
photopolymer、diffractive waveguide、
干渉露光などとの関係。

直接HOEを扱っていない場合でも、
応用できそうならその理由を書く。

【露光・プロセス】
レーザー波長、干渉露光、
記録方式、膜材料、屈折率変調、
加工方法などがabstractにあれば説明。

【検査・評価】
回折効率、角度選択性、
波長選択性、wavefront、
uniformity、scatterなど。

【実務で使えそうな点】
HOE膜の露光技術や
検査装置開発の観点から、
役立ちそうなポイントを説明。

abstractに存在しない数値や
条件は推測しないこと。

検索スコア:
{score}

Title:
{title}

Abstract:
{abstract}
"""

    response = (
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
        response
        .choices[0]
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

    for paper in papers:

        text = (
            paper.get(
                "title",
                ""
            )
            + " "
            + paper.get(
                "abstract",
                ""
            )
        )

        score = score_paper(
            text
        )

        # 以前は0点以下を除外していたが、
        # 今回は5点以上なら候補にする
        if score < 5:
            continue

        paper["score"] = score

        scored.append(paper)

    scored.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    print("")
    print("==========================")
    print("TOP 20 CANDIDATES")
    print("==========================")

    for paper in scored[:20]:

        print(
            paper["score"],
            "|",
            paper["source"],
            "|",
            paper["title"]
        )

    print("==========================")
    print("")

    selected = []

    for paper in scored:

        if len(selected) >= 5:
            break

        if paper["link"] in db:
            continue

        selected.append(
            paper
        )


    # ========================================================
    # Fallback
    #
    # スコア付き未送信論文が5本未満の場合、
    # score > 0 の関連論文から補充する。
    # ========================================================

    if len(selected) < 5:

        fallback = []

        for paper in papers:

            if paper["link"] in db:
                continue

            if paper in selected:
                continue

            text = (
                paper.get(
                    "title",
                    ""
                )
                + " "
                + paper.get(
                    "abstract",
                    ""
                )
            )

            score = score_paper(
                text
            )

            if score > 0:

                paper["score"] = score

                fallback.append(
                    paper
                )

        fallback.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        for paper in fallback:

            if len(selected) >= 5:
                break

            selected.append(
                paper
            )


    # ========================================================
    # Still nothing
    # ========================================================

    if len(selected) == 0:

        send_email(
            "本日はHOE・露光・検査・回折光学周辺で、"
            "未配信の候補論文を取得できませんでした。"
        )

        return


    # ========================================================
    # Build email
    # ========================================================

    body_lines = []

    body_lines.append(
        "HOE / Exposure / "
        "Metrology Paper Digest"
    )

    body_lines.append("")

    body_lines.append(
        "HOE膜・露光・検査に加え、"
        "回折光学・材料・周辺技術まで"
        "広めに検索しています。"
    )

    body_lines.append("")

    body_lines.append(
        "================================"
    )

    for index, paper in enumerate(
        selected,
        start=1
    ):

        summary = summarize(
            paper["title"],
            paper.get(
                "abstract",
                ""
            ),
            paper["score"]
        )

        body_lines.append("")
        body_lines.append(
            f"【{index}】"
        )

        body_lines.append(
            f"Score: "
            f"{paper['score']}"
        )

        body_lines.append(
            f"Source: "
            f"{paper['source']}"
        )

        body_lines.append("")

        body_lines.append(
            paper["title"]
        )

        body_lines.append(
            paper["link"]
        )

        body_lines.append("")

        body_lines.append(
            summary
        )

        body_lines.append("")

        body_lines.append(
            "================================"
        )

        db[
            paper["link"]
        ] = {
            "sent_at":
                datetime.now(
                    timezone.utc
                ).isoformat()
        }


    save_db(db)

    send_email(
        "\n".join(
            body_lines
        )
    )


if __name__ == "__main__":
    main()
