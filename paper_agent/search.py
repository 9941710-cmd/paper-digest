import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import feedparser
import requests
from dateutil import parser as dateparser
from urllib.parse import quote_plus


logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

# MUST include metasurface
META_TERMS = [
    "metasurface", "metasurfaces", '"meta surface"', '"meta-surface"'
]

# process-focused: want papers about fabrication / ALD / etch etc.
PROCESS_TERMS = [
    "ALD", '"atomic layer deposition"', "PEALD", '"plasma-enhanced ALD"',
    "ALE", '"atomic layer etching"',
    "NLD", '"neutral loop discharge"', "NLD-RIE",
    "RIE", '"reactive ion etching"', '"plasma etching"',
    "nanofabrication", "lithography", "nanoimprint", "NIL",
    "TiO2", "titania", '"titanium dioxide"', '"thin film"', "etching", "deposition", "patterning"
]

# Exclude common false-positive metasurface area for your use-case (通信/RIS)
EXCLUDE_TERMS = [
    "RIS", "reconfigurable intelligent surface", "wireless", "MIMO", "5G", "6G", "channel", "beamforming",
    "communication", "antenna", "network", "routing"
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _safe_get(d: Dict, k: str, default=None):
    v = d.get(k, default)
    return v if v is not None else default


def _arxiv_query(days_back: int) -> str:
    """
    arXiv API query string (search_query=...)
    We make metasurface MUST in query to reduce junk.
    """
    cats = "(cat:cond-mat.mtrl-sci OR cat:physics.app-ph OR cat:cond-mat.other OR cat:physics.ins-det)"
    meta = "(" + " OR ".join([f"all:{t}" if not t.startswith('"') else f"all:{t}" for t in META_TERMS]) + ")"
    proc = "(" + " OR ".join([f"all:{t}" if not t.startswith('"') else f"all:{t}" for t in PROCESS_TERMS]) + ")"
    # keep it simple: (cats) AND (meta) AND (proc)
    q = f"{cats} AND {meta} AND {proc}"
    return q


def _fetch_arxiv(days_back: int, max_results: int = 200) -> List[Dict]:
    q = _arxiv_query(days_back=days_back)
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query={quote_plus(q)}"
        f"&start=0&max_results={max_results}"
        "&sortBy=submittedDate&sortOrder=descending"
    )
    logger.info(f"Fetching arXiv: {url}")

    feed = feedparser.parse(url)
    results = []
    cutoff = datetime.now(JST) - timedelta(days=days_back)

    for e in feed.entries:
        # published like "2026-02-15T..."
        published = dateparser.parse(e.get("published", "")) if e.get("published") else None
        if published and published.astimezone(JST) < cutoff:
            continue

        link = None
        # prefer arXiv abs link
        for l in e.get("links", []):
            if l.get("rel") == "alternate" and "arxiv.org/abs/" in l.get("href", ""):
                link = l["href"]
                break
        if not link:
            link = e.get("link")

        authors = [a.get("name") for a in e.get("authors", []) if a.get("name")]
        title = _norm(e.get("title", ""))
        abstract = _norm(e.get("summary", ""))

        results.append({
            "title": title,
            "authors": authors,
            "published": published.astimezone(JST).strftime("%Y-%m-%d") if published else None,
            "source": "arXiv",
            "link": link,
            "doi": None,
            "abstract": abstract,
        })

    return results


def _fetch_openalex(days_back: int, per_page: int = 50, pages: int = 3) -> List[Dict]:
    """
    OpenAlex is a free scholarly index. No key required.
    We'll query metasurface + process keywords and pull DOI/publisher links when available.
    """
    cutoff_date = (datetime.now(JST) - timedelta(days=days_back)).date().isoformat()
    # OpenAlex uses filter=from_publication_date:YYYY-MM-DD
    # query= string in title/abstract
    query = "metasurface " + " ".join(["ALD", "etching", "deposition", "nanofabrication", "TiO2"])
    url = "https://api.openalex.org/works"
    results: List[Dict] = []

    for page in range(1, pages + 1):
        params = {
            "search": query,
            "filter": f"from_publication_date:{cutoff_date}",
            "per-page": per_page,
            "page": page,
        }
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
        except Exception as ex:
            logger.warning(f"OpenAlex fetch failed: {ex}")
            break

        for w in data.get("results", []):
            title = _norm(w.get("title", ""))
            if not title:
                continue
            doi = w.get("doi")
            authors = []
            for a in w.get("authorships", [])[:20]:
                n = a.get("author", {}).get("display_name")
                if n:
                    authors.append(n)

            pub_date = w.get("publication_date")
            abstract = ""
            # abstract_inverted_index exists sometimes; reconstruct roughly (optional)
            inv = w.get("abstract_inverted_index")
            if inv:
                # rebuild in order of position
                pos_to_word = {}
                for word, positions in inv.items():
                    for p in positions:
                        pos_to_word[p] = word
                abstract = " ".join(pos_to_word[p] for p in sorted(pos_to_word.keys()))
                abstract = _norm(abstract)

            # best link preference: DOI -> landing page
            link = None
            if doi:
                link = f"https://doi.org/{doi.replace('https://doi.org/','').replace('http://doi.org/','')}"
            else:
                primary = w.get("primary_location", {}).get("landing_page_url")
                link = primary or w.get("id")

            results.append({
                "title": title,
                "authors": authors,
                "published": pub_date,
                "source": "OpenAlex",
                "link": link,
                "doi": doi,
                "abstract": abstract,
            })

    return results


def _fetch_crossref(days_back: int, rows: int = 60) -> List[Dict]:
    """
    Crossref works API. No key required.
    Good for publisher/DOI hits; abstracts rarely present.
    """
    cutoff = datetime.now(JST) - timedelta(days=days_back)
    # Crossref query in bibliographic fields
    query = "metasurface ALD fabrication etching deposition TiO2"
    url = "https://api.crossref.org/works"
    params = {
        "query": query,
        "rows": rows,
        "sort": "published",
        "order": "desc",
    }

    results: List[Dict] = []
    try:
        r = requests.get(url, params=params, timeout=20, headers={"User-Agent": "paper-digest/1.0 (mailto:example@example.com)"})
        r.raise_for_status()
        data = r.json()
    except Exception as ex:
        logger.warning(f"Crossref fetch failed: {ex}")
        return results

    for item in data.get("message", {}).get("items", []):
        title_list = item.get("title") or []
        title = _norm(title_list[0]) if title_list else ""
        if not title:
            continue

        # published date
        pub = None
        if item.get("published-print", {}).get("date-parts"):
            dp = item["published-print"]["date-parts"][0]
            pub = "-".join(str(x).zfill(2) for x in dp)
        elif item.get("published-online", {}).get("date-parts"):
            dp = item["published-online"]["date-parts"][0]
            pub = "-".join(str(x).zfill(2) for x in dp)

        # filter by days_back if possible
        if pub:
            try:
                dt = dateparser.parse(pub).replace(tzinfo=JST)
                if dt < cutoff:
                    continue
            except Exception:
                pass

        doi = item.get("DOI")
        link = f"https://doi.org/{doi}" if doi else item.get("URL")

        authors = []
        for a in item.get("author", [])[:20]:
            given = a.get("given", "")
            family = a.get("family", "")
            nm = _norm(f"{given} {family}")
            if nm:
                authors.append(nm)

        results.append({
            "title": title,
            "authors": authors,
            "published": pub,
            "source": "Crossref",
            "link": link,
            "doi": doi,
            "abstract": "",  # Crossref usually doesn't provide
        })

    return results


def collect_candidates(days_back: int) -> List[Dict]:
    arxiv = _fetch_arxiv(days_back=days_back, max_results=200)
    openalex = _fetch_openalex(days_back=days_back, per_page=50, pages=3)
    crossref = _fetch_crossref(days_back=days_back, rows=80)

    # merge + de-dup by (doi or title)
    seen = set()
    out = []

    def key(p):
        if p.get("doi"):
            return ("doi", p["doi"].lower())
        return ("title", _norm(p.get("title", "")).lower())

    for src in (arxiv, openalex, crossref):
        for p in src:
            k = key(p)
            if k in seen:
                continue
            seen.add(k)
            out.append(p)

    return out
