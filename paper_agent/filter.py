import re
from typing import Dict, List

META_MUST = [
    "metasurface", "metasurfaces", "meta-surface", "meta surface"
]

# Strong process terms (must hit at least one)
PROCESS_STRONG = [
    "atomic layer deposition", "ald", "peald", "plasma-enhanced ald",
    "atomic layer etching", "ale",
    "neutral loop discharge", "nld", "nld-rie",
    "reactive ion etching", "rie", "plasma etching",
    "nanofabrication", "lithography", "nanoimprint", "nil",
    "thin film", "deposition", "etching", "patterning",
    "tio2", "titania", "titanium dioxide", "fabrication", "manufacturing", "structural", "resonator", "nanostructure"

]

# Exclude comms/RIS metasurface papers
EXCLUDE = [
    "reconfigurable intelligent surface", "wireless", "mimo", "5g", "6g",
    "beamforming", "channel", "communication", "antenna", "routing", "network",
    "multi-ris", "ris activation"
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _text(p: Dict) -> str:
    return (_norm(p.get("title","")) + " " + _norm(p.get("abstract",""))).lower()


def filter_process_metasurface_must(papers: List[Dict]) -> List[Dict]:
    out = []
    for p in papers:
        t = _text(p)

        # metasurface MUST
        if not any(k in t for k in META_MUST):
            continue

        # process keyword at least 1 hit (緩和版)
        if not any(k in t for k in PROCESS_STRONG):
            continue

        # exclude obvious communication RIS
        if any(k in t for k in EXCLUDE):
            continue

        out.append(p)
    return out




def _score(p: Dict) -> int:
    t = _text(p)
    score = 0
    # prefer ALD/NLD/etch
    for w in ["ald", "atomic layer deposition", "nld", "neutral loop discharge", "rie", "etch", "deposition", "tio2"]:
        if w in t:
            score += 2
    # prefer TiO2 specifically
    if "tio2" in t or "titania" in t:
        score += 3
    # bonus: has DOI / publisher link
    if p.get("doi"):
        score += 2
    # bonus: abstract present (better summarization)
    if p.get("abstract"):
        score += 1
    return score


def pick_top_n(papers: List[Dict], n: int = 5) -> List[Dict]:
    return sorted(papers, key=_score, reverse=True)[:n]


