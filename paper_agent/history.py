import json
import os
from typing import Dict, List, Set

HISTORY_FILE = "sent_db.json"


def _paper_key(p: Dict) -> str:
    # DOI優先、なければタイトル
    doi = (p.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    title = (p.get("title") or "").strip().lower()
    return f"title:{title}"


def load_sent() -> Set[str]:
    if not os.path.exists(HISTORY_FILE):
        return set()
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data) if isinstance(data, list) else set()
    except Exception:
        # 壊れてても落とさない
        return set()


def save_sent(sent: Set[str]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(list(sent)), f, ensure_ascii=False, indent=2)


def filter_new(papers: List[Dict]) -> List[Dict]:
    sent = load_sent()
    out = []
    for p in papers:
        k = _paper_key(p)
        if k and k not in sent:
            out.append(p)
    return out


def update_history(papers: List[Dict]) -> None:
    sent = load_sent()
    for p in papers:
        k = _paper_key(p)
        if k:
            sent.add(k)
    save_sent(sent)
