import logging
from . import config

logger = logging.getLogger(__name__)

def calculate_score(paper):
    """
    Calculate a relevance score based on keyword matches in title and abstract.
    """
    score = 0
    text = (paper['title'] + " " + paper['abstract']).lower()
    
    # High priority keywords (give more weight)
    # ALD/TiO2
    for k in config.KEYWORDS_ALD:
        if k.lower() in text:
            score += 2
            
    # RIE
    for k in config.KEYWORDS_RIE:
        if k.lower() in text:
            score += 2
            
    # Metasurface
    for k in config.KEYWORDS_METASURFACE:
        if k.lower() in text:
            score += 2
            
    # Specific microstructure process
    if "process" in text and "fabrication" in text:
        score += 1
        
    return score

def filter_papers(papers, top_n=5):
    """
    Filter papers, remove duplicates, score them, and return top N.
    """
    # Remove duplicates by DOI or Title
    seen_dois = set()
    unique_papers = []
    
    for paper in papers:
        doi = paper.get('doi')
        title = paper.get('title')
        
        if doi and doi != "N/A":
            if doi in seen_dois:
                continue
            seen_dois.add(doi)
        elif title:
            # Fallback to title check if DOI missing
            if title in seen_dois:
                continue
            seen_dois.add(title)
            
        unique_papers.append(paper)
        
    # Score papers
    scored_papers = []
    for paper in unique_papers:
        score = calculate_score(paper)
        # Filter out low relevance if needed, e.g. score > 0
        if score > 0:
            paper['score'] = score
            scored_papers.append(paper)
            
    # Sort by score desc, then date desc
    # (Assuming date format allows sorting, otherwise relying on score)
    scored_papers.sort(key=lambda x: x['score'], reverse=True)
    
    top_papers = scored_papers[:top_n]
    logger.info(f"Selected top {len(top_papers)} papers from {len(sys_papers if 'sys_papers' in locals() else papers)} candidates.")
    
    return top_papers
