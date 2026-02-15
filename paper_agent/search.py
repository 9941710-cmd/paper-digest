import arxiv
import logging
from . import config

logger = logging.getLogger(__name__)

def search_arxiv():
    """
    Search arXiv for papers related to the keywords defined in config.
    Returns a list of paper dictionaries.
    """
    results = []
    
    # Constructing a query. 
    # arXiv API supports boolean operators. We'll group by main topics.
    # Group 1: ALD / TiO2
    query_ald = ' OR '.join([f'abs:"{k}"' for k in config.KEYWORDS_ALD])
    query_rie = ' OR '.join([f'abs:"{k}"' for k in config.KEYWORDS_RIE])
    query_meta = ' OR '.join([f'abs:"{k}"' for k in config.KEYWORDS_METASURFACE])
    
    # We combine them with OR because we want papers from ANY of these fields.
    # However, a single query might be too long or complex. Let's do a combined query if possible, 
    # or separate queries if needed. Given the number of keywords, separate queries might be safer 
    # to avoid URL length limits, but 'arxiv' python lib handles post requests so length is less of an issue.
    # Let's try a combined query with appropriate parentheses.
    
    # Note: abs: searches abstract. ti: searches title. all: searches all.
    # Let's use all: for broader search then filter later, or ti/abs for precision.
    # User wants "Latest 5 papers".
    
    search_query = f'({query_ald} OR {query_rie} OR {query_meta})'
    
    # Sort by submittedDate (newest first)
    client = arxiv.Client()
    
    search = arxiv.Search(
        query = search_query,
        max_results = 20, # Get more than 5 to allow for filtering
        sort_by = arxiv.SortCriterion.SubmittedDate
    )
    
    for result in client.results(search):
        # Basic filtering: Check if "review" is in title (simple check before advanced filtering)
        if any(ex in result.title.lower() for ex in config.EXCLUDE_KEYWORDS):
            continue

        paper = {
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "published": result.published,
            "abstract": result.summary,
            "link": result.entry_id,
            "source": "arXiv",
            "doi": result.doi if result.doi else "N/A"
        }
        results.append(paper)
        
    logger.info(f"Fetched {len(results)} papers from arXiv")
    return results

def search_crossref():
    """
    Search Crossref API for papers.
    Crossref API is more complex. We'll use the 'works' endpoint.
    """
    # Placeholder for Crossref implementation
    # Crossref might require more specific handling or a library like 'habanero' or direct requests.
    # For now, let's focus on arXiv as the primary source and add Crossref if needed or time permits,
    # as Crossref API often returns metadata without full abstracts, making summarization hard.
    # However, the user requested it.
    
    # We will implement a basic requests-based search for Crossref.
    import requests
    import datetime
    
    # Get today's date for filtering
    # Crossref filter 'from-pub-date'
    
    results = []
    base_url = "https://api.crossref.org/works"
    
    # Join keywords with OR is not directly supported in simple query usually, 
    # but we can pass a general query string.
    # Let's pick the most important keywords for the query.
    query_params = {
        "query": "ALD TiO2 RIE Metasurface", # Broad query
        "filter": "type:journal-article", # only articles
        "sort": "published",
        "order": "desc",
        "rows": 20,
        "mailto": config.EMAIL_ADDRESS # Good etiquette for Crossref
    }
    
    try:
        response = requests.get(base_url, params=query_params)
        response.raise_for_status()
        data = response.json()
        
        for item in data.get('message', {}).get('items', []):
            title = item.get('title', [''])[0]
            # Simple exclusion
            if any(ex in title.lower() for ex in config.EXCLUDE_KEYWORDS):
                continue
                
            # Abstract checks - Crossref often doesn't give abstracts freely.
            # We might fallback to just title filtering or fetch more info if possible.
            # If no abstract, we mark it.
            
            paper = {
                "title": title,
                "authors": [f"{a.get('given', '')} {a.get('family', '')}" for a in item.get('author', [])],
                "published": item.get('created', {}).get('date-time', 'N/A'),
                "abstract": "Abstract not available via Crossref API", # Often the case
                "link": item.get('URL', ''),
                "source": "Crossref",
                "doi": item.get('DOI', '')
            }
            results.append(paper)
            
        logger.info(f"Fetched {len(results)} papers from Crossref")

    except Exception as e:
        logger.error(f"Crossref search failed: {e}")
        
    return results

def get_papers():
    """
    Orchestrate searches
    """
    papers = []
    # arXiv
    try:
        papers.extend(search_arxiv())
    except Exception as e:
        logger.error(f"arXiv search error: {e}")
        
    # Crossref
    try:
        papers.extend(search_crossref())
    except Exception as e:
        logger.error(f"Crossref search error: {e}")
        
    return papers
