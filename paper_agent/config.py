import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")

# Search Keywords
KEYWORDS_ALD = ["ALD", "atomic layer deposition", "PEALD", "plasma enhanced ALD", "TiO2", "titanium dioxide", "refractive index", "film quality", "low temperature", "defect"]
KEYWORDS_RIE = ["NLD-RIE", "neutral loop discharge", "RIE", "reactive ion etching", "Si etching", "SiO2 etching", "TiO2 etching", "aspect ratio dependent", "taper", "mask selectivity", "damage"]
KEYWORDS_METASURFACE = ["metasurface", "meta-surface", "nanoimprint", "NIL", "mass production", "lithography"]
KEYWORDS_TIO2_MICRO = ["TiO2 microstructure", "TiO2 nanostructure", "fabrication process"]

# Combine all keywords for broader initial search, or use them in specific queries
# For arXiv API, we need to construct a query string.
# We will combine these into a robust query in the search module.

# Exclude Keywords (Simple string matching in title/abstract)
EXCLUDE_KEYWORDS = ["review", "survey", "overview", "progress report"]

# Log Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

def configure_logging():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("paper_agent.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
