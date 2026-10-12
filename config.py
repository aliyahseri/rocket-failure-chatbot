"""
Configuration file for Rocket Failure Chatbot
All paths and hyperparameters are configurable here.
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent
BASE_DIR = PROJECT_ROOT
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
MODELS_DIR = BASE_DIR / "models"
TESTS_DIR = BASE_DIR / "tests"

# Create directories if they don't exist
for dir_path in [DATA_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR, TESTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data paths
CSV_PATH = DATA_DIR / "rocket_failures.csv"

# Model/artifact paths
TFIDF_MATRIX_PATH = MODELS_DIR / "tfidf_matrix.pkl"
BM25_INDEX_PATH = MODELS_DIR / "bm25_index.pkl"
INVERTED_INDEX_PATH = MODELS_DIR / "inverted_index.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
PREPROCESSED_DATA_PATH = MODELS_DIR / "preprocessed_data.pkl"
METADATA_PATH = MODELS_DIR / "metadata.pkl"

# TF-IDF parameters
TFIDF_PARAMS = {
    "min_df": 2,
    "max_df": 0.95,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "stop_words": "english"
}

# BM25 parameters
BM25_PARAMS = {
    "k1": 1.5,
    "b": 0.75,
    "epsilon": 0.25
}

# Retrieval parameters
RETRIEVAL_PARAMS = {
    "top_k": 3,
    "tfidf_weight": 0.4,
    "bm25_weight": 0.6,
    "min_score": 0.05
}

# Preprocessing parameters
PREPROCESSING_PARAMS = {
    "use_stemming": True,
    "use_lemmatization": False,
    "remove_domain_stopwords": True,
    "domain_stopwords": ["rocket", "launch", "vehicle", "incident"]
}

# Domain-specific synonym dictionary
SYNONYM_DICT = {
    "exploded": "combustion",
    "blew up": "explosion",
    "engine": "propulsion",
    "fuel": "propellant",
    "motor": "propulsion",
    "thrust": "propulsion",
    "burn": "combustion",
    "fire": "combustion",
    "crash": "failure",
    "destroyed": "failure",
    "lost": "failure",
    "malfunction": "failure",
    "breakdown": "failure"
}

# Intent classification patterns
INTENT_PATTERNS = {
    "FILTER_QUERY": [
        r"(show|list|display|find|get)\s+(me\s+)?(all\s+)?(failures|incidents)",
        r"(failures|incidents)\s+(in|from|by|during|at)",
        r"(what|which)\s+(failures|incidents)\s+(occurred|happened)"
    ],
    "FACTUAL_LOOKUP": [
        r"(what|how|why|when|where)\s+(caused|happened|occurred)",
        r"(tell|explain|describe)\s+(me\s+)?(about\s+)?(the\s+)?(incident|failure)",
        r"incident\s+[A-Z]{2,}-\d{4}",
        r"RFI-\d{4}"
    ],
    "COUNT_QUERY": [
        r"how\s+many\s+incidents",
        r"how\s+many\s+.*\s+involved",
        r"\bcount\b",
        r"count\s+the\s+number",
        r"(how\s+many|count|number\s+of)\s+(failures|incidents)",
        r"(total|sum)\s+(of\s+)?(failures|incidents)"
    ],
    "COMPARISON_QUERY": [
        r"(which|what)\s+(country|vehicle)\s+(had|has|more|most|less|fewer)",
        r"(compare|comparison|vs|versus)\s+between",
        r"(difference|differ)\s+between",
        r"\b(\w+)\s+and\s+(\w+)\b",
        r"which country has (more|most|the most)",
        r"most (rockets|failures|incidents)"
    ],
    "TREND_QUERY": [
        r"(what|which)\s+(failure|type|category)\s+(was|is)\s+(most|least|common|frequent)",
        r"(trend|pattern)\s+(over|across|during|in)",
        r"(most|least)\s+common\s+(in|during|for)"
    ],
    "STATISTICAL_QUERY": [
        r"most\s+common\s+(severity|failure|propellant|phase)",
        r"what\s+is\s+the\s+(most|least|average|median|mean)",
        r"(average|mean|median|mode|std|standard deviation)",
        r"(what\s+is\s+the)\s+(maximum|minimum|highest|lowest)",
        r"(statistics|stats|distribution)"
    ],
    "DEFINITION_QUERY": [
        r"explain\s+what\s+[\w\s]+means?",
        r"^explain\s+",
        r"what\s+does\s+\w+\s+mean",
        r"(what\s+is|what\s+does|define|definition|meaning)",
        r"(explain|describe)\s+(what\s+is\s+)?(nozzle|erosion|leak|rupture|malfunction)"
    ]
}

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = BASE_DIR / "chatbot.log"

# Streamlit app configuration
STREAMLIT_CONFIG = {
    "page_title": "Rocket Failure QA Chatbot",
    "page_icon": "🚀",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}
