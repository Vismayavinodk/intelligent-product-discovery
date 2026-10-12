"""
Central configuration module for the e-commerce IR system.
"""
import os
import random
import numpy as np
import torch

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Model configs
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# BM25 configs
BM25_K1 = 1.5
BM25_B = 0.75

# Retrieval & Ranking configs
RRF_K = 60
DEFAULT_TOP_K = 10
SIMILARITY_THRESHOLD = 0.85
PRF_TOP_DOCS = 10
PRF_EXPANSION_TERMS = 5
WORDNET_MAX_SYNONYMS = 3
SVD_N_COMPONENTS = 50

# Data configs
PRODUCT_TEXT_COLUMNS = ['name', 'brand', 'categories', 'primaryCategories', 'description']
REVIEW_COLUMNS = ['reviews.text', 'reviews.rating', 'reviews.title', 'reviews.username']

# Random seed
RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_directories_exist() -> None:
    """Ensure all required directories exist."""
    directories = [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Initialize
ensure_directories_exist()
set_seed()
