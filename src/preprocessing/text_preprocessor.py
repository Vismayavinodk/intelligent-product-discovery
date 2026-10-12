"""
Text preprocessing module.
"""
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from typing import List, Optional

def download_nltk_data() -> None:
    """Download required NLTK data if not already present."""
    resource_checks = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
    ]
    for path, name in resource_checks:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass

download_nltk_data()

def clean_text(text: str) -> str:
    """
    Lowercase, remove HTML tags, URLs, special chars, extra whitespace.
    """
    if not isinstance(text, str):
        return ""
        
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    
    # Remove special chars and punctuation
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize(text: str) -> List[str]:
    """Tokenize text using NLTK word_tokenize."""
    if not text.strip():
        return []
    try:
        return word_tokenize(text)
    except LookupError:
        # Fallback if punkt is not available
        return text.split()

def remove_stopwords(tokens: List[str]) -> List[str]:
    """Filter using NLTK English stopwords."""
    try:
        stop_words = set(stopwords.words('english'))
        return [t for t in tokens if t not in stop_words]
    except LookupError:
        return tokens

def get_wordnet_pos(treebank_tag: str) -> str:
    """Map POS tag to first character used by WordNetLemmatizer."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize(tokens: List[str]) -> List[str]:
    """Lemmatize tokens using WordNetLemmatizer with POS tag mapping."""
    if not tokens:
        return []
        
    lemmatizer = WordNetLemmatizer()
    try:
        tagged = nltk.pos_tag(tokens)
        return [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]
    except LookupError:
        # Fallback if tagger is not available
        return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess(text: str) -> List[str]:
    """
    Full pipeline: clean -> tokenize -> remove stopwords -> lemmatize.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    filtered = remove_stopwords(tokens)
    lemmatized = lemmatize(filtered)
    return lemmatized

def preprocess_to_string(text: str) -> str:
    """
    Full pipeline, but return as a space-separated string.
    """
    return ' '.join(preprocess(text))

class TextPreprocessor:
    """
    Wrapper for text preprocessing with optional caching.
    """
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._cache = {}
        
    def process(self, text: str) -> List[str]:
        """Process text, using cache if enabled."""
        if not isinstance(text, str):
            return []
            
        if self.use_cache and text in self._cache:
            return self._cache[text]
            
        result = preprocess(text)
        
        if self.use_cache:
            self._cache[text] = result
            
        return result
        
    def process_to_string(self, text: str) -> str:
        """Process text to string, using cache if enabled."""
        if not isinstance(text, str):
            return ""
            
        if self.use_cache and text in self._cache:
            return ' '.join(self._cache[text])
            
        result = preprocess(text)
        
        if self.use_cache:
            self._cache[text] = result
            
        return ' '.join(result)
        
    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache = {}

    # Aliases for compatibility
    def preprocess(self, text: str) -> str:
        """Alias for process_to_string (returns space-joined preprocessed tokens)."""
        return self.process_to_string(text)

    def preprocess_tokens(self, text: str) -> List[str]:
        """Alias for process (returns list of tokens)."""
        return self.process(text)

