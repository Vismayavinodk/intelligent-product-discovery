from typing import List, Dict, Any
from nltk.corpus import wordnet
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import PRF_TOP_DOCS, PRF_EXPANSION_TERMS, WORDNET_MAX_SYNONYMS

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class QueryExpander:
    """
    Handles query expansion using WordNet synonyms or Pseudo Relevance Feedback (PRF).
    """

    def __init__(self, bm25_retriever=None):
        """
        Initializes the QueryExpander.

        Args:
            bm25_retriever: An instance of BM25Retriever used for PRF.
        """
        self.bm25_retriever = bm25_retriever

    def expand_wordnet(self, query: str, max_synonyms: int = WORDNET_MAX_SYNONYMS) -> str:
        """
        Expands query by adding WordNet synonyms.

        Args:
            query: The original query.
            max_synonyms: Maximum number of synonyms to add per word.

        Returns:
            The expanded query string.
        """
        words = query.split()
        expanded_terms = set(words)
        
        for word in words:
            synonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    term = lemma.name().replace('_', ' ')
                    if term.lower() != word.lower():
                        synonyms.append(term)
            
            # Keep unique synonyms and limit count
            unique_synonyms = list(dict.fromkeys(synonyms))[:max_synonyms]
            expanded_terms.update(unique_synonyms)
            
        return ' '.join(expanded_terms)

    def expand_prf(self, query: str, top_docs: int = PRF_TOP_DOCS, expansion_terms: int = PRF_EXPANSION_TERMS) -> str:
        """
        Expands query using Pseudo Relevance Feedback.

        Args:
            query: The original query.
            top_docs: Number of top documents to consider for feedback.
            expansion_terms: Number of top terms to extract from the feedback documents.

        Returns:
            The expanded query string.
        """
        if not self.bm25_retriever:
            raise ValueError("BM25Retriever is required for PRF expansion.")
            
        # Retrieve top documents
        top_results = self.bm25_retriever.retrieve(query, top_k=top_docs)
        if not top_results:
            return query
            
        # Get the actual documents corresponding to the top product IDs
        # To do this correctly, we need access to the corpus.
        # Since we only have corpus_tokens in BM25Retriever, we'll reconstruct pseudo-docs
        
        # Getting indices of top docs from BM25 scores
        tokenized_query = self.bm25_retriever.preprocessor.preprocess(query).split()
        scores = self.bm25_retriever.bm25.get_scores(tokenized_query)
        import numpy as np
        top_indices = np.argsort(scores)[-top_docs:][::-1]
        
        # Reconstruct pseudo-documents from tokens
        feedback_docs = [" ".join(self.bm25_retriever.bm25.corpus[idx]) for idx in top_indices]
        
        if not feedback_docs:
            return query
            
        # Extract top TF-IDF terms from feedback docs
        vectorizer = TfidfVectorizer(max_features=expansion_terms, stop_words='english')
        vectorizer.fit(feedback_docs)
        
        new_terms = vectorizer.get_feature_names_out()
        
        # Combine original query and new terms
        expanded_query = query + " " + " ".join(new_terms)
        return expanded_query

    def expand(self, query: str, method: str = 'wordnet') -> str:
        """
        Expands the query using the specified method.

        Args:
            query: Original query.
            method: 'wordnet' or 'prf'.

        Returns:
            Expanded query string.
        """
        if method == 'wordnet':
            return self.expand_wordnet(query)
        elif method == 'prf':
            return self.expand_prf(query)
        else:
            raise ValueError(f"Unknown expansion method: {method}")
