"""
Custom pure-Python inverted index implementation for Information Retrieval.

Implements term-document inverted postings, log-normalized TF, IDF weighting,
document vector L2 normalization, and cosine similarity ranking from scratch
without scikit-learn.
"""

from collections import Counter, defaultdict
import math
import pickle
import re
from typing import Dict, List, Optional, Sequence, Tuple


class CustomInvertedIndex:
    """
    Inverted index constructed from scratch with TF-IDF vector space retrieval.
    """

    def __init__(self) -> None:
        """Initialize empty index structures."""
        # Mapping: term -> list of (doc_id, raw_term_frequency)
        self.index: Dict[str, List[Tuple[int, int]]] = {}
        # Document frequency: term -> number of docs containing the term
        self.document_frequency: Dict[str, int] = {}
        # Precomputed IDF values: term -> idf
        self.idf_values: Dict[str, float] = {}
        # Fast lookup: doc_id -> {term: raw_tf}
        self.doc_term_tfs: Dict[int, Dict[str, int]] = {}
        # L2 norm of doc TF-IDF vector: doc_id -> Euclidean length
        self.doc_lengths: Dict[int, float] = {}
        # Total number of indexed documents
        self.num_docs: int = 0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tokenize text into lowercase alphanumeric words.

        Args:
            text: Raw input text.

        Returns:
            List of normalized token strings.
        """
        if not isinstance(text, str):
            return []
        return re.findall(r"\b[a-zA-Z0-9'-]+\b", text.lower())

    def build(self, documents: Sequence[str]) -> None:
        """
        Build inverted index data structure, compute IDF, and document lengths.

        Args:
            documents: List or sequence of raw or preprocessed document strings.
        """
        self.index.clear()
        self.document_frequency.clear()
        self.idf_values.clear()
        self.doc_term_tfs.clear()
        self.doc_lengths.clear()

        self.num_docs = len(documents)
        if self.num_docs == 0:
            return

        # Phase 1: Build postings and term frequencies per document
        for doc_id, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            counts = Counter(tokens)
            self.doc_term_tfs[doc_id] = dict(counts)

            for term, count in counts.items():
                if term not in self.index:
                    self.index[term] = []
                    self.document_frequency[term] = 0
                self.index[term].append((doc_id, count))
                self.document_frequency[term] += 1

        # Phase 2: Compute IDF values for all terms
        for term in self.index:
            self.idf_values[term] = self._compute_idf(term)

        # Phase 3: Compute Euclidean lengths (L2 norms) for document vectors
        for doc_id, term_counts in self.doc_term_tfs.items():
            sum_sq = 0.0
            for term in term_counts:
                tfidf = self._compute_tfidf(term, doc_id)
                sum_sq += tfidf * tfidf
            self.doc_lengths[doc_id] = math.sqrt(sum_sq) if sum_sq > 0.0 else 0.0

    def _compute_tf(self, term: str, doc_id: int) -> float:
        """
        Compute log-normalized term frequency: 1 + log(tf) if tf > 0 else 0.

        Args:
            term: The vocabulary term.
            doc_id: The document index.

        Returns:
            Log-normalized term frequency.
        """
        raw_tf = self.doc_term_tfs.get(doc_id, {}).get(term.lower(), 0)
        if raw_tf <= 0:
            return 0.0
        return 1.0 + math.log(raw_tf)

    def _compute_idf(self, term: str) -> float:
        """
        Compute smoothed inverse document frequency for a term.

        Args:
            term: The vocabulary term.

        Returns:
            Smoothed IDF score: log((N + 1) / (df + 1)) + 1.
        """
        df = self.get_document_frequency(term)
        if df == 0 or self.num_docs == 0:
            return 0.0
        return math.log((self.num_docs + 1.0) / (df + 1.0)) + 1.0

    def _compute_tfidf(self, term: str, doc_id: int) -> float:
        """
        Compute TF-IDF weight for a term in a document: tf * idf.

        Args:
            term: Vocabulary term.
            doc_id: Document index.

        Returns:
            TF-IDF weight.
        """
        term_clean = term.lower()
        tf = self._compute_tf(term_clean, doc_id)
        if tf == 0.0:
            return 0.0
        idf = self.idf_values.get(term_clean, self._compute_idf(term_clean))
        return tf * idf

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Compute TF-IDF cosine similarity for the query against all documents.

        Args:
            query: User search query string.
            top_k: Number of highest ranking documents to return.

        Returns:
            List of (doc_index, cosine_similarity_score) tuples sorted descending.
        """
        if self.num_docs == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        query_counts = Counter(query_tokens)

        # Compute query vector weights and query L2 length
        query_weights: Dict[str, float] = {}
        query_sum_sq = 0.0

        for term, count in query_counts.items():
            if term not in self.index:
                continue
            tf = 1.0 + math.log(count)
            idf = self.idf_values.get(term, self._compute_idf(term))
            weight = tf * idf
            query_weights[term] = weight
            query_sum_sq += weight * weight

        query_len = math.sqrt(query_sum_sq)
        if query_len <= 0.0:
            return []

        # Accumulate dot product using inverted index postings
        dot_products: Dict[int, float] = defaultdict(float)

        for term, q_weight in query_weights.items():
            postings = self.get_postings(term)
            for doc_id, _ in postings:
                d_weight = self._compute_tfidf(term, doc_id)
                dot_products[doc_id] += q_weight * d_weight

        # Normalize by query length and doc length to obtain cosine similarity
        scored_docs: List[Tuple[int, float]] = []
        for doc_id, dot in dot_products.items():
            d_len = self.doc_lengths.get(doc_id, 0.0)
            if d_len > 0.0:
                cosine = dot / (query_len * d_len)
                scored_docs.append((doc_id, float(cosine)))

        scored_docs.sort(key=lambda item: item[1], reverse=True)
        return scored_docs[:top_k]

    def get_postings(self, term: str) -> List[Tuple[int, int]]:
        """
        Get the postings list for a term.

        Args:
            term: The query term.

        Returns:
            List of (doc_id, term_frequency) tuples for the term.
        """
        return self.index.get(term.lower(), [])

    def get_vocabulary(self) -> List[str]:
        """
        Get all indexed unique vocabulary terms.

        Returns:
            Sorted list of all terms in the inverted index.
        """
        return sorted(list(self.index.keys()))

    def get_document_frequency(self, term: str) -> int:
        """
        Get the document frequency (number of documents containing the term).

        Args:
            term: The query term.

        Returns:
            Number of documents containing the term.
        """
        return self.document_frequency.get(term.lower(), 0)

    def save(self, path: str) -> None:
        """
        Save the custom inverted index instance to disk using pickle.

        Args:
            path: Target file path.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "CustomInvertedIndex":
        """
        Load a saved custom inverted index instance from disk.

        Args:
            path: Source file path.

        Returns:
            Loaded CustomInvertedIndex instance.
        """
        with open(path, "rb") as f:
            return pickle.load(f)
