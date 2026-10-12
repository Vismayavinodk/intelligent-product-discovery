import pickle
import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

class InvertedIndex:
    """
    Inverted index for TF-IDF based retrieval.
    """

    def __init__(self):
        """
        Initializes the TF-IDF vectorizer and storage for the document matrix.
        """
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None

    def build(self, documents: List[str]) -> None:
        """
        Builds the TF-IDF inverted index from a list of documents.

        Args:
            documents: List of text documents.
        """
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """
        Searches the index for the given query using cosine similarity.

        Args:
            query: The search query string.
            top_k: Number of top results to return.

        Returns:
            List of tuples containing (document_index, similarity_score) sorted in descending order of score.
        """
        if self.tfidf_matrix is None:
            raise ValueError("Index not built yet.")
        
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def save(self, path: str) -> None:
        """
        Saves the index to disk.

        Args:
            path: Path to save the index to.
        """
        with open(path, 'wb') as f:
            pickle.dump({'vectorizer': self.vectorizer, 'tfidf_matrix': self.tfidf_matrix}, f)

    @classmethod
    def load(cls, path: str) -> 'InvertedIndex':
        """
        Loads an index from disk.

        Args:
            path: Path to load the index from.

        Returns:
            An instance of InvertedIndex.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        index = cls()
        index.vectorizer = data['vectorizer']
        index.tfidf_matrix = data['tfidf_matrix']
        return index

    def get_feature_names(self) -> List[str]:
        """
        Returns the list of feature names (vocabulary).
        """
        if self.vectorizer is None:
            return []
        return self.vectorizer.get_feature_names_out().tolist()

    def get_document_vector(self, doc_idx: int) -> sp.csr_matrix:
        """
        Returns the sparse TF-IDF vector for a specific document.

        Args:
            doc_idx: Index of the document.

        Returns:
            Sparse matrix representation of the document vector.
        """
        if self.tfidf_matrix is None:
            raise ValueError("Index not built yet.")
        return self.tfidf_matrix[doc_idx]
