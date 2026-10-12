from typing import List, Tuple
from src.indexing.embedding_index import EmbeddingIndex
from src.config import DEFAULT_TOP_K

class SemanticRetriever:
    """
    Retriever class that uses an EmbeddingIndex to fetch relevant documents based on semantic similarity.
    """

    def __init__(self, embedding_index: EmbeddingIndex, product_ids: List[str]):
        """
        Initializes the semantic retriever.

        Args:
            embedding_index: Built EmbeddingIndex instance.
            product_ids: List of product IDs corresponding to the documents in the index.
        """
        self.embedding_index = embedding_index
        self.product_ids = product_ids

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Tuple[str, float]]:
        """
        Retrieves the top-k product IDs for a given query using semantic similarity.

        Args:
            query: The search query string.
            top_k: Number of top results to return.

        Returns:
            List of tuples (product_id, score).
        """
        results = self.embedding_index.search(query, top_k)
        mapped_results = []
        for doc_idx, score in results:
            mapped_results.append((self.product_ids[doc_idx], score))
        return mapped_results
