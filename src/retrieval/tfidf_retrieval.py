from typing import List, Tuple
from src.indexing.inverted_index import InvertedIndex
from src.config import DEFAULT_TOP_K

class TFIDFRetriever:
    """
    Retriever class that uses an InvertedIndex to fetch relevant documents.
    """

    def __init__(self, inverted_index: InvertedIndex, product_ids: List[str]):
        """
        Initializes the retriever with an index and a mapping to product IDs.

        Args:
            inverted_index: Built InvertedIndex instance.
            product_ids: List of product IDs corresponding to the documents in the index.
        """
        self.inverted_index = inverted_index
        self.product_ids = product_ids

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Tuple[str, float]]:
        """
        Retrieves the top-k product IDs for a given query using TF-IDF.

        Args:
            query: The search query string.
            top_k: Number of top results to return.

        Returns:
            List of tuples (product_id, score).
        """
        results = self.inverted_index.search(query, top_k)
        mapped_results = []
        for doc_idx, score in results:
            mapped_results.append((self.product_ids[doc_idx], score))
        return mapped_results
