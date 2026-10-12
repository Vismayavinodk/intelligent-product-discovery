from typing import List, Tuple
from rank_bm25 import BM25Okapi
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.config import DEFAULT_TOP_K, BM25_K1, BM25_B
import numpy as np

class BM25Retriever:
    """
    BM25 based retriever for text search.
    """

    def __init__(self, corpus_tokens: List[List[str]], product_ids: List[str], k1: float = BM25_K1, b: float = BM25_B):
        """
        Initializes the BM25 retriever.

        Args:
            corpus_tokens: List of tokenized documents.
            product_ids: List of corresponding product IDs.
            k1: BM25 k1 parameter.
            b: BM25 b parameter.
        """
        self.product_ids = product_ids
        self.bm25 = BM25Okapi(corpus_tokens, k1=k1, b=b)
        self.preprocessor = TextPreprocessor() # assuming default initialization is fine

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Tuple[str, float]]:
        """
        Retrieves top documents using BM25 scoring.

        Args:
            query: The raw search query.
            top_k: Number of top documents to return.

        Returns:
            List of (product_id, score) tuples.
        """
        # Tokenize the query using the preprocessor
        tokenized_query = self.preprocessor.preprocess(query).split()
        
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices with non-zero scores
        sorted_indices = np.argsort(scores)[::-1]
        results = []
        for idx in sorted_indices:
            if scores[idx] > 0.0:
                results.append((self.product_ids[idx], float(scores[idx])))
                if len(results) >= top_k:
                    break
            else:
                break
            
        return results
