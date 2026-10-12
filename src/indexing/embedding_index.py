import pickle
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import EMBEDDING_MODEL

class EmbeddingIndex:
    """
    Dense embedding index for semantic retrieval.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initializes the embedding index.

        Args:
            model_name: Name of the SentenceTransformer model to use.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.embeddings = None

    def build(self, documents: List[str], batch_size: int = 32) -> None:
        """
        Builds the embedding index by encoding all documents.

        Args:
            documents: List of text documents.
            batch_size: Batch size for encoding.
        """
        self.embeddings = self.model.encode(documents, batch_size=batch_size, show_progress_bar=True)
        # Ensure it's a 2D numpy array
        if not isinstance(self.embeddings, np.ndarray):
            self.embeddings = np.array(self.embeddings)

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """
        Searches the embedding index for the given query using cosine similarity.

        Args:
            query: The search query string.
            top_k: Number of top results to return.

        Returns:
            List of tuples containing (document_index, similarity_score).
        """
        if self.embeddings is None:
            raise ValueError("Index not built yet.")
            
        query_vec = self.model.encode([query])
        similarities = cosine_similarity(query_vec, self.embeddings).flatten()
        
        sorted_indices = similarities.argsort()[::-1]
        results = []
        for idx in sorted_indices:
            if similarities[idx] >= 0.28:
                results.append((int(idx), float(similarities[idx])))
                if len(results) >= top_k:
                    break
            else:
                break
        return results

    def save(self, path: str) -> None:
        """
        Saves the embeddings and model info to disk.

        Args:
            path: Path to save the data to.
        """
        with open(path, 'wb') as f:
            pickle.dump({'model_name': self.model_name, 'embeddings': self.embeddings}, f)

    @classmethod
    def load(cls, path: str) -> 'EmbeddingIndex':
        """
        Loads the embedding index from disk.

        Args:
            path: Path to load the data from.

        Returns:
            An instance of EmbeddingIndex.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        index = cls(model_name=data['model_name'])
        index.embeddings = data['embeddings']
        return index

    def get_embedding(self, doc_idx: int) -> np.ndarray:
        """
        Returns the embedding vector for a specific document.

        Args:
            doc_idx: Index of the document.

        Returns:
            Numpy array of the embedding.
        """
        if self.embeddings is None:
            raise ValueError("Index not built yet.")
        return self.embeddings[doc_idx]
