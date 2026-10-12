"""
Duplicate detection module.
"""
import numpy as np
from typing import List, Set, Dict, Any
from datasketch import MinHash, MinHashLSH
from sklearn.metrics.pairwise import cosine_similarity
from src.config import SIMILARITY_THRESHOLD
from src.preprocessing.text_preprocessor import TextPreprocessor

class DuplicateDetector:
    """Detects duplicate products using MinHashLSH and Embeddings."""
    
    def __init__(self, similarity_threshold: float = SIMILARITY_THRESHOLD) -> None:
        """
        Initialize the duplicate detector.
        
        Args:
            similarity_threshold: Threshold for similarity.
        """
        self.similarity_threshold = similarity_threshold
        self.preprocessor = TextPreprocessor()
        
    def _compute_minhash(self, text: str, num_perm: int = 128) -> MinHash:
        """
        Create MinHash signature from text shingles.
        
        Args:
            text: Input text.
            num_perm: Number of permutations.
            
        Returns:
            MinHash signature.
        """
        m = MinHash(num_perm=num_perm)
        tokens = self.preprocessor.process(text)
        if not tokens:
            m.update("empty".encode('utf8'))
            return m
        # Create word-level shingles (3-grams of tokens)
        for i in range(max(1, len(tokens) - 2)):
            shingle = ' '.join(tokens[i:i + 3])
            m.update(shingle.encode('utf8'))
        return m
        
    def find_duplicates_lsh(self, documents: List[str], product_ids: List[str]) -> List[Set[str]]:
        """
        Use MinHashLSH to find near-duplicate clusters.
        
        Args:
            documents: List of document strings.
            product_ids: List of product IDs.
            
        Returns:
            List of sets containing duplicate product IDs.
        """
        if not documents or len(documents) != len(product_ids):
            return []
            
        lsh = MinHashLSH(threshold=self.similarity_threshold, num_perm=128)
        minhashes = {}
        
        for i, text in enumerate(documents):
            m = self._compute_minhash(text)
            lsh.insert(product_ids[i], m)
            minhashes[product_ids[i]] = m
            
        clusters = []
        visited = set()
        
        for pid, m in minhashes.items():
            if pid in visited:
                continue
                
            result = lsh.query(m)
            if len(result) > 1:
                cluster = set(result)
                clusters.append(cluster)
                visited.update(cluster)
                
        return clusters
        
    def find_duplicates_embedding(self, embeddings: np.ndarray, product_ids: List[str], threshold: float = 0.85) -> List[Set[str]]:
        """
        Pairwise cosine similarity on embeddings to cluster similar products.
        
        Args:
            embeddings: Numpy array of embeddings.
            product_ids: List of product IDs.
            threshold: Similarity threshold.
            
        Returns:
            List of sets containing duplicate product IDs.
        """
        if len(embeddings) == 0 or len(embeddings) != len(product_ids):
            return []
            
        sim_matrix = cosine_similarity(embeddings)
        clusters = []
        visited = set()
        
        for i in range(len(product_ids)):
            if product_ids[i] in visited:
                continue
                
            similar_indices = np.where(sim_matrix[i] >= threshold)[0]
            if len(similar_indices) > 1:
                cluster = set(product_ids[idx] for idx in similar_indices)
                clusters.append(cluster)
                visited.update(cluster)
                
        return clusters
        
    def find_all_duplicates(self, documents: List[str], embeddings: np.ndarray, product_ids: List[str]) -> Dict[str, Any]:
        """
        Combine both methods, return union of duplicate clusters with method attribution.
        
        Args:
            documents: List of document strings.
            embeddings: Numpy array of embeddings.
            product_ids: List of product IDs.
            
        Returns:
            Dict with union of duplicate clusters.
        """
        lsh_clusters = self.find_duplicates_lsh(documents, product_ids)
        emb_clusters = self.find_duplicates_embedding(embeddings, product_ids, self.similarity_threshold)
        
        all_clusters = []
        for c in lsh_clusters:
            all_clusters.append({'cluster': c, 'method': 'lsh'})
            
        for c in emb_clusters:
            exists = any(c.issubset(ac['cluster']) or ac['cluster'].issubset(c) for ac in all_clusters)
            if not exists:
                all_clusters.append({'cluster': c, 'method': 'embedding'})
                
        return {'duplicates': all_clusters}
