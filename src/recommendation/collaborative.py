"""
Collaborative filtering recommender module.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple
from sklearn.decomposition import TruncatedSVD
from src.config import SVD_N_COMPONENTS

logger = logging.getLogger(__name__)

class CollaborativeRecommender:
    """Recommends products based on collaborative filtering using TruncatedSVD."""
    
    def __init__(self, n_components: int = SVD_N_COMPONENTS) -> None:
        """
        Initialize the collaborative recommender.
        
        Args:
            n_components: Number of components for SVD.
        """
        self.n_components = n_components
        self.user_item_matrix = None
        self.user_ids = []
        self.product_ids = []
        self.user_factors = None
        self.item_factors = None
        self.svd = TruncatedSVD(n_components=self.n_components)
        
    def build(self, ratings_df: pd.DataFrame) -> None:
        """
        Build user-item matrix and apply TruncatedSVD.
        
        Args:
            ratings_df: DataFrame containing user_id, product_id, rating.
        """
        if ratings_df.empty:
            raise ValueError("Ratings DataFrame is empty.")
            
        pivot_table = ratings_df.pivot_table(index='user_id', columns='product_id', values='rating', fill_value=0)
        self.user_ids = pivot_table.index.tolist()
        self.product_ids = pivot_table.columns.tolist()
        self.user_item_matrix = pivot_table.values
        
        n_comps = min(self.n_components, self.user_item_matrix.shape[1] - 1, self.user_item_matrix.shape[0] - 1)
        if n_comps < 1:
            raise ValueError("Not enough data to perform SVD.")
        
        self.svd = TruncatedSVD(n_components=n_comps)
        self.user_factors = self.svd.fit_transform(self.user_item_matrix)
        self.item_factors = self.svd.components_
        
    def predict_rating(self, user_id: str, product_id: str) -> float:
        """
        Predict rating for a user and product.
        
        Args:
            user_id: ID of the user.
            product_id: ID of the product.
            
        Returns:
            Predicted rating.
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model not built. Call build first.")
            
        if user_id not in self.user_ids or product_id not in self.product_ids:
            return 0.0
            
        u_idx = self.user_ids.index(user_id)
        i_idx = self.product_ids.index(product_id)
        
        pred = np.dot(self.user_factors[u_idx, :], self.item_factors[:, i_idx])
        return float(pred)
        
    def recommend(self, user_id: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Predict ratings for all unseen products and return top-n.
        
        Args:
            user_id: ID of the user.
            top_n: Number of recommendations.
            
        Returns:
            List of tuples (product_id, predicted_rating).
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model not built. Call build first.")
            
        if user_id not in self.user_ids:
            logger.warning(f"Cold-start: User {user_id} not found.")
            return []
            
        u_idx = self.user_ids.index(user_id)
        user_ratings = self.user_item_matrix[u_idx, :]
        unseen_mask = user_ratings == 0
        
        all_preds = np.dot(self.user_factors[u_idx, :], self.item_factors)
        
        unseen_indices = np.where(unseen_mask)[0]
        unseen_preds = all_preds[unseen_indices]
        
        top_indices = unseen_indices[np.argsort(unseen_preds)[::-1][:top_n]]
        
        recommendations = []
        for idx in top_indices:
            pred = float(all_preds[idx])
            if pred > 0:
                recommendations.append((self.product_ids[idx], pred))
                
        return recommendations
