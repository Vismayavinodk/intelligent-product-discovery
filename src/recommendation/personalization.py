"""
Personalization engine module.
"""
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional

class PersonalizationEngine:
    """Engine for personalizing recommendations based on user history."""
    
    def __init__(self) -> None:
        """Initialize the personalization engine."""
        self.history: Dict[str, List[Dict[str, Any]]] = {}
        
    def update_history(self, user_id: str, product_id: str, action: str = 'view', metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Track user interactions.
        
        Args:
            user_id: ID of the user.
            product_id: ID of the product.
            action: Type of action ('view', 'rate', 'search').
            metadata: Additional metadata.
        """
        if user_id not in self.history:
            self.history[user_id] = []
            
        interaction = {
            'product_id': product_id,
            'action': action,
            'metadata': metadata or {}
        }
        self.history[user_id].append(interaction)
        
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Compute category affinities, brand affinities, avg preferred rating from history.
        
        Args:
            user_id: ID of the user.
            
        Returns:
            Dictionary containing user preferences.
        """
        if user_id not in self.history or not self.history[user_id]:
            return {'categories': {}, 'brands': {}, 'avg_rating': 0.0}
            
        categories: Dict[str, int] = {}
        brands: Dict[str, int] = {}
        total_rating = 0.0
        rating_count = 0
        
        for item in self.history[user_id]:
            meta = item.get('metadata', {})
            
            cat = meta.get('category')
            if cat:
                categories[cat] = categories.get(cat, 0) + 1
                
            brand = meta.get('brand')
            if brand:
                brands[brand] = brands.get(brand, 0) + 1
                
            if item['action'] == 'rate':
                rating = meta.get('rating')
                if rating is not None:
                    total_rating += float(rating)
                    rating_count += 1
                    
        avg_rating = total_rating / rating_count if rating_count > 0 else 0.0
        
        return {
            'categories': categories,
            'brands': brands,
            'avg_rating': avg_rating
        }
        
    def personalize(self, user_id: str, ranked_results: List[Tuple[str, float]], products_df: pd.DataFrame) -> List[Tuple[str, float]]:
        """
        Re-rank results by boosting scores for products matching user preferences.
        
        Args:
            user_id: ID of the user.
            ranked_results: List of (product_id, score).
            products_df: DataFrame containing product details.
            
        Returns:
            Re-ranked list of tuples.
        """
        profile = self.get_user_profile(user_id)
        
        if not profile['categories'] and not profile['brands']:
            return ranked_results
            
        personalized = []
        for pid, score in ranked_results:
            boost = 1.0
            
            prod_rows = products_df[products_df['product_id'] == pid]
            if not prod_rows.empty:
                prod = prod_rows.iloc[0]
                
                cat = prod.get('category')
                if cat and cat in profile['categories']:
                    boost += 0.1 * profile['categories'][cat]
                    
                brand = prod.get('brand')
                if brand and brand in profile['brands']:
                    boost += 0.1 * profile['brands'][brand]
                    
                rating = prod.get('rating')
                if pd.notna(rating) and profile['avg_rating'] > 0:
                    diff = abs(float(rating) - profile['avg_rating'])
                    if diff < 1.0:
                        boost += 0.05
                        
            personalized.append((pid, score * boost))
            
        return sorted(personalized, key=lambda x: x[1], reverse=True)
        
    def clear_history(self, user_id: str) -> None:
        """
        Reset user history.
        
        Args:
            user_id: ID of the user.
        """
        if user_id in self.history:
            del self.history[user_id]
