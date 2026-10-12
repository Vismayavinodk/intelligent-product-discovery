"""
Content-based recommender module.
"""
import pandas as pd
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import SIMILARITY_THRESHOLD
from src.preprocessing.text_preprocessor import TextPreprocessor

class ContentBasedRecommender:
    """Recommends products based on their content features."""
    
    def __init__(self) -> None:
        """Initialize the content-based recommender."""
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.products_df = None
        
    def build_profile(self, products_df: pd.DataFrame) -> None:
        """
        Build TF-IDF vectors from product features.
        
        Args:
            products_df: DataFrame containing product details.
        """
        self.products_df = products_df.copy()
        
        def combine_features(row):
            features = [
                str(row.get('name', '')),
                str(row.get('brand', '')),
                str(row.get('categories', '')),
                str(row.get('description', ''))
            ]
            return ' '.join(features)
            
        combined_text = self.products_df.apply(combine_features, axis=1)
        processed_text = [self.preprocessor.process_to_string(text) for text in combined_text]
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_text)
        
    def recommend(self, product_id: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Find most similar products using cosine similarity on TF-IDF vectors.
        
        Args:
            product_id: ID of the product.
            top_n: Number of recommendations.
            
        Returns:
            List of tuples (product_id, similarity_score).
        """
        if self.tfidf_matrix is None or self.products_df is None:
            raise ValueError("Profile not built. Call build_profile first.")
            
        if product_id not in self.products_df['product_id'].values:
            return []
            
        idx = self.products_df.index[self.products_df['product_id'] == product_id].tolist()[0]
        query_vector = self.tfidf_matrix[idx]
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        similar_indices = similarities.argsort()[::-1]
        
        recommendations = []
        for i in similar_indices:
            if i != idx and similarities[i] > 0:
                recommendations.append((self.products_df.iloc[i]['product_id'], float(similarities[i])))
                if len(recommendations) >= top_n:
                    break
                    
        return recommendations
        
    def recommend_from_text(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Find products similar to a text description.
        
        Args:
            text: Query text.
            top_n: Number of recommendations.
            
        Returns:
            List of tuples (product_id, similarity_score).
        """
        if self.tfidf_matrix is None or self.products_df is None:
            raise ValueError("Profile not built. Call build_profile first.")
            
        processed_text = self.preprocessor.process_to_string(text)
        query_vector = self.vectorizer.transform([processed_text])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        similar_indices = similarities.argsort()[::-1]
        
        recommendations = []
        for i in similar_indices:
            if similarities[i] > 0:
                recommendations.append((self.products_df.iloc[i]['product_id'], float(similarities[i])))
                if len(recommendations) >= top_n:
                    break
                    
        return recommendations
