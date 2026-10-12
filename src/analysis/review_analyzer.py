"""
Review analysis module.
"""
from typing import List, Dict, Any
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from src.preprocessing.text_preprocessor import TextPreprocessor

class ReviewAnalyzer:
    """Analyzes product reviews."""
    
    def __init__(self) -> None:
        """Initialize the review analyzer."""
        self.preprocessor = TextPreprocessor()
        
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a text using TextBlob.
        
        Args:
            text: Text to analyze.
            
        Returns:
            Dict containing polarity and subjectivity.
        """
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
        
    def analyze_product_reviews(self, reviews: List[str]) -> Dict[str, Any]:
        """
        Compute average polarity, sentiment distribution, and top representative sentences.
        
        Args:
            reviews: List of review strings.
            
        Returns:
            Dict containing analysis results.
        """
        if not reviews:
            return {
                'avg_polarity': 0.0,
                'distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
                'representative': []
            }
            
        polarities = []
        distribution = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        for review in reviews:
            sent = self.analyze_sentiment(review)
            pol = sent['polarity']
            polarities.append(pol)
            
            if pol > 0.1:
                distribution['positive'] += 1
            elif pol < -0.1:
                distribution['negative'] += 1
            else:
                distribution['neutral'] += 1
                
        avg_polarity = sum(polarities) / len(polarities)
        
        return {
            'avg_polarity': avg_polarity,
            'distribution': distribution,
            'representative': reviews[:3]  # Simplified representative sentences
        }
        
    def extract_attributes(self, reviews: List[str]) -> List[str]:
        """
        Extract frequently mentioned noun phrases as product attributes.
        
        Args:
            reviews: List of review strings.
            
        Returns:
            List of noun phrases.
        """
        phrases = {}
        for review in reviews:
            blob = TextBlob(review)
            for np_ in blob.noun_phrases:
                phrases[np_] = phrases.get(np_, 0) + 1
                
        sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
        return [p[0] for p in sorted_phrases[:10]]
        
    def summarize_reviews(self, reviews: List[str], top_k: int = 5) -> List[str]:
        """
        Rank sentences by TF-IDF importance, return top-k summary sentences.
        
        Args:
            reviews: List of review strings.
            top_k: Number of sentences to return.
            
        Returns:
            List of summary sentences.
        """
        if not reviews:
            return []
            
        sentences = []
        for review in reviews:
            blob = TextBlob(review)
            sentences.extend([str(s) for s in blob.sentences])
            
        if not sentences:
            return []
            
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            
            top_indices = scores.argsort()[::-1][:min(top_k, len(sentences))]
            return [sentences[i] for i in top_indices]
        except ValueError:
            return sentences[:top_k]
            
    def full_analysis(self, product_name: str, reviews: List[str]) -> Dict[str, Any]:
        """
        Complete analysis combining all above.
        
        Args:
            product_name: Name of the product.
            reviews: List of review strings.
            
        Returns:
            Dict containing full analysis.
        """
        return {
            'product_name': product_name,
            'sentiment': self.analyze_product_reviews(reviews),
            'attributes': self.extract_attributes(reviews),
            'summary': self.summarize_reviews(reviews)
        }
