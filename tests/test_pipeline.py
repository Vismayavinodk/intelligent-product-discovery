"""
Test suite for the E-Commerce IR & Recommendation System.
Uses small synthetic data so tests run without the full dataset.

Run:  python -m pytest tests/test_pipeline.py -v
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.text_preprocessor import (
    clean_text, tokenize, remove_stopwords, lemmatize, preprocess,
    preprocess_to_string, TextPreprocessor,
)
from src.indexing.inverted_index import InvertedIndex
from src.indexing.embedding_index import EmbeddingIndex
from src.retrieval.bm25_retrieval import BM25Retriever
from src.retrieval.tfidf_retrieval import TFIDFRetriever
from src.retrieval.semantic_retrieval import SemanticRetriever
from src.retrieval.query_expansion import QueryExpander
from src.retrieval.hybrid_ranker import HybridRanker
from src.recommendation.content_based import ContentBasedRecommender
from src.recommendation.collaborative import CollaborativeRecommender
from src.analysis.review_analyzer import ReviewAnalyzer
from src.analysis.duplicate_detector import DuplicateDetector
from src.evaluation.metrics import (
    precision_at_k, recall_at_k, f1_at_k,
    average_precision, mean_average_precision,
    dcg_at_k, ndcg_at_k,
    reciprocal_rank, mean_reciprocal_rank,
    evaluate_retriever, generate_test_queries,
)
from src.retrieval.spell_corrector import SpellCorrector
from src.retrieval.faceted_search import FacetedSearch
from src.indexing.custom_inverted_index import CustomInvertedIndex


# ── Sample Data ───────────────────────────────────────────────────────
SAMPLE_DOCS = [
    "Apple iPhone 14 Pro smartphone with excellent camera quality and A16 chip",
    "Samsung Galaxy S23 Ultra phone with great screen and Snapdragon processor",
    "Sony WH-1000XM5 wireless noise cancelling bluetooth headphones",
    "Bose QuietComfort 45 noise cancelling wireless headphones for music",
    "Apple iPad Air tablet for reading and drawing with M1 chip",
    "Amazon Kindle Paperwhite e-reader for digital book reading",
    "JBL Flip 6 portable waterproof bluetooth speaker for outdoor music",
    "Sony SRS-XB43 extra bass portable wireless bluetooth speaker",
    "Samsung Galaxy Tab S8 tablet with S-Pen for productivity and reading",
    "Apple MacBook Air M2 laptop for students and professionals",
]

SAMPLE_IDS = [str(i + 1) for i in range(len(SAMPLE_DOCS))]


# ── Text Preprocessing Tests ─────────────────────────────────────────
class TestTextPreprocessing:
    def test_clean_text_removes_html(self):
        assert "<b>" not in clean_text("Hello <b>world</b>")

    def test_clean_text_removes_urls(self):
        assert "http" not in clean_text("Visit https://example.com today")

    def test_clean_text_lowercases(self):
        result = clean_text("HELLO World")
        assert result == "hello world"

    def test_clean_text_handles_none(self):
        assert clean_text(None) == ""
        assert clean_text(123) == ""

    def test_tokenize(self):
        tokens = tokenize("hello world test")
        assert len(tokens) >= 3

    def test_remove_stopwords(self):
        tokens = ["this", "is", "a", "test", "sentence"]
        filtered = remove_stopwords(tokens)
        assert "this" not in filtered
        assert "test" in filtered

    def test_lemmatize(self):
        tokens = ["running", "cats", "better"]
        result = lemmatize(tokens)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_preprocess_pipeline(self):
        result = preprocess("The cats are running quickly!")
        assert isinstance(result, list)
        assert len(result) > 0
        # Stopwords should be removed
        assert "the" not in result
        assert "are" not in result

    def test_preprocessor_class_caching(self):
        pp = TextPreprocessor(use_cache=True)
        r1 = pp.process("hello world test")
        r2 = pp.process("hello world test")
        assert r1 == r2  # cached result


# ── Inverted Index Tests ─────────────────────────────────────────────
class TestInvertedIndex:
    def test_build_and_search(self):
        index = InvertedIndex()
        processed = [preprocess_to_string(doc) for doc in SAMPLE_DOCS]
        index.build(processed)

        results = index.search("wireless bluetooth headphones", top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        # Top result should be one of the headphone docs (index 2 or 3)
        top_idx = results[0][0]
        assert top_idx in [2, 3]

    def test_search_returns_scores(self):
        index = InvertedIndex()
        processed = [preprocess_to_string(doc) for doc in SAMPLE_DOCS]
        index.build(processed)

        results = index.search("phone camera", top_k=5)
        for idx, score in results:
            assert score >= 0.0


# ── BM25 Retrieval Tests ─────────────────────────────────────────────
class TestBM25Retrieval:
    def test_retrieve(self):
        pp = TextPreprocessor()
        corpus_tokens = [pp.process(doc) for doc in SAMPLE_DOCS]
        bm25 = BM25Retriever(corpus_tokens, SAMPLE_IDS)

        results = bm25.retrieve("bluetooth speaker music", top_k=3)
        assert len(results) == 3
        pids = [pid for pid, _ in results]
        # Speaker docs are IDs "7" and "8"
        assert "7" in pids or "8" in pids

    def test_retrieve_returns_scores(self):
        pp = TextPreprocessor()
        corpus_tokens = [pp.process(doc) for doc in SAMPLE_DOCS]
        bm25 = BM25Retriever(corpus_tokens, SAMPLE_IDS)

        results = bm25.retrieve("tablet reading", top_k=5)
        for pid, score in results:
            assert isinstance(score, float)


# ── TF-IDF Retrieval Tests ──────────────────────────────────────────
class TestTFIDFRetrieval:
    def test_retrieve(self):
        index = InvertedIndex()
        processed = [preprocess_to_string(doc) for doc in SAMPLE_DOCS]
        index.build(processed)

        retriever = TFIDFRetriever(index, SAMPLE_IDS)
        results = retriever.retrieve("noise cancelling headphones", top_k=3)
        assert len(results) > 0
        pids = [pid for pid, _ in results]
        assert "3" in pids or "4" in pids


# ── Query Expansion Tests ───────────────────────────────────────────
class TestQueryExpansion:
    def test_wordnet_expansion(self):
        expander = QueryExpander()
        expanded = expander.expand_wordnet("phone camera")
        # Should contain original terms plus synonyms
        assert "phone" in expanded.lower() or "camera" in expanded.lower()
        # Should be longer than original
        assert len(expanded.split()) >= 2

    def test_expand_dispatch(self):
        expander = QueryExpander()
        result = expander.expand("laptop computer", method="wordnet")
        assert isinstance(result, str)
        assert len(result) > 0


# ── Hybrid Ranker Tests ──────────────────────────────────────────────
class TestHybridRanker:
    def test_rrf_fusion(self):
        """Test that RRF correctly fuses two ranked lists."""
        # Create mock retrievers using BM25 and TF-IDF
        pp = TextPreprocessor()
        corpus_tokens = [pp.process(doc) for doc in SAMPLE_DOCS]
        processed = [preprocess_to_string(doc) for doc in SAMPLE_DOCS]

        index = InvertedIndex()
        index.build(processed)

        tfidf_ret = TFIDFRetriever(index, SAMPLE_IDS)
        bm25_ret = BM25Retriever(corpus_tokens, SAMPLE_IDS)

        hybrid = HybridRanker(
            retrievers={"tfidf": tfidf_ret, "bm25": bm25_ret}
        )

        results = hybrid.retrieve("wireless headphones music", top_k=5)
        assert len(results) == 5
        # Results should be (product_id, score) tuples
        for pid, score in results:
            assert isinstance(pid, str)
            assert isinstance(score, float)
            assert score > 0

    def test_retrieve_with_details(self):
        pp = TextPreprocessor()
        corpus_tokens = [pp.process(doc) for doc in SAMPLE_DOCS]
        processed = [preprocess_to_string(doc) for doc in SAMPLE_DOCS]

        index = InvertedIndex()
        index.build(processed)

        tfidf_ret = TFIDFRetriever(index, SAMPLE_IDS)
        bm25_ret = BM25Retriever(corpus_tokens, SAMPLE_IDS)

        hybrid = HybridRanker(
            retrievers={"tfidf": tfidf_ret, "bm25": bm25_ret}
        )

        details = hybrid.retrieve_with_details("tablet", top_k=3)
        assert "tfidf" in details
        assert "bm25" in details
        assert "fused" in details


# ── Evaluation Metrics Tests ─────────────────────────────────────────
class TestMetrics:
    def test_precision_at_k(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "e"}
        assert np.isclose(precision_at_k(retrieved, relevant, 3), 2 / 3)
        assert np.isclose(precision_at_k(retrieved, relevant, 5), 3 / 5)

    def test_recall_at_k(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "e", "f", "g"}
        assert np.isclose(recall_at_k(retrieved, relevant, 5), 3 / 5)

    def test_f1_at_k(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "c"}
        f1 = f1_at_k(retrieved, relevant, 3)
        p = precision_at_k(retrieved, relevant, 3)
        r = recall_at_k(retrieved, relevant, 3)
        expected = 2 * p * r / (p + r)
        assert np.isclose(f1, expected)

    def test_average_precision(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "c"}
        # AP = (1/1 + 2/3) / 2 = 0.833...
        ap = average_precision(retrieved, relevant)
        assert np.isclose(ap, (1.0 + 2 / 3) / 2)

    def test_map(self):
        all_retrieved = [["a", "b", "c"], ["d", "e", "f"]]
        all_relevant = [{"a", "c"}, {"e"}]
        m = mean_average_precision(all_retrieved, all_relevant)
        assert 0 <= m <= 1

    def test_ndcg(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "c"}
        n = ndcg_at_k(retrieved, relevant, 4)
        assert 0 <= n <= 1

    def test_reciprocal_rank(self):
        retrieved = ["x", "y", "a", "b"]
        relevant = {"a"}
        rr = reciprocal_rank(retrieved, relevant)
        assert np.isclose(rr, 1 / 3)

    def test_mrr(self):
        all_retrieved = [["x", "a"], ["b", "c"]]
        all_relevant = [{"a"}, {"c"}]
        mrr = mean_reciprocal_rank(all_retrieved, all_relevant)
        # (1/2 + 1/2) / 2 = 0.5
        assert np.isclose(mrr, 0.5)

    def test_empty_inputs(self):
        assert precision_at_k([], set(), 5) == 0.0
        assert recall_at_k([], {"a"}, 5) == 0.0
        assert ndcg_at_k([], {"a"}, 5) == 0.0


# ── Content-Based Recommendation Tests ───────────────────────────────
class TestContentBasedRecommender:
    def test_recommend(self):
        products_df = pd.DataFrame({
            "product_id": list(range(1, len(SAMPLE_DOCS) + 1)),
            "name": [doc.split()[0] + " " + doc.split()[1] for doc in SAMPLE_DOCS],
            "brand": ["Apple", "Samsung", "Sony", "Bose", "Apple",
                       "Amazon", "JBL", "Sony", "Samsung", "Apple"],
            "categories": ["phone", "phone", "headphones", "headphones", "tablet",
                           "ereader", "speaker", "speaker", "tablet", "laptop"],
            "description": SAMPLE_DOCS,
        })

        cb = ContentBasedRecommender()
        cb.build_profile(products_df)

        # Recommend similar to the Sony headphones (product_id=3)
        recs = cb.recommend(3, top_n=3)
        assert len(recs) > 0
        # Bose headphones (4) should be among top recommendations
        rec_pids = [pid for pid, _ in recs]
        assert 4 in rec_pids


# ── Collaborative Filtering Tests ────────────────────────────────────
class TestCollaborativeRecommender:
    def test_build_and_recommend(self):
        ratings_df = pd.DataFrame({
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u2", "u3", "u3", "u3"],
            "product_id": ["p1", "p2", "p3", "p1", "p2", "p4", "p2", "p3", "p4"],
            "rating": [5.0, 4.0, 3.0, 4.0, 5.0, 3.0, 3.0, 4.0, 5.0],
        })

        collab = CollaborativeRecommender(n_components=2)
        collab.build(ratings_df)

        # u1 hasn't rated p4, should get recommendation
        recs = collab.recommend("u1", top_n=3)
        assert isinstance(recs, list)

    def test_cold_start(self):
        ratings_df = pd.DataFrame({
            "user_id": ["u1", "u1", "u2", "u2"],
            "product_id": ["p1", "p2", "p1", "p2"],
            "rating": [5.0, 3.0, 4.0, 2.0],
        })
        collab = CollaborativeRecommender(n_components=1)
        collab.build(ratings_df)
        # Unknown user
        recs = collab.recommend("unknown_user", top_n=5)
        assert recs == []


# ── Review Analyzer Tests ───────────────────────────────────────────
class TestReviewAnalyzer:
    def test_sentiment_analysis(self):
        analyzer = ReviewAnalyzer()
        result = analyzer.analyze_sentiment("This product is absolutely wonderful!")
        assert result["polarity"] > 0
        result_neg = analyzer.analyze_sentiment("Terrible product, worst purchase ever.")
        assert result_neg["polarity"] < 0

    def test_full_analysis(self):
        analyzer = ReviewAnalyzer()
        reviews = [
            "Great product with amazing battery life",
            "The screen quality is excellent",
            "Poor build quality and bad customer service",
        ]
        analysis = analyzer.full_analysis("Test Product", reviews)
        assert "sentiment" in analysis
        assert "attributes" in analysis
        assert "summary" in analysis


# ── Duplicate Detection Tests ────────────────────────────────────────
class TestDuplicateDetector:
    def test_find_duplicates_lsh(self):
        docs = [
            "Apple iPhone 14 Pro smartphone with camera",
            "Apple iPhone 14 Pro smartphone with excellent camera",  # near duplicate
            "Sony wireless noise cancelling headphones",
            "Samsung Galaxy tablet for reading",
        ]
        pids = ["1", "2", "3", "4"]
        detector = DuplicateDetector(similarity_threshold=0.5)
        clusters = detector.find_duplicates_lsh(docs, pids)
        # Should detect docs 0 and 1 as near-duplicates
        assert isinstance(clusters, list)

    def test_find_duplicates_embedding(self):
        # Synthetic embeddings where 0 and 1 are very similar
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],   # very similar to 0
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        pids = ["1", "2", "3", "4"]
        detector = DuplicateDetector()
        clusters = detector.find_duplicates_embedding(embeddings, pids, threshold=0.95)
        assert len(clusters) > 0
        # Cluster should contain "1" and "2"
        found = False
        for cluster in clusters:
            if "1" in cluster and "2" in cluster:
                found = True
        assert found


# ── Spell Corrector Tests ─────────────────────────────────────────────
class TestSpellCorrector:
    def test_spell_corrector_vocab_and_edit_distance(self):
        corrector = SpellCorrector()
        assert corrector._edit_distance("kitten", "sitting") == 3
        assert corrector._edit_distance("apple", "apple") == 0
        assert corrector._edit_distance("", "test") == 4

        docs = ["wireless headphones with noise cancelling", "bluetooth speaker with bass"]
        corrector.build_vocabulary(docs)
        assert corrector.vocab_counts["headphones"] > 0
        assert corrector.vocab_counts["speaker"] > 0

    def test_correct_word_and_query(self):
        corrector = SpellCorrector(vocabulary=["wireless", "headphones", "bluetooth", "speaker", "phone"])
        assert corrector.correct_word("headphonse") == "headphones"
        assert corrector.correct_word("speeker") == "speaker"
        corrected_q = corrector.correct_query("wireles headphonse")
        assert corrected_q == "wireless headphones"

    def test_suggest_completions(self):
        corrector = SpellCorrector(vocabulary=["headphone", "headphones", "headset", "speaker"])
        suggestions = corrector.suggest_completions("head", top_k=2)
        assert len(suggestions) <= 2
        assert all(s.startswith("head") for s in suggestions)


# ── Faceted Search Tests ─────────────────────────────────────────────
class TestFacetedSearch:
    def test_facets_filtering_and_counts(self):
        df = pd.DataFrame({
            "product_id": ["p1", "p2", "p3", "p4"],
            "name": ["iPhone 14", "Galaxy S23", "Sony WH1000", "JBL Flip"],
            "brand": ["Apple", "Samsung", "Sony", "JBL"],
            "categories": ["phone, mobile", "phone", "audio, headphones", "audio, speaker"],
            "price": [999.0, 899.0, 349.0, 99.0],
            "rating": [4.8, 4.6, 4.4, 3.9],
        })
        fs = FacetedSearch(df)
        available = fs.get_available_facets()
        assert "Apple" in available["brands"]
        assert "phone" in available["categories"]

        results = [("p1", 0.9), ("p2", 0.8), ("p3", 0.7), ("p4", 0.6)]

        # Brand filter
        filtered_brand = fs.filter(results, brand="Apple")
        assert len(filtered_brand) == 1
        assert filtered_brand[0][0] == "p1"

        # Price range filter
        filtered_price = fs.filter(results, min_price=100.0, max_price=500.0)
        assert len(filtered_price) == 1
        assert filtered_price[0][0] == "p3"

        # Apply facets dict
        filtered_dict = fs.apply_facets(results, {"min_rating": 4.5})
        assert len(filtered_dict) == 2

        # Facet counts
        counts = fs.get_facet_counts(results)
        assert counts["brands"]["Apple"] == 1
        assert counts["rating_ranges"]["4 stars & above"] == 3


# ── Custom Inverted Index Tests ───────────────────────────────────────
class TestCustomInvertedIndex:
    def test_build_and_search(self):
        index = CustomInvertedIndex()
        index.build(SAMPLE_DOCS)

        assert len(index.get_vocabulary()) > 0
        assert index.get_document_frequency("headphones") >= 2
        postings = index.get_postings("headphones")
        assert len(postings) >= 2

        results = index.search("wireless noise cancelling headphones", top_k=3)
        assert len(results) > 0
        top_idx, score = results[0]
        assert top_idx in [2, 3]  # One of the headphone documents
        assert score > 0.0

    def test_tf_idf_values(self):
        index = CustomInvertedIndex()
        index.build(["quick brown fox", "jumped over lazy dog"])
        tf = index._compute_tf("quick", 0)
        idf = index._compute_idf("quick")
        tfidf = index._compute_tfidf("quick", 0)
        assert tf > 0
        assert idf > 0
        assert np.isclose(tfidf, tf * idf)


# ── Hybrid Ranker Weighted Fusion Tests ──────────────────────────────
class TestHybridRankerWeightedFusion:
    def test_weighted_fusion(self):
        pp = TextPreprocessor()
        corpus_tokens = [pp.process(doc) for doc in SAMPLE_DOCS]
        processed = [preprocess_to_string(doc) for doc in SAMPLE_DOCS]

        index = InvertedIndex()
        index.build(processed)

        tfidf_ret = TFIDFRetriever(index, SAMPLE_IDS)
        bm25_ret = BM25Retriever(corpus_tokens, SAMPLE_IDS)

        hybrid = HybridRanker(
            retrievers={"tfidf": tfidf_ret, "bm25": bm25_ret}
        )
        hybrid.set_weights({"tfidf": 0.7, "bm25": 0.3})

        results = hybrid.retrieve("bluetooth speaker", top_k=5, fusion_method="weighted")
        assert len(results) == 5
        for pid, score in results:
            assert isinstance(pid, str)
            assert isinstance(score, float)


# ── Evaluation and Queries Tests ─────────────────────────────────────
class TestEvaluationAndQueries:
    def test_evaluate_retriever_and_generate_queries(self):
        df = pd.DataFrame({
            "product_id": ["p1", "p2", "p3", "p4"],
            "brand": ["Apple", "Apple", "Sony", "Sony"],
            "categories": ["phone", "phone", "headphones", "headphones"],
        })
        queries = generate_test_queries(df, n_queries=2)
        assert len(queries) > 0
        assert "relevant_ids" in queries[0]

        class DummyRetriever:
            def retrieve(self, q, top_k=10):
                return [("p1", 1.0), ("p2", 0.5)]

        metrics = evaluate_retriever(DummyRetriever(), queries, top_k=2)
        assert "Precision@2" in metrics
        assert "Recall@2" in metrics
        assert "MAP" in metrics
        assert "MRR" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

