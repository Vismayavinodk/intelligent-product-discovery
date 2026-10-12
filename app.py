"""
Streamlit Web Application for E-Commerce IR & Recommendation System.
====================================================================
Interactive user interface providing:
- Multi-method Search (TF-IDF, BM25, Dense Semantic, Hybrid RRF)
- Query Expansion (WordNet & PRF) and Spell Correction
- Facet Filtering (Brand, Category, Price, Rating)
- Product Recommendations (Content-Based, Collaborative SVD, Personalization)
- Review & Sentiment Analysis with Key Attribute Extraction
- Duplicate & Near-Duplicate Detection (MinHash LSH & Embeddings)
- Evaluation Benchmark Dashboard with Visual Metrics Comparison
"""

import os
import sys
import pickle
import traceback
from typing import List, Dict, Tuple, Any, Optional, Set

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import config
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.indexing.inverted_index import InvertedIndex
from src.indexing.embedding_index import EmbeddingIndex
from src.retrieval.tfidf_retrieval import TFIDFRetriever
from src.retrieval.bm25_retrieval import BM25Retriever
from src.retrieval.semantic_retrieval import SemanticRetriever
from src.retrieval.query_expansion import QueryExpander
from src.retrieval.hybrid_ranker import HybridRanker
from src.recommendation.content_based import ContentBasedRecommender
from src.recommendation.collaborative import CollaborativeRecommender
from src.recommendation.personalization import PersonalizationEngine
from src.analysis.review_analyzer import ReviewAnalyzer
from src.analysis.duplicate_detector import DuplicateDetector
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    average_precision,
    mean_average_precision,
    ndcg_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce IR & Recommender Engine",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .product-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease-in-out;
    }
    .product-card:hover {
        border-color: #3B82F6;
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.15);
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 0.4rem;
    }
    .badge-brand { background-color: #DBEAFE; color: #1D4ED8; }
    .badge-category { background-color: #FEF3C7; color: #B45309; }
    .badge-score { background-color: #D1FAE5; color: #065F46; }
    .badge-price { background-color: #F3E8FF; color: #6B21A8; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE DATASET GENERATOR (Fallback if real data is missing)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def get_sample_products() -> pd.DataFrame:
    """Generate a rich sample dataset of e-commerce products for demo purposes."""
    sample_records = [
        {
            "product_id": 1,
            "name": "Apple iPhone 14 Pro 128GB Deep Purple",
            "brand": "Apple",
            "categories": "Smartphones, Electronics, Mobile",
            "price": 999.00,
            "avg_rating": 4.8,
            "review_count": 340,
            "description": "Super Retina XDR display with ProMotion, Dynamic Island, A16 Bionic chip, and 48MP Pro camera system with Photonic Engine.",
            "reviews_text": "Incredible camera quality and battery life. Dynamic Island is super handy. Highly recommend for photography enthusiasts. Beautiful OLED screen.",
        },
        {
            "product_id": 2,
            "name": "Samsung Galaxy S23 Ultra 256GB Phantom Black",
            "brand": "Samsung",
            "categories": "Smartphones, Electronics, Mobile",
            "price": 1199.99,
            "avg_rating": 4.7,
            "review_count": 290,
            "description": "Flagship smartphone featuring 200MP camera, built-in S Pen stylus, Snapdragon 8 Gen 2 processor, and 6.8 inch Dynamic AMOLED 2X display.",
            "reviews_text": "The zoom camera is out of this world. S-Pen is extremely useful for note taking and work. Battery lasts well over a full day.",
        },
        {
            "product_id": 3,
            "name": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
            "brand": "Sony",
            "categories": "Headphones, Audio, Electronics",
            "price": 398.00,
            "avg_rating": 4.6,
            "review_count": 512,
            "description": "Industry leading active noise cancellation with dual processors, 8 microphones, crystal clear hands-free calling, and 30-hour battery life.",
            "reviews_text": "Best noise cancelling headphones ever made. Super lightweight and comfortable for long flights. Crisp highs and punchy bass.",
        },
        {
            "product_id": 4,
            "name": "Bose QuietComfort 45 Bluetooth Wireless Headphones",
            "brand": "Bose",
            "categories": "Headphones, Audio, Electronics",
            "price": 329.00,
            "avg_rating": 4.5,
            "review_count": 420,
            "description": "Iconic quiet comfort with acoustic noise cancelling technology, Aware mode for environmental sound, lightweight materials, and 24-hour battery.",
            "reviews_text": "Supreme comfort and plush ear cushions. Excellent sound balance and very easy Bluetooth pairing. Noise cancellation is top tier.",
        },
        {
            "product_id": 5,
            "name": "Apple iPad Air 5th Generation 64GB Space Gray",
            "brand": "Apple",
            "categories": "Tablets, Electronics, Computers",
            "price": 599.00,
            "avg_rating": 4.8,
            "review_count": 380,
            "description": "10.9-inch Liquid Retina display powered by the revolutionary Apple M1 chip with Touch ID and support for Apple Pencil 2.",
            "reviews_text": "Lightning fast tablet for drawing and reading books. M1 chip handles heavy editing effortlessly. Screen colors are gorgeous.",
        },
        {
            "product_id": 6,
            "name": "Amazon Kindle Paperwhite 16GB Waterproof E-Reader",
            "brand": "Amazon",
            "categories": "E-Readers, Electronics, Books",
            "price": 149.99,
            "avg_rating": 4.7,
            "review_count": 890,
            "description": "6.8-inch glare-free display with adjustable warm light, up to 10 weeks of battery life, and waterproof IPX8 design for digital reading.",
            "reviews_text": "The warm light feature is fantastic for bedtime reading. Waterproof so I read by the pool without worry. Battery lasts for weeks.",
        },
        {
            "product_id": 7,
            "name": "JBL Flip 6 Portable Waterproof Bluetooth Speaker",
            "brand": "JBL",
            "categories": "Speakers, Audio, Electronics",
            "price": 129.95,
            "avg_rating": 4.6,
            "review_count": 610,
            "description": "Bold audio sound with 2-way speaker system, IP67 waterproof and dustproof protection, and 12 hours of playtime on a single charge.",
            "reviews_text": "Surprisingly deep bass for its compact size. Took it camping and survived water splashes and sand. Great party speaker.",
        },
        {
            "product_id": 8,
            "name": "Sony SRS-XB43 Extra Bass Wireless Bluetooth Speaker",
            "brand": "Sony",
            "categories": "Speakers, Audio, Electronics",
            "price": 278.00,
            "avg_rating": 4.5,
            "review_count": 215,
            "description": "Heavy bass portable speaker with multi-color party lighting, X-Balanced Speaker Unit, IP67 rating, and 24-hour battery.",
            "reviews_text": "Thumping extra bass and vibrant party lights. Built like a tank. Sound fills large outdoor spaces with ease.",
        },
        {
            "product_id": 9,
            "name": "Samsung Galaxy Tab S8 11-inch Android Tablet",
            "brand": "Samsung",
            "categories": "Tablets, Electronics, Computers",
            "price": 699.99,
            "avg_rating": 4.5,
            "review_count": 180,
            "description": "Fast Snapdragon 8 Gen 1 processor, ultra wide front camera, 120Hz display, and bundled low-latency S Pen for productivity.",
            "reviews_text": "Love the bundled stylus. Great tablet for student notes and streaming movies. Samsung DeX mode feels like a real computer.",
        },
        {
            "product_id": 10,
            "name": "Apple MacBook Air 13.6-inch Laptop M2 Chip 256GB",
            "brand": "Apple",
            "categories": "Laptops, Computers, Electronics",
            "price": 1099.00,
            "avg_rating": 4.9,
            "review_count": 750,
            "description": "Strikingly thin design with fast M2 processor, 13.6-inch Liquid Retina display, MagSafe charging, and silent fanless operation.",
            "reviews_text": "Incredible battery life lasting 16 hours. The M2 processor is blazing fast for programming and video editing. Whisper quiet.",
        },
        {
            "product_id": 11,
            "name": "Dell XPS 13 Plus Intel Core i7 16GB RAM 512GB SSD",
            "brand": "Dell",
            "categories": "Laptops, Computers, Electronics",
            "price": 1399.00,
            "avg_rating": 4.3,
            "review_count": 160,
            "description": "Premium ultraportable Windows laptop with capacitive touch function row, borderless keyboard, zero-lattice design, and 4K display.",
            "reviews_text": "Sleek and futuristic design. Screen is vivid and crisp. Keyboard feels nice, though battery life is average under heavy load.",
        },
        {
            "product_id": 12,
            "name": "Apple AirPods Pro 2nd Gen Wireless Earbuds",
            "brand": "Apple",
            "categories": "Headphones, Audio, Electronics",
            "price": 249.00,
            "avg_rating": 4.7,
            "review_count": 920,
            "description": "H2 chip powered noise cancellation, Adaptive Transparency, personalized spatial audio with dynamic head tracking, and MagSafe case.",
            "reviews_text": "Transparency mode sounds 100% natural. Noise cancelling is a huge step up from gen 1. Seamless integration with iPhone and Mac.",
        },
        {
            "product_id": 13,
            "name": "Google Pixel 7 Pro 5G 128GB Obsidian",
            "brand": "Google",
            "categories": "Smartphones, Electronics, Mobile",
            "price": 899.00,
            "avg_rating": 4.4,
            "review_count": 310,
            "description": "Google Tensor G2 chip, triple rear camera system with 5x telephoto lens, Macro Focus, and pure Android experience with speedy updates.",
            "reviews_text": "Stock Android software is clean with zero bloat. Real Tone photos look stunning. Dictation and live translation work like magic.",
        },
        {
            "product_id": 14,
            "name": "Amazon Echo Dot 5th Gen Smart Speaker with Alexa",
            "brand": "Amazon",
            "categories": "Speakers, Smart Home, Audio",
            "price": 49.99,
            "avg_rating": 4.6,
            "review_count": 1200,
            "description": "Best sounding Echo Dot yet with vibrant sound, Alexa voice assistant, motion sensor, and smart home hub capabilities.",
            "reviews_text": "Significantly better bass than 4th gen. Alexa controls my smart lights flawlessly. Perfect bedside smart speaker.",
        },
        {
            "product_id": 15,
            "name": "Apple Watch Series 8 GPS 41mm Midnight Aluminum",
            "brand": "Apple",
            "categories": "Wearables, Electronics, Smartwatches",
            "price": 399.00,
            "avg_rating": 4.8,
            "review_count": 450,
            "description": "Advanced health sensors for ECG, blood oxygen, temperature sensing, Crash Detection, and comprehensive fitness tracking.",
            "reviews_text": "Comprehensive fitness tracking and sleep analysis. Crash detection provides peace of mind. Daily battery lasts comfortably.",
        },
    ]
    df = pd.DataFrame(sample_records)

    def _create_doc(row: pd.Series) -> str:
        parts = [
            str(row["name"]),
            str(row["brand"]),
            str(row["categories"]),
            str(row["description"]),
            str(row["reviews_text"]),
        ]
        return " ".join(p for p in parts if p.strip())

    df["document"] = df.apply(_create_doc, axis=1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# DATA & INDEX CACHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset() -> Tuple[pd.DataFrame, bool]:
    """
    Load dataset from processed CSV if available; otherwise fallback to sample dataset.
    Returns (DataFrame, is_sample_data).
    """
    processed_csv = os.path.join(config.PROCESSED_DATA_DIR, "products.csv")
    if os.path.exists(processed_csv):
        try:
            df = pd.read_csv(processed_csv)
            if not df.empty and "document" in df.columns and "product_id" in df.columns:
                return df, False
        except Exception as err:
            st.warning(f"Could not load processed dataset: {err}. Using sample data.")

    return get_sample_products(), True


@st.cache_resource
def load_or_build_indices(df: pd.DataFrame):
    """
    Load precomputed indices or build them in memory.
    Cached as a Streamlit resource for fast persistent access.
    """
    product_ids = df["product_id"].astype(str).tolist()
    documents = df["document"].tolist()
    preprocessor = TextPreprocessor()

    # Preprocessed string documents for TF-IDF
    processed_docs = [preprocessor.process_to_string(doc) for doc in documents]

    # Inverted Index & TFIDF
    tfidf_path = os.path.join(config.MODEL_DIR, "tfidf_index.pkl")
    inv_index = InvertedIndex()
    if os.path.exists(tfidf_path):
        try:
            inv_index = InvertedIndex.load(tfidf_path)
        except Exception:
            inv_index.build(processed_docs)
    else:
        inv_index.build(processed_docs)
    tfidf_ret = TFIDFRetriever(inv_index, product_ids)

    # BM25 Retriever
    corpus_tokens_path = os.path.join(config.MODEL_DIR, "corpus_tokens.pkl")
    if os.path.exists(corpus_tokens_path):
        try:
            with open(corpus_tokens_path, "rb") as f:
                corpus_tokens = pickle.load(f)
        except Exception:
            corpus_tokens = [preprocessor.process(doc) for doc in documents]
    else:
        corpus_tokens = [preprocessor.process(doc) for doc in documents]
    bm25_ret = BM25Retriever(corpus_tokens, product_ids)

    # Dense Semantic Retriever
    emb_path = os.path.join(config.MODEL_DIR, "embedding_index.npz")
    semantic_ret = None
    emb_index = None
    try:
        emb_index = EmbeddingIndex(model_name=config.EMBEDDING_MODEL)
        if os.path.exists(emb_path):
            try:
                emb_index = EmbeddingIndex.load(emb_path)
            except Exception:
                emb_index.build(documents)
        else:
            emb_index.build(documents)
        semantic_ret = SemanticRetriever(emb_index, product_ids)
    except Exception as exc:
        # Fallback if sentence-transformers / PyTorch / CUDA cannot load
        emb_index = None
        semantic_ret = None

    # Query Expander
    expander = QueryExpander(bm25_retriever=bm25_ret)

    # Hybrid Ranker
    active_retrievers = {"tfidf": tfidf_ret, "bm25": bm25_ret}
    if semantic_ret is not None:
        active_retrievers["semantic"] = semantic_ret
    hybrid = HybridRanker(retrievers=active_retrievers)

    # Content-Based Recommender
    cb_recommender = ContentBasedRecommender()
    cb_recommender.build_profile(df)

    # Collaborative Filtering Recommender
    collab_recommender = CollaborativeRecommender()
    ratings_records = []
    for _, row in df.iterrows():
        rating_val = row.get("avg_rating")
        pid_val = row["product_id"]
        if pd.notna(rating_val) and float(rating_val) > 0:
            # Generate deterministic synthetic interactions for collaborative evaluation
            for u in range(1, 4):
                user_name = f"user_{((pid_val * 7 + u) % 15) + 1}"
                adj_rating = min(5.0, max(1.0, float(rating_val) + ((u - 2) * 0.5)))
                ratings_records.append(
                    {"user_id": user_name, "product_id": pid_val, "rating": adj_rating}
                )

    if ratings_records:
        try:
            ratings_df = pd.DataFrame(ratings_records)
            collab_recommender.build(ratings_df)
        except Exception:
            collab_recommender = None
    else:
        collab_recommender = None

    return {
        "preprocessor": preprocessor,
        "tfidf": tfidf_ret,
        "bm25": bm25_ret,
        "semantic": semantic_ret,
        "expander": expander,
        "hybrid": hybrid,
        "emb_index": emb_index,
        "content_based": cb_recommender,
        "collaborative": collab_recommender,
    }


def spell_check(query: str) -> Tuple[str, bool]:
    """Perform basic spelling correction via TextBlob."""
    try:
        from textblob import TextBlob

        corrected = str(TextBlob(query).correct())
        if corrected.strip().lower() != query.strip().lower():
            return corrected, True
    except Exception:
        pass
    return query, False


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SEARCH
# ─────────────────────────────────────────────────────────────────────────────
def render_search_page(df: pd.DataFrame, components: Dict[str, Any]) -> None:
    """Render the Interactive Search interface."""
    st.markdown('<div class="main-title">🔍 Product Search</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Explore retrieval methods, query expansion, and multi-facet filtering.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        query_input = st.text_input(
            "Search Query",
            placeholder="e.g. noise cancelling wireless headphones, tablet for reading",
            key="search_query_input",
        )
    with col2:
        method_options = ["Hybrid", "BM25", "TF-IDF", "All"]
        if components.get("semantic") is not None:
            method_options.insert(2, "Semantic")
        selected_method = st.selectbox("Retrieval Method", method_options, index=0)

    # Search Options Row
    col_opt1, col_opt2, col_opt3 = st.columns([1, 1, 1])
    with col_opt1:
        top_k = st.slider("Results Count (Top-K)", min_value=1, max_value=50, value=10)
    with col_opt2:
        enable_spell = st.checkbox("Spell Correction", value=False)
    with col_opt3:
        enable_expansion = st.checkbox("Query Expansion", value=False)
        expansion_method = "wordnet"
        if enable_expansion:
            expansion_method = st.radio(
                "Expansion Type", ["WordNet", "PRF"], horizontal=True
            ).lower()

    # Facet Filters in an Expander
    with st.expander("🎯 Facet Filters (Brand, Category, Price, Rating)", expanded=False):
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)

        all_brands = ["All"] + sorted([str(b) for b in df["brand"].dropna().unique() if str(b).strip()])
        with fcol1:
            sel_brand = st.selectbox("Brand", all_brands)

        all_cats = ["All"]
        raw_cats = []
        for cat_str in df["categories"].dropna():
            for item in str(cat_str).split(","):
                item_clean = item.strip()
                if item_clean and item_clean not in raw_cats:
                    raw_cats.append(item_clean)
        all_cats += sorted(raw_cats)
        with fcol2:
            sel_category = st.selectbox("Category", all_cats)

        min_p = float(df["price"].min()) if "price" in df.columns and pd.notna(df["price"].min()) else 0.0
        max_p = float(df["price"].max()) if "price" in df.columns and pd.notna(df["price"].max()) else 2000.0
        with fcol3:
            price_range = st.slider("Price Range ($)", min_value=0.0, max_value=max(max_p, 100.0), value=(0.0, max(max_p, 100.0)))

        with fcol4:
            min_rating = st.slider("Minimum Rating", min_value=0.0, max_value=5.0, value=0.0, step=0.1)

    if not query_input.strip():
        st.info("💡 Type a query above to search through the product catalogue.")
        return

    # Process Query
    effective_query = query_input.strip()
    if enable_spell:
        corrected_q, has_changed = spell_check(effective_query)
        if has_changed:
            st.info(f"✨ Auto-corrected query: **{corrected_q}** (original: *{effective_query}*)")
            effective_query = corrected_q

    if enable_expansion:
        expander: QueryExpander = components["expander"]
        try:
            expanded_q = expander.expand(effective_query, method=expansion_method)
            st.caption(f"Expanded Query: `{expanded_q}`")
            effective_query = expanded_q
        except Exception as err:
            st.warning(f"Query expansion note: {err}")

    # Retrieval Execution
    retrievers = {
        "TF-IDF": components["tfidf"],
        "BM25": components["bm25"],
    }
    if components.get("semantic") is not None:
        retrievers["Semantic"] = components["semantic"]

    # Filter helper
    def matches_facets(row: pd.Series) -> bool:
        if sel_brand != "All" and str(row.get("brand", "")).lower() != sel_brand.lower():
            return False
        if sel_category != "All" and sel_category.lower() not in str(row.get("categories", "")).lower():
            return False
        price_val = row.get("price")
        if pd.notna(price_val):
            try:
                p_flt = float(price_val)
                if p_flt < price_range[0] or p_flt > price_range[1]:
                    return False
            except ValueError:
                pass
        rating_val = row.get("avg_rating")
        if pd.notna(rating_val):
            try:
                if float(rating_val) < min_rating:
                    return False
            except ValueError:
                pass
        return True

    if selected_method == "All":
        st.subheader("📊 Comparative Search Results Across Methods")
        all_rankings = {}
        for name, ret in retrievers.items():
            all_rankings[name] = ret.retrieve(effective_query, top_k=top_k * 2)
        all_rankings["Hybrid"] = components["hybrid"].retrieve(effective_query, top_k=top_k * 2)

        # Collect unique product IDs across methods
        all_pids = []
        for r_list in all_rankings.values():
            for pid, _ in r_list:
                if pid not in all_pids:
                    all_pids.append(pid)

        comp_rows = []
        for pid in all_pids:
            row_match = df[df["product_id"].astype(str) == str(pid)]
            if row_match.empty or not matches_facets(row_match.iloc[0]):
                continue
            item = row_match.iloc[0]
            scores_dict = {"ID": pid, "Name": str(item.get("name", ""))[:55], "Brand": item.get("brand", "")}
            for method_name, r_list in all_rankings.items():
                match_score = next((score for p, score in r_list if str(p) == str(pid)), None)
                scores_dict[method_name] = f"{match_score:.4f}" if match_score is not None else "-"
            comp_rows.append(scores_dict)

        if comp_rows:
            comp_df = pd.DataFrame(comp_rows).head(top_k)
            st.dataframe(comp_df, use_container_width=True)
        else:
            st.warning("No products matched the search query and facet filter criteria.")

    else:
        # Single Method Retrieval
        if selected_method == "Hybrid":
            results = components["hybrid"].retrieve(effective_query, top_k=top_k * 3)
        elif selected_method == "TF-IDF":
            results = components["tfidf"].retrieve(effective_query, top_k=top_k * 3)
        elif selected_method == "BM25":
            results = components["bm25"].retrieve(effective_query, top_k=top_k * 3)
        elif selected_method == "Semantic":
            results = components["semantic"].retrieve(effective_query, top_k=top_k * 3)
        else:
            results = []

        # Filter by facets and limit to top_k
        filtered_results = []
        for pid, score in results:
            match = df[df["product_id"].astype(str) == str(pid)]
            if not match.empty and matches_facets(match.iloc[0]):
                filtered_results.append((match.iloc[0], score))
                if len(filtered_results) >= top_k:
                    break

        st.markdown(f"**Found {len(filtered_results)} matching products** (Method: `{selected_method}`)")

        if not filtered_results:
            st.warning("No products matched the given query and facet filters.")
            return

        for rank, (item, score) in enumerate(filtered_results, 1):
            name = item.get("name", "Product")
            brand = item.get("brand", "Unknown")
            categories = item.get("categories", "General")
            price = f"${float(item['price']):.2f}" if pd.notna(item.get("price")) else "N/A"
            rating = f"⭐ {float(item['avg_rating']):.1f}" if pd.notna(item.get("avg_rating")) else "No rating"
            rev_count = int(item.get("review_count", 0)) if pd.notna(item.get("review_count")) else 0
            desc = str(item.get("description", ""))

            with st.container():
                st.markdown(
                    f"""
                    <div class="product-card">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                            <h4 style="margin:0; color:#1E3A8A;">#{rank}. {name}</h4>
                            <span class="badge badge-score">Relevance: {score:.4f}</span>
                        </div>
                        <div style="margin-top:0.4rem; margin-bottom:0.6rem;">
                            <span class="badge badge-brand">🏷️ {brand}</span>
                            <span class="badge badge-category">📁 {categories}</span>
                            <span class="badge badge-price">💰 {price}</span>
                            <span class="badge" style="background-color:#FEF3C7; color:#92400E;">{rating} ({rev_count} reviews)</span>
                        </div>
                        <p style="color:#4B5563; font-size:0.92rem; margin:0;">{desc}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
def render_recommendations_page(df: pd.DataFrame, components: Dict[str, Any]) -> None:
    """Render Content-Based, Collaborative, and Personalized Recommendations."""
    st.markdown('<div class="main-title">🎯 Product Recommendations</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Explore Content-Based Similarity, Matrix Factorization (SVD), and Personalization Boosts.</div>',
        unsafe_allow_html=True,
    )

    cb_recommender: ContentBasedRecommender = components["content_based"]
    collab_recommender: Optional[CollaborativeRecommender] = components["collaborative"]

    # Select Product
    product_options = {
        f"[{row['product_id']}] {str(row['name'])[:60]}": row["product_id"]
        for _, row in df.iterrows()
    }

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_label = st.selectbox("Select Target Product", list(product_options.keys()))
        selected_pid = product_options[selected_label]
    with col2:
        top_n = st.slider("Number of Recommendations", min_value=1, max_value=20, value=5)

    # Show Target Product Details
    target_row = df[df["product_id"] == selected_pid].iloc[0]
    with st.expander("📌 Selected Product Details", expanded=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Brand", str(target_row.get("brand", "N/A")))
        m2.metric("Category", str(target_row.get("categories", "N/A"))[:20])
        price_val = f"${float(target_row['price']):.2f}" if pd.notna(target_row.get("price")) else "N/A"
        m3.metric("Price", price_val)
        rating_val = f"⭐ {float(target_row['avg_rating']):.1f}" if pd.notna(target_row.get("avg_rating")) else "N/A"
        m4.metric("Rating", rating_val)
        st.write(f"**Description:** {target_row.get('description', 'No description available.')}")

    # Personalization Toggle
    st.markdown("---")
    st.subheader("⚙️ Recommendation Settings & Personalization")
    enable_pers = st.toggle("Enable User History Personalization Engine", value=False)

    pers_engine = PersonalizationEngine()
    simulated_user_id = f"user_{selected_pid % 10 + 1}"

    if enable_pers:
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.info(f"Simulating Profile for Active User: `{simulated_user_id}`")
            pref_brand = st.selectbox(
                "Simulated Brand Affinity",
                sorted([str(b) for b in df["brand"].dropna().unique()]),
            )
        with pcol2:
            pref_cat = st.selectbox(
                "Simulated Category Affinity",
                sorted([str(c).split(",")[0].strip() for c in df["categories"].dropna().unique()]),
            )

        # Seed personalization engine
        pers_engine.update_history(
            simulated_user_id,
            str(selected_pid),
            action="view",
            metadata={"brand": pref_brand, "category": pref_cat, "rating": 5.0},
        )
        pers_engine.update_history(
            simulated_user_id,
            str(selected_pid),
            action="rate",
            metadata={"brand": pref_brand, "category": pref_cat, "rating": 5.0},
        )

    # Compute Recommendations
    recs_col1, recs_col2 = st.columns(2)

    with recs_col1:
        st.subheader("📄 Content-Based Recommendations")
        st.caption("Matches features (name, category, brand, description) using TF-IDF & Cosine Similarity.")
        cb_recs = cb_recommender.recommend(selected_pid, top_n=top_n * 2)

        if enable_pers and cb_recs:
            cb_recs = pers_engine.personalize(simulated_user_id, cb_recs, df)[:top_n]
        else:
            cb_recs = cb_recs[:top_n]

        if cb_recs:
            cb_data = []
            for r_pid, sim_score in cb_recs:
                match = df[df["product_id"] == r_pid]
                if not match.empty:
                    m_row = match.iloc[0]
                    cb_data.append(
                        {
                            "ID": r_pid,
                            "Name": str(m_row.get("name", ""))[:45],
                            "Brand": m_row.get("brand", ""),
                            "Price": f"${float(m_row['price']):.2f}" if pd.notna(m_row.get("price")) else "N/A",
                            "Score": f"{sim_score:.4f}",
                        }
                    )
            st.dataframe(pd.DataFrame(cb_data), use_container_width=True)
        else:
            st.info("No content-based recommendations found.")

    with recs_col2:
        st.subheader("👥 Collaborative Filtering (SVD)")
        st.caption("Predicts unseen preferences via TruncatedSVD Matrix Factorization.")
        if collab_recommender is not None:
            user_key = f"user_{selected_pid % 15 + 1}"
            collab_recs = collab_recommender.recommend(user_key, top_n=top_n * 2)

            if enable_pers and collab_recs:
                collab_recs = pers_engine.personalize(simulated_user_id, collab_recs, df)[:top_n]
            else:
                collab_recs = collab_recs[:top_n]

            if collab_recs:
                cf_data = []
                for r_pid, pred_score in collab_recs:
                    match = df[df["product_id"] == r_pid]
                    if not match.empty:
                        m_row = match.iloc[0]
                        cf_data.append(
                            {
                                "ID": r_pid,
                                "Name": str(m_row.get("name", ""))[:45],
                                "Brand": m_row.get("brand", ""),
                                "Price": f"${float(m_row['price']):.2f}" if pd.notna(m_row.get("price")) else "N/A",
                                "Pred Rating": f"{pred_score:.3f}",
                            }
                        )
                st.dataframe(pd.DataFrame(cf_data), use_container_width=True)
            else:
                st.info(f"No unseen product recommendations for `{user_key}`.")
        else:
            st.warning("Collaborative recommender could not be built with current ratings.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def render_analysis_page(df: pd.DataFrame, components: Dict[str, Any]) -> None:
    """Render Review Sentiment Analysis, Attribute Extraction, and Duplicate Detection."""
    st.markdown('<div class="main-title">📊 Review Analysis & Duplicate Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Extract customer sentiments, key aspects, summaries, and near-duplicate product clusters.</div>',
        unsafe_allow_html=True,
    )

    tab_review, tab_dupes = st.tabs(["💬 Review Sentiment & Attributes", "🔍 Duplicate Product Detection"])

    # --- TAB 1: REVIEW ANALYSIS ---
    with tab_review:
        analyzer = ReviewAnalyzer()
        prod_options = {
            f"[{row['product_id']}] {str(row['name'])[:60]}": row["product_id"]
            for _, row in df.iterrows()
            if pd.notna(row.get("reviews_text")) and len(str(row.get("reviews_text")).strip()) > 5
        }

        if not prod_options:
            st.warning("No products with text reviews available for analysis.")
            return

        selected_label = st.selectbox("Select Product to Analyze", list(prod_options.keys()))
        selected_pid = prod_options[selected_label]
        target_row = df[df["product_id"] == selected_pid].iloc[0]

        reviews_raw = str(target_row.get("reviews_text", ""))
        reviews_list = [r.strip() for r in reviews_raw.split(". ") if len(r.strip()) > 5]
        if not reviews_list:
            reviews_list = [reviews_raw]

        analysis = analyzer.full_analysis(str(target_row.get("name", "")), reviews_list)
        sent_data = analysis["sentiment"]

        # Top Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        avg_pol = sent_data.get("avg_polarity", 0.0)
        pol_label = "Positive" if avg_pol > 0.1 else ("Negative" if avg_pol < -0.1 else "Neutral")
        col1.metric("Average Polarity", f"{avg_pol:.3f}", delta=pol_label)
        dist = sent_data.get("distribution", {"positive": 0, "neutral": 0, "negative": 0})
        col2.metric("Positive Reviews", dist.get("positive", 0))
        col3.metric("Neutral Reviews", dist.get("neutral", 0))
        col4.metric("Negative Reviews", dist.get("negative", 0))

        # Sentiment Chart & Attributes
        c_left, c_right = st.columns([1, 1])

        with c_left:
            st.subheader("Sentiment Distribution")
            fig, ax = plt.subplots(figsize=(5, 4))
            labels = ["Positive", "Neutral", "Negative"]
            counts = [dist.get("positive", 0), dist.get("neutral", 0), dist.get("negative", 0)]
            colors = ["#10B981", "#9CA3AF", "#EF4444"]

            if sum(counts) > 0:
                ax.pie(
                    counts,
                    labels=labels,
                    autopct="%1.1f%%",
                    colors=colors,
                    startangle=140,
                    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
                )
                ax.axis("equal")
            else:
                ax.text(0.5, 0.5, "No review text available", ha="center")
            st.pyplot(fig)
            plt.close(fig)

        with c_right:
            st.subheader("🏷️ Extracted Key Attributes")
            attributes = analysis.get("attributes", [])
            if attributes:
                st.markdown(
                    "".join(
                        [
                            f'<span class="badge" style="background-color:#E0E7FF; color:#3730A3; margin-bottom:0.4rem;">{attr}</span> '
                            for attr in attributes
                        ]
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.write("No distinct noun-phrase attributes extracted.")

            st.subheader("📝 Review Summary Sentences")
            summary_sentences = analysis.get("summary", [])
            if summary_sentences:
                for sent in summary_sentences:
                    st.info(f"“{sent}”")
            else:
                st.write("No summary sentences available.")

    # --- TAB 2: DUPLICATE DETECTION ---
    with tab_dupes:
        st.subheader("Near-Duplicate Product Clustering")
        st.write(
            "Detect duplicate and near-duplicate catalog listings using **MinHash Locality-Sensitive Hashing (LSH)** "
            "and dense semantic embedding cosine similarity."
        )

        thresh = st.slider("Similarity Threshold", min_value=0.5, max_value=0.99, value=0.75, step=0.05)
        run_detection = st.button("🚀 Run Duplicate Detection on Catalog", type="primary")

        if run_detection:
            detector = DuplicateDetector(similarity_threshold=thresh)
            docs = df["document"].tolist()
            pids = df["product_id"].astype(str).tolist()

            with st.spinner("Executing MinHash LSH and pairwise comparisons..."):
                lsh_clusters = detector.find_duplicates_lsh(docs, pids)

            st.metric("Detected Duplicate Clusters", len(lsh_clusters))

            if lsh_clusters:
                for i, cluster in enumerate(lsh_clusters, 1):
                    with st.expander(f"Cluster #{i} ({len(cluster)} products)", expanded=True):
                        for pid in cluster:
                            match = df[df["product_id"].astype(str) == str(pid)]
                            if not match.empty:
                                p_row = match.iloc[0]
                                st.markdown(
                                    f"- **[{pid}] {p_row['name']}** — *{p_row.get('brand', 'Unknown')}* (`${p_row.get('price', 'N/A')}`)"
                                )
            else:
                st.success("No duplicate or near-duplicate listings found at the selected threshold.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def render_evaluation_page(df: pd.DataFrame, components: Dict[str, Any]) -> None:
    """Render Information Retrieval Evaluation Benchmark Dashboard."""
    st.markdown('<div class="main-title">📈 IR Evaluation Benchmark</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Benchmark TF-IDF, BM25, Semantic, and Hybrid RRF retrievers across standard IR metrics.</div>',
        unsafe_allow_html=True,
    )

    eval_k = st.slider("Evaluation K (Cut-off rank)", min_value=3, max_value=20, value=5)
    run_eval_btn = st.button("⚡ Run Full Evaluation Benchmark", type="primary")

    if not run_eval_btn and "eval_results" not in st.session_state:
        st.info("Click the button above to run the IR evaluation pipeline on synthetic/catalog ground truth queries.")
        return

    if run_eval_btn:
        with st.spinner("Generating test queries and computing IR metrics..."):
            # Generate test queries from categories or brands
            test_queries = []
            categories = df["categories"].dropna().unique()
            for cat in categories[:8]:
                cat_clean = str(cat).split(",")[0].strip()
                relevant = set(
                    df[df["categories"].str.contains(cat_clean, case=False, na=False)][
                        "product_id"
                    ]
                    .astype(str)
                    .tolist()
                )
                if len(relevant) >= 2:
                    test_queries.append({"query": cat_clean, "relevant_ids": relevant})

            if len(test_queries) < 3:
                # Fallback to brand queries
                for brand in df["brand"].dropna().unique()[:5]:
                    rel = set(
                        df[df["brand"] == brand]["product_id"].astype(str).tolist()
                    )
                    if len(rel) >= 2:
                        test_queries.append({"query": str(brand), "relevant_ids": rel})

            retrievers_to_eval = {
                "TF-IDF": components["tfidf"],
                "BM25": components["bm25"],
            }
            if components.get("semantic") is not None:
                retrievers_to_eval["Semantic"] = components["semantic"]
            retrievers_to_eval["Hybrid (RRF)"] = components["hybrid"]

            all_results = {}
            detailed_queries = []

            for ret_name, retriever in retrievers_to_eval.items():
                all_retrieved = []
                all_relevant = []

                for tq in test_queries:
                    res = retriever.retrieve(tq["query"], top_k=eval_k)
                    ret_ids = [str(pid) for pid, _ in res]
                    all_retrieved.append(ret_ids)
                    all_relevant.append(tq["relevant_ids"])

                    if ret_name == "Hybrid (RRF)":
                        detailed_queries.append(
                            {
                                "Query": tq["query"],
                                "Retrieved": ", ".join(ret_ids),
                                "Relevant Count": len(tq["relevant_ids"]),
                                f"P@{eval_k}": f"{precision_at_k(ret_ids, tq['relevant_ids'], eval_k):.3f}",
                                f"R@{eval_k}": f"{recall_at_k(ret_ids, tq['relevant_ids'], eval_k):.3f}",
                                "AP": f"{average_precision(ret_ids, tq['relevant_ids']):.3f}",
                            }
                        )

                p = np.mean([precision_at_k(r, rel, eval_k) for r, rel in zip(all_retrieved, all_relevant)])
                r = np.mean([recall_at_k(r, rel, eval_k) for r, rel in zip(all_retrieved, all_relevant)])
                f1 = np.mean([f1_at_k(r, rel, eval_k) for r, rel in zip(all_retrieved, all_relevant)])
                m_ap = mean_average_precision(all_retrieved, all_relevant)
                mrr = mean_reciprocal_rank(all_retrieved, all_relevant)
                ndcg = np.mean([ndcg_at_k(r, rel, eval_k) for r, rel in zip(all_retrieved, all_relevant)])

                all_results[ret_name] = {
                    f"Precision@{eval_k}": p,
                    f"Recall@{eval_k}": r,
                    f"F1@{eval_k}": f1,
                    "MAP": m_ap,
                    "MRR": mrr,
                    f"NDCG@{eval_k}": ndcg,
                }

            results_df = pd.DataFrame(all_results).T
            st.session_state["eval_results"] = results_df
            st.session_state["eval_queries"] = pd.DataFrame(detailed_queries)

    results_df = st.session_state["eval_results"]
    detailed_df = st.session_state["eval_queries"]

    # Styled DataFrame Table
    st.subheader("📋 Overall Retrieval Performance Comparison")
    st.dataframe(
        results_df.style.highlight_max(axis=0, color="#D1FAE5").format("{:.4f}"),
        use_container_width=True,
    )

    # Bar Chart Comparison
    st.subheader("📊 Visual Metrics Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    results_df.T.plot(kind="bar", ax=ax, colormap="viridis", width=0.8)
    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.set_title(f"Evaluation Metrics Comparison (Cut-off K={eval_k})")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Per-Query Breakdown
    st.subheader("🔍 Per-Query Breakdown (Hybrid Ranker)")
    st.dataframe(detailed_df, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────────────────────────────────────
def render_about_page() -> None:
    """Render the About & System Architecture page."""
    st.markdown('<div class="main-title">ℹ️ About the System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Intelligent E-Commerce Information Retrieval and Recommendation Engine.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(
            """
            ### 📌 Project Overview
            This application is an end-to-end e-commerce search, ranking, recommendation, and catalog analysis platform.
            It demonstrates key Information Retrieval (IR) paradigms:
            - **Lexical Matching**: Sparse TF-IDF inverted index and Okapi BM25 ranking.
            - **Dense Semantic Retrieval**: Transformer-based text embeddings (`all-MiniLM-L6-v2`).
            - **Hybrid Rank Fusion**: Reciprocal Rank Fusion (RRF) combining sparse and dense ranking scores.
            - **Query Understanding**: Spell checking via TextBlob and query expansion using WordNet and PRF.
            - **Catalog Analytics**: Review sentiment extraction, aspect extraction, and duplicate detection with MinHash LSH.
            """
        )

        st.markdown(
            """
            ### 🛠️ Technologies & Libraries
            - **UI & Presentation**: Streamlit, Matplotlib, Seaborn
            - **Natural Language Processing**: NLTK (Tokenization, WordNet, Lemmatization), TextBlob
            - **Information Retrieval**: Rank-BM25, Scikit-Learn (TF-IDF vectorizer, SVD, Cosine Similarity)
            - **Deep Learning**: HuggingFace Sentence-Transformers, PyTorch
            - **Near-Duplicate Detection**: DataSketch (MinHash & MinHash LSH)
            """
        )

    with col2:
        st.markdown(
            """
            ### 🏗️ Architecture Pipeline
            """
        )
        st.markdown(
            """
```mermaid
flowchart TD
    A["Raw Product Catalog & Reviews"] --> B["Text Preprocessor"]
    B --> C["Index Builder"]
    C --> D["TF-IDF Inverted Index"]
    C --> E["Okapi BM25 Index"]
    C --> F["Sentence Embeddings Index"]
    G["User Search Query"] --> H["Spell Check & Query Expansion"]
    H --> I["Multi-Method Retrievers"]
    D --> I
    E --> I
    F --> I
    I --> J["Reciprocal Rank Fusion (RRF)"]
    J --> K["Ranked Results & Facet Filtering"]
```
            """
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION ROUTING
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """Application entry point and routing coordinator."""
    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/clouds/200/shopping-cart.png", width=110)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Search", "Recommendations", "Analysis", "Evaluation", "About"],
        index=0,
    )

    # Load dataset & indices
    with st.spinner("Initializing datasets and indexing engines..."):
        df, is_sample = load_dataset()
        components = load_or_build_indices(df)

    if is_sample:
        st.sidebar.info(
            "ℹ️ Using built-in sample catalog. Preprocess raw data with `python main.py preprocess && python main.py index` for full dataset."
        )

    st.sidebar.markdown("---")
    st.sidebar.caption("E-Commerce IR & Recommender Engine v1.0")

    # Routing
    if page == "Search":
        render_search_page(df, components)
    elif page == "Recommendations":
        render_recommendations_page(df, components)
    elif page == "Analysis":
        render_analysis_page(df, components)
    elif page == "Evaluation":
        render_evaluation_page(df, components)
    elif page == "About":
        render_about_page()


if __name__ == "__main__":
    main()
