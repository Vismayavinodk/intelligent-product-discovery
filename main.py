"""
E-Commerce Information Retrieval & Recommendation System
=========================================================
CLI entry point for the full pipeline.

Usage:
    python main.py preprocess
    python main.py index
    python main.py search
    python main.py recommend
    python main.py evaluate
    python main.py demo
    python main.py app
    python main.py ui
"""

import argparse
import os
import sys
import pickle
import warnings

import pandas as pd
import numpy as np

# ── Add project root to path ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from src import config
from src.preprocessing.data_loader import (
    load_raw_data, merge_product_reviews, create_product_corpus,
    save_processed_data, load_processed_data,
)
from src.preprocessing.text_preprocessor import (
    preprocess, preprocess_to_string, TextPreprocessor,
)
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
    precision_at_k, recall_at_k, f1_at_k,
    average_precision, mean_average_precision,
    ndcg_at_k, reciprocal_rank, mean_reciprocal_rank,
    plot_comparison,
)

# ── Helpers ───────────────────────────────────────────────────────────
PROCESSED_CSV = os.path.join(config.PROCESSED_DATA_DIR, "products.csv")
TFIDF_INDEX_PATH = os.path.join(config.MODEL_DIR, "tfidf_index.pkl")
EMBEDDING_INDEX_PATH = os.path.join(config.MODEL_DIR, "embedding_index.npz")
BM25_PATH = os.path.join(config.MODEL_DIR, "bm25.pkl")
CORPUS_TOKENS_PATH = os.path.join(config.MODEL_DIR, "corpus_tokens.pkl")


def header(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def sub_header(text: str) -> None:
    print(f"\n--- {text} ---")


def load_products() -> pd.DataFrame:
    """Load processed products DataFrame, or raise."""
    if not os.path.exists(PROCESSED_CSV):
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_CSV}.\n"
            "Run  python main.py preprocess  first."
        )
    return load_processed_data(PROCESSED_CSV)


def load_all_indices(products_df: pd.DataFrame):
    """Load / reconstruct all retrieval components."""
    product_ids = products_df["product_id"].astype(str).tolist()
    documents = products_df["document"].tolist()

    # TF-IDF index
    inv_index = InvertedIndex()
    if os.path.exists(TFIDF_INDEX_PATH):
        inv_index = InvertedIndex.load(TFIDF_INDEX_PATH)
    else:
        raise FileNotFoundError("TF-IDF index not found. Run  python main.py index  first.")

    # Embedding index
    emb_index = EmbeddingIndex()
    if os.path.exists(EMBEDDING_INDEX_PATH):
        emb_index = EmbeddingIndex.load(EMBEDDING_INDEX_PATH)
    else:
        raise FileNotFoundError("Embedding index not found. Run  python main.py index  first.")

    # BM25 needs corpus tokens
    if os.path.exists(CORPUS_TOKENS_PATH):
        with open(CORPUS_TOKENS_PATH, "rb") as f:
            corpus_tokens = pickle.load(f)
    else:
        preprocessor = TextPreprocessor()
        corpus_tokens = [preprocessor.process(doc) for doc in documents]

    # Build retrievers
    tfidf_ret = TFIDFRetriever(inv_index, product_ids)
    bm25_ret = BM25Retriever(corpus_tokens, product_ids)
    semantic_ret = SemanticRetriever(emb_index, product_ids)

    # Query expander
    expander = QueryExpander(bm25_retriever=bm25_ret)

    # Hybrid ranker
    hybrid = HybridRanker(
        retrievers={"tfidf": tfidf_ret, "bm25": bm25_ret, "semantic": semantic_ret}
    )

    return tfidf_ret, bm25_ret, semantic_ret, expander, hybrid, emb_index


# ── Mode: preprocess ─────────────────────────────────────────────────
def run_preprocess(_args):
    header("PREPROCESSING DATA")
    config.ensure_directories_exist()

    # Find raw data
    raw_files = [f for f in os.listdir(config.RAW_DATA_DIR)
                 if f.endswith(".csv")]
    if not raw_files:
        print(f"[ERROR] No CSV files found in {config.RAW_DATA_DIR}")
        print("        Please download the dataset and place CSV file(s) there.")
        return

    raw_path = os.path.join(config.RAW_DATA_DIR, raw_files[0])
    print(f"[1/4] Loading raw data from {raw_path} ...")
    df = load_raw_data(raw_path)
    print(f"       Loaded {len(df)} rows, {len(df.columns)} columns")

    print("[2/4] Merging product reviews ...")
    df = merge_product_reviews(df)
    print(f"       {len(df)} unique products after merging")

    print("[3/4] Creating product corpus ...")
    products = create_product_corpus(df)
    print(f"       Corpus ready — {len(products)} products")

    print(f"[4/4] Saving processed data to {PROCESSED_CSV} ...")
    save_processed_data(products, PROCESSED_CSV)

    # Print sample
    print("\n  Sample products:")
    for _, row in products.head(3).iterrows():
        name = str(row.get("name", ""))[:60]
        brand = str(row.get("brand", ""))[:20]
        print(f"    [{row['product_id']}] {name}  |  brand={brand}")

    print("\n[DONE] Preprocessing complete.")


# ── Mode: index ──────────────────────────────────────────────────────
def run_index(_args):
    header("BUILDING INDICES")
    products = load_products()
    documents = products["document"].tolist()
    preprocessor = TextPreprocessor()

    # Preprocessed string documents for TF-IDF
    print("[1/4] Preprocessing documents ...")
    processed_docs = [preprocessor.process_to_string(doc) for doc in documents]

    # TF-IDF inverted index
    print("[2/4] Building TF-IDF inverted index ...")
    inv_index = InvertedIndex()
    inv_index.build(processed_docs)
    inv_index.save(TFIDF_INDEX_PATH)
    print(f"       Saved to {TFIDF_INDEX_PATH}")

    # Corpus tokens for BM25
    print("[3/4] Tokenizing corpus for BM25 ...")
    corpus_tokens = [preprocessor.process(doc) for doc in documents]
    with open(CORPUS_TOKENS_PATH, "wb") as f:
        pickle.dump(corpus_tokens, f)
    print(f"       Saved {len(corpus_tokens)} tokenized documents")

    # Embedding index
    print("[4/4] Building sentence-transformer embeddings ...")
    print(f"       Model: {config.EMBEDDING_MODEL}")
    emb_index = EmbeddingIndex(model_name=config.EMBEDDING_MODEL)
    emb_index.build(documents)
    emb_index.save(EMBEDDING_INDEX_PATH)
    print(f"       Saved to {EMBEDDING_INDEX_PATH}")

    print("\n[DONE] All indices built and saved.")


# ── Mode: search ─────────────────────────────────────────────────────
def run_search(_args):
    header("INTERACTIVE SEARCH")
    products = load_products()
    tfidf_ret, bm25_ret, semantic_ret, expander, hybrid, _ = load_all_indices(products)

    print("Type a natural-language query. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("quit", "exit"):
            break

        top_k = config.DEFAULT_TOP_K

        # Retrieve from each method
        tfidf_results = tfidf_ret.retrieve(query, top_k)
        bm25_results = bm25_ret.retrieve(query, top_k)
        semantic_results = semantic_ret.retrieve(query, top_k)
        hybrid_results = hybrid.retrieve(query, top_k)

        # Query expansion
        expanded_wn = expander.expand(query, method="wordnet")
        expanded_results = bm25_ret.retrieve(expanded_wn, top_k)

        def _name(pid):
            row = products[products["product_id"] == int(pid)]
            if len(row) > 0:
                return str(row.iloc[0]["name"])[:50]
            return str(pid)

        methods = [
            ("TF-IDF", tfidf_results),
            ("BM25", bm25_results),
            ("Semantic", semantic_results),
            ("Expanded (WordNet)", expanded_results),
            ("Hybrid (RRF)", hybrid_results),
        ]

        for method_name, results in methods:
            sub_header(method_name)
            for rank, (pid, score) in enumerate(results[:5], 1):
                print(f"  {rank}. [{pid}] {_name(pid)}  (score={score:.4f})")

        print()


# ── Mode: recommend ──────────────────────────────────────────────────
def run_recommend(_args):
    header("RECOMMENDATION ENGINE")
    products = load_products()

    # Content-based
    sub_header("Building content-based recommender")
    cb = ContentBasedRecommender()
    cb.build_profile(products)

    # Collaborative
    sub_header("Building collaborative recommender")
    collab = None
    if "reviews_text" in products.columns:
        # Build synthetic ratings if the dataset doesn't have per-user data
        # For demo, we use the aggregated data
        ratings_data = []
        for _, row in products.iterrows():
            if pd.notna(row.get("avg_rating")) and row.get("avg_rating", 0) > 0:
                ratings_data.append({
                    "user_id": f"user_{row['product_id'] % 50}",
                    "product_id": row["product_id"],
                    "rating": float(row["avg_rating"]),
                })
        if ratings_data:
            ratings_df = pd.DataFrame(ratings_data)
            collab = CollaborativeRecommender()
            try:
                collab.build(ratings_df)
                print(f"  Built from {len(ratings_df)} ratings")
            except Exception as e:
                print(f"  Collaborative filtering skipped: {e}")
                collab = None

    print("\nEnter a product ID to get recommendations. Type 'quit' to stop.\n")

    while True:
        try:
            inp = input("Product ID> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not inp or inp.lower() in ("quit", "exit"):
            break

        try:
            pid = int(inp)
        except ValueError:
            # Search by name
            matches = products[products["name"].str.contains(inp, case=False, na=False)]
            if len(matches) == 0:
                print("  No matching product found.")
                continue
            pid = matches.iloc[0]["product_id"]
            print(f"  Matched: [{pid}] {matches.iloc[0]['name'][:60]}")

        sub_header("Content-Based Recommendations")
        recs = cb.recommend(pid, top_n=5)
        for rank, (rec_pid, score) in enumerate(recs, 1):
            row = products[products["product_id"] == rec_pid]
            name = str(row.iloc[0]["name"])[:50] if len(row) > 0 else str(rec_pid)
            print(f"  {rank}. [{rec_pid}] {name}  (sim={score:.4f})")

        if collab:
            sub_header("Collaborative Recommendations")
            user_id = f"user_{pid % 50}"
            crecs = collab.recommend(user_id, top_n=5)
            for rank, (rec_pid, pred) in enumerate(crecs, 1):
                row = products[products["product_id"] == rec_pid]
                name = str(row.iloc[0]["name"])[:50] if len(row) > 0 else str(rec_pid)
                print(f"  {rank}. [{rec_pid}] {name}  (pred={pred:.4f})")

        print()


# ── Mode: evaluate ───────────────────────────────────────────────────
def run_evaluate(_args):
    header("EVALUATION")
    products = load_products()
    tfidf_ret, bm25_ret, semantic_ret, expander, hybrid, _ = load_all_indices(products)

    # Generate test queries from categories
    sub_header("Generating test queries")
    categories = products["categories"].dropna().unique()
    categories = [c for c in categories if isinstance(c, str) and len(c.strip()) > 2]

    if len(categories) == 0:
        print("  No categories available for test query generation.")
        print("  Using product names as queries instead.")
        sample = products.sample(min(20, len(products)))
        test_queries = []
        for _, row in sample.iterrows():
            name_words = str(row["name"]).split()[:3]
            query = " ".join(name_words)
            # Relevant = same brand products
            relevant = set(
                products[products["brand"] == row.get("brand", "")]["product_id"]
                .astype(str).tolist()
            )
            if relevant:
                test_queries.append({"query": query, "relevant_ids": relevant})
    else:
        np.random.seed(config.RANDOM_SEED)
        sampled = np.random.choice(categories, min(25, len(categories)), replace=False)
        test_queries = []
        for cat in sampled:
            relevant = set(
                products[products["categories"].str.contains(str(cat), case=False, na=False)]
                ["product_id"].astype(str).tolist()
            )
            if len(relevant) >= 2:
                test_queries.append({"query": str(cat), "relevant_ids": relevant})

    print(f"  Generated {len(test_queries)} test queries")

    # Evaluate each retriever
    retrievers = {
        "TF-IDF": tfidf_ret,
        "BM25": bm25_ret,
        "Semantic": semantic_ret,
        "Hybrid (RRF)": hybrid,
    }

    top_k = config.DEFAULT_TOP_K
    all_results = {}

    for name, retriever in retrievers.items():
        sub_header(f"Evaluating {name}")
        all_retrieved = []
        all_relevant = []

        for tq in test_queries:
            results = retriever.retrieve(tq["query"], top_k=top_k)
            retrieved_ids = [str(pid) for pid, _ in results]
            all_retrieved.append(retrieved_ids)
            all_relevant.append(tq["relevant_ids"])

        p = np.mean([precision_at_k(r, rel, top_k)
                      for r, rel in zip(all_retrieved, all_relevant)])
        r = np.mean([recall_at_k(r, rel, top_k)
                      for r, rel in zip(all_retrieved, all_relevant)])
        f1 = np.mean([f1_at_k(r, rel, top_k)
                       for r, rel in zip(all_retrieved, all_relevant)])
        m_ap = mean_average_precision(all_retrieved, all_relevant)
        mrr = mean_reciprocal_rank(all_retrieved, all_relevant)
        ndcg = np.mean([ndcg_at_k(r, rel, top_k)
                         for r, rel in zip(all_retrieved, all_relevant)])

        metrics = {
            f"Precision@{top_k}": p,
            f"Recall@{top_k}": r,
            f"F1@{top_k}": f1,
            "MAP": m_ap,
            "MRR": mrr,
            f"NDCG@{top_k}": ndcg,
        }
        all_results[name] = metrics

        for metric_name, value in metrics.items():
            print(f"  {metric_name:20s}: {value:.4f}")

    # Comparison table
    results_df = pd.DataFrame(all_results).T
    sub_header("Comparison Table")
    print(results_df.to_string(float_format="{:.4f}".format))

    # Save plot
    plot_path = os.path.join(config.DATA_DIR, "evaluation_comparison.png")
    try:
        plot_comparison(results_df, save_path=plot_path)
        print(f"\n  Plot saved to {plot_path}")
    except Exception as e:
        print(f"\n  Could not save plot: {e}")

    print("\n[DONE] Evaluation complete.")


# ── Mode: demo ───────────────────────────────────────────────────────
def run_demo(_args):
    header("FULL SYSTEM DEMO")
    products = load_products()
    tfidf_ret, bm25_ret, semantic_ret, expander, hybrid, emb_index = load_all_indices(products)

    # ── 1. Search Demo ────────────────────────────────────────────────
    demo_queries = [
        "wireless bluetooth headphones",
        "tablet for reading books",
        "best speaker for music",
    ]

    for query in demo_queries:
        sub_header(f'Search: "{query}"')

        results = hybrid.retrieve_with_details(query, top_k=5)

        for method, ranked in results.items():
            print(f"\n  [{method.upper()}]")
            for rank, (pid, score) in enumerate(ranked[:3], 1):
                row = products[products["product_id"] == int(pid)]
                name = str(row.iloc[0]["name"])[:50] if len(row) > 0 else pid
                print(f"    {rank}. {name}  ({score:.4f})")

    # ── 2. Query Expansion Demo ───────────────────────────────────────
    sub_header("Query Expansion Demo")
    original = "cheap laptop"
    expanded_wn = expander.expand(original, method="wordnet")
    print(f"  Original:  {original}")
    print(f"  Expanded (WordNet): {expanded_wn}")
    try:
        expanded_prf = expander.expand(original, method="prf")
        print(f"  Expanded (PRF):     {expanded_prf}")
    except Exception as e:
        print(f"  PRF expansion: {e}")

    # ── 3. Recommendation Demo ────────────────────────────────────────
    sub_header("Content-Based Recommendation Demo")
    cb = ContentBasedRecommender()
    cb.build_profile(products)

    sample_pid = products.iloc[0]["product_id"]
    sample_name = str(products.iloc[0]["name"])[:50]
    print(f"  Recommending similar to: [{sample_pid}] {sample_name}")
    recs = cb.recommend(sample_pid, top_n=5)
    for rank, (rec_pid, score) in enumerate(recs, 1):
        row = products[products["product_id"] == rec_pid]
        name = str(row.iloc[0]["name"])[:50] if len(row) > 0 else str(rec_pid)
        print(f"    {rank}. [{rec_pid}] {name}  (sim={score:.4f})")

    # ── 4. Review Analysis Demo ───────────────────────────────────────
    sub_header("Review Analysis Demo")
    analyzer = ReviewAnalyzer()
    sample = products[products["reviews_text"].str.len() > 50].head(1)
    if len(sample) > 0:
        row = sample.iloc[0]
        reviews = str(row["reviews_text"]).split(". ")[:10]
        analysis = analyzer.full_analysis(str(row["name"])[:50], reviews)
        print(f"  Product: {analysis['product_name']}")
        sent = analysis["sentiment"]
        print(f"  Avg Polarity: {sent['avg_polarity']:.3f}")
        print(f"  Distribution: {sent['distribution']}")
        if analysis["attributes"]:
            print(f"  Key Attributes: {', '.join(analysis['attributes'][:5])}")
    else:
        print("  No reviews available for analysis demo.")

    # ── 5. Duplicate Detection Demo ───────────────────────────────────
    sub_header("Duplicate Detection Demo")
    detector = DuplicateDetector()
    documents = products["document"].tolist()
    pids = products["product_id"].astype(str).tolist()

    lsh_dupes = detector.find_duplicates_lsh(documents[:200], pids[:200])
    print(f"  Scanned {min(200, len(documents))} products")
    print(f"  Found {len(lsh_dupes)} duplicate clusters (LSH)")
    for i, cluster in enumerate(lsh_dupes[:3]):
        print(f"    Cluster {i + 1}: {cluster}")

    print("\n[DONE] Demo complete.")


# ── Mode: app / ui ───────────────────────────────────────────────────
def run_app(_args):
    header("STARTING STREAMLIT WEB UI")
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    print(f"Executing: {' '.join(cmd)}\n")
    import subprocess
    try:
        subprocess.run(cmd)
    except FileNotFoundError:
        print("[ERROR] Streamlit is not installed or not found in PATH.")
        print("        Install with: pip install streamlit")
    except KeyboardInterrupt:
        print("\n[INFO] Streamlit app closed.")


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="E-Commerce IR & Recommendation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  preprocess   Load & preprocess raw CSV data
  index        Build TF-IDF, BM25, and embedding indices
  search       Interactive search with all retrieval methods
  recommend    Product recommendations (content-based & collaborative)
  evaluate     Run evaluation suite with all metrics
  demo         Full demo showcasing all features
  app / ui     Launch the Streamlit web application
        """,
    )
    parser.add_argument(
        "mode",
        choices=["preprocess", "index", "search", "recommend", "evaluate", "demo", "app", "ui"],
        help="Mode to run the system in.",
    )

    args = parser.parse_args()

    dispatch = {
        "preprocess": run_preprocess,
        "index": run_index,
        "search": run_search,
        "recommend": run_recommend,
        "evaluate": run_evaluate,
        "demo": run_demo,
        "app": run_app,
        "ui": run_app,
    }

    try:
        dispatch[args.mode](args)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
