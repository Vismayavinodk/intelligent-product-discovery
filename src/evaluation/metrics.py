import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Calculate Precision@k."""
    if not retrieved or not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_retrieved = set(retrieved_k).intersection(relevant)
    return len(relevant_retrieved) / min(k, len(retrieved_k))

def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Calculate Recall@k."""
    if not retrieved or not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_retrieved = set(retrieved_k).intersection(relevant)
    return len(relevant_retrieved) / len(relevant)

def f1_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Calculate F1-score@k."""
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)

def average_precision(retrieved: list, relevant: set) -> float:
    """Calculate Average Precision."""
    if not retrieved or not relevant:
        return 0.0
    ap = 0.0
    relevant_count = 0
    for i, item in enumerate(retrieved):
        if item in relevant:
            relevant_count += 1
            ap += relevant_count / (i + 1)
    return ap / len(relevant)

def mean_average_precision(all_retrieved: list[list], all_relevant: list[set]) -> float:
    """Calculate Mean Average Precision (MAP)."""
    if not all_retrieved or not all_relevant:
        return 0.0
    ap_sum = sum(average_precision(ret, rel) for ret, rel in zip(all_retrieved, all_relevant))
    return ap_sum / len(all_retrieved)

def dcg_at_k(scores: list[float], k: int) -> float:
    """Calculate Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, score in enumerate(scores[:k]):
        dcg += score / math.log2(i + 2)
    return dcg

def ndcg_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k (binary relevance)."""
    if not retrieved or not relevant:
        return 0.0
    # Binary relevance scores
    relevance_scores = [1.0 if item in relevant else 0.0 for item in retrieved]
    dcg = dcg_at_k(relevance_scores, k)
    
    # Ideal DCG
    ideal_scores = sorted([1.0] * min(len(relevant), k) + [0.0] * max(0, k - len(relevant)), reverse=True)
    idcg = dcg_at_k(ideal_scores, k)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg

def reciprocal_rank(retrieved: list, relevant: set) -> float:
    """Calculate Reciprocal Rank."""
    for i, item in enumerate(retrieved):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0

def mean_reciprocal_rank(all_retrieved: list[list], all_relevant: list[set]) -> float:
    """Calculate Mean Reciprocal Rank (MRR)."""
    if not all_retrieved or not all_relevant:
        return 0.0
    rr_sum = sum(reciprocal_rank(ret, rel) for ret, rel in zip(all_retrieved, all_relevant))
    return rr_sum / len(all_retrieved)

def evaluate_retriever(retriever, test_queries: list[dict], top_k: int = 10) -> dict:
    """
    Evaluate a retriever on a set of test queries.
    
    test_queries: list of dicts with 'query' (str) and 'relevant_ids' (set)
    """
    all_retrieved = []
    all_relevant = []
    
    for tq in test_queries:
        query = tq['query']
        relevant_ids = set(str(x) for x in tq.get('relevant_ids', []))
        
        # Support retrieve() as primary API, fallback to search()
        if hasattr(retriever, 'retrieve'):
            results = retriever.retrieve(query, top_k=top_k)
        elif hasattr(retriever, 'search'):
            results = retriever.search(query, top_k=top_k)
        else:
            raise AttributeError("Retriever must implement retrieve() or search()")
            
        retrieved_ids = []
        for res in results:
            if isinstance(res, tuple):
                retrieved_ids.append(str(res[0]))
            elif isinstance(res, dict):
                retrieved_ids.append(str(res.get('product_id', res.get('id', ''))))
            else:
                retrieved_ids.append(str(res))
        
        all_retrieved.append(retrieved_ids)
        all_relevant.append(relevant_ids)
    
    p_k = np.mean([precision_at_k(ret, rel, top_k) for ret, rel in zip(all_retrieved, all_relevant)])
    r_k = np.mean([recall_at_k(ret, rel, top_k) for ret, rel in zip(all_retrieved, all_relevant)])
    f1_k = np.mean([f1_at_k(ret, rel, top_k) for ret, rel in zip(all_retrieved, all_relevant)])
    map_score = mean_average_precision(all_retrieved, all_relevant)
    mrr_score = mean_reciprocal_rank(all_retrieved, all_relevant)
    ndcg_k = np.mean([ndcg_at_k(ret, rel, top_k) for ret, rel in zip(all_retrieved, all_relevant)])
    
    return {
        f'Precision@{top_k}': float(p_k),
        f'Recall@{top_k}': float(r_k),
        f'F1@{top_k}': float(f1_k),
        'MAP': float(map_score),
        'MRR': float(mrr_score),
        f'NDCG@{top_k}': float(ndcg_k)
    }

def compare_retrievers(retrievers: dict, test_queries: list[dict], top_k: int = 10) -> pd.DataFrame:
    """Evaluate multiple retrievers and return a comparison table."""
    results = {}
    for name, retriever in retrievers.items():
        print(f"Evaluating {name}...")
        results[name] = evaluate_retriever(retriever, test_queries, top_k=top_k)
    
    df = pd.DataFrame(results).T
    return df

def plot_comparison(results_df: pd.DataFrame, save_path: str = None):
    """Plot bar chart comparing metrics across retrievers."""
    df_t = results_df.T
    
    plt.figure(figsize=(12, 6))
    df_t.plot(kind='bar', figsize=(12, 6), colormap='viridis')
    plt.title('Retriever Comparison')
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.xticks(rotation=45)
    plt.legend(title='Retrievers', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def generate_test_queries(products_df: pd.DataFrame, n_queries: int = 25) -> list[dict]:
    """Auto-generate test queries from product categories or brands."""
    queries = []
    
    cat_col = 'categories' if 'categories' in products_df.columns else ('category' if 'category' in products_df.columns else None)
    id_col = 'product_id' if 'product_id' in products_df.columns else ('id' if 'id' in products_df.columns else None)
    brand_col = 'brand' if 'brand' in products_df.columns else None
    
    if id_col is None:
        raise ValueError("DataFrame must contain an 'id' or 'product_id' column.")
    
    if cat_col and cat_col in products_df.columns:
        categories = products_df[cat_col].dropna().unique()
        sampled_cats = np.random.choice(categories, min(n_queries, len(categories)), replace=False)
        for cat in sampled_cats:
            cat_str = str(cat)
            exact_match = products_df[products_df[cat_col] == cat]
            if len(exact_match) > 0:
                relevant = set(exact_match[id_col].astype(str).tolist())
            else:
                relevant = set(
                    products_df[products_df[cat_col].astype(str).str.contains(cat_str, case=False, na=False)][id_col].astype(str).tolist()
                )
            if relevant:
                queries.append({
                    'query': cat_str,
                    'relevant_ids': relevant
                })
            
    if len(queries) < n_queries and brand_col and brand_col in products_df.columns:
        brands = products_df[brand_col].dropna().unique()
        sampled_brands = np.random.choice(brands, min(n_queries - len(queries), len(brands)), replace=False)
        for brand in sampled_brands:
            brand_str = str(brand)
            relevant = set(products_df[products_df[brand_col] == brand][id_col].astype(str).tolist())
            if relevant:
                queries.append({
                    'query': brand_str,
                    'relevant_ids': relevant
                })
            
    return queries
