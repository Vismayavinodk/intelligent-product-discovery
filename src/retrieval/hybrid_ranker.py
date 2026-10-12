from typing import List, Tuple, Dict, Any, Optional, Union
from src.config import DEFAULT_TOP_K, RRF_K
from collections import defaultdict

class HybridRanker:
    """
    Ranks documents using an ensemble of retrievers via Reciprocal Rank Fusion (RRF)
    or score-normalized Weighted Fusion.
    """

    def __init__(
        self,
        retrievers: Dict[str, Any],
        rrf_k: int = RRF_K,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initializes the hybrid ranker.

        Args:
            retrievers: Dictionary mapping retriever names to retriever instances (e.g., TFIDF, BM25, Semantic).
            rrf_k: Constant used in RRF scoring.
            weights: Optional dictionary mapping retriever names to fusion weights.
        """
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.weights = weights if weights is not None else {name: 1.0 for name in retrievers.keys()}

    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Set default weights for weighted score fusion.

        Args:
            weights: Dictionary mapping retriever names to float weights.
        """
        self.weights = weights

    def _reciprocal_rank_fusion(self, rankings: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
        """
        Fuses multiple ranked lists using Reciprocal Rank Fusion.

        Args:
            rankings: List of ranked lists. Each ranked list is a list of (product_id, score) tuples.
            k: RRF constant.

        Returns:
            Fused ranked list of (product_id, rrf_score) tuples, sorted by rrf_score descending.
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        
        for ranked_list in rankings:
            for rank, (doc_id, _) in enumerate(ranked_list):
                # RRF score is 1 / (k + rank) where rank is 1-indexed
                rrf_scores[doc_id] += 1.0 / (k + rank + 1)
                
        # Sort by RRF score descending
        fused_ranking = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        return fused_ranking

    def weighted_fusion(
        self,
        rankings: Union[Dict[str, List[Tuple[str, float]]], List[List[Tuple[str, float]]]],
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Normalize each ranker's scores to [0, 1], then compute weighted sum.

        Args:
            rankings: Either a dictionary mapping retriever names to their ranked lists of (doc_id, score),
                      or a list of ranked lists.
            weights: Dictionary mapping retriever names to their respective weights.
                     Defaults to self.weights or equal weights (1.0).

        Returns:
            Fused ranked list of (product_id, score) tuples, sorted descending.
        """
        active_weights = weights if weights is not None else self.weights
        if active_weights is None:
            active_weights = {}

        # Standardize rankings to dict {name: ranked_list}
        if isinstance(rankings, dict):
            rankings_dict = rankings
        elif isinstance(rankings, (list, tuple)):
            rankings_dict = {}
            retriever_keys = list(self.retrievers.keys())
            for idx, r_list in enumerate(rankings):
                key = retriever_keys[idx] if idx < len(retriever_keys) else f"ranker_{idx}"
                rankings_dict[key] = r_list
        else:
            raise TypeError("Rankings must be a dict or a list of ranked lists.")

        combined_scores: Dict[str, float] = defaultdict(float)

        for name, ranked_list in rankings_dict.items():
            if not ranked_list:
                continue

            weight = float(active_weights.get(name, 1.0))
            if weight <= 0.0:
                continue

            scores = [score for _, score in ranked_list]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score

            for doc_id, score in ranked_list:
                if score_range > 1e-9:
                    norm_score = (score - min_score) / score_range
                else:
                    norm_score = 1.0 if max_score > 0 else 0.0

                combined_scores[doc_id] += weight * norm_score

        fused_ranking = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        return fused_ranking

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        methods: List[str] = None,
        fusion_method: str = "rrf",
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Retrieves top documents by fusing results from selected retrievers.

        Args:
            query: The search query.
            top_k: Number of final top results to return.
            methods: List of retriever names to use. If None, uses all available retrievers.
            fusion_method: 'rrf' for Reciprocal Rank Fusion, or 'weighted' for score-normalized weighted fusion.
            weights: Optional weights dict when fusion_method='weighted'.

        Returns:
            Fused list of (product_id, score) tuples.
        """
        if methods is None:
            methods = list(self.retrievers.keys())

        for method in methods:
            if method not in self.retrievers:
                raise ValueError(f"Retriever '{method}' not found.")

        rankings_dict = {}
        all_rankings = []
        for method in methods:
            ranking = self.retrievers[method].retrieve(query, top_k=top_k)
            rankings_dict[method] = ranking
            all_rankings.append(ranking)

        fusion_type = fusion_method.lower().strip()
        if fusion_type == "rrf":
            fused_results = self._reciprocal_rank_fusion(all_rankings, k=self.rrf_k)
        elif fusion_type == "weighted":
            fused_results = self.weighted_fusion(rankings_dict, weights=weights)
        else:
            raise ValueError(f"Unknown fusion method '{fusion_method}'. Supported methods: 'rrf', 'weighted'.")

        return fused_results[:top_k]

    def retrieve_with_details(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        fusion_method: str = "rrf",
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieves documents and returns results from each individual retriever plus the fused result.

        Args:
            query: The search query.
            top_k: Number of results to return.
            fusion_method: 'rrf' or 'weighted'.
            weights: Optional weights for weighted fusion.

        Returns:
            Dictionary containing individual rankings and the fused ranking.
        """
        details = {}
        all_rankings = []
        rankings_dict = {}

        for method_name, retriever in self.retrievers.items():
            ranking = retriever.retrieve(query, top_k=top_k)
            details[method_name] = ranking
            rankings_dict[method_name] = ranking
            all_rankings.append(ranking)

        fusion_type = fusion_method.lower().strip()
        if fusion_type == "rrf":
            fused_ranking = self._reciprocal_rank_fusion(all_rankings, k=self.rrf_k)[:top_k]
        elif fusion_type == "weighted":
            fused_ranking = self.weighted_fusion(rankings_dict, weights=weights)[:top_k]
        else:
            raise ValueError(f"Unknown fusion method '{fusion_method}'. Supported methods: 'rrf', 'weighted'.")

        details['fused'] = fused_ranking
        return details
