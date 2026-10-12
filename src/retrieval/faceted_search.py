"""
Faceted search module for e-commerce product filtering.

Provides multi-dimensional faceted filtering (brand, category, price range,
rating) over information retrieval result sets, along with facet aggregation counts.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class FacetedSearch:
    """
    Applies faceted filtering and aggregation over product search results.
    """

    def __init__(self, products_df: pd.DataFrame) -> None:
        """
        Initialize FacetedSearch with a product DataFrame.

        Args:
            products_df: DataFrame containing product metadata (product_id/id,
                         brand, categories/category, price, rating).
        """
        self.products_df = products_df.copy()

        # Identify ID column
        if "product_id" in self.products_df.columns:
            self.id_col = "product_id"
        elif "id" in self.products_df.columns:
            self.id_col = "id"
        else:
            raise ValueError("products_df must contain 'product_id' or 'id' column.")

        # Identify category column
        if "categories" in self.products_df.columns:
            self.cat_col = "categories"
        elif "category" in self.products_df.columns:
            self.cat_col = "category"
        else:
            self.cat_col = None

        # Identify brand column
        self.brand_col = "brand" if "brand" in self.products_df.columns else None

        # Identify price column
        self.price_col = "price" if "price" in self.products_df.columns else None
        if self.price_col:
            self.products_df[self.price_col] = pd.to_numeric(
                self.products_df[self.price_col], errors="coerce"
            )

        # Identify rating column
        if "rating" in self.products_df.columns:
            self.rating_col = "rating"
        elif "avg_rating" in self.products_df.columns:
            self.rating_col = "avg_rating"
        elif "reviews_rating" in self.products_df.columns:
            self.rating_col = "reviews_rating"
        else:
            self.rating_col = None

        if self.rating_col:
            self.products_df[self.rating_col] = pd.to_numeric(
                self.products_df[self.rating_col], errors="coerce"
            )

        # Pre-build lookup dictionary by string ID for fast filtering
        self.products_df["_str_id"] = self.products_df[self.id_col].astype(str)
        self._lookup: Dict[str, pd.Series] = {
            row["_str_id"]: row for _, row in self.products_df.iterrows()
        }

    def get_available_facets(self) -> Dict[str, Any]:
        """
        Return available filter options across the full catalog.

        Returns:
            Dictionary with available brands, categories, price range, and rating options.
        """
        facets: Dict[str, Any] = {
            "brands": [],
            "categories": [],
            "price_range": {"min": 0.0, "max": 0.0},
            "rating_ranges": [1.0, 2.0, 3.0, 4.0, 4.5],
        }

        if self.brand_col and self.brand_col in self.products_df.columns:
            brands = self.products_df[self.brand_col].dropna().unique().tolist()
            facets["brands"] = sorted([str(b) for b in brands if str(b).strip()])

        if self.cat_col and self.cat_col in self.products_df.columns:
            raw_cats = self.products_df[self.cat_col].dropna().unique().tolist()
            unique_cats = set()
            for cat in raw_cats:
                cat_str = str(cat).strip()
                if "," in cat_str:
                    unique_cats.update([c.strip() for c in cat_str.split(",") if c.strip()])
                elif cat_str:
                    unique_cats.add(cat_str)
            facets["categories"] = sorted(list(unique_cats))

        if self.price_col and self.price_col in self.products_df.columns:
            valid_prices = self.products_df[self.price_col].dropna()
            if not valid_prices.empty:
                facets["price_range"] = {
                    "min": float(valid_prices.min()),
                    "max": float(valid_prices.max()),
                }

        return facets

    def filter(
        self,
        results: List[Tuple[Any, float]],
        brand: Optional[Union[str, List[str]]] = None,
        category: Optional[Union[str, List[str]]] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_rating: Optional[float] = None,
    ) -> List[Tuple[Any, float]]:
        """
        Filter retrieval results by specified facets.

        Args:
            results: List of (product_id, score) tuples.
            brand: Brand name or list of brand names to filter by.
            category: Category name or list of categories to filter by.
            min_price: Minimum price threshold.
            max_price: Maximum price threshold.
            min_rating: Minimum rating threshold.

        Returns:
            Filtered list of (product_id, score) tuples.
        """
        if not results:
            return []

        # Prepare brand filters
        target_brands = None
        if brand is not None:
            if isinstance(brand, str):
                target_brands = {brand.lower().strip()}
            elif isinstance(brand, (list, tuple, set)):
                target_brands = {str(b).lower().strip() for b in brand}

        # Prepare category filters
        target_categories = None
        if category is not None:
            if isinstance(category, str):
                target_categories = [category.lower().strip()]
            elif isinstance(category, (list, tuple, set)):
                target_categories = [str(c).lower().strip() for c in category]

        filtered_results: List[Tuple[Any, float]] = []

        for item in results:
            pid = item[0] if isinstance(item, tuple) else item
            str_pid = str(pid)

            if str_pid not in self._lookup:
                continue

            row = self._lookup[str_pid]

            # Brand filter
            if target_brands and self.brand_col:
                row_brand = str(row.get(self.brand_col, "")).lower().strip()
                if row_brand not in target_brands:
                    continue

            # Category filter
            if target_categories and self.cat_col:
                row_cat = str(row.get(self.cat_col, "")).lower().strip()
                if not any(tc in row_cat for tc in target_categories):
                    continue

            # Price filter
            if self.price_col:
                price_val = row.get(self.price_col)
                if pd.notna(price_val):
                    price_float = float(price_val)
                    if min_price is not None and price_float < min_price:
                        continue
                    if max_price is not None and price_float > max_price:
                        continue
                elif min_price is not None or max_price is not None:
                    # Missing price when price filter is requested
                    continue

            # Rating filter
            if self.rating_col and min_rating is not None:
                rating_val = row.get(self.rating_col)
                if pd.notna(rating_val):
                    if float(rating_val) < min_rating:
                        continue
                else:
                    continue

            filtered_results.append(item)

        return filtered_results

    def apply_facets(
        self,
        results: List[Tuple[Any, float]],
        facets: Dict[str, Any],
    ) -> List[Tuple[Any, float]]:
        """
        Apply multiple facets from a dictionary.

        Args:
            results: List of (product_id, score) tuples.
            facets: Dictionary containing facet filter keys ('brand', 'category',
                    'min_price', 'max_price', 'min_rating').

        Returns:
            Filtered list of (product_id, score) tuples.
        """
        if not facets:
            return results

        brand = facets.get("brand", facets.get("brands"))
        category = facets.get("category", facets.get("categories"))
        min_price = facets.get("min_price", facets.get("price_min"))
        max_price = facets.get("max_price", facets.get("price_max"))
        min_rating = facets.get("min_rating", facets.get("rating_min"))

        return self.filter(
            results=results,
            brand=brand,
            category=category,
            min_price=float(min_price) if min_price is not None else None,
            max_price=float(max_price) if max_price is not None else None,
            min_rating=float(min_rating) if min_rating is not None else None,
        )

    def get_facet_counts(
        self,
        results: List[Tuple[Any, float]],
        products_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Dict[str, int]]:
        """
        Count how many retrieved results fall into each facet value.

        Args:
            results: List of (product_id, score) tuples.
            products_df: Optional custom product DataFrame to compute counts over.

        Returns:
            Dictionary mapping facet categories to value-frequency dictionaries.
        """
        df = products_df if products_df is not None else self.products_df
        id_col = self.id_col if products_df is None else (
            "product_id" if "product_id" in df.columns else "id"
        )
        cat_col = self.cat_col if products_df is None else (
            "categories" if "categories" in df.columns else (
                "category" if "category" in df.columns else None
            )
        )
        brand_col = self.brand_col if products_df is None else (
            "brand" if "brand" in df.columns else None
        )
        price_col = self.price_col if products_df is None else (
            "price" if "price" in df.columns else None
        )
        rating_col = self.rating_col if products_df is None else (
            "rating" if "rating" in df.columns else (
                "avg_rating" if "avg_rating" in df.columns else None
            )
        )

        pids = [str(item[0] if isinstance(item, tuple) else item) for item in results]
        matched_df = df[df[id_col].astype(str).isin(set(pids))].copy()

        counts: Dict[str, Dict[str, int]] = {
            "brands": {},
            "categories": {},
            "price_ranges": {
                "Under $25": 0,
                "$25 to $50": 0,
                "$50 to $100": 0,
                "$100 to $200": 0,
                "$200+": 0,
            },
            "rating_ranges": {
                "4 stars & above": 0,
                "3 stars & above": 0,
                "2 stars & above": 0,
                "1 star & above": 0,
            },
        }

        if matched_df.empty:
            return counts

        # Brand counts
        if brand_col and brand_col in matched_df.columns:
            brand_series = matched_df[brand_col].dropna().astype(str)
            for brand_name, cnt in brand_series.value_counts().items():
                if brand_name.strip():
                    counts["brands"][str(brand_name)] = int(cnt)

        # Category counts
        if cat_col and cat_col in matched_df.columns:
            for cat_entry in matched_df[cat_col].dropna().astype(str):
                cat_list = [c.strip() for c in cat_entry.split(",") if c.strip()]
                for cat_val in cat_list:
                    counts["categories"][cat_val] = counts["categories"].get(cat_val, 0) + 1

        # Price range counts
        if price_col and price_col in matched_df.columns:
            numeric_prices = pd.to_numeric(matched_df[price_col], errors="coerce").dropna()
            for p in numeric_prices:
                if p < 25.0:
                    counts["price_ranges"]["Under $25"] += 1
                elif 25.0 <= p < 50.0:
                    counts["price_ranges"]["$25 to $50"] += 1
                elif 50.0 <= p < 100.0:
                    counts["price_ranges"]["$50 to $100"] += 1
                elif 100.0 <= p < 200.0:
                    counts["price_ranges"]["$100 to $200"] += 1
                else:
                    counts["price_ranges"]["$200+"] += 1

        # Rating range counts
        if rating_col and rating_col in matched_df.columns:
            numeric_ratings = pd.to_numeric(matched_df[rating_col], errors="coerce").dropna()
            for r in numeric_ratings:
                if r >= 4.0:
                    counts["rating_ranges"]["4 stars & above"] += 1
                if r >= 3.0:
                    counts["rating_ranges"]["3 stars & above"] += 1
                if r >= 2.0:
                    counts["rating_ranges"]["2 stars & above"] += 1
                if r >= 1.0:
                    counts["rating_ranges"]["1 star & above"] += 1

        return counts
