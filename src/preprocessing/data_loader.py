"""
Data loading and transformation module.
"""
import pandas as pd
import numpy as np
from typing import Optional
from src.config import PRODUCT_TEXT_COLUMNS, REVIEW_COLUMNS

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw CSV data, handling encoding issues and dropping rows with no product name.
    
    Args:
        filepath: Path to the raw CSV file.
        
    Returns:
        pd.DataFrame containing the loaded and filtered data.
    """
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, encoding='latin1')
        except Exception as e:
            df = pd.read_csv(filepath, encoding='cp1252')
            
    # Drop rows without a product name
    df = df.dropna(subset=['name'])
    return df

def merge_product_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by product and aggregate reviews.
    
    Args:
        df: DataFrame containing products and reviews.
        
    Returns:
        pd.DataFrame with aggregated reviews per product.
    """
    if 'reviews.text' not in df.columns:
        # No reviews in this dataset
        return df.drop_duplicates(subset=['id', 'name'])
        
    # Function to concatenate valid text
    def agg_text(series):
        return ' '.join(str(x) for x in series.dropna() if str(x).strip())
        
    # Grouping by name (or id if available and unique per product)
    groupby_cols = ['name', 'brand']
    if 'id' in df.columns:
        groupby_cols = ['id'] + groupby_cols
        
    # Separate product info from review info
    product_cols = [c for c in df.columns if not c.startswith('reviews.') and c not in groupby_cols]
    
    aggs = {}
    if 'reviews.text' in df.columns:
        aggs['reviews.text'] = agg_text
    if 'reviews.rating' in df.columns:
        aggs['reviews.rating'] = ['mean', 'count']
        
    for col in product_cols:
        aggs[col] = 'first'
        
    grouped = df.groupby(groupby_cols, as_index=False).agg(aggs)
    
    # Flatten multi-index columns if they exist
    if isinstance(grouped.columns, pd.MultiIndex):
        grouped.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in grouped.columns]
        
    # Rename for consistency
    rename_map = {
        'reviews.text_agg_text': 'reviews_text',
        'reviews.text_join': 'reviews_text',
        'reviews.rating_mean': 'avg_rating',
        'reviews.rating_count': 'review_count'
    }
    grouped = grouped.rename(columns=rename_map)
    
    # Handle single aggregation cases
    if 'reviews.text' in grouped.columns:
        grouped = grouped.rename(columns={'reviews.text': 'reviews_text'})
    
    return grouped

def create_product_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a unified 'document' column by combining product text fields.
    
    Args:
        df: DataFrame with product info and aggregated reviews.
        
    Returns:
        pd.DataFrame with the unified 'document' column and clean structure.
    """
    clean_df = pd.DataFrame()
    
    # Assign sequential product_id
    clean_df['product_id'] = range(1, len(df) + 1)
    
    # Extract known columns if they exist
    cols_to_extract = {
        'name': 'name',
        'brand': 'brand',
        'categories': 'categories',
        'primaryCategories': 'categories', # Fallback
        'description': 'description',
        'prices.amountMax': 'price',
        'avg_rating': 'avg_rating',
        'review_count': 'review_count',
        'reviews_text': 'reviews_text'
    }
    
    for df_col, clean_col in cols_to_extract.items():
        if df_col in df.columns and clean_col not in clean_df.columns:
            clean_df[clean_col] = df[df_col]
            
    # Fill NA for text columns
    text_columns = ['name', 'brand', 'categories', 'description', 'reviews_text']
    for col in text_columns:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].fillna('')
        else:
            clean_df[col] = ''
            
    # Create the document column
    def combine_text(row):
        parts = [
            str(row['name']),
            str(row['brand']),
            str(row['categories']),
            str(row['description']),
            str(row['reviews_text'])
        ]
        return ' '.join(p for p in parts if p.strip())
        
    clean_df['document'] = clean_df.apply(combine_text, axis=1)
    
    return clean_df

def save_processed_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save processed DataFrame to CSV.
    
    Args:
        df: Processed DataFrame.
        filepath: Output file path.
    """
    df.to_csv(filepath, index=False, encoding='utf-8')

def load_processed_data(filepath: str) -> pd.DataFrame:
    """
    Load processed DataFrame from CSV.
    
    Args:
        filepath: Input file path.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(filepath, encoding='utf-8')
