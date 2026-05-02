"""
Seller-level feature engineering for performance analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def create_seller_features(df_orders: pd.DataFrame,
                           df_order_items: pd.DataFrame,
                           df_sellers: pd.DataFrame,
                           df_reviews: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Create comprehensive seller-level features.
    
    Args:
        df_orders: Orders dataset
        df_order_items: Order items dataset
        df_sellers: Sellers dataset with location info
        df_reviews: Order reviews dataset (optional)
    
    Returns:
        DataFrame with seller features including:
        - total_orders: Number of orders fulfilled
        - total_revenue: Total sales value
        - avg_delivery_days: Average delivery time
        - unique_customers: Number of unique customers
        - seller_segment: Segmented seller classification
    """
    
    # Merge order items with orders
    df = df_order_items.merge(df_orders, on='order_id')
    
    # Filter delivered orders if status column exists
    if 'order_status' in df.columns:
        df = df[df['order_status'] == 'delivered']
    
    # Calculate delivery days if available
    if 'order_delivered_customer_date' in df.columns and 'order_purchase_timestamp' in df.columns:
        df['delivery_days'] = (
            pd.to_datetime(df['order_delivered_customer_date']) - 
            pd.to_datetime(df['order_purchase_timestamp'])
        ).dt.days
    
    # Aggregate by seller
    seller_features = df.groupby('seller_id').agg({
        'order_id': 'nunique',
        'price': ['sum', 'mean', 'std'],
        'freight_value': ['sum', 'mean'],
    }).reset_index()
    
    # Flatten column names
    seller_features.columns = [
        'seller_id', 'total_orders', 'total_revenue', 'avg_price', 'std_price',
        'total_freight', 'avg_freight'
    ]
    
    # Add unique customer count
    unique_customers = df.groupby('seller_id')['customer_id'].nunique().reset_index()
    unique_customers.columns = ['seller_id', 'unique_customers']
    seller_features = seller_features.merge(unique_customers, on='seller_id')
    
    # Add repeat customer rate
    repeat_customers = df.groupby(['seller_id', 'customer_id']).size().reset_index()
    repeat_customers = repeat_customers[repeat_customers[0] > 1].groupby('seller_id')['customer_id'].nunique().reset_index()
    repeat_customers.columns = ['seller_id', 'repeat_customers']
    seller_features = seller_features.merge(repeat_customers, on='seller_id', how='left')
    seller_features['repeat_customers'] = seller_features['repeat_customers'].fillna(0)
    seller_features['repeat_rate'] = seller_features['repeat_customers'] / seller_features['unique_customers']
    
    # Add delivery time metrics
    if 'delivery_days' in df.columns:
        delivery_agg = df.groupby('seller_id')['delivery_days'].agg([
            'mean', 'median', 'std', 'min', 'max'
        ]).reset_index()
        delivery_agg.columns = ['seller_id', 'avg_delivery_days', 'median_delivery_days',
                                'std_delivery_days', 'min_delivery_days', 'max_delivery_days']
        seller_features = seller_features.merge(delivery_agg, on='seller_id', how='left')
    
    # Add seller location
    seller_features = seller_features.merge(
        df_sellers[['seller_id', 'seller_state', 'seller_city']],
        on='seller_id',
        how='left'
    )
    
    # Calculate order frequency (orders per day, if date available)
    if 'order_purchase_timestamp' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_purchase_timestamp']).dt.date
        first_order = df.groupby('seller_id')['order_date'].min().reset_index()
        last_order = df.groupby('seller_id')['order_date'].max().reset_index()
        
        first_order.columns = ['seller_id', 'first_order_date']
        last_order.columns = ['seller_id', 'last_order_date']
        
        seller_features = seller_features.merge(first_order, on='seller_id')
        seller_features = seller_features.merge(last_order, on='seller_id')
        
        # Convert to datetime
        seller_features['first_order_date'] = pd.to_datetime(seller_features['first_order_date'])
        seller_features['last_order_date'] = pd.to_datetime(seller_features['last_order_date'])
        
        # Calculate active days and order frequency
        active_days = (seller_features['last_order_date'] - seller_features['first_order_date']).dt.days
        active_days = active_days.replace(0, 1)  # Avoid division by zero
        seller_features['orders_per_day'] = seller_features['total_orders'] / active_days
    
    # Add review scores if available
    if df_reviews is not None:
        review_agg = df_reviews.groupby('seller_id')['review_score'].agg(['mean', 'std', 'count']).reset_index()
        review_agg.columns = ['seller_id', 'avg_review_score', 'std_review_score', 'review_count']
        seller_features = seller_features.merge(review_agg, on='seller_id', how='left')
    
    return seller_features


def segment_sellers(seller_features: pd.DataFrame) -> pd.DataFrame:
    """
    Segment sellers into categories based on performance metrics.
    
    Segmentation criteria:
    - Platinum: High volume, fast delivery, high revenue
    - Gold: Medium-high volume, good delivery
    - Silver: Medium volume, average delivery
    - Bronze: Low volume or slow delivery
    - New: Low order count
    - At Risk: Poor delivery performance or low reviews
    
    Returns:
        DataFrame with 'seller_segment' and 'segment_score' columns
    """
    seller_features = seller_features.copy()
    
    # Initialize conditions
    conditions = []
    choices = []
    
    # Platinum: Top 10% by orders AND fast delivery
    high_volume_threshold = seller_features['total_orders'].quantile(0.9)
    fast_delivery_threshold = seller_features['avg_delivery_days'].quantile(0.2) if 'avg_delivery_days' in seller_features.columns else 10
    
    conditions.append(
        (seller_features['total_orders'] >= high_volume_threshold) &
        (seller_features.get('avg_delivery_days', 100) <= fast_delivery_threshold)
    )
    choices.append('Platinum')
    
    # Gold: Top 30% by orders OR high revenue
    gold_threshold = seller_features['total_orders'].quantile(0.7)
    conditions.append(seller_features['total_orders'] >= gold_threshold)
    choices.append('Gold')
    
    # Silver: Top 60% by orders
    silver_threshold = seller_features['total_orders'].quantile(0.4)
    conditions.append(seller_features['total_orders'] >= silver_threshold)
    choices.append('Silver')
    
    # At Risk: Poor performance (slow delivery or low reviews)
    if 'avg_review_score' in seller_features.columns:
        at_risk_condition = (
            (seller_features.get('avg_delivery_days', 0) > 20) |
            (seller_features['avg_review_score'] < 3)
        )
    else:
        at_risk_condition = (seller_features.get('avg_delivery_days', 0) > 20)
    
    conditions.append(at_risk_condition & ~seller_features['seller_segment'].isin(['Platinum', 'Gold', 'Silver']))
    choices.append('At Risk')
    
    # Default to Bronze
    conditions.append(pd.Series([True] * len(seller_features)))
    choices.append('Bronze')
    
    seller_features['seller_segment'] = np.select(conditions, choices, default='Bronze')
    
    # Add segment score (1-5, where 5 is best)
    segment_scores = {
        'Platinum': 5,
        'Gold': 4,
        'Silver': 3,
        'Bronze': 2,
        'At Risk': 1,
        'New': 2
    }
    seller_features['segment_score'] = seller_features['seller_segment'].map(segment_scores)
    
    return seller_features


def calculate_seller_risk_score(seller_features: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate risk score for each seller based on multiple factors.
    
    Risk factors:
    - Delivery performance (higher days = higher risk)
    - Volume (low volume = higher risk)
    - Review scores (low scores = higher risk)
    - Repeat rate (low repeat = higher risk)
    
    Returns:
        DataFrame with 'risk_score' (0-100, higher = more risky)
    """
    seller_features = seller_features.copy()
    
    risk_components = []
    weights = {'delivery': 0.35, 'volume': 0.25, 'review': 0.25, 'repeat': 0.15}
    
    # Delivery risk (higher days = higher risk)
    if 'avg_delivery_days' in seller_features.columns:
        max_delivery = seller_features['avg_delivery_days'].max()
        delivery_risk = (seller_features['avg_delivery_days'] / max_delivery) * 100
        delivery_risk = delivery_risk.fillna(50)
        risk_components.append(weights['delivery'] * delivery_risk)
    
    # Volume risk (lower volume = higher risk)
    max_orders = seller_features['total_orders'].max()
    volume_risk = (1 - seller_features['total_orders'] / max_orders) * 100
    risk_components.append(weights['volume'] * volume_risk)
    
    # Review risk (lower score = higher risk)
    if 'avg_review_score' in seller_features.columns:
        review_risk = (5 - seller_features['avg_review_score']) / 4 * 100
        review_risk = review_risk.fillna(50)
        risk_components.append(weights['review'] * review_risk)
    else:
        risk_components.append(weights['review'] * 50)  # Neutral if no reviews
    
    # Repeat rate risk (lower repeat = higher risk)
    if 'repeat_rate' in seller_features.columns:
        repeat_risk = (1 - seller_features['repeat_rate']) * 100
        repeat_risk = repeat_risk.fillna(50)
        risk_components.append(weights['repeat'] * repeat_risk)
    else:
        risk_components.append(weights['repeat'] * 50)
    
    seller_features['risk_score'] = sum(risk_components)
    
    # Risk category
    seller_features['risk_category'] = pd.cut(
        seller_features['risk_score'],
        bins=[0, 25, 50, 75, 101],
        labels=['Low', 'Medium-Low', 'Medium-High', 'High']
    )
    
    return seller_features


def get_seller_summary_stats(seller_features: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for sellers.
    
    Returns:
        Dictionary with seller summary metrics
    """
    stats = {
        'total_sellers': len(seller_features),
        'total_orders': seller_features['total_orders'].sum(),
        'total_revenue': seller_features['total_revenue'].sum(),
        'avg_orders_per_seller': seller_features['total_orders'].mean(),
        'median_orders_per_seller': seller_features['total_orders'].median(),
        'avg_revenue_per_seller': seller_features['total_revenue'].mean(),
        'top_10_percent_revenue': seller_features['total_revenue'].quantile(0.9),
        'sellers_with_fast_delivery': (seller_features.get('avg_delivery_days', 100) <= 10).sum() if 'avg_delivery_days' in seller_features.columns else 0,
        'sellers_with_slow_delivery': (seller_features.get('avg_delivery_days', 0) >= 20).sum() if 'avg_delivery_days' in seller_features.columns else 0,
        'segment_distribution': seller_features['seller_segment'].value_counts().to_dict() if 'seller_segment' in seller_features.columns else {},
    }
    
    return stats


def identify_top_sellers(seller_features: pd.DataFrame, 
                          metric: str = 'total_revenue', 
                          n: int = 10) -> pd.DataFrame:
    """
    Identify top N sellers based on specified metric.
    
    Args:
        seller_features: DataFrame with seller features
        metric: Column name to rank by ('total_revenue', 'total_orders', 'avg_review_score')
        n: Number of top sellers to return
    
    Returns:
        DataFrame with top N sellers
    """
    if metric not in seller_features.columns:
        raise ValueError(f"Metric '{metric}' not found in seller_features")
    
    return seller_features.nlargest(n, metric)[['seller_id', metric, 'seller_state', 'seller_segment']]