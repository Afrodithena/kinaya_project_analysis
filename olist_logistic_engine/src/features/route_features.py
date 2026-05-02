"""
Route-level feature engineering for logistics network analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .geospatial import add_distance_columns, add_region_columns, create_distance_bins


def create_route_features(df_orders: pd.DataFrame,
                          df_order_items: pd.DataFrame,
                          df_sellers: pd.DataFrame,
                          df_customers: pd.DataFrame,
                          state_centroids: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive route-level features from raw order data.
    
    Args:
        df_orders: Orders dataset
        df_order_items: Order items dataset
        df_sellers: Sellers dataset
        df_customers: Customers dataset
        state_centroids: State centroids with coordinates
    
    Returns:
        DataFrame with route features including:
        - order_count: Number of orders per route
        - total_revenue: Total order value
        - avg_price: Average item price
        - avg_freight: Average freight value
        - distance_km: Geographic distance
        - delivery_performance: Delivery time metrics
    """
    
    # Merge datasets
    df = df_order_items.merge(df_orders, on='order_id')
    df = df.merge(df_sellers[['seller_id', 'seller_state']], on='seller_id')
    df = df.merge(df_customers[['customer_id', 'customer_state']], on='customer_id')
    
    # Filter delivered orders
    if 'order_status' in df.columns:
        df = df[df['order_status'] == 'delivered']
    
    # Calculate delivery time if available
    if 'order_delivered_customer_date' in df.columns and 'order_purchase_timestamp' in df.columns:
        df['delivery_days'] = (
            pd.to_datetime(df['order_delivered_customer_date']) - 
            pd.to_datetime(df['order_purchase_timestamp'])
        ).dt.days
    
    # Aggregate by route (seller_state -> customer_state)
    route_features = df.groupby(['seller_state', 'customer_state']).agg({
        'order_id': 'count',
        'price': ['sum', 'mean'],
        'freight_value': ['sum', 'mean'],
    }).reset_index()
    
    # Flatten column names
    route_features.columns = [
        'seller_state', 'customer_state',
        'order_count', 'total_revenue', 'avg_price',
        'total_freight', 'avg_freight'
    ]
    
    # Add delivery time metrics
    if 'delivery_days' in df.columns:
        delivery_agg = df.groupby(['seller_state', 'customer_state'])['delivery_days'].agg([
            'mean', 'median', 'std', 'min', 'max'
        ]).reset_index()
        delivery_agg.columns = ['seller_state', 'customer_state', 
                                'avg_delivery_days', 'median_delivery_days',
                                'std_delivery_days', 'min_delivery_days', 'max_delivery_days']
        
        route_features = route_features.merge(delivery_agg, on=['seller_state', 'customer_state'])
    
    # Add distance and region columns
    route_features = add_distance_columns(route_features, state_centroids)
    route_features = add_region_columns(route_features)
    
    # Add distance bins
    route_features['distance_bin'] = create_distance_bins(
        route_features['distance_km'].values,
        bins=[0, 500, 1000, 1500, 2000, 3000]
    )
    
    # Add derived features
    route_features['freight_per_km'] = route_features['avg_freight'] / route_features['distance_km']
    route_features['revenue_per_order'] = route_features['total_revenue'] / route_features['order_count']
    route_features['freight_to_price_ratio'] = route_features['avg_freight'] / route_features['avg_price']
    
    return route_features


def calculate_route_density(df_orders: pd.DataFrame,
                            df_sellers: pd.DataFrame,
                            df_customers: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate route density metrics (orders per seller, orders per customer).
    
    Returns:
        DataFrame with density metrics per route
    """
    
    # Seller density
    seller_orders = df_orders.groupby('seller_id').size().reset_index()
    seller_orders.columns = ['seller_id', 'seller_order_count']
    
    # Customer density
    customer_orders = df_orders.groupby('customer_id').size().reset_index()
    customer_orders.columns = ['customer_id', 'customer_order_count']
    
    return seller_orders, customer_orders


def identify_high_volume_routes(route_features: pd.DataFrame, 
                                 threshold_percentile: int = 80) -> pd.DataFrame:
    """
    Identify high-volume routes based on order count threshold.
    
    Args:
        route_features: DataFrame with route features
        threshold_percentile: Percentile to use as threshold (default: 80)
    
    Returns:
        DataFrame with binary flag 'is_high_volume'
    """
    route_features = route_features.copy()
    threshold = route_features['order_count'].quantile(threshold_percentile / 100)
    route_features['is_high_volume'] = (route_features['order_count'] >= threshold).astype(int)
    route_features['volume_tier'] = pd.cut(
        route_features['order_count'],
        bins=[0, 50, 200, 500, 1000, float('inf')],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    return route_features


def classify_performance(route_features: pd.DataFrame,
                          fast_threshold: int = 10,
                          normal_threshold: int = 15,
                          slow_threshold: int = 20) -> pd.DataFrame:
    """
    Classify route performance based on average delivery days.
    
    Args:
        route_features: DataFrame with 'avg_delivery_days' column
        fast_threshold: Days under which route is 'Fast'
        normal_threshold: Days under which route is 'Normal'
        slow_threshold: Days under which route is 'Slow'
    
    Returns:
        DataFrame with 'performance' and 'performance_score' columns
    """
    route_features = route_features.copy()
    
    conditions = [
        route_features['avg_delivery_days'] <= fast_threshold,
        route_features['avg_delivery_days'] <= normal_threshold,
        route_features['avg_delivery_days'] <= slow_threshold,
        route_features['avg_delivery_days'] > slow_threshold
    ]
    
    choices = ['Fast', 'Normal', 'Slow', 'Critical']
    route_features['performance'] = np.select(conditions, choices, default='Unknown')
    
    # Performance score (lower is better, normalized to 0-100)
    max_days = route_features['avg_delivery_days'].max()
    route_features['performance_score'] = 100 * (1 - route_features['avg_delivery_days'] / max_days)
    
    return route_features


def calculate_growth_potential(route_features: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate growth potential score for each route.
    
    Growth potential is based on:
    - Current order volume
    - Geographic distance (longer routes with high volume have more potential)
    - Freight cost (higher freight per km indicates optimization potential)
    
    Returns:
        DataFrame with 'growth_potential' and 'priority_score' columns
    """
    route_features = route_features.copy()
    
    # Normalize metrics
    order_norm = route_features['order_count'] / route_features['order_count'].max()
    distance_norm = route_features['distance_km'] / route_features['distance_km'].max()
    freight_norm = route_features['freight_per_km'] / route_features['freight_per_km'].max()
    
    # Calculate growth potential (weighted combination)
    route_features['growth_potential'] = (
        0.4 * order_norm +
        0.3 * distance_norm +
        0.3 * freight_norm
    ) * 100
    
    # Priority score for warehouse placement
    # Higher score = more beneficial to optimize with new warehouse
    delivery_performance_norm = 1 - (route_features['avg_delivery_days'] / route_features['avg_delivery_days'].max())
    route_features['priority_score'] = (
        0.5 * order_norm +
        0.3 * delivery_performance_norm +
        0.2 * freight_norm
    ) * 100
    
    return route_features


def get_route_summary_stats(route_features: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for all routes.
    
    Returns:
        Dictionary with summary metrics
    """
    stats = {
        'total_routes': len(route_features),
        'unique_seller_states': route_features['seller_state'].nunique(),
        'unique_customer_states': route_features['customer_state'].nunique(),
        'total_orders': route_features['order_count'].sum(),
        'total_revenue': route_features['total_revenue'].sum(),
        'avg_distance_km': route_features['distance_km'].mean(),
        'median_distance_km': route_features['distance_km'].median(),
        'avg_delivery_days': route_features.get('avg_delivery_days', pd.Series([0])).mean(),
        'avg_freight_per_km': route_features['freight_per_km'].mean(),
        'high_volume_routes': (route_features['order_count'] > 500).sum(),
        'critical_performance': (route_features.get('performance', '') == 'Critical').sum(),
        'top_route': route_features.nlargest(1, 'order_count')[['seller_state', 'customer_state', 'order_count']].to_dict('records')[0] if len(route_features) > 0 else {}
    }
    
    return stats