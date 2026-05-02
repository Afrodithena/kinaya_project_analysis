"""
Geospatial utility functions for distance calculations and coordinate processing
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from math import radians, sin, cos, sqrt, atan2

# Earth's radius in kilometers
EARTH_RADIUS_KM = 6371

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    
    Args:
        lat1, lon1: Latitude and longitude of point 1 (in degrees)
        lat2, lon2: Latitude and longitude of point 2 (in degrees)
    
    Returns:
        Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = EARTH_RADIUS_KM * c
    return distance


def calculate_distance_batch(df: pd.DataFrame, 
                             lat1_col: str, 
                             lon1_col: str, 
                             lat2_col: str, 
                             lon2_col: str) -> np.ndarray:
    """
    Calculate distances for multiple rows in a DataFrame.
    
    Args:
        df: DataFrame containing coordinate columns
        lat1_col, lon1_col: Column names for first point
        lat2_col, lon2_col: Column names for second point
    
    Returns:
        Array of distances in kilometers
    """
    lat1_rad = np.radians(df[lat1_col].values)
    lon1_rad = np.radians(df[lon1_col].values)
    lat2_rad = np.radians(df[lat2_col].values)
    lon2_rad = np.radians(df[lon2_col].values)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    distances = EARTH_RADIUS_KM * c
    return distances


def haversine_vectorized(lat1: np.ndarray, 
                          lon1: np.ndarray, 
                          lat2: np.ndarray, 
                          lon2: np.ndarray) -> np.ndarray:
    """
    Vectorized haversine distance calculation using numpy.
    
    Args:
        lat1, lon1: Arrays of coordinates for first points
        lat2, lon2: Arrays of coordinates for second points
    
    Returns:
        Array of distances in kilometers
    """
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return EARTH_RADIUS_KM * c


def get_state_coordinates(state_code: str, state_centroids: pd.DataFrame) -> Tuple[float, float]:
    """
    Get latitude and longitude for a given state code.
    
    Args:
        state_code: Brazilian state code (e.g., 'SP', 'RJ')
        state_centroids: DataFrame with columns 'state', 'lat', 'lng'
    
    Returns:
        Tuple of (latitude, longitude)
    """
    row = state_centroids[state_centroids['state'] == state_code]
    if row.empty:
        raise ValueError(f"State code '{state_code}' not found in centroids")
    
    return row.iloc[0]['lat'], row.iloc[0]['lng']


def add_distance_columns(df: pd.DataFrame, state_centroids: pd.DataFrame) -> pd.DataFrame:
    """
    Add distance columns to a DataFrame with seller and customer states.
    
    Args:
        df: DataFrame with 'seller_state' and 'customer_state' columns
        state_centroids: DataFrame with state centroids
    
    Returns:
        DataFrame with added 'distance_km' column
    """
    # Merge coordinates for sellers
    df = df.merge(
        state_centroids[['state', 'lat', 'lng']].rename(
            columns={'state': 'seller_state', 'lat': 'seller_lat', 'lng': 'seller_lng'}
        ),
        on='seller_state',
        how='left'
    )
    
    # Merge coordinates for customers
    df = df.merge(
        state_centroids[['state', 'lat', 'lng']].rename(
            columns={'state': 'customer_state', 'lat': 'customer_lat', 'lng': 'customer_lng'}
        ),
        on='customer_state',
        how='left'
    )
    
    # Calculate distance
    df['distance_km'] = calculate_distance_batch(
        df, 'seller_lat', 'seller_lng', 'customer_lat', 'customer_lng'
    )
    
    return df


def calculate_centrality(df: pd.DataFrame, 
                          lat_col: str = 'lat', 
                          lng_col: str = 'lng') -> pd.DataFrame:
    """
    Calculate geographic centrality metrics for nodes.
    
    Args:
        df: DataFrame with latitude and longitude columns
        lat_col, lng_col: Column names for coordinates
    
    Returns:
        DataFrame with added 'centrality_score' column (lower = more central)
    """
    coords = df[[lat_col, lng_col]].values
    n_points = len(coords)
    
    centrality_scores = []
    for i in range(n_points):
        # Calculate average distance to all other points
        distances = []
        for j in range(n_points):
            if i != j:
                dist = haversine_distance(
                    coords[i, 0], coords[i, 1],
                    coords[j, 0], coords[j, 1]
                )
                distances.append(dist)
        
        avg_distance = np.mean(distances) if distances else 0
        centrality_scores.append(avg_distance)
    
    df = df.copy()
    df['centrality_score'] = centrality_scores
    return df


def create_distance_bins(distance_km: np.ndarray, bins: Optional[List[float]] = None) -> np.ndarray:
    """
    Create distance bins for categorical analysis.
    
    Args:
        distance_km: Array of distances in kilometers
        bins: Custom bin edges (default: [0, 500, 1000, 1500, 2000, 3000])
    
    Returns:
        Array of bin labels
    """
    if bins is None:
        bins = [0, 500, 1000, 1500, 2000, 3000, 10000]
    
    labels = [f'{bins[i]}-{bins[i+1]}km' for i in range(len(bins)-2)]
    labels.append(f'>{bins[-2]}km')
    
    return np.digitize(distance_km, bins, right=False)


def is_same_state(seller_state: str, customer_state: str) -> bool:
    """
    Check if seller and customer are in the same state.
    
    Returns:
        True if same state, False otherwise
    """
    return seller_state == customer_state


def get_region_from_state(state_code: str) -> str:
    """
    Map Brazilian state to geographic region.
    
    Args:
        state_code: Brazilian state code
    
    Returns:
        Region name: 'North', 'Northeast', 'Central-West', 'Southeast', 'South'
    """
    north = ['AM', 'RR', 'AP', 'PA', 'TO', 'RO', 'AC']
    northeast = ['MA', 'PI', 'CE', 'RN', 'PB', 'PE', 'AL', 'SE', 'BA']
    central_west = ['MT', 'MS', 'GO', 'DF']
    southeast = ['SP', 'RJ', 'MG', 'ES']
    south = ['PR', 'SC', 'RS']
    
    if state_code in north:
        return 'North'
    elif state_code in northeast:
        return 'Northeast'
    elif state_code in central_west:
        return 'Central-West'
    elif state_code in southeast:
        return 'Southeast'
    elif state_code in south:
        return 'South'
    else:
        return 'Unknown'


def add_region_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add region columns to DataFrame based on seller and customer states.
    
    Args:
        df: DataFrame with 'seller_state' and 'customer_state' columns
    
    Returns:
        DataFrame with added 'seller_region' and 'customer_region' columns
    """
    df = df.copy()
    df['seller_region'] = df['seller_state'].apply(get_region_from_state)
    df['customer_region'] = df['customer_state'].apply(get_region_from_state)
    df['is_interstate'] = (df['seller_state'] != df['customer_state']).astype(int)
    df['is_interregion'] = (df['seller_region'] != df['customer_region']).astype(int)
    
    return df