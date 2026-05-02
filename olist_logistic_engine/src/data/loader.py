"""
Data loader module for Olist Logistics Engine
Loads all parquet files and model from the data directory
"""

import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, Optional

# Root project directory (3 levels up from this file)
# src/data/loader.py -> src/ -> olist_logistic_engine/
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"

def load_network_data() -> pd.DataFrame:
    """
    Load network data with route information
    
    Returns:
        DataFrame with columns: seller_state, customer_state, order_count, 
        avg_delivery_days, performance, seller_lat, seller_lng, customer_lat, customer_lng
    """
    file_path = DATA_DIR / "network_data.parquet"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Network data not found at {file_path}")
    
    df = pd.read_parquet(file_path)
    
    # Add latitude/longitude columns if not present (for backward compatibility)
    if 'seller_lat' not in df.columns and 'seller_state' in df.columns:
        # Load state centroids to get coordinates
        state_centroids = load_state_centroids()
        if not state_centroids.empty:
            # Merge seller coordinates
            df = df.merge(
                state_centroids[['state', 'lat', 'lng']].rename(
                    columns={'state': 'seller_state', 'lat': 'seller_lat', 'lng': 'seller_lng'}
                ),
                on='seller_state',
                how='left'
            )
            # Merge customer coordinates
            df = df.merge(
                state_centroids[['state', 'lat', 'lng']].rename(
                    columns={'state': 'customer_state', 'lat': 'customer_lat', 'lng': 'customer_lng'}
                ),
                on='customer_state',
                how='left'
            )
    
    return df


def load_route_features() -> pd.DataFrame:
    """
    Load route-level aggregated features
    
    Returns:
        DataFrame with route features for modeling
    """
    file_path = DATA_DIR / "route_features.parquet"
    
    if not file_path.exists():
        print(f"Warning: Route features not found at {file_path}")
        return pd.DataFrame()
    
    return pd.read_parquet(file_path)


def load_seller_features() -> pd.DataFrame:
    """
    Load seller performance metrics
    
    Returns:
        DataFrame with seller-level features
    """
    file_path = DATA_DIR / "seller_features.parquet"
    
    if not file_path.exists():
        print(f"Warning: Seller features not found at {file_path}")
        return pd.DataFrame()
    
    return pd.read_parquet(file_path)


def load_category_features() -> pd.DataFrame:
    """
    Load product category features
    
    Returns:
        DataFrame with category-level analysis
    """
    file_path = DATA_DIR / "category_features.parquet"
    
    if not file_path.exists():
        print(f"Warning: Category features not found at {file_path}")
        return pd.DataFrame()
    
    return pd.read_parquet(file_path)


def load_warehouse_candidates() -> Optional[pd.DataFrame]:
    """
    Load warehouse candidate locations from clustering analysis
    
    Returns:
        DataFrame with warehouse locations or None if not found
    """
    file_path = DATA_DIR / "warehouse_candidates.parquet"
    
    if not file_path.exists():
        print(f"Warning: Warehouse candidates not found at {file_path}")
        return None
    
    return pd.read_parquet(file_path)


def load_route_priority() -> pd.DataFrame:
    """
    Load route priority/ranking data
    
    Returns:
        DataFrame with prioritized routes for optimization
    """
    file_path = DATA_DIR / "route_priority.parquet"
    
    if not file_path.exists():
        print(f"Warning: Route priority not found at {file_path}")
        return pd.DataFrame()
    
    return pd.read_parquet(file_path)


def load_state_centroids() -> pd.DataFrame:
    """
    Load state centroids with geographic coordinates
    
    Returns:
        DataFrame with columns: state, lat, lng, radius
    """
    file_path = DATA_DIR / "state_centroids.parquet"
    
    if not file_path.exists():
        print(f"Warning: State centroids not found at {file_path}")
        # Return default state centroids if file doesn't exist
        return _get_default_state_centroids()
    
    return pd.read_parquet(file_path)


def load_cost_benefit_summary() -> pd.DataFrame:
    """
    Load cost-benefit analysis summary
    
    Returns:
        DataFrame with cost-benefit metrics for warehouse placement
    """
    file_path = DATA_DIR / "cost_benefit_summary.parquet"
    
    if not file_path.exists():
        print(f"Warning: Cost-benefit summary not found at {file_path}")
        return pd.DataFrame()
    
    return pd.read_parquet(file_path)


def load_risk_model():
    """
    Load trained risk classification model
    
    Returns:
        Trained scikit-learn model (RandomForest or similar)
    """
    file_path = DATA_DIR / "route_risk_model.pkl"
    
    if not file_path.exists():
        print(f"Warning: Risk model not found at {file_path}")
        return None
    
    return joblib.load(file_path)


def load_what_if_simulation() -> pd.DataFrame:
    """
    Load what-if simulation results for warehouse optimization
    
    Returns:
        DataFrame with simulation results
    """
    file_path = DATA_DIR / "what_if_simulation.parquet"
    
    if not file_path.exists():
        print(f"Warning: What-if simulation not found at {file_path}")
        return pd.DataFrame()
    
    return pd.read_parquet(file_path)


def _get_default_state_centroids() -> pd.DataFrame:
    """
    Return default state centroids if file not found
    
    Returns:
        DataFrame with default Brazilian state centroids
    """
    return pd.DataFrame({
        'state': ['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'PE', 'CE', 'AM', 'PA', 
                  'GO', 'DF', 'ES', 'MT', 'MS', 'MA', 'PB', 'RN', 'AL', 'SE', 'PI', 
                  'RO', 'AC', 'RR', 'AP', 'TO'],
        'lat': [-23.55, -22.91, -19.92, -30.03, -25.43, -27.60, -12.97, -8.05, -3.73,
                -3.07, -1.45, -16.68, -15.80, -20.32, -15.60, -20.44, -2.53, -7.12,
                -5.79, -9.66, -10.91, -5.09, -8.76, -9.97, 2.82, 0.03, -10.18],
        'lng': [-46.63, -43.20, -43.94, -51.23, -49.27, -48.52, -38.51, -34.90, -38.52,
                -60.00, -48.50, -49.25, -47.86, -40.34, -56.10, -54.64, -44.30, -34.86,
                -35.21, -35.74, -37.07, -42.80, -63.90, -67.81, -60.67, -51.05, -48.33],
        'radius': 15000
    })


def load_all() -> Dict[str, Any]:
    """
    Load all available data files at once
    
    Returns:
        Dictionary with all loaded data:
        - network: DataFrame
        - route_features: DataFrame
        - seller_features: DataFrame
        - category_features: DataFrame
        - warehouse_candidates: DataFrame or None
        - route_priority: DataFrame
        - state_centroids: DataFrame
        - cost_benefit_summary: DataFrame
        - risk_model: model or None
        - what_if_simulation: DataFrame
    """
    return {
        'network': load_network_data(),
        'route_features': load_route_features(),
        'seller_features': load_seller_features(),
        'category_features': load_category_features(),
        'warehouse_candidates': load_warehouse_candidates(),
        'route_priority': load_route_priority(),
        'state_centroids': load_state_centroids(),
        'cost_benefit_summary': load_cost_benefit_summary(),
        'risk_model': load_risk_model(),
        'what_if_simulation': load_what_if_simulation()
    }


def check_data_availability() -> Dict[str, bool]:
    """
    Check which data files are available
    
    Returns:
        Dictionary with file names as keys and availability as values
    """
    files = [
        'network_data.parquet',
        'route_features.parquet',
        'seller_features.parquet',
        'category_features.parquet',
        'warehouse_candidates.parquet',
        'route_priority.parquet',
        'state_centroids.parquet',
        'cost_benefit_summary.parquet',
        'route_risk_model.pkl',
        'what_if_simulation.parquet'
    ]
    
    availability = {}
    for file in files:
        file_path = DATA_DIR / file
        availability[file] = file_path.exists()
    
    return availability


# For quick testing
if __name__ == "__main__":
    print(f"Data directory: {DATA_DIR}")
    print("\nChecking data availability:")
    availability = check_data_availability()
    for file, exists in availability.items():
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {file}")
    
    print("\nLoading data...")
    data = load_all()
    print(f"Network data loaded: {len(data['network'])} routes")
    print(f"State centroids loaded: {len(data['state_centroids'])} states")
    
    if data['warehouse_candidates'] is not None:
        print(f"Warehouse candidates: {len(data['warehouse_candidates'])} locations")
    else:
        print("Warehouse candidates: Not available")