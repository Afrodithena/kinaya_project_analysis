"""
Warehouse optimization using clustering algorithms
Finds optimal warehouse locations based on order volume and geography
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def render_warehouse_optimization(data=None): 
    """Render warehouse optimization dashboard"""
    
    st.markdown('<div class="page-header">Warehouse Optimization</div>', unsafe_allow_html=True)
    st.markdown("Strategic analysis for optimal warehouse placement to reduce delivery times")
    
    if data is None:
        @st.cache_data
        def get_data():
            return load_all()
        data = get_data()
    
    warehouse_candidates = data.get('warehouse_candidates', None)
    df_network = data.get('network', pd.DataFrame())


class WarehouseOptimizer:
    """
    Optimize warehouse locations using clustering algorithms.
    Finds optimal locations based on order volume, geography, and delivery performance.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize warehouse optimizer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.kmeans_model = None
        self.scaler = StandardScaler()
        self.optimal_locations = None
        self.cluster_labels = None
    
    def prepare_features(self, 
                         df_orders: pd.DataFrame,
                         df_sellers: pd.DataFrame,
                         df_customers: pd.DataFrame,
                         use_volume_weight: bool = True) -> np.ndarray:
        """
        Prepare features for clustering.
        
        Args:
            df_orders: Orders dataset
            df_sellers: Sellers dataset with coordinates
            df_customers: Customers dataset with coordinates
            use_volume_weight: Whether to weight points by order volume
        
        Returns:
            Feature matrix for clustering
        """
        # Aggregate orders by seller location
        seller_orders = df_orders.groupby('seller_id').size().reset_index()
        seller_orders.columns = ['seller_id', 'order_volume']
        
        seller_data = df_sellers.merge(seller_orders, on='seller_id', how='left')
        seller_data['order_volume'] = seller_data['order_volume'].fillna(0)
        
        # Get coordinates
        coordinates = seller_data[['seller_lat', 'seller_lng']].values
        
        if use_volume_weight:
            # Repeat coordinates based on order volume (weighted clustering)
            weights = seller_data['order_volume'].values.astype(int)
            weights = np.maximum(weights, 1)  # Minimum weight of 1
            
            weighted_coords = []
            for i, (lat, lng) in enumerate(coordinates):
                weight = weights[i]
                weighted_coords.extend([[lat, lng]] * weight)
            
            features = np.array(weighted_coords)
        else:
            features = coordinates
        
        return features
    
    def find_optimal_k(self, features: np.ndarray, max_k: int = 15) -> Dict[str, Any]:
        """
        Find optimal number of clusters using elbow method and silhouette score.
        
        Args:
            features: Feature matrix
            max_k: Maximum number of clusters to test
        
        Returns:
            Dictionary with optimal k and metrics
        """
        from sklearn.metrics import silhouette_score
        
        inertias = []
        silhouette_scores = []
        
        for k in range(2, min(max_k + 1, len(features) - 1)):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)
            
            if len(np.unique(kmeans.labels_)) > 1:
                score = silhouette_score(features, kmeans.labels_)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(-1)
        
        # Find elbow point (simplified)
        if len(inertias) > 1:
            deltas = np.diff(inertias)
            deltas_2 = np.diff(deltas)
            optimal_k_elbow = np.argmax(deltas_2) + 3 if len(deltas_2) > 0 else 5
        else:
            optimal_k_elbow = 5
        
        # Find k with best silhouette score
        optimal_k_silhouette = np.argmax(silhouette_scores) + 2 if silhouette_scores else 5
        
        return {
            'elbow_k': optimal_k_elbow,
            'silhouette_k': optimal_k_silhouette,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores
        }
    
    def optimize_kmeans(self, 
                        features: np.ndarray, 
                        n_clusters: int = 5) -> pd.DataFrame:
        """
        Perform K-Means clustering to find warehouse locations.
        
        Args:
            features: Feature matrix
            n_clusters: Number of clusters (warehouses)
        
        Returns:
            DataFrame with warehouse candidate locations
        """
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        self.cluster_labels = self.kmeans_model.fit_predict(features)
        
        # Get cluster centers
        centers = self.kmeans_model.cluster_centers_
        
        # Create warehouse candidates DataFrame
        warehouse_candidates = pd.DataFrame({
            'warehouse_id': [f'WH_{i+1:02d}' for i in range(n_clusters)],
            'lat': centers[:, 0],
            'lng': centers[:, 1],
            'cluster_size': [np.sum(self.cluster_labels == i) for i in range(n_clusters)]
        })
        
        # Calculate coverage area (approximate)
        for i, center in enumerate(centers):
            cluster_points = features[self.cluster_labels == i]
            if len(cluster_points) > 0:
                distances = np.sqrt(np.sum((cluster_points - center) ** 2, axis=1))
                warehouse_candidates.loc[i, 'coverage_radius_km'] = np.percentile(distances, 90)
            else:
                warehouse_candidates.loc[i, 'coverage_radius_km'] = 500
        
        self.optimal_locations = warehouse_candidates
        return warehouse_candidates
    
    def optimize_dbscan(self, 
                        features: np.ndarray, 
                        eps: float = 0.5, 
                        min_samples: int = 5) -> pd.DataFrame:
        """
        Perform DBSCAN clustering for warehouse locations.
        DBSCAN can find arbitrary shaped clusters and identify outliers.
        
        Args:
            features: Feature matrix
            eps: Maximum distance between two samples
            min_samples: Minimum samples in neighborhood
        
        Returns:
            DataFrame with warehouse candidate locations
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(features)
        
        # Get unique clusters (excluding noise points labeled -1)
        unique_clusters = [l for l in np.unique(labels) if l != -1]
        
        warehouse_candidates = []
        for cluster_id in unique_clusters:
            cluster_points = features[labels == cluster_id]
            if len(cluster_points) > 0:
                center = cluster_points.mean(axis=0)
                warehouse_candidates.append({
                    'warehouse_id': f'WH_DB_{cluster_id+1:02d}',
                    'lat': center[0],
                    'lng': center[1],
                    'cluster_size': len(cluster_points),
                    'cluster_type': 'DBSCAN'
                })
        
        # Also consider high-density noise points as candidates
        noise_points = features[labels == -1]
        if len(noise_points) > 0:
            # Cluster noise points separately
            if len(noise_points) >= min_samples:
                noise_kmeans = KMeans(n_clusters=min(3, len(noise_points)), random_state=self.random_state)
                noise_labels = noise_kmeans.fit_predict(noise_points)
                
                for i in range(noise_labels.max() + 1):
                    cluster_points = noise_points[noise_labels == i]
                    if len(cluster_points) > 0:
                        center = cluster_points.mean(axis=0)
                        warehouse_candidates.append({
                            'warehouse_id': f'WH_NOISE_{i+1:02d}',
                            'lat': center[0],
                            'lng': center[1],
                            'cluster_size': len(cluster_points),
                            'cluster_type': 'Noise'
                        })
        
        result_df = pd.DataFrame(warehouse_candidates)
        if len(result_df) == 0:
            return pd.DataFrame()
        
        result_df['coverage_radius_km'] = 300
        return result_df
    
    def evaluate_locations(self, 
                           warehouse_candidates: pd.DataFrame,
                           df_orders: pd.DataFrame,
                           df_sellers: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate warehouse candidate locations.
        
        Args:
            warehouse_candidates: DataFrame with warehouse locations
            df_orders: Orders dataset
            df_sellers: Sellers dataset
        
        Returns:
            Dictionary with evaluation metrics
        """
        from ..features.geospatial import haversine_distance
        
        metrics = {
            'total_warehouses': len(warehouse_candidates),
            'avg_cluster_size': warehouse_candidates['cluster_size'].mean(),
            'total_points_covered': warehouse_candidates['cluster_size'].sum(),
            'warehouse_details': []
        }
        
        # Calculate potential coverage
        seller_coords = df_sellers[['seller_lat', 'seller_lng']].values
        
        for _, warehouse in warehouse_candidates.iterrows():
            distances = []
            for seller in seller_coords:
                dist = haversine_distance(
                    warehouse['lat'], warehouse['lng'],
                    seller[0], seller[1]
                )
                distances.append(dist)
            
            nearby_sellers = sum(1 for d in distances if d <= 300)
            
            metrics['warehouse_details'].append({
                'warehouse_id': warehouse['warehouse_id'],
                'nearby_sellers_300km': nearby_sellers,
                'coverage_percentage': (nearby_sellers / len(seller_coords)) * 100,
                'avg_distance_to_sellers': np.mean(distances)
            })
        
        return metrics
    
    def get_priority_routes(self, 
                            df_network: pd.DataFrame,
                            warehouse_candidates: pd.DataFrame,
                            top_n: int = 10) -> pd.DataFrame:
        """
        Identify priority routes for warehouse optimization.
        
        Args:
            df_network: Network data with routes
            warehouse_candidates: Warehouse candidate locations
            top_n: Number of top priority routes to return
        
        Returns:
            DataFrame with priority routes
        """
        df = df_network.copy()
        
        # Calculate potential benefit score
        if 'avg_delivery_days' in df.columns:
            delivery_score = (df['avg_delivery_days'] - df['avg_delivery_days'].min()) / \
                            (df['avg_delivery_days'].max() - df['avg_delivery_days'].min())
        else:
            delivery_score = 0.5
        
        if 'order_count' in df.columns:
            volume_score = df['order_count'] / df['order_count'].max()
        else:
            volume_score = 0.5
        
        if 'distance_km' in df.columns:
            distance_score = df['distance_km'] / df['distance_km'].max()
        else:
            distance_score = 0.5
        
        # Combined priority score
        df['priority_score'] = (
            0.4 * volume_score +
            0.35 * delivery_score +
            0.25 * distance_score
        ) * 100
        
        df = df.sort_values('priority_score', ascending=False)
        
        return df.head(top_n)
    
    def save_results(self, filepath: str):
        """
        Save warehouse optimization results.
        
        Args:
            filepath: Path to save results
        """
        if self.optimal_locations is not None:
            self.optimal_locations.to_parquet(filepath)
            print(f"Warehouse candidates saved to {filepath}")
        else:
            print("No optimization results to save")


def find_optimal_warehouses(df_orders: pd.DataFrame,
                            df_sellers: pd.DataFrame,
                            df_customers: pd.DataFrame,
                            n_warehouses: int = 5,
                            method: str = 'kmeans') -> pd.DataFrame:
    """
    Convenience function to find optimal warehouse locations.
    
    Args:
        df_orders: Orders dataset
        df_sellers: Sellers dataset
        df_customers: Customers dataset
        n_warehouses: Number of warehouses to find (for KMeans)
        method: 'kmeans' or 'dbscan'
    
    Returns:
        DataFrame with warehouse candidate locations
    """
    optimizer = WarehouseOptimizer()
    
    features = optimizer.prepare_features(df_orders, df_sellers, df_customers)
    
    if method == 'kmeans':
        warehouses = optimizer.optimize_kmeans(features, n_clusters=n_warehouses)
    elif method == 'dbscan':
        warehouses = optimizer.optimize_dbscan(features, eps=0.3, min_samples=10)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return warehouses


if __name__ == "__main__":
    print("WarehouseOptimizer module loaded")