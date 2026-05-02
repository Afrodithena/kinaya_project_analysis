"""
Constants and configuration values for Olist Logistics Engine
"""

# State centroids (latitude, longitude) for Brazilian states
STATE_CENTROIDS = {
    # Southeast
    'SP': (-23.55, -46.63),
    'RJ': (-22.91, -43.20),
    'MG': (-19.92, -43.94),
    'ES': (-20.32, -40.34),
    
    # South
    'RS': (-30.03, -51.23),
    'PR': (-25.43, -49.27),
    'SC': (-27.60, -48.52),
    
    # Northeast
    'BA': (-12.97, -38.51),
    'PE': (-8.05, -34.90),
    'CE': (-3.73, -38.52),
    'MA': (-2.53, -44.30),
    'PB': (-7.12, -34.86),
    'RN': (-5.79, -35.21),
    'AL': (-9.66, -35.74),
    'SE': (-10.91, -37.07),
    'PI': (-5.09, -42.80),
    
    # North
    'AM': (-3.07, -60.00),
    'PA': (-1.45, -48.50),
    'RO': (-8.76, -63.90),
    'AC': (-9.97, -67.81),
    'RR': (2.82, -60.67),
    'AP': (0.03, -51.05),
    'TO': (-10.18, -48.33),
    
    # Central-West
    'GO': (-16.68, -49.25),
    'DF': (-15.80, -47.86),
    'MT': (-15.60, -56.10),
    'MS': (-20.44, -54.64),
}

# Delivery performance thresholds (in days)
DELIVERY_THRESHOLDS = {
    'fast': 10,
    'normal': 15,
    'slow': 20,
    'critical': 30
}

# Performance category labels
PERFORMANCE_CATEGORIES = ['Fast', 'Normal', 'Slow', 'Critical']

# Color mapping for routes (RGBA format)
PERFORMANCE_COLORS = {
    'Fast': [0, 255, 0, 180],        # Green
    'Normal': [255, 255, 0, 200],    # Yellow
    'Slow': [255, 165, 0, 220],      # Orange
    'Critical': [255, 0, 0, 220],    # Red
    'Default': [100, 100, 150, 150]  # Default color
}

# Node (state) visualization colors
NODE_COLORS = {
    'fill': [50, 50, 80, 120],
    'line': [255, 255, 255, 80],
    'hover': [255, 255, 255, 150]
}

# Warehouse visualization colors
WAREHOUSE_COLORS = {
    'fill': [255, 100, 0, 200],
    'line': [255, 200, 0, 255],
    'text': [255, 255, 255, 255]
}

# Map visualization defaults
MAP_DEFAULTS = {
    'default_lat': -15.78,      # Center of Brazil
    'default_lng': -47.93,
    'default_zoom': 3.5,
    'default_pitch': 40,
    'map_style': 'mapbox://styles/mapbox/dark-v11',
    'max_routes': 800,
    'warehouse_radius': 12000,
    'node_min_radius': 5000,
    'node_max_radius': 50000
}

# Animation settings
ANIMATION_SETTINGS = {
    'frames': 60,
    'dash_array': [4, 8],
    'speed_slow': 0.5,
    'speed_normal': 0.3,
    'speed_fast': 0.1
}

# Filter defaults
FILTER_DEFAULTS = {
    'min_orders': 100,
    'min_orders_range': (50, 2000),
    'performance_options': ['Fast', 'Normal', 'Slow', 'Critical']
}

# Geographic regions mapping
STATE_TO_REGION = {
    # North
    'AM': 'North', 'RR': 'North', 'AP': 'North', 'PA': 'North', 
    'TO': 'North', 'RO': 'North', 'AC': 'North',
    
    # Northeast
    'MA': 'Northeast', 'PI': 'Northeast', 'CE': 'Northeast', 
    'RN': 'Northeast', 'PB': 'Northeast', 'PE': 'Northeast',
    'AL': 'Northeast', 'SE': 'Northeast', 'BA': 'Northeast',
    
    # Central-West
    'MT': 'Central-West', 'MS': 'Central-West', 'GO': 'Central-West', 'DF': 'Central-West',
    
    # Southeast
    'SP': 'Southeast', 'RJ': 'Southeast', 'MG': 'Southeast', 'ES': 'Southeast',
    
    # South
    'PR': 'South', 'SC': 'South', 'RS': 'South'
}

# Risk score thresholds
RISK_THRESHOLDS = {
    'low': 25,
    'medium_low': 50,
    'medium_high': 75,
    'high': 100
}

# Seller segmentation thresholds
SELLER_SEGMENTS = {
    'platinum_orders_percentile': 90,
    'gold_orders_percentile': 70,
    'silver_orders_percentile': 40,
    'fast_delivery_days': 10,
    'slow_delivery_days': 20
}

# Model parameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    },
    'kmeans': {
        'n_clusters': 5,
        'random_state': 42,
        'n_init': 10
    },
    'dbscan': {
        'eps': 0.5,
        'min_samples': 5
    }
}

# Distance calculation constants
EARTH_RADIUS_KM = 6371

# Distance bins for route categorization
DISTANCE_BINS = [0, 500, 1000, 1500, 2000, 3000, 5000]
DISTANCE_BIN_LABELS = ['0-500km', '500-1000km', '1000-1500km', '1500-2000km', '2000-3000km', '3000+km']

# File paths relative to data directory
DATA_FILES = {
    'network_data': 'network_data.parquet',
    'route_features': 'route_features.parquet',
    'seller_features': 'seller_features.parquet',
    'category_features': 'category_features.parquet',
    'warehouse_candidates': 'warehouse_candidates.parquet',
    'route_priority': 'route_priority.parquet',
    'state_centroids': 'state_centroids.parquet',
    'cost_benefit_summary': 'cost_benefit_summary.parquet',
    'route_risk_model': 'route_risk_model.pkl',
    'what_if_simulation': 'what_if_simulation.parquet'
}