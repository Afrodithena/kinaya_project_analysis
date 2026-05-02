"""
Recommendation module for stock selection engine
"""
from .engine import (
    compute_stock_features,
    build_stock_features_dataframe,
    normalize_features,
    get_goal_weights,
    adjust_weights_for_risk,
    score_stocks,
    generate_recommendation
)