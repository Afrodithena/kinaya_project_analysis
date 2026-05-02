"""
Helper utility functions for Olist Logistics Engine
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import hashlib
import json


def format_number(value: Union[int, float], decimals: int = 0) -> str:
    """
    Format large numbers with K, M, B suffixes.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
    
    Returns:
        Formatted string (e.g., '1.5K', '2.3M')
    """
    if value is None or pd.isna(value):
        return 'N/A'
    
    abs_value = abs(value)
    
    if abs_value >= 1e9:
        formatted = f"{value / 1e9:.{decimals}f}B"
    elif abs_value >= 1e6:
        formatted = f"{value / 1e6:.{decimals}f}M"
    elif abs_value >= 1e3:
        formatted = f"{value / 1e3:.{decimals}f}K"
    else:
        formatted = f"{value:.{decimals}f}"
    
    return formatted


def format_currency(value: Union[int, float], currency: str = 'BRL') -> str:
    """
    Format currency values.
    
    Args:
        value: Number to format
        currency: Currency code ('BRL', 'USD')
    
    Returns:
        Formatted currency string
    """
    if value is None or pd.isna(value):
        return 'N/A'
    
    symbol = 'R$' if currency == 'BRL' else '$'
    
    if abs(value) >= 1e6:
        return f"{symbol} {value/1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"{symbol} {value/1e3:.0f}K"
    else:
        return f"{symbol} {value:,.2f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format percentage values.
    
    Args:
        value: Number to format (0-100)
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    if value is None or pd.isna(value):
        return 'N/A'
    
    return f"{value:.{decimals}f}%"


def calculate_growth_rate(current: float, previous: float) -> float:
    """
    Calculate growth rate between two values.
    
    Args:
        current: Current value
        previous: Previous value
    
    Returns:
        Growth rate as percentage
    """
    if previous == 0 or pd.isna(previous):
        return 0
    
    return ((current - previous) / abs(previous)) * 100


def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
    """
    Safe division that handles division by zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
    
    Returns:
        Division result or default value
    """
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


def get_date_range(days_back: int, end_date: Optional[datetime] = None) -> tuple:
    """
    Get date range for filtering.
    
    Args:
        days_back: Number of days to go back
        end_date: End date (defaults to today)
    
    Returns:
        Tuple of (start_date, end_date)
    """
    if end_date is None:
        end_date = datetime.now()
    
    start_date = end_date - timedelta(days=days_back)
    
    return start_date, end_date


def detect_outliers_iqr(data: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using IQR method.
    
    Args:
        data: Series of values
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        Boolean Series where True indicates outlier
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (data < lower_bound) | (data > upper_bound)


def detect_outliers_zscore(data: pd.Series, threshold: float = 3) -> pd.Series:
    """
    Detect outliers using Z-score method.
    
    Args:
        data: Series of values
        threshold: Z-score threshold (default 3)
    
    Returns:
        Boolean Series where True indicates outlier
    """
    z_scores = np.abs((data - data.mean()) / data.std())
    return z_scores > threshold


def normalize_series(data: pd.Series, method: str = 'minmax') -> pd.Series:
    """
    Normalize a series to 0-1 range.
    
    Args:
        data: Series to normalize
        method: 'minmax' or 'zscore'
    
    Returns:
        Normalized series
    """
    if method == 'minmax':
        min_val = data.min()
        max_val = data.max()
        if min_val == max_val:
            return pd.Series([0.5] * len(data), index=data.index)
        return (data - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean = data.mean()
        std = data.std()
        if std == 0:
            return pd.Series([0] * len(data), index=data.index)
        return (data - mean) / std
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def get_top_values(df: pd.DataFrame, 
                   column: str, 
                   n: int = 10, 
                   ascending: bool = False) -> pd.DataFrame:
    """
    Get top N values from a DataFrame column.
    
    Args:
        df: DataFrame
        column: Column to sort by
        n: Number of top values
        ascending: Sort ascending or descending
    
    Returns:
        DataFrame with top N rows
    """
    return df.nlargest(n, column) if not ascending else df.nsmallest(n, column)


def create_hash_id(*args, length: int = 8) -> str:
    """
    Create a hash ID from input arguments.
    
    Args:
        *args: Values to hash
        length: Desired hash length
    
    Returns:
        Hash string
    """
    combined = ''.join(str(arg) for arg in args)
    hash_obj = hashlib.md5(combined.encode())
    return hash_obj.hexdigest()[:length]


def safe_json_loads(json_string: str, default: Any = None) -> Any:
    """
    Safely load JSON string.
    
    Args:
        json_string: JSON string to parse
        default: Default value if parsing fails
    
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return default


def aggregate_by_period(df: pd.DataFrame, 
                        date_col: str, 
                        value_col: str, 
                        period: str = 'M') -> pd.DataFrame:
    """
    Aggregate data by time period.
    
    Args:
        df: DataFrame with date column
        date_col: Name of date column
        value_col: Name of value column to aggregate
        period: 'D' (day), 'W' (week), 'M' (month), 'Q' (quarter), 'Y' (year)
    
    Returns:
        Aggregated DataFrame
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    df['period'] = df[date_col].dt.to_period(period)
    
    aggregated = df.groupby('period')[value_col].agg(['sum', 'mean', 'count']).reset_index()
    aggregated['period'] = aggregated['period'].astype(str)
    
    return aggregated


def calculate_moving_average(data: pd.Series, window: int = 7) -> pd.Series:
    """
    Calculate moving average.
    
    Args:
        data: Series of values
        window: Window size
    
    Returns:
        Moving average series
    """
    return data.rolling(window=window, min_periods=1).mean()


def get_state_list() -> List[str]:
    """
    Get list of all Brazilian state codes.
    
    Returns:
        List of state codes
    """
    from .constants import STATE_CENTROIDS
    return list(STATE_CENTROIDS.keys())


def get_region_from_state(state_code: str) -> str:
    """
    Get region name from state code.
    
    Args:
        state_code: Brazilian state code
    
    Returns:
        Region name
    """
    from .constants import STATE_TO_REGION
    return STATE_TO_REGION.get(state_code, 'Unknown')


def is_valid_state(state_code: str) -> bool:
    """
    Check if state code is valid.
    
    Args:
        state_code: State code to validate
    
    Returns:
        True if valid, False otherwise
    """
    from .constants import STATE_CENTROIDS
    return state_code in STATE_CENTROIDS


def get_state_centroid(state_code: str) -> tuple:
    """
    Get centroid coordinates for a state.
    
    Args:
        state_code: Brazilian state code
    
    Returns:
        Tuple of (latitude, longitude)
    """
    from .constants import STATE_CENTROIDS
    return STATE_CENTROIDS.get(state_code, (-15.78, -47.93))


def validate_dataframe_columns(df: pd.DataFrame, 
                                required_columns: List[str]) -> List[str]:
    """
    Validate that DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        List of missing columns (empty if all present)
    """
    missing = [col for col in required_columns if col not in df.columns]
    return missing


def fill_missing_values(df: pd.DataFrame, 
                        strategy: str = 'median') -> pd.DataFrame:
    """
    Fill missing values in DataFrame.
    
    Args:
        df: DataFrame with missing values
        strategy: 'median', 'mean', 'zero', 'forward'
    
    Returns:
        DataFrame with filled values
    """
    df = df.copy()
    
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if strategy == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif strategy == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == 'zero':
                df[col] = df[col].fillna(0)
            elif strategy == 'forward':
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    return df


def truncate_string(text: str, max_length: int = 50, suffix: str = '...') -> str:
    """
    Truncate string to maximum length.
    
    Args:
        text: String to truncate
        max_length: Maximum length
        suffix: Suffix to add for truncated strings
    
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def get_file_size_str(filepath: str) -> str:
    """
    Get human-readable file size.
    
    Args:
        filepath: Path to file
    
    Returns:
        File size string (e.g., '1.5 MB')
    """
    from pathlib import Path
    
    size_bytes = Path(filepath).stat().st_size
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} TB"


if __name__ == "__main__":
    print("Helpers module loaded")
    print(f"Test format_number: {format_number(1523456)}")
    print(f"Test format_currency: {format_currency(1523456)}")