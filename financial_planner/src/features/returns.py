"""
Returns calculation module for stock price data.
Computes daily and multi-period rolling returns.
"""

import pandas as pd


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily and rolling returns for stock price data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column and datetime index
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional return columns:
        - daily_return: daily percentage change
        - return_5d: 5-day rolling return
        - return_21d: 21-day rolling return (~1 trading month)
        - return_63d: 63-day rolling return (~3 trading months)
    """
    df = df.copy()
    
    # Daily return (%)
    df["daily_return"] = df["close"].pct_change() * 100
    
    # Multi-period rolling returns (%)
    df["return_5d"] = df["close"].pct_change(5) * 100
    df["return_21d"] = df["close"].pct_change(21) * 100
    df["return_63d"] = df["close"].pct_change(63) * 100
    
    return df