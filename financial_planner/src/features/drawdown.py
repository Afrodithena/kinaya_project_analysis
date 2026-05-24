"""
Drawdown calculation module.
"""

import pandas as pd
import numpy as np


def calculate_drawdown(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """
    Calculate drawdown and maximum drawdown from peak to trough.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price column
    price_col : str, default 'close'
        Name of the price column to use
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - cummax: cumulative maximum price
        - drawdown: current decline from peak (%)
        - max_drawdown: historical worst drawdown (%)
    """
    df = df.copy()
    df["cummax"] = df[price_col].cummax()
    df["drawdown"] = (df[price_col] - df["cummax"]) / df["cummax"] * 100
    df["max_drawdown"] = df["drawdown"].cummin()
    
    return df


def get_max_drawdown(df: pd.DataFrame, price_col: str = 'close') -> float:
    """
    Get the maximum drawdown value.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price column
    price_col : str, default 'close'
        Name of the price column to use
    
    Returns
    -------
    float
        Maximum drawdown percentage
    """
    df = calculate_drawdown(df, price_col=price_col)
    return round(df['max_drawdown'].min(), 2)


def get_current_drawdown(df: pd.DataFrame, price_col: str = 'close') -> float:
    """
    Get the current drawdown from all-time high.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price column
    price_col : str, default 'close'
        Name of the price column to use
    
    Returns
    -------
    float
        Current drawdown percentage
    """
    df = calculate_drawdown(df, price_col=price_col)
    return round(df['drawdown'].iloc[-1], 2)