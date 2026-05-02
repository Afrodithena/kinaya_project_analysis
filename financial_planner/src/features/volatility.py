"""
Volatility calculation module.
Computes rolling volatility and risk classification for stocks.
"""

import pandas as pd
import numpy as np

from src.config import RISK_THRESHOLDS, TRADING_DAYS_PER_YEAR


def calculate_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate rolling volatility and assign risk classification.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'daily_return' column
    window : int, default 20
        Rolling window size in trading days (20 days ~ 1 trading month)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - volatility_20d: rolling standard deviation of daily returns
        - volatility_annual: annualized volatility (multiplied by sqrt(252))
        - risk_level: categorical label (Low, Medium, High)
    
    Risk thresholds (from config):
        Low: volatility < 1.5%
        Medium: 1.5% <= volatility < 3.0%
        High: volatility >= 3.0%
    """
    df = df.copy()
    
    # Rolling volatility
    df["volatility_20d"] = df["daily_return"].rolling(window=window).std()
    
    # Annualized volatility (252 trading days per year)
    df["volatility_annual"] = df["volatility_20d"] * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    # Risk classification based on daily volatility
    df["risk_level"] = pd.cut(
        df["volatility_20d"],
        bins=[0, RISK_THRESHOLDS["low"], RISK_THRESHOLDS["medium"], 100],
        labels=["Low", "Medium", "High"]
    )
    
    return df