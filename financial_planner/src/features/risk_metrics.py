"""
Risk metrics calculation module.
Computes Value at Risk (VaR), Expected Shortfall, drawdown, and recovery time.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def calculate_var_es(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate Value at Risk (VaR) and Expected Shortfall (CVaR).
    
    Parameters
    ----------
    returns : pd.Series
        Series of daily returns in percentage
    confidence_level : float, default 0.95
        Confidence level for VaR (0.95 for 95%, 0.99 for 99%)
    
    Returns
    -------
    Tuple[float, float]
        (VaR, Expected Shortfall) in percentage
    """
    returns = returns.dropna()
    var = returns.quantile(1 - confidence_level)
    cvar = returns[returns <= var].mean()
    
    return var, cvar


def calculate_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate drawdown and maximum drawdown from peak to trough.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - cummax: cumulative maximum price
        - drawdown: current decline from peak (%)
        - max_drawdown: historical worst drawdown (%)
    """
    df = df.copy()
    df["cummax"] = df["close"].cummax()
    df["drawdown"] = (df["close"] - df["cummax"]) / df["cummax"] * 100
    df["max_drawdown"] = df["drawdown"].cummin()
    
    return df


def calculate_recovery_time(
    df: pd.DataFrame,
    crisis_start: str = "2020-03-01",
    crisis_end: str = "2020-06-30"
    ) -> dict:
    """
    Calculate recovery time and drop magnitude from COVID-19 crisis.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    crisis_start : str
        Start date of crisis period
    crisis_end : str
        End date of crisis period
    
    Returns
    -------
    dict
        Dictionary containing:
        - max_drop_%: maximum decline percentage during crisis
        - recovery_days: days to recover to pre-crisis price
        - recovered: boolean indicating if recovery occurred
    """
    df = df.copy()
    df = df[df.index >= "2019-01-01"]
    
    # Price just before crisis
    pre_crisis_price = df.loc[:crisis_start, "close"].iloc[-1]
    
    # Find trough during crisis
    crisis_df = df.loc[crisis_start:crisis_end]
    trough_price = crisis_df["close"].min()
    trough_date = crisis_df["close"].idxmin()
    max_drop = ((trough_price - pre_crisis_price) / pre_crisis_price) * 100
    
    # Find recovery date
    post_crisis_df = df.loc[crisis_end:]
    recovery_df = post_crisis_df[post_crisis_df["close"] >= pre_crisis_price]
    
    if len(recovery_df) > 0:
        recovery_date = recovery_df.index[0]
        recovery_days = (recovery_date - trough_date).days
        recovered = True
    else:
        recovery_days = None
        recovered = False
    
    return {
        "max_drop_%": round(max_drop, 2),
        "recovery_days": recovery_days,
        "recovered": recovered
    }