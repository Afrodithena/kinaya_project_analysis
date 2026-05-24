"""
Volatility calculation module.
Computes rolling volatility, risk classification, and related metrics for stocks.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List


# Default risk thresholds (daily volatility in percentage)
DEFAULT_RISK_THRESHOLDS = {
    'low': 1.5,      # Volatility below 1.5% = Low Risk
    'medium': 3.0,   # Volatility 1.5% to 3.0% = Medium Risk
    'high': 100      # Volatility above 3.0% = High Risk
}

# Default trading days per year for annualization
DEFAULT_TRADING_DAYS = 252


def calculate_volatility(
    df: pd.DataFrame,
    window: int = 20,
    annualize: bool = True,
    trading_days: int = DEFAULT_TRADING_DAYS
) -> pd.DataFrame:
    """
    Calculate rolling volatility and assign risk classification.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'daily_return' column
    window : int, default 20
        Rolling window size in trading days (20 days = 1 trading month)
    annualize : bool, default True
        Whether to calculate annualized volatility
    trading_days : int, default 252
        Number of trading days per year for annualization
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - volatility_{window}d: rolling standard deviation of daily returns
        - volatility_annual: annualized volatility (if annualize=True)
        - risk_level: categorical label (Low, Medium, High)
    """
    df = df.copy()
    
    # Rolling volatility
    vol_col = f"volatility_{window}d"
    df[vol_col] = df["daily_return"].rolling(window=window).std()
    
    # Annualized volatility
    if annualize:
        df["volatility_annual"] = df[vol_col] * np.sqrt(trading_days)
    
    return df


def classify_risk(
    df: pd.DataFrame,
    volatility_col: str = "volatility_20d",
    thresholds: Dict[str, float] = None
) -> pd.Series:
    """
    Classify risk levels based on volatility thresholds.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with volatility column
    volatility_col : str, default "volatility_20d"
        Name of the volatility column to classify
    thresholds : dict, optional
        Custom risk thresholds (keys: 'low', 'medium')
    
    Returns
    -------
    pd.Series
        Risk level classifications (Low, Medium, High)
    """
    if thresholds is None:
        thresholds = DEFAULT_RISK_THRESHOLDS
    
    risk_levels = pd.cut(
        df[volatility_col],
        bins=[0, thresholds['low'], thresholds['medium'], thresholds['high']],
        labels=["Low", "Medium", "High"]
    )
    
    return risk_levels


def get_volatility_summary(
    df: pd.DataFrame,
    window: int = 20,
    annualize: bool = True
) -> Dict[str, float]:
    """
    Get comprehensive volatility statistics for a stock.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'daily_return' column
    window : int, default 20
        Rolling window size
    annualize : bool, default True
        Whether to calculate annualized volatility
    
    Returns
    -------
    dict
        Volatility statistics including mean, std, min, max, percentiles
    """
    df = calculate_volatility(df, window=window, annualize=annualize)
    
    vol_col = f"volatility_{window}d"
    vol_series = df[vol_col].dropna()
    
    summary = {
        "mean": round(vol_series.mean(), 2),
        "std": round(vol_series.std(), 2),
        "min": round(vol_series.min(), 2),
        "max": round(vol_series.max(), 2),
        "p25": round(vol_series.quantile(0.25), 2),
        "median": round(vol_series.quantile(0.50), 2),
        "p75": round(vol_series.quantile(0.75), 2),
        "p95": round(vol_series.quantile(0.95), 2)
    }
    
    if annualize and "volatility_annual" in df.columns:
        annual_series = df["volatility_annual"].dropna()
        summary["annual_mean"] = round(annual_series.mean(), 2)
        summary["annual_std"] = round(annual_series.std(), 2)
    
    return summary


def get_risk_distribution(
    df: pd.DataFrame,
    volatility_col: str = "volatility_20d",
    thresholds: Dict[str, float] = None
) -> Dict[str, int]:
    """
    Get distribution of risk classifications.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with volatility column
    volatility_col : str, default "volatility_20d"
        Name of the volatility column
    thresholds : dict, optional
        Custom risk thresholds
    
    Returns
    -------
    dict
        Count of days in each risk category
    """
    risk_levels = classify_risk(df, volatility_col=volatility_col, thresholds=thresholds)
    
    distribution = {
        "Low": (risk_levels == "Low").sum(),
        "Medium": (risk_levels == "Medium").sum(),
        "High": (risk_levels == "High").sum()
    }
    
    return distribution


def calculate_rolling_volatility_series(
    df: pd.DataFrame,
    windows: List[int] = [10, 20, 63, 252],
    annualize: bool = True
) -> pd.DataFrame:
    """
    Calculate rolling volatility for multiple windows simultaneously.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'daily_return' column
    windows : list, default [10, 20, 63, 252]
        List of rolling window sizes
    annualize : bool, default True
        Whether to annualize volatility
    
    Returns
    -------
    pd.DataFrame
        DataFrame with volatility columns for each window
    """
    df_result = df.copy()
    
    for window in windows:
        vol_col = f"volatility_{window}d"
        df_result[vol_col] = df_result["daily_return"].rolling(window=window).std()
        
        if annualize:
            df_result[f"volatility_{window}d_annual"] = df_result[vol_col] * np.sqrt(DEFAULT_TRADING_DAYS)
    
    return df_result


def get_average_volatility_by_period(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    window: int = 20
) -> float:
    """
    Calculate average volatility for a specific time period.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'daily_return' column
    start_date : str
        Start date of the period
    end_date : str
        End date of the period
    window : int, default 20
        Rolling window size
    
    Returns
    -------
    float
        Average volatility during the specified period
    """
    df = calculate_volatility(df, window=window, annualize=False)
    
    period_df = df.loc[start_date:end_date]
    vol_col = f"volatility_{window}d"
    vol_series = period_df[vol_col].dropna()
    
    if len(vol_series) == 0:
        return 0.0
    
    return round(vol_series.mean(), 2)


def compare_volatility(
    stocks_data: dict,
    tickers: List[str],
    window: int = 20
) -> pd.DataFrame:
    """
    Compare volatility across multiple stocks.
    
    Parameters
    ----------
    stocks_data : dict
        Dictionary of stock dataframes with 'daily_return' column
    tickers : list
        List of stock tickers to compare
    window : int, default 20
        Rolling window size
    
    Returns
    -------
    pd.DataFrame
        Comparison of volatility metrics across stocks
    """
    results = []
    
    for ticker in tickers:
        if ticker not in stocks_data:
            continue
        
        df = stocks_data[ticker].copy()
        df = calculate_volatility(df, window=window)
        
        vol_col = f"volatility_{window}d"
        vol_series = df[vol_col].dropna()
        
        if len(vol_series) > 0:
            risk_levels = classify_risk(df, volatility_col=vol_col)
            most_common_risk = risk_levels.mode().iloc[0] if len(risk_levels) > 0 else "Unknown"
            
            results.append({
                'Ticker': ticker,
                'Volatility_Mean': round(vol_series.mean(), 2),
                'Volatility_Std': round(vol_series.std(), 2),
                'Volatility_Annual_Mean': round(vol_series.mean() * np.sqrt(DEFAULT_TRADING_DAYS), 2),
                'Max_Volatility': round(vol_series.max(), 2),
                'Most_Common_Risk': most_common_risk,
                'Percent_Low': round((risk_levels == "Low").sum() / len(risk_levels) * 100, 1),
                'Percent_Medium': round((risk_levels == "Medium").sum() / len(risk_levels) * 100, 1),
                'Percent_High': round((risk_levels == "High").sum() / len(risk_levels) * 100, 1)
            })
    
    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values('Volatility_Mean', ascending=True)
    
    return df_result


def calculate_volatility_regime(
    df: pd.DataFrame,
    window: int = 20,
    lookback: int = 252
) -> pd.Series:
    """
    Identify volatility regime (normal vs elevated).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'daily_return' column
    window : int, default 20
        Rolling window for current volatility
    lookback : int, default 252
        Lookback period for baseline volatility (1 year)
    
    Returns
    -------
    pd.Series
        Volatility regime labels: 'Normal', 'Elevated', 'Extreme'
    """
    df = calculate_volatility(df, window=window, annualize=False)
    
    vol_col = f"volatility_{window}d"
    current_vol = df[vol_col]
    
    # Calculate baseline (median volatility over lookback period)
    baseline_vol = current_vol.rolling(window=lookback).median()
    
    # Define regimes based on multiples of baseline
    regime = pd.Series(index=df.index, dtype='object')
    
    regime[current_vol <= baseline_vol * 1.5] = 'Normal'
    regime[(current_vol > baseline_vol * 1.5) & (current_vol <= baseline_vol * 2.5)] = 'Elevated'
    regime[current_vol > baseline_vol * 2.5] = 'Extreme'
    
    return regime


def get_risk_level_from_volatility(
    volatility: float,
    thresholds: Dict[str, float] = None
) -> str:
    """
    Get risk level from a single volatility value.
    
    Parameters
    ----------
    volatility : float
        Daily volatility percentage
    thresholds : dict, optional
        Custom risk thresholds
    
    Returns
    -------
    str
        Risk level: 'Low', 'Medium', or 'High'
    """
    if thresholds is None:
        thresholds = DEFAULT_RISK_THRESHOLDS
    
    if volatility < thresholds['low']:
        return 'Low'
    elif volatility < thresholds['medium']:
        return 'Medium'
    else:
        return 'High'


if __name__ == "__main__":
    print("=" * 60)
    print("VOLATILITY CALCULATION MODULE")
    print("=" * 60)
    print("\nThis module provides volatility calculation functions.")
    print("\nAvailable functions:")
    print("  - calculate_volatility(): Add rolling volatility columns")
    print("  - classify_risk(): Classify risk levels based on volatility")
    print("  - get_volatility_summary(): Get volatility statistics")
    print("  - get_risk_distribution(): Get distribution of risk classifications")
    print("  - calculate_rolling_volatility_series(): Multi-window volatility")
    print("  - get_average_volatility_by_period(): Period-specific volatility")
    print("  - compare_volatility(): Compare volatility across stocks")
    print("  - calculate_volatility_regime(): Identify volatility regime")
    print("  - get_risk_level_from_volatility(): Quick risk level lookup")