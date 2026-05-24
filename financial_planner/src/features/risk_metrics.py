"""
Risk metrics calculation module.
Computes Value at Risk (VaR), Expected Shortfall, drawdown, recovery time,
volatility, and other risk-related metrics.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List


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
        Confidence level for VaR (0.95 for 95 percent, 0.99 for 99 percent)
    
    Returns
    -------
    Tuple[float, float]
        (VaR, Expected Shortfall) in percentage
    """
    returns = returns.dropna()
    
    if len(returns) == 0:
        return 0.0, 0.0
    
    var = returns.quantile(1 - confidence_level)
    cvar = returns[returns <= var].mean()
    
    return round(var, 2), round(cvar, 2)


def calculate_multilevel_var_es(returns: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Calculate VaR and ES at multiple confidence levels.
    
    Parameters
    ----------
    returns : pd.Series
        Series of daily returns in percentage
    
    Returns
    -------
    dict
        Dictionary with confidence levels as keys and VaR/ES as values
    """
    results = {}
    
    for conf in [0.90, 0.95, 0.99]:
        var, cvar = calculate_var_es(returns, confidence_level=conf)
        results[f"{int(conf*100)} Percent"] = {
            'VaR': var,
            'Expected Shortfall': cvar
        }
    
    return results


def calculate_drawdown(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """
    Calculate drawdown and maximum drawdown from peak to trough.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price column
    price_col : str, default 'close'
        Name of the price column to use ('close' or 'adjusted_close')
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - cummax: cumulative maximum price
        - drawdown: current decline from peak (percentage)
        - max_drawdown: historical worst drawdown (percentage)
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


def calculate_recovery_time(
    df: pd.DataFrame,
    price_col: str = 'close',
    crisis_start: str = "2020-03-01",
    crisis_end: str = "2020-06-30",
    pre_crisis_days: int = 30
) -> Dict:
    """
    Calculate recovery time and drop magnitude from crisis period.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price column
    price_col : str, default 'close'
        Name of the price column to use
    crisis_start : str, default "2020-03-01"
        Start date of crisis period
    crisis_end : str, default "2020-06-30"
        End date of crisis period
    pre_crisis_days : int, default 30
        Number of days before crisis to consider for baseline
    
    Returns
    -------
    dict
        Dictionary containing:
        - pre_crisis_price: price before crisis
        - trough_price: lowest price during crisis
        - trough_date: date of trough
        - max_drop_percent: maximum decline percentage
        - recovery_days: days to recover (None if not recovered)
        - recovered: boolean indicating if recovery occurred
        - resilience_score: score from 0 to 100 (higher is more resilient)
        - risk_category: classification based on resilience score
    """
    df = df.copy()
    df = df[df.index >= "2019-01-01"]
    
    # Get pre-crisis baseline (highest in the pre_crisis_days period)
    pre_crisis_df = df.loc[:crisis_start].tail(pre_crisis_days)
    if len(pre_crisis_df) == 0:
        pre_crisis_price = df[price_col].iloc[0]
    else:
        pre_crisis_price = pre_crisis_df[price_col].max()
    
    # Find trough during crisis
    crisis_df = df.loc[crisis_start:crisis_end]
    if len(crisis_df) == 0:
        return {
            "error": "No data available for crisis period",
            "max_drop_percent": 0,
            "recovery_days": None,
            "recovered": False,
            "resilience_score": 0,
            "risk_category": "Unknown"
        }
    
    trough_price = crisis_df[price_col].min()
    trough_date = crisis_df[price_col].idxmin()
    max_drop = ((trough_price - pre_crisis_price) / pre_crisis_price) * 100
    
    # Find recovery date
    post_crisis_df = df.loc[crisis_end:]
    recovery_df = post_crisis_df[post_crisis_df[price_col] >= pre_crisis_price]
    
    if len(recovery_df) > 0:
        recovery_date = recovery_df.index[0]
        recovery_days = (recovery_date - trough_date).days
        recovered = True
    else:
        recovery_days = None
        recovered = False
    
    # Calculate resilience score (0 to 100)
    # Higher score means more resilient (fast recovery, smaller drop)
    drop_factor = max(0, 100 + max_drop)  # -80 percent -> 20, -20 percent -> 80
    if recovered and recovery_days is not None:
        speed_factor = max(0, 100 - (recovery_days / 10))  # 10 days -> 99, 500 days -> 50
        resilience_score = (drop_factor * 0.3) + (speed_factor * 0.7)
    else:
        resilience_score = drop_factor * 0.5
    
    resilience_score = min(100, max(0, round(resilience_score, 1)))
    
    # Determine risk category
    if resilience_score > 75:
        risk_category = "Bulletproof"
    elif resilience_score > 40:
        risk_category = "Average"
    else:
        risk_category = "Fragile"
    
    return {
        "pre_crisis_price": round(pre_crisis_price, 2),
        "trough_price": round(trough_price, 2),
        "trough_date": trough_date.strftime("%Y-%m-%d"),
        "max_drop_percent": round(max_drop, 2),
        "recovery_days": recovery_days,
        "recovered": recovered,
        "resilience_score": resilience_score,
        "risk_category": risk_category
    }


def calculate_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate rolling volatility.
    
    Parameters
    ----------
    returns : pd.Series
        Series of daily returns in percentage
    window : int, default 20
        Rolling window size in days
    annualize : bool, default True
        Whether to annualize volatility (multiply by sqrt of 252)
    
    Returns
    -------
    pd.Series
        Rolling volatility series
    """
    rolling_std = returns.rolling(window=window).std()
    
    if annualize:
        rolling_std = rolling_std * np.sqrt(252)
    
    return rolling_std


def get_annualized_volatility(returns: pd.Series) -> float:
    """
    Calculate annualized volatility from daily returns.
    
    Parameters
    ----------
    returns : pd.Series
        Series of daily returns in percentage
    
    Returns
    -------
    float
        Annualized volatility percentage
    """
    daily_std = returns.std()
    annual_vol = daily_std * np.sqrt(252)
    return round(annual_vol, 2)


def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calculate beta (market sensitivity) of a stock.
    
    Parameters
    ----------
    stock_returns : pd.Series
        Daily returns of the stock
    market_returns : pd.Series
        Daily returns of the market index (e.g., LQ45)
    
    Returns
    -------
    float
        Beta value (1.0 = moves with market, greater than 1 = more volatile than market)
    """
    aligned = pd.DataFrame({
        'stock': stock_returns,
        'market': market_returns
    }).dropna()
    
    if len(aligned) < 2:
        return 1.0
    
    covariance = aligned['stock'].cov(aligned['market'])
    market_variance = aligned['market'].var()
    
    if market_variance == 0:
        return 1.0
    
    beta = covariance / market_variance
    return round(beta, 2)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
    annualize: bool = True
) -> float:
    """
    Calculate Sharpe ratio (risk-adjusted return).
    
    Parameters
    ----------
    returns : pd.Series
        Series of daily returns in percentage
    risk_free_rate : float, default 0.05
        Annual risk-free rate (e.g., 0.05 for 5 percent)
    annualize : bool, default True
        Whether to annualize the ratio
    
    Returns
    -------
    float
        Sharpe ratio
    """
    returns = returns.dropna()
    
    if len(returns) == 0:
        return 0.0
    
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    excess_returns = returns / 100 - daily_rf
    
    if annualize:
        annual_return = excess_returns.mean() * 252
        annual_vol = excess_returns.std() * np.sqrt(252)
    else:
        annual_return = excess_returns.mean()
        annual_vol = excess_returns.std()
    
    if annual_vol == 0:
        return 0.0
    
    sharpe = annual_return / annual_vol
    return round(sharpe, 3)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
    annualize: bool = True
) -> float:
    """
    Calculate Sortino ratio (downside risk-adjusted return).
    Only penalizes negative returns (downside volatility).
    
    Parameters
    ----------
    returns : pd.Series
        Series of daily returns in percentage
    risk_free_rate : float, default 0.05
        Annual risk-free rate (e.g., 0.05 for 5 percent)
    annualize : bool, default True
        Whether to annualize the ratio
    
    Returns
    -------
    float
        Sortino ratio
    """
    returns = returns.dropna()
    
    if len(returns) == 0:
        return 0.0
    
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    excess_returns = returns / 100 - daily_rf
    
    # Calculate downside deviation (only negative returns)
    negative_returns = excess_returns[excess_returns < 0]
    
    if annualize:
        annual_return = excess_returns.mean() * 252
        downside_dev = negative_returns.std() * np.sqrt(252)
    else:
        annual_return = excess_returns.mean()
        downside_dev = negative_returns.std()
    
    if downside_dev == 0:
        return 0.0
    
    sortino = annual_return / downside_dev
    return round(sortino, 3)


def calculate_max_drawdown_duration(df: pd.DataFrame, price_col: str = 'close') -> int:
    """
    Calculate the duration (in days) of the maximum drawdown period.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price column
    price_col : str, default 'close'
        Name of the price column to use
    
    Returns
    -------
    int
        Number of days from peak to trough during max drawdown
    """
    df = calculate_drawdown(df, price_col=price_col)
    
    # Find the start of max drawdown (peak before trough)
    max_dd_idx = df['max_drawdown'].idxmin()
    
    # Find the peak that led to this trough
    pre_dd_df = df.loc[:max_dd_idx]
    peak_idx = pre_dd_df[pre_dd_df['cummax'] == pre_dd_df['cummax'].max()].index[0]
    
    duration = (max_dd_idx - peak_idx).days
    return duration


def get_risk_summary(returns: pd.Series, df: pd.DataFrame = None) -> Dict:
    """
    Get comprehensive risk summary for a stock.
    
    Parameters
    ----------
    returns : pd.Series
        Series of daily returns
    df : pd.DataFrame, optional
        DataFrame with price column for drawdown calculation
    
    Returns
    -------
    dict
        Comprehensive risk metrics
    """
    var_95, cvar_95 = calculate_var_es(returns, confidence_level=0.95)
    var_99, cvar_99 = calculate_var_es(returns, confidence_level=0.99)
    
    summary = {
        'volatility_annual': get_annualized_volatility(returns),
        'volatility_daily': round(returns.std(), 2),
        'var_95': var_95,
        'cvar_95': cvar_95,
        'var_99': var_99,
        'cvar_99': cvar_99,
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'sortino_ratio': calculate_sortino_ratio(returns),
        'max_daily_gain': round(returns.max(), 2),
        'max_daily_loss': round(returns.min(), 2)
    }
    
    if df is not None:
        summary['max_drawdown'] = get_max_drawdown(df)
        summary['current_drawdown'] = get_current_drawdown(df)
    
    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("RISK METRICS CALCULATION MODULE")
    print("=" * 60)
    print("\nThis module provides risk calculation functions.")
    print("\nAvailable functions:")
    print("  - calculate_var_es(): VaR and Expected Shortfall")
    print("  - calculate_multilevel_var_es(): VaR at multiple confidence levels")
    print("  - calculate_drawdown(): Drawdown analysis")
    print("  - get_max_drawdown(): Maximum drawdown value")
    print("  - get_current_drawdown(): Current drawdown from peak")
    print("  - calculate_recovery_time(): Crisis recovery analysis")
    print("  - calculate_volatility(): Rolling volatility")
    print("  - get_annualized_volatility(): Annualized volatility")
    print("  - calculate_beta(): Market beta")
    print("  - calculate_sharpe_ratio(): Sharpe ratio")
    print("  - calculate_sortino_ratio(): Sortino ratio")
    print("  - calculate_max_drawdown_duration(): Duration of max drawdown")
    print("  - get_risk_summary(): Comprehensive risk summary")