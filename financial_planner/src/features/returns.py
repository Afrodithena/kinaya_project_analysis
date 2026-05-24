"""
Returns calculation module for stock price data.
Computes daily and multi-period rolling returns using adjusted close prices.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple


def calculate_returns(df: pd.DataFrame, use_adjusted: bool = True) -> pd.DataFrame:
    """
    Calculate daily and rolling returns for stock price data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' or 'adjusted_close' column and datetime index
    use_adjusted : bool, default True
        Whether to use 'adjusted_close' column if available
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional return columns:
        - daily_return: daily percentage change
        - return_5d: 5-day rolling return (~1 week)
        - return_21d: 21-day rolling return (~1 trading month)
        - return_63d: 63-day rolling return (~3 trading months)
        - return_252d: 252-day rolling return (~1 trading year)
    """
    df = df.copy()
    
    # Determine which price column to use
    if use_adjusted and 'adjusted_close' in df.columns:
        price_col = 'adjusted_close'
    else:
        price_col = 'close'
    
    # Daily return (percentage)
    df["daily_return"] = df[price_col].pct_change() * 100
    
    # Multi-period rolling returns (percentage)
    df["return_5d"] = df[price_col].pct_change(5) * 100      # One week
    df["return_21d"] = df[price_col].pct_change(21) * 100    # One month
    df["return_63d"] = df[price_col].pct_change(63) * 100    # Three months
    df["return_252d"] = df[price_col].pct_change(252) * 100  # One year
    
    return df


def calculate_log_returns(df: pd.DataFrame, use_adjusted: bool = True) -> pd.DataFrame:
    """
    Calculate log returns (continuously compounded returns).
    Log returns are time-additive and useful for time series analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price column
    use_adjusted : bool, default True
        Whether to use 'adjusted_close' column if available
        
    Returns
    -------
    pd.DataFrame
        DataFrame with log return columns
    """
    df = df.copy()
    
    if use_adjusted and 'adjusted_close' in df.columns:
        price_col = 'adjusted_close'
    else:
        price_col = 'close'
    
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1)) * 100
    df["log_return_5d"] = np.log(df[price_col] / df[price_col].shift(5)) * 100
    df["log_return_21d"] = np.log(df[price_col] / df[price_col].shift(21)) * 100
    df["log_return_63d"] = np.log(df[price_col] / df[price_col].shift(63)) * 100
    
    return df


def calculate_cumulative_return(df: pd.DataFrame, use_adjusted: bool = True) -> pd.Series:
    """
    Calculate cumulative return from start of data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price column
    use_adjusted : bool, default True
        Whether to use 'adjusted_close' column if available
        
    Returns
    -------
    pd.Series
        Cumulative return series (as multiplier, e.g., 1.0 = 0 percent return)
    """
    df = df.copy()
    
    if use_adjusted and 'adjusted_close' in df.columns:
        price_col = 'adjusted_close'
    else:
        price_col = 'close'
    
    cumulative_return = df[price_col] / df[price_col].iloc[0]
    return cumulative_return


def calculate_annualized_return(df: pd.DataFrame, use_adjusted: bool = True) -> float:
    """
    Calculate annualized return from price data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price column
    use_adjusted : bool, default True
        Whether to use 'adjusted_close' column if available
        
    Returns
    -------
    float
        Annualized return percentage
    """
    df = df.copy()
    
    if use_adjusted and 'adjusted_close' in df.columns:
        price_col = 'adjusted_close'
    else:
        price_col = 'close'
    
    start_price = df[price_col].iloc[0]
    end_price = df[price_col].iloc[-1]
    
    total_return = (end_price / start_price) - 1
    
    # Calculate number of years
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25
    
    if years > 0 and total_return > -1:
        annualized_return = ((1 + total_return) ** (1 / years) - 1) * 100
    else:
        annualized_return = 0
    
    return round(annualized_return, 2)


def get_return_summary(df: pd.DataFrame, use_adjusted: bool = True) -> pd.DataFrame:
    """
    Get comprehensive return statistics for a stock.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price column and datetime index
    use_adjusted : bool, default True
        Whether to use 'adjusted_close' column if available
        
    Returns
    -------
    pd.DataFrame
        Summary statistics of returns
    """
    df = calculate_returns(df, use_adjusted=use_adjusted)
    
    return_columns = ['daily_return', 'return_5d', 'return_21d', 'return_63d', 'return_252d']
    existing_cols = [col for col in return_columns if col in df.columns]
    
    summary = []
    for col in existing_cols:
        series = df[col].dropna()
        if len(series) > 0:
            summary.append({
                'Period': col.replace('_', ' ').title(),
                'Mean Percent': round(series.mean(), 2),
                'Std Percent': round(series.std(), 2),
                'Min Percent': round(series.min(), 2),
                'Max Percent': round(series.max(), 2),
                'Positive Percent': round((series > 0).sum() / len(series) * 100, 1),
                'Negative Percent': round((series < 0).sum() / len(series) * 100, 1)
            })
    
    return pd.DataFrame(summary)


def get_rolling_return_stats(df: pd.DataFrame, window: int = 21, use_adjusted: bool = True) -> pd.DataFrame:
    """
    Get rolling return statistics over a specified window.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price column
    window : int
        Rolling window size in days
    use_adjusted : bool, default True
        Whether to use 'adjusted_close' column if available
        
    Returns
    -------
    pd.DataFrame
        Rolling return statistics
    """
    df = df.copy()
    
    if use_adjusted and 'adjusted_close' in df.columns:
        price_col = 'adjusted_close'
    else:
        price_col = 'close'
    
    rolling_returns = df[price_col].pct_change(window) * 100
    
    stats = pd.DataFrame({
        'Mean': [rolling_returns.mean()],
        'Std': [rolling_returns.std()],
        'Min': [rolling_returns.min()],
        '25th': [rolling_returns.quantile(0.25)],
        'Median': [rolling_returns.quantile(0.50)],
        '75th': [rolling_returns.quantile(0.75)],
        'Max': [rolling_returns.max()]
    })
    
    return stats


def compare_returns(stocks_data: dict, tickers: List[str], period: str = 'daily') -> pd.DataFrame:
    """
    Compare returns across multiple stocks.
    
    Parameters
    ----------
    stocks_data : dict
        Dictionary of stock dataframes
    tickers : list
        List of stock tickers to compare
    period : str
        Return period: 'daily', '5d', '21d', '63d', '252d'
        
    Returns
    -------
    pd.DataFrame
        Comparison of returns across stocks
    """
    period_map = {
        'daily': 'daily_return',
        '5d': 'return_5d',
        '21d': 'return_21d',
        '63d': 'return_63d',
        '252d': 'return_252d'
    }
    
    return_col = period_map.get(period, 'daily_return')
    
    results = []
    for ticker in tickers:
        if ticker not in stocks_data:
            continue
        
        df = calculate_returns(stocks_data[ticker])
        returns = df[return_col].dropna()
        
        if len(returns) > 0:
            results.append({
                'Ticker': ticker,
                'Mean Percent': round(returns.mean(), 2),
                'Std Percent': round(returns.std(), 2),
                'Sharpe Approx': round(returns.mean() / returns.std() if returns.std() > 0 else 0, 2),
                'Positive Days Percent': round((returns > 0).sum() / len(returns) * 100, 1),
                'Best Day Percent': round(returns.max(), 2),
                'Worst Day Percent': round(returns.min(), 2)
            })
    
    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values('Mean Percent', ascending=False)
    
    return df_result


def calculate_rolling_mean_return(df: pd.DataFrame, window: int = 20, use_adjusted: bool = True) -> pd.Series:
    """
    Calculate rolling mean of returns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price column
    window : int
        Rolling window size
    use_adjusted : bool, default True
        Whether to use 'adjusted_close' column if available
        
    Returns
    -------
    pd.Series
        Rolling mean of daily returns
    """
    df = calculate_returns(df, use_adjusted=use_adjusted)
    return df['daily_return'].rolling(window=window).mean()


def calculate_rolling_std_return(df: pd.DataFrame, window: int = 20, use_adjusted: bool = True) -> pd.Series:
    """
    Calculate rolling standard deviation of returns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price column
    window : int
        Rolling window size
    use_adjusted : bool, default True
        Whether to use 'adjusted_close' column if available
        
    Returns
    -------
    pd.Series
        Rolling standard deviation of daily returns
    """
    df = calculate_returns(df, use_adjusted=use_adjusted)
    return df['daily_return'].rolling(window=window).std()


if __name__ == "__main__":
    print("=" * 60)
    print("RETURNS CALCULATION MODULE")
    print("=" * 60)
    print("\nThis module provides functions for calculating stock returns.")
    print("\nAvailable functions:")
    print("  - calculate_returns(): Add daily and rolling returns to dataframe")
    print("  - calculate_log_returns(): Calculate log returns")
    print("  - calculate_cumulative_return(): Get cumulative return from start")
    print("  - calculate_annualized_return(): Calculate annualized return")
    print("  - get_return_summary(): Get comprehensive return statistics")
    print("  - get_rolling_return_stats(): Get rolling return statistics")
    print("  - compare_returns(): Compare returns across multiple stocks")
    print("  - calculate_rolling_mean_return(): Calculate rolling mean of returns")
    print("  - calculate_rolling_std_return(): Calculate rolling standard deviation")