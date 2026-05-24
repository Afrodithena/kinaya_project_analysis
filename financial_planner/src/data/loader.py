"""
Data loading and cleaning module for LQ45 stock data.
Handles CSV loading, pickle caching, and BEI trading calendar integration.
"""

import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from datetime import datetime

# Default tickers if config not available
DEFAULT_TICKERS = [
    "ADRO", "AKRA", "AMRT", "ANTM", "ARTO", "ASII", "BBCA", "BBNI", "BBRI",
    "BBTN", "BMRI", "BRIS", "BRPT", "CPIN", "CTRA", "ESSA", "EXCL", "GGRM",
    "HRUM", "ICBP", "INCO", "INDF", "INKP", "ISAT", "ITMG", "JPFA", "JSMR",
    "KLBF", "MAPA", "MAPI", "MDKA", "MEDC", "PGAS", "PTBA", "SIDO", "SMGR",
    "TLKM", "TOWR", "UNTR", "UNVR"
]


def load_stock_data(file_path: str) -> pd.DataFrame:
    """
    Load a single stock CSV file and prepare base structure.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file
    
    Returns
    -------
    pd.DataFrame
        DataFrame with date as index, delisting_date column removed
    """
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    
    # Remove useless columns
    if "delisting_date" in df.columns:
        df = df.drop(columns=["delisting_date"])
    
    return df


def load_all_stocks(
    data_path: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    use_cache: bool = True
) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    """
    Load all LQ45 stocks from CSV files or pickle cache.
    
    Parameters
    ----------
    data_path : str, optional
        Path to data directory. If None, auto-detect.
    tickers : list, optional
        List of ticker symbols to load. If None, use DEFAULT_TICKERS.
    use_cache : bool, default True
        Whether to use pickle cache for faster loading
    
    Returns
    -------
    tuple
        (list of ticker symbols, dictionary of stock dataframes)
    """
    
    if tickers is None:
        tickers = DEFAULT_TICKERS
    
    possible_cache_paths = [
        Path(__file__).parent.parent / "data" / "all_stocks_data.pkl",
        Path.cwd() / "financial_planner" / "data" / "all_stocks_data.pkl",
        Path.cwd() / "data" / "all_stocks_data.pkl",
        Path("/mount/src/kinaya_project_analysis/financial_planner/data/all_stocks_data.pkl"),
    ]
    
    # Try to load from cache if enabled
    if use_cache:
        for cache_path in possible_cache_paths:
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        cached_data = pickle.load(f)
                    
                    # Handle different cache formats
                    if isinstance(cached_data, dict):
                        # Direct dictionary format
                        loaded_tickers = [k for k in cached_data.keys() if not k.startswith('_')]
                        print(f"Loaded {len(loaded_tickers)} stocks from cache: {cache_path}")
                        return loaded_tickers, cached_data
                    
                    elif isinstance(cached_data, tuple) and len(cached_data) == 2:
                        # (tickers_list, data_dict) format
                        tickers_list, stocks_dict = cached_data
                        print(f"Loaded {len(tickers_list)} stocks from cache: {cache_path}")
                        return tickers_list, stocks_dict
                    
                    else:
                        print(f"Unknown cache format in {cache_path}, skipping...")
                        
                except Exception as e:
                    print(f"Failed to load cache from {cache_path}: {e}")
    
    # Find CSV directory
    possible_csv_paths = [
        Path(__file__).parent.parent / "data" / "raw",
        Path(__file__).parent.parent / "data",
        Path.cwd() / "financial_planner" / "data" / "raw",
        Path.cwd() / "data" / "raw",
        Path.cwd() / "data",
    ]
    
    if data_path:
        possible_csv_paths.insert(0, Path(data_path))
    
    csv_path = None
    for path in possible_csv_paths:
        if path.exists():
            csv_path = path
            break
    
    if csv_path is None:
        print("No data directory found. Please check your data path.")
        return [], {}
    
    # Load all stocks from CSV
    print(f"Loading stocks from CSV directory: {csv_path}")
    all_stocks_data = {}
    loaded_count = 0
    
    for ticker in tickers:
        file_path = csv_path / f"{ticker}.csv"
        if file_path.exists():
            try:
                df = load_stock_data(str(file_path))
                all_stocks_data[ticker] = df
                loaded_count += 1
            except Exception as e:
                print(f"Error loading {ticker}: {e}")
    
    print(f"Loaded {loaded_count} out of {len(tickers)} stocks from CSV")
    
    # Optionally save to cache for faster future loads
    if use_cache and loaded_count > 0:
        cache_dir = Path(__file__).parent.parent / "data"
        cache_dir.mkdir(exist_ok=True)
        cache_path = cache_dir / "all_stocks_data.pkl"
        
        try:
            with open(cache_path, "wb") as f:
                pickle.dump((list(all_stocks_data.keys()), all_stocks_data), f)
            print(f"Saved cache to {cache_path}")
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    return list(all_stocks_data.keys()), all_stocks_data


def save_to_cache(stocks_data: Dict[str, pd.DataFrame], cache_path: Optional[Path] = None) -> bool:
    """
    Save stock data to pickle cache.
    
    Parameters
    ----------
    stocks_data : dict
        Dictionary of stock dataframes
    cache_path : Path, optional
        Path to save cache file. If None, uses default location.
    
    Returns
    -------
    bool
        True if save successful, False otherwise
    """
    if cache_path is None:
        cache_path = Path(__file__).parent.parent / "data" / "all_stocks_data.pkl"
    
    cache_path.parent.mkdir(exist_ok=True)
    
    try:
        with open(cache_path, "wb") as f:
            pickle.dump((list(stocks_data.keys()), stocks_data), f)
        print(f"Saved {len(stocks_data)} stocks to {cache_path}")
        return True
    except Exception as e:
        print(f"Failed to save cache: {e}")
        return False


def add_bei_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add BEI trading calendar columns.
    
    BEI operates Monday to Friday only.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns: day_of_week, day_name, is_trading_day
    """
    df = df.copy()
    df["day_of_week"] = df.index.dayofweek
    df["day_name"] = df.index.day_name()
    df["is_trading_day"] = df["day_of_week"] <= 4  # Monday=0 to Friday=4
    
    return df


def filter_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to only BEI trading days (Monday to Friday).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only trading days
    """
    df = add_bei_calendar(df)
    return df[df["is_trading_day"]].copy()


def add_month_end_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add month-end and quarter-end flags for calendar effect analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns: is_month_end, is_quarter_end
    """
    df = df.copy()
    
    # Month-end detection
    df["year_month"] = df.index.to_period("M")
    df["is_month_end"] = False
    
    for period in df["year_month"].unique():
        month_data = df[df["year_month"] == period]
        if len(month_data) > 0:
            last_day = month_data[month_data["is_trading_day"]].index[-1]
            df.loc[last_day, "is_month_end"] = True
    
    # Quarter-end detection
    df["year_quarter"] = df.index.to_period("Q")
    df["is_quarter_end"] = False
    
    for period in df["year_quarter"].unique():
        quarter_data = df[df["year_quarter"] == period]
        if len(quarter_data) > 0:
            last_day = quarter_data[quarter_data["is_trading_day"]].index[-1]
            df.loc[last_day, "is_quarter_end"] = True
    
    # Drop temporary columns
    df = df.drop(["year_month", "year_quarter"], axis=1)
    
    return df


def calculate_trading_days_per_year(df: pd.DataFrame) -> Dict[int, int]:
    """
    Calculate number of trading days per year.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index and is_trading_day column
    
    Returns
    -------
    dict
        Dictionary with year as key and trading days count as value
    """
    trading_days = df[df["is_trading_day"]].groupby(df.index.year).size()
    return trading_days.to_dict()


def get_trading_calendar_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary of BEI trading calendar statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index and trading calendar columns
    
    Returns
    -------
    pd.DataFrame
        Summary table of calendar statistics
    """
    df = add_bei_calendar(df)
    
    trading_days = calculate_trading_days_per_year(df)
    
    summary = pd.DataFrame([
        {
            "Metric": "Average trading days per year",
            "Value": f"{sum(trading_days.values()) // len(trading_days):.0f} days",
            "Note": "252 days is standard for volatility annualization"
        },
        {
            "Metric": "Monday average return",
            "Value": f"{df[df['day_name'] == 'Monday']['daily_return'].mean():.3f}%" if 'daily_return' in df.columns else "N/A",
            "Note": "Negative but small effect"
        },
        {
            "Metric": "Friday average return",
            "Value": f"{df[df['day_name'] == 'Friday']['daily_return'].mean():.3f}%" if 'daily_return' in df.columns else "N/A",
            "Note": "Positive, higher than Monday"
        },
        {
            "Metric": "Weekend trading days",
            "Value": "0",
            "Note": "BEI operates Monday to Friday only"
        }
    ])
    
    return summary


# For backward compatibility with existing imports
def add_crisis_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add crisis period flags for bootstrap weighting.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
    
    Returns
    -------
    pd.DataFrame
        DataFrame with is_crisis and bootstrap_weight columns
    """
    df = df.copy()
    
    crisis_start = '2020-03-01'
    crisis_end = '2020-06-30'
    
    df['is_crisis'] = (df.index >= crisis_start) & (df.index <= crisis_end)
    df['bootstrap_weight'] = 1.0
    df.loc[df['is_crisis'], 'bootstrap_weight'] = 3.0
    
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("DATA LOADER TEST")
    print("=" * 60)
    
    tickers, stocks = load_all_stocks(use_cache=True)
    
    print(f"\nLoaded {len(tickers)} stocks")
    print(f"First 5 tickers: {tickers[:5]}")
    
    if stocks:
        sample_stock = list(stocks.keys())[0]
        sample_df = stocks[sample_stock]
        print(f"\nSample data for {sample_stock}:")
        print(f"  Shape: {sample_df.shape}")
        print(f"  Date range: {sample_df.index.min()} to {sample_df.index.max()}")
        print(f"  Columns: {list(sample_df.columns)}")