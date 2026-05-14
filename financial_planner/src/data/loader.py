"""
Data loading and cleaning module for LQ45 stock data.
Handles CSV loading, pickle caching, and BEI trading calendar integration.
"""

import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple, Dict, List, Optional

from src.config import TICKERS, DATA_DIR


def load_stock_data(file_path: str) -> pd.DataFrame:
    """Load a single stock CSV file and prepare base structure."""
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    
    if "delisting_date" in df.columns:
        df = df.drop(columns=["delisting_date"])
    
    return df


def load_all_stocks(data_path: Optional[str] = None) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    """
    Load all LQ45 stocks from CSV files or pickle cache.
    """
    if data_path is None:
        data_path = str(DATA_DIR / "raw")
    
    pickle_path = DATA_DIR / "all_stocks_data.pkl"
    
    # DEBUG: Print paths
    print(f"[DEBUG] DATA_DIR: {DATA_DIR}")
    print(f"[DEBUG] pickle_path: {pickle_path}")
    print(f"[DEBUG] pickle_path.exists(): {pickle_path.exists()}")
    
    # Use pickle cache if available
    if pickle_path.exists():
        print(f"[DEBUG] Loading pickle from: {pickle_path}")
        with open(pickle_path, "rb") as f:
            all_stocks_data = pickle.load(f)
        
        stock_tickers = [k for k in all_stocks_data.keys() if not k.startswith('_')]
        print(f"[DEBUG] Loaded {len(stock_tickers)} stocks from pickle")
        return stock_tickers, all_stocks_data
    
    print(f"[DEBUG] Pickle not found at {pickle_path}, falling back to CSV...")

def add_bei_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add BEI (Bursa Efek Indonesia) trading calendar columns.
    
    BEI operates Monday to Friday only. Add columns:
    - day_of_week: 0=Monday to 6=Sunday
    - day_name: Monday, Tuesday, etc.
    - is_trading_day: True for Monday-Friday
    """
    df = df.copy()
    df["day_of_week"] = df.index.dayofweek
    df["day_name"] = df.index.day_name()
    df["is_trading_day"] = df["day_of_week"] <= 4
    
    return df


def filter_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe to only BEI trading days (Monday to Friday)."""
    df = add_bei_calendar(df)
    return df[df["is_trading_day"]].copy()