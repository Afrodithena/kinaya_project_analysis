"""
Data loading and cleaning module for LQ45 stock data.
Handles CSV loading, pickle caching, and BEI trading calendar integration.
"""

import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple, Dict, List, Optional

from src.config import TICKERS


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
    
    # ============================================
    # CARI FILE PICKLE DI LOKASI YANG BENAR
    # ============================================
    possible_paths = [
        Path(__file__).parent.parent / "data" / "all_stocks_data.pkl",  # financial_planner/data/
        Path.cwd() / "financial_planner/data/all_stocks_data.pkl",
        Path("/mount/src/kinaya_project_analysis/financial_planner/data/all_stocks_data.pkl"),
    ]
    
    pickle_path = None
    for path in possible_paths:
        if path.exists():
            pickle_path = path
            print(f"[INFO] Found pickle at: {pickle_path}")
            break
    
    # Jika pickle ditemukan, load
    if pickle_path is not None:
        try:
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
            
            # Handle berbagai kemungkinan format
            if isinstance(data, dict):
                # Langsung dictionary
                stock_tickers = [k for k in data.keys() if not k.startswith('_')]
                print(f"[INFO] Loaded {len(stock_tickers)} stocks from pickle")
                return stock_tickers, data
            
            elif isinstance(data, tuple) and len(data) == 2:
                # Format (tickers_list, data_dict)
                tickers, stocks_dict = data
                print(f"[INFO] Loaded {len(tickers)} stocks from pickle (tuple format)")
                return tickers, stocks_dict
            
            else:
                print(f"[WARN] Unknown pickle format: {type(data)}")
                
        except Exception as e:
            print(f"[ERROR] Failed to load pickle: {e}")
    
    # ============================================
    # FALLBACK: LOAD DARI CSV
    # ============================================
    print("[INFO] Pickle not found. Loading from CSV files...")
    
    # Cari folder CSV
    csv_dirs = [
        Path(__file__).parent.parent / "data" / "raw",
        Path.cwd() / "data" / "raw",
    ]
    
    csv_path = None
    for path in csv_dirs:
        if path.exists():
            csv_path = path
            break
    
    if csv_path is None:
        print("[ERROR] No CSV directory found!")
        return [], {}
    
    all_stocks_data = {}
    for ticker in TICKERS:
        file_path = csv_path / f"{ticker}.csv"
        if file_path.exists():
            try:
                df = load_stock_data(str(file_path))
                all_stocks_data[ticker] = df
            except Exception as e:
                print(f"[WARN] Error loading {ticker}: {e}")
    
    print(f"[INFO] Loaded {len(all_stocks_data)} stocks from CSV")
    return list(all_stocks_data.keys()), all_stocks_data


def add_bei_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Add BEI trading calendar columns."""
    df = df.copy()
    df["day_of_week"] = df.index.dayofweek
    df["day_name"] = df.index.day_name()
    df["is_trading_day"] = df["day_of_week"] <= 4
    return df


def filter_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only BEI trading days."""
    df = add_bei_calendar(df)
    return df[df["is_trading_day"]].copy()