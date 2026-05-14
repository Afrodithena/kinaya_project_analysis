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
    
    Priority:
    1. Load from pickle file if exists (faster)
    2. Otherwise load from individual CSV files
    
    Returns:
        Tuple of (list of tickers, dictionary of stock dataframes)
    """
    
    # ============================================
    # CARI PICKLE FILE DI MULTIPLE LOCATIONS
    # ============================================
    possible_pickle_paths = [
        Path(__file__).parent.parent / "data" / "all_stocks_data.pkl",  # financial_planner/data/
        Path.cwd() / "data" / "all_stocks_data.pkl",                    # current working directory/data
        Path.cwd() / "financial_planner/data/all_stocks_data.pkl",       # explicit path
        Path("/mount/src/kinaya_project_analysis/financial_planner/data/all_stocks_data.pkl"), # Streamlit Cloud
    ]
    
    pickle_path = None
    for path in possible_pickle_paths:
        if path.exists():
            pickle_path = path
            print(f"✅ Found pickle at: {pickle_path}")
            break
    
    # Use pickle cache if available
    if pickle_path is not None and pickle_path.exists():
        try:
            with open(pickle_path, "rb") as f:
                all_stocks_data = pickle.load(f)
            
            # Handle different pickle formats
            if isinstance(all_stocks_data, dict):
                # Direct dictionary format
                stock_tickers = [k for k in all_stocks_data.keys() if not k.startswith('_')]
                print(f"✅ Loaded {len(stock_tickers)} stocks from pickle (dict format)")
                return stock_tickers, all_stocks_data
            
            elif isinstance(all_stocks_data, tuple) and len(all_stocks_data) == 2:
                # Tuple format (tickers, data_dict)
                tickers, data_dict = all_stocks_data
                if isinstance(tickers, list) and isinstance(data_dict, dict):
                    print(f"✅ Loaded {len(tickers)} stocks from pickle (tuple format)")
                    return tickers, data_dict
            
            else:
                print(f"⚠️ Unknown pickle format: {type(all_stocks_data)}")
                
        except Exception as e:
            print(f"⚠️ Error loading pickle: {e}")
    
    # ============================================
    # FALLBACK: LOAD FROM CSV
    # ============================================
    print("⚠️ Pickle not found or failed. Falling back to CSV loading...")
    
    # Tentukan folder data CSV
    if data_path is None:
        possible_csv_paths = [
            Path(__file__).parent.parent / "data" / "raw",
            Path.cwd() / "data" / "raw",
            Path.cwd() / "financial_planner/data/raw",
        ]
        
        for csv_path in possible_csv_paths:
            if csv_path.exists():
                data_path = str(csv_path)
                print(f"✅ Found CSV data at: {data_path}")
                break
        else:
            print("❌ CSV data directory not found!")
            return [], {}
    
    all_stocks_data = {}
    
    for ticker in TICKERS:
        file_path = f"{data_path}/{ticker}.csv"
        
        try:
            df = load_stock_data(file_path)
            all_stocks_data[ticker] = df
        except FileNotFoundError:
            print(f"Warning: {ticker}.csv not found, skipping")
        except Exception as e:
            print(f"Error loading {ticker}: {e}")
    
    print(f"✅ Loaded {len(all_stocks_data)} stocks from CSV")
    return list(all_stocks_data.keys()), all_stocks_data


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