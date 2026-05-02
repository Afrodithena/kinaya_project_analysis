"""
Configuration for Financial Planning Engine

Contains global constants, file paths, and parameter defaults used across modules.
"""

from pathlib import Path

# Path configuration
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"

# LQ45 stock tickers (40 stocks with complete 2019-2025 data)
TICKERS = [
    "ADRO", "AKRA", "AMRT", "ANTM", "ARTO", "ASII",
    "BBCA", "BBNI", "BBRI", "BBTN", "BMRI", "BRIS", "BRPT",
    "CPIN", "CTRA", "ESSA", "EXCL", "GGRM", "HRUM", "ICBP",
    "INCO", "INDF", "INKP", "ISAT", "ITMG", "JPFA", "JSMR",
    "KLBF", "MAPA", "MAPI", "MDKA", "MEDC", "PGAS", "PTBA",
    "SIDO", "SMGR", "TLKM", "TOWR", "UNTR", "UNVR"
]

# Data period filters
DATE_RANGE = {
    "start": "2019-07-01",
    "end": "2025-02-28"
}

# Market conventions
TRADING_DAYS_PER_YEAR = 252

# Risk thresholds for daily volatility (%)
RISK_THRESHOLDS = {
    "low": 1.5,
    "medium": 3.0
}

# Simulation defaults
DEFAULT_SIMULATIONS = 10000
CRISIS_WEIGHT = 3.0
CRISIS_PERIOD = ("2020-03-01", "2020-06-30")

# Inflation assumptions
INFLATION_RATES = {
    "general": {"low": 2.0, "medium": 3.5, "high": 5.0},
    "education": {"low": 5.0, "medium": 10.0, "high": 15.0}
}

# KPR hidden costs (as percentage of house price or loan amount)
KPR_HIDDEN_COSTS = {
    "bphtb": 0.05,           # 5% of house price
    "notary": 0.01,          # 1% of house price
    "bank_provision": 0.01   # 1% of loan amount
}