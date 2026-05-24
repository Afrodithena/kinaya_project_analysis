"""
Configuration for Financial Planning Engine

Contains global constants, file paths, and parameter defaults used across modules.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = ROOT_DIR / "outputs"

# Ensure directories exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, CACHE_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# STOCK TICKERS
# ============================================================================

# LQ45 stock tickers (40 stocks with complete 2019-2025 data)
TICKERS: List[str] = [
    "ADRO", "AKRA", "AMRT", "ANTM", "ARTO", "ASII",
    "BBCA", "BBNI", "BBRI", "BBTN", "BMRI", "BRIS", "BRPT",
    "CPIN", "CTRA", "ESSA", "EXCL", "GGRM", "HRUM", "ICBP",
    "INCO", "INDF", "INKP", "ISAT", "ITMG", "JPFA", "JSMR",
    "KLBF", "MAPA", "MAPI", "MDKA", "MEDC", "PGAS", "PTBA",
    "SIDO", "SMGR", "TLKM", "TOWR", "UNTR", "UNVR"
]

# Excluded stocks (insufficient historical data)
EXCLUDED_TICKERS: List[str] = [
    "AMMN", "BUKA", "EMTK", "GOTO", "INDY", "INTP",
    "MBMA", "MTEL", "PGEO", "PTMP", "SCMA", "SRTG",
    "TBIG", "TINS", "TPIA"
]

# Focus stocks for detailed analysis
FOCUS_STOCKS: List[str] = [
    "BBRI", "BBCA", "TLKM", "BMRI", "BBNI", "ASII",
    "PGAS", "GGRM", "UNVR", "UNTR", "PTBA", "BBTN",
    "ANTM", "ADRO", "MDKA", "ARTO"
]


# ============================================================================
# DATA PERIOD FILTERS
# ============================================================================

DATE_RANGE: Dict[str, str] = {
    "start": "2019-07-01",
    "end": "2025-02-28"
}

# Period definitions for regime analysis
PERIODS: Dict[str, Tuple[str, str]] = {
    "pre_covid": ("2019-07-01", "2020-02-28"),
    "covid_crash": ("2020-03-01", "2020-06-30"),
    "post_covid": ("2020-07-01", "2025-02-28")
}


# ============================================================================
# MARKET CONVENTIONS
# ============================================================================

TRADING_DAYS_PER_YEAR: int = 252
RISK_FREE_RATE: float = 0.05  # 5 percent annual (BI 7-Day Reverse Repo Rate average)


# ============================================================================
# RISK THRESHOLDS (Daily Volatility Percentage)
# ============================================================================

RISK_THRESHOLDS: Dict[str, float] = {
    "low": 1.5,      # Below 1.5% = Low Risk
    "medium": 3.0    # 1.5% to 3.0% = Medium Risk, above 3.0% = High Risk
}

# Volatility classification labels
RISK_LABELS: Dict[str, str] = {
    "low": "Low Risk",
    "medium": "Medium Risk",
    "high": "High Risk"
}


# ============================================================================
# SIMULATION DEFAULTS
# ============================================================================

DEFAULT_SIMULATIONS: int = 10000
CRISIS_WEIGHT: float = 3.0
CRISIS_PERIOD: Tuple[str, str] = ("2020-03-01", "2020-06-30")

# Expected returns by risk profile (annual percentage)
EXPECTED_RETURNS: Dict[str, float] = {
    "Conservative": 0.08,   # 8 percent per year
    "Moderate": 0.10,       # 10 percent per year
    "Aggressive": 0.12      # 12 percent per year
}

# Expected volatility by risk profile (annual percentage)
EXPECTED_VOLATILITY: Dict[str, float] = {
    "Conservative": 0.12,   # 12 percent annual volatility
    "Moderate": 0.18,       # 18 percent annual volatility
    "Aggressive": 0.25      # 25 percent annual volatility
}


# ============================================================================
# GOAL-BASED ALLOCATIONS (Percentage)
# ============================================================================

GOAL_ALLOCATIONS: Dict[str, Dict[str, int]] = {
    "Wedding": {
        "low_risk": 80,
        "medium_risk": 20,
        "high_risk": 0,
        "expected_return": 8,
        "max_drawdown": -15,
        "time_horizon_years": "1-3"
    },
    "KPR": {
        "low_risk": 50,
        "medium_risk": 40,
        "high_risk": 10,
        "expected_return": 10,
        "max_drawdown": -25,
        "time_horizon_years": "3-5"
    },
    "Education": {
        "low_risk": 30,
        "medium_risk": 40,
        "high_risk": 30,
        "expected_return": 12,
        "max_drawdown": -35,
        "time_horizon_years": "10-18"
    }
}


# ============================================================================
# INFLATION ASSUMPTIONS (Annual Percentage)
# ============================================================================

INFLATION_RATES: Dict[str, Dict[str, float]] = {
    "general": {
        "low": 0.02,      # 2 percent
        "medium": 0.035,  # 3.5 percent
        "high": 0.05      # 5 percent
    },
    "education": {
        "low": 0.05,      # 5 percent
        "medium": 0.10,   # 10 percent
        "high": 0.15      # 15 percent
    }
}

# Default inflation rates by scenario
DEFAULT_INFLATION: Dict[str, float] = {
    "general": 0.035,     # 3.5 percent for general planning
    "education": 0.10     # 10 percent for education planning
}


# ============================================================================
# KPR HIDDEN COSTS (Percentage of house price or loan amount)
# ============================================================================

KPR_HIDDEN_COSTS: Dict[str, float] = {
    "bphtb": 0.05,           # 5 percent of house price
    "notary": 0.01,          # 1 percent of house price
    "bank_provision": 0.01   # 1 percent of loan amount
}

# Default KPR parameters
KPR_DEFAULTS: Dict[str, float] = {
    "down_payment_percent": 20,
    "loan_term_years": 15,
    "annual_interest_rate": 0.07,
    "monthly_installment": 5_000_000
}


# ============================================================================
# DIVIDEND TAX AND TRANSACTION COSTS
# ============================================================================

# Indonesian stock market transaction costs
TRANSACTION_COSTS: Dict[str, float] = {
    "broker_fee_buy": 0.0015,      # 0.15 percent for buying
    "broker_fee_sell": 0.0025,     # 0.25 percent for selling
    "vat_rate": 0.11,              # 11 percent VAT on broker fees
    "capital_gain_tax": 0.001,     # 0.1 percent final tax on capital gains
    "dividend_tax": 0.10           # 10 percent final withholding tax
}

# Minimum recommended capital for diversification
MINIMUM_RECOMMENDED_CAPITAL: int = 100_000_000  # Rp 100 million
LOT_SIZE: int = 100  # 1 lot = 100 shares


# ============================================================================
# SECTOR CLASSIFICATION
# ============================================================================

SECTOR_MAPPING: Dict[str, str] = {
    # Banking
    "BBCA": "Banking", "BBRI": "Banking", "BMRI": "Banking",
    "BBNI": "Banking", "BBTN": "Banking", "BRIS": "Banking",
    # Consumer
    "ICBP": "Consumer", "INDF": "Consumer", "UNVR": "Consumer",
    "CPIN": "Consumer", "JPFA": "Consumer", "GGRM": "Consumer",
    "SIDO": "Consumer", "KLBF": "Healthcare",
    # Energy & Mining
    "ADRO": "Energy", "ITMG": "Energy", "PTBA": "Energy",
    "MEDC": "Energy", "ESSA": "Energy", "PGAS": "Energy",
    "HRUM": "Energy", "MDKA": "Mining", "INCO": "Mining",
    "ANTM": "Mining", "BRPT": "Chemical",
    # Telecom
    "TLKM": "Telecom", "ISAT": "Telecom", "EXCL": "Telecom",
    "TOWR": "Telecom Tower",
    # Others
    "ASII": "Automotive", "UNTR": "Heavy Equipment",
    "AKRA": "Trading", "AMRT": "Retail", "MAPA": "Retail",
    "MAPI": "Retail", "CTRA": "Property", "INKP": "Pulp & Paper",
    "SMGR": "Cement", "JSMR": "Infrastructure", "ARTO": "Digital Banking"
}


# ============================================================================
# VISUALIZATION DEFAULTS
# ============================================================================

PLOT_STYLE: str = "seaborn-v0_8-whitegrid"
FIGURE_DPI: int = 150
FIGURE_FIGSIZE: Tuple[int, int] = (12, 6)

# Color palette (blue theme)
COLORS: Dict[str, str] = {
    "primary": "#2c5aa0",
    "secondary": "#4a7bc4",
    "light": "#7ba5d9",
    "pale": "#e8eff7",
    "accent": "#1e3a6b",
    "danger": "#c0392b",
    "warning": "#e67e22",
    "success": "#27ae60",
    "neutral": "#7f8c8d",
    "dark": "#2c3e50",
    "border": "#d0d7de"
}


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_ticker(ticker: str) -> bool:
    """
    Check if a ticker is in the valid LQ45 list.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
        
    Returns
    -------
    bool
        True if ticker is valid
    """
    return ticker in TICKERS


def validate_risk_profile(profile: str) -> bool:
    """
    Check if a risk profile is valid.
    
    Parameters
    ----------
    profile : str
        Risk profile name
        
    Returns
    -------
    bool
        True if profile is valid
    """
    return profile in EXPECTED_RETURNS


def validate_goal(goal: str) -> bool:
    """
    Check if a goal is valid.
    
    Parameters
    ----------
    goal : str
        Goal name (Wedding, KPR, Education)
        
    Returns
    -------
    bool
        True if goal is valid
    """
    return goal in GOAL_ALLOCATIONS


if __name__ == "__main__":
    print("=" * 60)
    print("CONFIGURATION MODULE")
    print("=" * 60)
    print(f"\nRoot Directory: {ROOT_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Number of Tickers: {len(TICKERS)}")
    print(f"Number of Excluded Tickers: {len(EXCLUDED_TICKERS)}")
    print(f"Number of Focus Stocks: {len(FOCUS_STOCKS)}")
    print(f"\nTrading Days per Year: {TRADING_DAYS_PER_YEAR}")
    print(f"Risk-Free Rate: {RISK_FREE_RATE * 100:.0f}%")
    print(f"Default Simulations: {DEFAULT_SIMULATIONS:,}")
    print(f"Crisis Weight: {CRISIS_WEIGHT}x")
    print(f"\nRisk Thresholds: Low < {RISK_THRESHOLDS['low']}%, Medium < {RISK_THRESHOLDS['medium']}%")
    print(f"\nMinimum Recommended Capital: Rp {MINIMUM_RECOMMENDED_CAPITAL:,.0f}")