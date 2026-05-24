"""
Constants and static data mappings for LQ45 stocks.

Contains:
- SECTOR_MAP: Stock ticker to sector classification
- RISK_THRESHOLDS: Volatility thresholds for risk classification
- RISK_ALLOCATIONS: Portfolio allocation by risk profile
- EXPECTED_RETURNS: Expected annual return by risk profile
- DIVIDEND_DATA: Historical dividend per share (IDR) for 2019-2024
- TRANSACTION_COSTS: Broker fees, taxes, and other costs
"""

# ============================================================================
# SECTOR CLASSIFICATION
# ============================================================================

SECTOR_MAP = {
    # Banking
    "BBCA": "Banking", "BBRI": "Banking", "BMRI": "Banking", "BBNI": "Banking", 
    "BBTN": "Banking", "BRIS": "Banking",
    # Consumer
    "ICBP": "Consumer", "INDF": "Consumer", "UNVR": "Consumer", "CPIN": "Consumer",
    "JPFA": "Consumer", "KLBF": "Consumer", "MAPI": "Consumer", "AMRT": "Consumer",
    "GGRM": "Consumer", "SIDO": "Consumer",
    # Energy
    "ADRO": "Energy", "ITMG": "Energy", "PTBA": "Energy", "MEDC": "Energy",
    "PGAS": "Energy", "PGEO": "Energy", "ESSA": "Energy", "HRUM": "Energy",
    # Mining
    "ANTM": "Mining", "INCO": "Mining", "MDKA": "Mining", "BRPT": "Chemical",
    # Property
    "CTRA": "Property",
    # Telecommunications
    "TLKM": "Telecom", "EXCL": "Telecom", "ISAT": "Telecom", "TOWR": "Telecom Tower",
    "MTEL": "Telecom",
    # Automotive
    "ASII": "Automotive", "UNTR": "Heavy Equipment",
    # Technology
    "ARTO": "Digital Banking", "GOTO": "Technology",
    # Other sectors
    "INKP": "Pulp & Paper", "SMGR": "Cement", "AKRA": "Trading", "JSMR": "Infrastructure",
    "MAPA": "Retail", "ACES": "Retail"
}

# ============================================================================
# RISK THRESHOLDS (Daily volatility percentage)
# ============================================================================

RISK_THRESHOLDS = {
    "low": 1.5,      # Below 1.5% = Low Risk
    "medium": 3.0    # 1.5% to 3.0% = Medium Risk, above 3.0% = High Risk
}

# ============================================================================
# RISK LABELS
# ============================================================================

RISK_LABELS = {
    "low": "Low Risk",
    "medium": "Medium Risk",
    "high": "High Risk"
}

# ============================================================================
# PORTFOLIO ALLOCATION BY RISK PROFILE (Percentage)
# ============================================================================

RISK_ALLOCATIONS = {
    "Conservative": {"low": 0.80, "medium": 0.20, "high": 0.00},
    "Moderate": {"low": 0.60, "medium": 0.30, "high": 0.10},
    "Aggressive": {"low": 0.40, "medium": 0.40, "high": 0.20}
}

# ============================================================================
# EXPECTED ANNUAL RETURNS BY RISK PROFILE (Decimal)
# ============================================================================

EXPECTED_RETURNS = {
    "Conservative": 0.08,   # 8 percent per year
    "Moderate": 0.10,       # 10 percent per year
    "Aggressive": 0.12      # 12 percent per year
}

# ============================================================================
# EXPECTED VOLATILITY BY RISK PROFILE (Annual percentage)
# ============================================================================

EXPECTED_VOLATILITY = {
    "Conservative": 0.12,   # 12 percent annual volatility
    "Moderate": 0.18,       # 18 percent annual volatility
    "Aggressive": 0.25      # 25 percent annual volatility
}

# ============================================================================
# MARKET CONVENTIONS
# ============================================================================

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.05  # 5 percent annual (BI 7-Day Reverse Repo Rate average)

# ============================================================================
# SIMULATION DEFAULTS
# ============================================================================

DEFAULT_SIMULATIONS = 10000
CRISIS_WEIGHT = 3.0
CRISIS_PERIOD = ("2020-03-01", "2020-06-30")

# ============================================================================
# INFLATION ASSUMPTIONS (Annual percentage)
# ============================================================================

INFLATION_RATES = {
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

# ============================================================================
# KPR HIDDEN COSTS (Percentage of house price or loan amount)
# ============================================================================

KPR_HIDDEN_COSTS = {
    "bphtb": 0.05,           # 5 percent of house price
    "notary": 0.01,          # 1 percent of house price
    "bank_provision": 0.01   # 1 percent of loan amount
}

# Default KPR parameters
KPR_DEFAULTS = {
    "down_payment_percent": 20,
    "loan_term_years": 15,
    "annual_interest_rate": 0.07,
    "monthly_installment": 5_000_000
}

# ============================================================================
# TRANSACTION COSTS (Indonesian stock market)
# ============================================================================

TRANSACTION_COSTS = {
    "broker_fee_buy": 0.0015,      # 0.15 percent for buying
    "broker_fee_sell": 0.0025,     # 0.25 percent for selling
    "vat_rate": 0.11,              # 11 percent VAT on broker fees
    "capital_gain_tax": 0.001,     # 0.1 percent final tax on capital gains
    "dividend_tax": 0.10           # 10 percent final withholding tax
}

# ============================================================================
# DEFAULT VALUES
# ============================================================================

MINIMUM_RECOMMENDED_CAPITAL = 100_000_000  # Rp 100 million
LOT_SIZE = 100  # 1 lot = 100 shares

# ============================================================================
# DATA PERIOD
# ============================================================================

DATE_RANGE = {
    "start": "2019-07-01",
    "end": "2025-02-28"
}

# Period definitions for regime analysis
PERIODS = {
    "pre_covid": ("2019-07-01", "2020-02-28"),
    "covid_crash": ("2020-03-01", "2020-06-30"),
    "post_covid": ("2020-07-01", "2025-02-28")
}

# ============================================================================
# HISTORICAL DIVIDEND PER SHARE (IDR) FOR 2019-2024
# Sources: IDX, KSEI, company annual reports
# ============================================================================

DIVIDEND_DATA = {
    "ADRO": {2019: 114, 2020: 55, 2021: 160, 2022: 481, 2023: 299, 2024: 251},
    "AKRA": {2019: 100, 2020: 125, 2021: 29, 2022: 100, 2023: 125, 2024: 125},
    "AMRT": {2019: 13, 2020: 14, 2021: 19, 2022: 24, 2023: 28, 2024: 31},
    "ANTM": {2019: 3, 2020: 0, 2021: 16, 2022: 38, 2023: 79, 2024: 128},
    "ASII": {2019: 214, 2020: 157, 2021: 114, 2022: 239, 2023: 640, 2024: 519},
    "BBCA": {2019: 555, 2020: 530, 2021: 145, 2022: 170, 2023: 205, 2024: 227},
    "BBNI": {2019: 206, 2020: 44, 2021: 47, 2022: 146, 2023: 392, 2024: 280},
    "BBRI": {2019: 168, 2020: 98, 2021: 121, 2022: 174, 2023: 288, 2024: 312},
    "BBTN": {2019: 0, 2020: 0, 2021: 4, 2022: 22, 2023: 43, 2024: 49},
    "BMRI": {2019: 353, 2020: 165, 2021: 220, 2022: 360, 2023: 529, 2024: 354},
    "BRPT": {2019: 0, 2020: 0, 2021: 0, 2022: 1, 2023: 2, 2024: 0},
    "CPIN": {2019: 118, 2020: 81, 2021: 108, 2022: 108, 2023: 100, 2024: 100},
    "CTRA": {2019: 8, 2020: 0, 2021: 14, 2022: 15, 2023: 17, 2024: 18},
    "EXCL": {2019: 0, 2020: 20, 2021: 31, 2022: 13, 2023: 42, 2024: 48},
    "ICBP": {2019: 236, 2020: 215, 2021: 215, 2022: 215, 2023: 188, 2024: 200},
    "INCO": {2019: 0, 2020: 0, 2021: 0, 2022: 130, 2023: 118, 2024: 94},
    "INDF": {2019: 278, 2020: 278, 2021: 278, 2022: 278, 2023: 257, 2024: 267},
    "INKP": {2019: 50, 2020: 50, 2021: 50, 2022: 50, 2023: 50, 2024: 50},
    "ISAT": {2019: 0, 2020: 0, 2021: 0, 2022: 248, 2023: 344, 2024: 344},
    "ITMG": {2019: 2045, 2020: 402, 2021: 1218, 2022: 3040, 2023: 6413, 2024: 4407},
    "JPFA": {2019: 50, 2020: 20, 2021: 40, 2022: 60, 2023: 50, 2024: 50},
    "KLBF": {2019: 26, 2020: 20, 2021: 28, 2022: 35, 2023: 38, 2024: 31},
    "MAPI": {2019: 10, 2020: 0, 2021: 0, 2022: 0, 2023: 8, 2024: 8},
    "MDKA": {2019: 0, 2020: 0, 2021: 0, 2022: 0, 2023: 0, 2024: 0},
    "MEDC": {2019: 0, 2020: 0, 2021: 0, 2022: 15, 2023: 15, 2024: 15},
    "PGAS": {2019: 41, 2020: 0, 2021: 0, 2022: 124, 2023: 141, 2024: 148},
    "PTBA": {2019: 326, 2020: 74, 2021: 688, 2022: 1094, 2023: 397, 2024: 397},
    "SMGR": {2019: 207, 2020: 101, 2021: 164, 2022: 245, 2023: 165, 2024: 165},
    "TLKM": {2019: 154, 2020: 168, 2021: 149, 2022: 167, 2023: 176, 2024: 178},
    "TOWR": {2019: 20, 2020: 25, 2021: 24, 2022: 28, 2023: 30, 2024: 30},
    "UNTR": {2019: 1113, 2020: 648, 2021: 1240, 2022: 7003, 2023: 2270, 2024: 2270},
    "UNVR": {2019: 915, 2020: 187, 2021: 150, 2022: 141, 2023: 132, 2024: 128},
}

# ============================================================================
# STOCK SPLIT ADJUSTMENTS (For dividend adjustment)
# ============================================================================

STOCK_SPLITS = {
    "AKRA": ("2022-01-12", 5),
    "ARTO": ("2020-03-30", 4),
    "BBCA": ("2021-10-13", 5),
    "BBNI": ("2023-10-06", 4),
    "BMRI": ("2023-04-04", 4),
    "BRPT": ("2019-08-06", 5),
    "GGRM": ("2023-07-17", 5),
    "HRUM": ("2022-06-02", 5),
    "ISAT": ("2024-10-14", 5),
    "MAPA": ("2023-07-17", 5),
    "MDKA": ("2019-10-18", 5),
    "SIDO": ("2020-09-14", 5),
    "UNVR": ("2020-01-02", 5),
}

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_ticker(ticker: str, tickers_list: list) -> bool:
    """
    Check if a ticker is in the valid list.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    tickers_list : list
        List of valid tickers
    
    Returns
    -------
    bool
        True if ticker is valid
    """
    return ticker in tickers_list


def validate_risk_profile(profile: str) -> bool:
    """
    Check if a risk profile is valid.
    
    Parameters
    ----------
    profile : str
        Risk profile name (Conservative, Moderate, Aggressive)
    
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
    valid_goals = ["Wedding", "KPR", "Education", "Wedding Fund", "KPR Down Payment", "Child Education"]
    return any(g in goal for g in valid_goals)


def get_risk_threshold(level: str) -> float:
    """
    Get risk threshold value.
    
    Parameters
    ----------
    level : str
        Risk level ('low' or 'medium')
    
    Returns
    -------
    float
        Threshold value
    """
    return RISK_THRESHOLDS.get(level, 0.0)


def get_expected_return(risk_profile: str) -> float:
    """
    Get expected return for a risk profile.
    
    Parameters
    ----------
    risk_profile : str
        Risk profile name
    
    Returns
    -------
    float
        Expected annual return (decimal)
    """
    return EXPECTED_RETURNS.get(risk_profile, 0.10)


def get_allocation(risk_profile: str) -> dict:
    """
    Get portfolio allocation for a risk profile.
    
    Parameters
    ----------
    risk_profile : str
        Risk profile name
    
    Returns
    -------
    dict
        Allocation percentages for low, medium, high risk
    """
    return RISK_ALLOCATIONS.get(risk_profile, RISK_ALLOCATIONS["Moderate"])


if __name__ == "__main__":
    print("=" * 60)
    print("CONSTANTS MODULE")
    print("=" * 60)
    print(f"\nNumber of stocks in SECTOR_MAP: {len(SECTOR_MAP)}")
    print(f"Number of stocks with dividend data: {len(DIVIDEND_DATA)}")
    print(f"Number of stock splits recorded: {len(STOCK_SPLITS)}")
    print(f"\nRisk thresholds: Low < {RISK_THRESHOLDS['low']}%, Medium < {RISK_THRESHOLDS['medium']}%")
    print(f"Risk-free rate: {RISK_FREE_RATE * 100:.0f}%")
    print(f"Trading days per year: {TRADING_DAYS_PER_YEAR}")
    print(f"Default simulations: {DEFAULT_SIMULATIONS:,}")
    print(f"Crisis weight: {CRISIS_WEIGHT}x")