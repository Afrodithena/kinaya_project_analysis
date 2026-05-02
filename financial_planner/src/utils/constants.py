"""
Constants and static data mappings for LQ45 stocks.

Contains:
- SECTOR_MAP: Stock ticker to sector classification
- RISK_ALLOCATIONS: Portfolio allocation by risk profile
- EXPECTED_RETURNS: Expected annual return by risk profile
- DIVIDEND_DATA: Historical dividend per share (IDR) for 2019-2024
"""

# Sector classification for LQ45 stocks
SECTOR_MAP = {
    # Banking
    "BBCA": "Bank", "BBRI": "Bank", "BMRI": "Bank", "BBNI": "Bank", "BBTN": "Bank",
    # Consumer
    "ICBP": "Consumer", "INDF": "Consumer", "UNVR": "Consumer", "CPIN": "Consumer",
    "JPFA": "Consumer", "KLBF": "Consumer", "MAPI": "Consumer", "AMRT": "Consumer",
    # Energy
    "ADRO": "Energy", "ITMG": "Energy", "PTBA": "Energy", "MEDC": "Energy",
    "PGAS": "Energy", "PGEO": "Energy",
    # Mining
    "ANTM": "Mining", "INCO": "Mining", "MDKA": "Mining", "BRPT": "Mining",
    # Property
    "CTRA": "Property",
    # Telecommunications
    "TLKM": "Telecom", "EXCL": "Telecom", "ISAT": "Telecom", "TOWR": "Telecom",
    # Automotive
    "ASII": "Automotive", "UNTR": "Automotive",
    # Other sectors
    "INKP": "Pulp & Paper", "SMGR": "Cement", "AKRA": "Logistics"
}

# Portfolio allocation by risk profile (percentage)
RISK_ALLOCATIONS = {
    "Conservative": {"low": 0.80, "medium": 0.20, "high": 0.00},
    "Moderate": {"low": 0.60, "medium": 0.30, "high": 0.10},
    "Aggressive": {"low": 0.40, "medium": 0.40, "high": 0.20}
}

# Expected annual returns by risk profile (decimal)
EXPECTED_RETURNS = {
    "Conservative": 0.08,
    "Moderate": 0.10,
    "Aggressive": 0.12
}

# Historical dividend per share (IDR) for 2019-2024
# Sources: IDX, KSEI, company annual reports
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