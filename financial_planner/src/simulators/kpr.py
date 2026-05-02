"""
KPR Readiness Simulator (3-5 years)

Simulates down payment accumulation for house purchase, including hidden costs
(BPHTB, notary fees, bank provisions) and dividend synergy from bank stocks.
"""

from dataclasses import dataclass
from typing import Tuple

from src.config import KPR_HIDDEN_COSTS


@dataclass
class KPRResult:
    """Container for KPR simulator results."""
    
    down_payment: float
    bphtb: float
    notary_fee: float
    bank_provision: float
    total_needed: float
    total_saved: float
    progress_pct: float
    readiness_score: str
    dividend_coverage_months: float


class KPRSimulator:
    """
    KPR readiness simulator for 3-5 year horizon.
    
    Strategy: 60% low risk + 30% medium risk + 10% high risk stocks.
    Hidden costs included:
        - BPHTB: 5% of house price
        - Notary fee: 1% of house price
        - Bank provision: 1% of loan amount
    
    Dividend synergy: Bank stocks (BBCA, BBRI, BMRI) typically yield 3-4%,
    which can cover 2-4 months of KPR payments annually.
    """
    
    @staticmethod
    def calculate_hidden_costs(house_price: float, dp_percent: float) -> Tuple[float, float, float, float, float]:
        """
        Calculate all costs for KPR down payment.
        
        Parameters
        ----------
        house_price : float
            Total house price (IDR)
        dp_percent : float
            Down payment percentage (5-30%)
            
        Returns
        -------
        Tuple[float, float, float, float, float]
            (down_payment, bphtb, notary_fee, bank_provision, total_needed)
        """
        down_payment = house_price * dp_percent / 100
        bphtb = house_price * KPR_HIDDEN_COSTS["bphtb"]
        notary_fee = house_price * KPR_HIDDEN_COSTS["notary"]
        bank_provision = (house_price - down_payment) * KPR_HIDDEN_COSTS["bank_provision"]
        total_needed = down_payment + bphtb + notary_fee + bank_provision
        
        return down_payment, bphtb, notary_fee, bank_provision, total_needed
    
    @staticmethod
    def dividend_coverage(investment_amount: float, avg_dividend_yield: float, monthly_kpr: float) -> float:
        """
        Calculate how many months of KPR payments dividends can cover.
        
        Parameters
        ----------
        investment_amount : float
            Total investment amount (IDR)
        avg_dividend_yield : float
            Average dividend yield of portfolio (e.g., 0.035 for 3.5%)
        monthly_kpr : float
            Monthly KPR installment (IDR)
            
        Returns
        -------
        float
            Number of months covered by annual dividends
        """
        if monthly_kpr <= 0:
            return 0.0
        
        yearly_dividend = investment_amount * avg_dividend_yield
        return yearly_dividend / monthly_kpr
    
    def simulate(
        self,
        house_price: float,
        dp_percent: float,
        monthly_kpr: float,
        years: float,
        monthly_saving: float,
        avg_dividend_yield: float = 0.035
    ) -> KPRResult:
        """
        Run KPR readiness simulation.
        
        Parameters
        ----------
        house_price : float
            Total house price (IDR)
        dp_percent : float
            Down payment percentage (5-30%)
        monthly_kpr : float
            Target monthly KPR installment (IDR)
        years : float
            Years to save for down payment (3-5)
        monthly_saving : float
            Monthly saving capacity (IDR)
        avg_dividend_yield : float, default 0.035
            Average dividend yield of portfolio (3.5%)
            
        Returns
        -------
        KPRResult
            Container with simulation results including readiness score
        """
        # Calculate all costs
        down_payment, bphtb, notary_fee, bank_provision, total_needed = self.calculate_hidden_costs(
            house_price, dp_percent
        )
        
        # Total savings accumulated
        total_saved = monthly_saving * years * 12
        
        # Progress and readiness
        progress = min(100.0, (total_saved / total_needed) * 100) if total_needed > 0 else 0.0
        readiness = "READY" if total_saved >= total_needed else "NOT READY"
        
        # Dividend coverage calculation
        coverage = self.dividend_coverage(total_saved, avg_dividend_yield, monthly_kpr)
        
        return KPRResult(
            down_payment=down_payment,
            bphtb=bphtb,
            notary_fee=notary_fee,
            bank_provision=bank_provision,
            total_needed=total_needed,
            total_saved=total_saved,
            progress_pct=round(progress, 1),
            readiness_score=readiness,
            dividend_coverage_months=round(coverage, 1)
        )