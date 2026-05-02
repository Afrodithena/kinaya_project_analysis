"""
Wedding Fund Simulator (1-3 years)

Simulates short-term investment outcomes for wedding fund goals.
Strategy: 80% low risk + 20% medium risk stocks.
"""

from dataclasses import dataclass
from typing import List

from src.simulation.bootstrap import BootstrapSimulator
from src.utils.constants import EXPECTED_RETURNS


@dataclass
class WeddingResult:
    """Container for wedding simulator results."""
    
    required_monthly: float
    is_sufficient: bool
    probability_success: float
    status: str
    expected_return: float
    recommended_stocks: List[str]


class WeddingSimulator:
    """
    Wedding fund simulator for 1-3 year horizon.
    
    Strategy: 80% low risk + 20% medium risk stocks.
    Recommended stocks adapt to risk profile:
        - Conservative: BBCA, TLKM, ASII
        - Moderate: BBCA, BBRI, ICBP
        - Aggressive: BBCA, BBRI, ADRO
    """
    
    STOCK_RECOMMENDATIONS = {
        "Conservative": ["BBCA", "TLKM", "ASII"],
        "Moderate": ["BBCA", "BBRI", "ICBP"],
        "Aggressive": ["BBCA", "BBRI", "ADRO"]
    }
    
    def __init__(self, bootstrap_simulator: BootstrapSimulator):
        """
        Initialize wedding simulator with bootstrap engine.
        
        Parameters
        ----------
        bootstrap_simulator : BootstrapSimulator
            Bootstrap engine for Monte Carlo projections
        """
        self.bootstrap = bootstrap_simulator
    
    @classmethod
    def calculate_required_saving(cls, target_amount: float, years: float, annual_return: float) -> float:
        """
        Calculate required monthly saving to reach target amount.
        
        Uses future value of annuity formula:
            PMT = FV * r / ((1 + r)^n - 1)
        
        Parameters
        ----------
        target_amount : float
            Target wedding fund amount (IDR)
        years : float
            Investment horizon in years
        annual_return : float
            Expected annual return rate
            
        Returns
        -------
        float
            Required monthly saving (IDR)
        """
        months = years * 12
        monthly_rate = annual_return / 12
        
        if monthly_rate == 0:
            return target_amount / months
        
        return target_amount * monthly_rate / ((1 + monthly_rate) ** months - 1)
    
    def simulate(
        self,
        target_amount: float,
        years: float,
        monthly_saving: float,
        risk_profile: str
    ) -> WeddingResult:
        """
        Run wedding fund simulation.
        
        Parameters
        ----------
        target_amount : float
            Target wedding fund amount (IDR)
        years : float
            Investment horizon (1-3 years)
        monthly_saving : float
            User's monthly saving capacity (IDR)
        risk_profile : str
            Risk tolerance ('Conservative', 'Moderate', 'Aggressive')
            
        Returns
        -------
        WeddingResult
            Container with simulation results including success probability
        """
        expected_return = EXPECTED_RETURNS[risk_profile]
        required = self.calculate_required_saving(target_amount, years, expected_return)
        
        initial_investment = monthly_saving * years * 12
        bootstrap_result = self.bootstrap.simulate_with_target(initial_investment, target_amount, years)
        
        is_sufficient = monthly_saving >= required
        status = "ON TRACK" if is_sufficient else "NEEDS INCREASE"
        
        return WeddingResult(
            required_monthly=round(required, 0),
            is_sufficient=is_sufficient,
            probability_success=round(bootstrap_result.probability_success, 1),
            status=status,
            expected_return=round(expected_return * 100, 1),
            recommended_stocks=self.STOCK_RECOMMENDATIONS.get(risk_profile, ["BBCA"])
        )