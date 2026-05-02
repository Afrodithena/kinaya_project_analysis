"""
Education Fund Simulator (10-18 years)

Simulates long-term investment outcomes for child education funding,
incorporating education inflation and consumer stocks as inflation hedge.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class EducationResult:
    """Container for education simulator results."""
    
    future_cost: float
    total_target: float
    total_saved: float
    projected_value: float
    progress_pct: float
    pessimistic: float
    optimistic: float
    recommended_stocks: List[str]


class EducationSimulator:
    """
    Education fund simulator for 10-18 year horizon.
    
    Strategy: 40% low risk + 30% medium risk + 30% consumer stocks.
    Consumer stocks (ICBP, INDF, UNVR) act as natural hedge against education inflation.
    
    Scenario returns assumptions:
        - Pessimistic: 6% annual return
        - Moderate: 10% annual return
        - Optimistic: 14% annual return
    """
    
    SCENARIO_RETURNS = {
        "pessimistic": 0.06,
        "moderate": 0.10,
        "optimistic": 0.14
    }
    
    RECOMMENDED_STOCKS = ["ICBP", "INDF", "UNVR"]
    
    @staticmethod
    def calculate_future_cost(current_cost: float, years: float, inflation_rate: float) -> float:
        """
        Calculate future university cost with compounding inflation.
        
        Parameters
        ----------
        current_cost : float
            Current annual university cost (IDR)
        years : float
            Years until college enrollment
        inflation_rate : float
            Annual education inflation rate (default 10%)
            
        Returns
        -------
        float
            Projected annual cost at enrollment year
        """
        return current_cost * ((1 + inflation_rate) ** years)
    
    @staticmethod
    def project_value(saved_amount: float, years: float, annual_return: float) -> float:
        """
        Project future value of accumulated savings.
        
        Parameters
        ----------
        saved_amount : float
            Total savings accumulated (IDR)
        years : float
            Investment horizon in years
        annual_return : float
            Expected annual return rate
        
        Returns
        -------
        float
            Projected future value (IDR)
        """
        return saved_amount * (1 + annual_return) ** (years / 5)
    
    def simulate(
        self,
        current_cost: float,
        child_age: int,
        years: float,
        monthly_saving: float,
        inflation_rate: float = 0.10
    ) -> EducationResult:
        """
        Run education fund simulation.
        
        Parameters
        ----------
        current_cost : float
            Current annual university cost (IDR)
        child_age : int
            Current age of child
        years : float
            Years until college (typically 10-18)
        monthly_saving : float
            Monthly saving amount (IDR)
        inflation_rate : float, default 0.10
            Education inflation rate (5-15% range)
            
        Returns
        -------
        EducationResult
            Container with projected costs, savings, and scenario analysis
        """
        # Project future university cost
        future_cost = self.calculate_future_cost(current_cost, years, inflation_rate)
        total_target = future_cost * 4  # 4 years of university
        
        # Total savings accumulated
        total_saved = monthly_saving * years * 12
        
        # Projected value under moderate scenario (10% return)
        moderate = self.project_value(total_saved, years, self.SCENARIO_RETURNS["moderate"])
        
        # Progress percentage
        progress = min(100.0, (moderate / total_target) * 100) if total_target > 0 else 0.0
        
        # Scenario analysis
        pessimistic = self.project_value(total_saved, years, self.SCENARIO_RETURNS["pessimistic"])
        optimistic = self.project_value(total_saved, years, self.SCENARIO_RETURNS["optimistic"])
        
        return EducationResult(
            future_cost=future_cost,
            total_target=total_target,
            total_saved=total_saved,
            projected_value=moderate,
            progress_pct=round(progress, 1),
            pessimistic=pessimistic,
            optimistic=optimistic,
            recommended_stocks=self.RECOMMENDED_STOCKS
        )