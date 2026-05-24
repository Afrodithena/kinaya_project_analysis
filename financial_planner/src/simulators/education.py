"""
Education Fund Simulator (10-18 years)

Simulates long-term investment outcomes for child education funding,
incorporating education inflation, dividend income, and consumer stocks as inflation hedge.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class EducationResult:
    """Container for education simulator results."""
    
    future_annual_cost: float
    total_target_4year: float
    total_saved: float
    projected_investment_value: float
    progress_percent: float
    pessimistic_value: float
    optimistic_value: float
    recommended_stocks: List[str]
    years_until_college: int
    monthly_saving: float
    inflation_rate: float
    annual_dividend_income: float = 0.0
    dividend_coverage_percent: float = 0.0
    monthly_breakdown: List[Dict] = field(default_factory=list)
    scenario_analysis: Dict[str, float] = field(default_factory=dict)


@dataclass
class EducationMonthlyPlan:
    """Container for monthly saving plan."""
    
    year: int
    child_age: int
    education_level: str
    annual_cost: float
    monthly_cost: float
    recommended_saving: float
    shortfall: float


class EducationSimulator:
    """
    Education fund simulator for 10-18 year horizon.
    
    Strategy: 40 percent low risk plus 30 percent medium risk plus 30 percent consumer stocks.
    Consumer stocks (ICBP, INDF, UNVR) act as natural hedge against education inflation.
    
    Scenario returns assumptions:
        - Pessimistic: 6 percent annual return
        - Moderate: 10 percent annual return
        - Optimistic: 14 percent annual return
    """
    
    SCENARIO_RETURNS = {
        "pessimistic": 0.06,
        "moderate": 0.10,
        "optimistic": 0.14
    }
    
    # Education levels with typical age ranges
    EDUCATION_LEVELS = {
        "SD": {"age_start": 7, "duration": 6, "base_cost": 5_000_000},
        "SMP": {"age_start": 13, "duration": 3, "base_cost": 7_000_000},
        "SMA": {"age_start": 16, "duration": 3, "base_cost": 9_000_000},
        "PTN": {"age_start": 19, "duration": 4, "base_cost": 25_000_000}
    }
    
    # Recommended stocks for education fund
    RECOMMENDED_STOCKS = ["ICBP", "INDF", "UNVR", "BBCA", "BBRI"]
    
    # Consumer stocks for inflation hedge
    CONSUMER_STOCKS = ["ICBP", "INDF", "UNVR"]
    
    def __init__(
        self,
        dividend_data: Optional[Dict] = None,
        all_stocks_data: Optional[Dict] = None
    ):
        """
        Initialize Education Simulator.
        
        Parameters
        ----------
        dividend_data : dict, optional
            Dictionary of dividend data for KPR-like coverage analysis
        all_stocks_data : dict, optional
            Dictionary of stock dataframes for price lookup
        """
        self.dividend_data = dividend_data or {}
        self.all_stocks_data = all_stocks_data or {}
    
    @staticmethod
    def calculate_future_cost(
        current_cost: float,
        years: float,
        inflation_rate: float
    ) -> float:
        """
        Calculate future university cost with compounding inflation.
        
        Parameters
        ----------
        current_cost : float
            Current annual university cost (IDR)
        years : float
            Years until college enrollment
        inflation_rate : float
            Annual education inflation rate (default 10 percent)
            
        Returns
        -------
        float
            Projected annual cost at enrollment year
        """
        return current_cost * ((1 + inflation_rate) ** years)
    
    @staticmethod
    def calculate_monthly_saving_needed(
        target_amount: float,
        years: float,
        annual_return: float
    ) -> float:
        """
        Calculate monthly saving needed to reach target.
        
        Parameters
        ----------
        target_amount : float
            Target amount to achieve (IDR)
        years : float
            Investment horizon in years
        annual_return : float
            Expected annual return rate
            
        Returns
        -------
        float
            Required monthly saving amount (IDR)
        """
        months = years * 12
        monthly_rate = annual_return / 12
        
        if monthly_rate <= 0:
            return target_amount / months
        
        monthly_saving = target_amount * monthly_rate / ((1 + monthly_rate) ** months - 1)
        return monthly_saving
    
    @staticmethod
    def project_value(
        saved_amount: float,
        years: float,
        annual_return: float
    ) -> float:
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
        return saved_amount * (1 + annual_return) ** years
    
    def get_dividend_yield(self, stock: str, current_price: float) -> float:
        """
        Calculate average dividend yield from historical data.
        
        Parameters
        ----------
        stock : str
            Stock ticker symbol
        current_price : float
            Current stock price
            
        Returns
        -------
        float
            Average dividend yield percentage
        """
        if stock not in self.dividend_data:
            return 0.0
        
        div_values = list(self.dividend_data[stock].values())
        if not div_values:
            return 0.0
        
        avg_dps = sum(div_values) / len(div_values)
        return (avg_dps / current_price) * 100 if current_price > 0 else 0
    
    def calculate_dividend_income(
        self,
        investment_amount: float,
        stock: str
    ) -> Tuple[float, float]:
        """
        Calculate annual dividend income from investment.
        
        Parameters
        ----------
        investment_amount : float
            Total investment amount (IDR)
        stock : str
            Stock ticker symbol
            
        Returns
        -------
        tuple
            (dividend_yield_percent, annual_dividend_income)
        """
        if stock not in self.all_stocks_data:
            return 0.0, 0.0
        
        df = self.all_stocks_data[stock]
        if 'adjusted_close' in df.columns:
            current_price = df['adjusted_close'].iloc[-1]
        else:
            current_price = df['close'].iloc[-1]
        
        dividend_yield = self.get_dividend_yield(stock, current_price)
        annual_dividend = investment_amount * (dividend_yield / 100)
        
        return dividend_yield, annual_dividend
    
    def simulate(
        self,
        current_cost: float,
        child_age: int,
        years: float,
        monthly_saving: float,
        inflation_rate: float = 0.10,
        annual_return: float = None
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
            Education inflation rate (5-15 percent range)
        annual_return : float, optional
            Custom annual return (overrides moderate scenario)
            
        Returns
        -------
        EducationResult
            Container with projected costs, savings, and scenario analysis
        """
        if annual_return is None:
            annual_return = self.SCENARIO_RETURNS["moderate"]
        
        # Project future university cost
        future_annual_cost = self.calculate_future_cost(current_cost, years, inflation_rate)
        total_target = future_annual_cost * 4
        
        # Total savings accumulated
        total_saved = monthly_saving * years * 12
        
        # Projected value under selected scenario
        projected_value = self.project_value(total_saved, years, annual_return)
        
        # Progress percentage
        progress = min(100.0, (projected_value / total_target) * 100) if total_target > 0 else 0.0
        
        # Scenario analysis
        scenario_analysis = {}
        for scenario, ret in self.SCENARIO_RETURNS.items():
            scenario_value = self.project_value(total_saved, years, ret)
            scenario_analysis[scenario] = round(scenario_value, 0)
            scenario_analysis[f"{scenario}_coverage_pct"] = round(
                min(100.0, (scenario_value / total_target) * 100), 1
            ) if total_target > 0 else 0.0
        
        # Dividend income from recommended stocks (using first consumer stock as proxy)
        dividend_yield = 0.0
        annual_dividend = 0.0
        if self.CONSUMER_STOCKS and self.CONSUMER_STOCKS[0] in self.all_stocks_data:
            dividend_yield, annual_dividend = self.calculate_dividend_income(
                projected_value, self.CONSUMER_STOCKS[0]
            )
        
        dividend_coverage = (annual_dividend / (future_annual_cost)) * 100 if future_annual_cost > 0 else 0
        
        return EducationResult(
            future_annual_cost=round(future_annual_cost, 0),
            total_target_4year=round(total_target, 0),
            total_saved=round(total_saved, 0),
            projected_investment_value=round(projected_value, 0),
            progress_percent=round(progress, 1),
            pessimistic_value=round(scenario_analysis.get("pessimistic", 0), 0),
            optimistic_value=round(scenario_analysis.get("optimistic", 0), 0),
            recommended_stocks=self.RECOMMENDED_STOCKS,
            years_until_college=int(years),
            monthly_saving=monthly_saving,
            inflation_rate=inflation_rate,
            annual_dividend_income=round(annual_dividend, 0),
            dividend_coverage_percent=round(dividend_coverage, 1),
            monthly_breakdown=[],
            scenario_analysis=scenario_analysis
        )
    
    def simulate_with_breakdown(
        self,
        child_age: int,
        monthly_saving: float,
        target_level: str = "PTN",
        inflation_rate: float = 0.10,
        annual_return: float = 0.10
    ) -> List[EducationMonthlyPlan]:
        """
        Simulate education funding with yearly breakdown by education level.
        
        Parameters
        ----------
        child_age : int
            Current age of child
        monthly_saving : float
            Monthly saving amount (IDR)
        target_level : str, default "PTN"
            Target education level (SD, SMP, SMA, PTN)
        inflation_rate : float, default 0.10
            Annual education inflation rate
        annual_return : float, default 0.10
            Expected annual return rate
            
        Returns
        -------
        list
            List of EducationMonthlyPlan for each year
        """
        results = []
        levels_order = ["SD", "SMP", "SMA", "PTN"]
        
        # Find index of target level
        target_idx = levels_order.index(target_level) if target_level in levels_order else 3
        
        cumulative_savings = 0
        current_age = child_age
        
        for i in range(target_idx + 1):
            level = levels_order[i]
            level_info = self.EDUCATION_LEVELS[level]
            
            years_until_start = max(0, level_info["age_start"] - current_age)
            
            if years_until_start <= 0:
                # Already past this level
                continue
            
            # Calculate future cost at start of this level
            future_annual_cost = level_info["base_cost"] * ((1 + inflation_rate) ** years_until_start)
            
            # Project savings by start year
            months_until_start = years_until_start * 12
            future_savings = self.project_value(
                monthly_saving * months_until_start,
                years_until_start,
                annual_return
            )
            
            # Calculate recommended monthly saving to cover this level
            total_level_cost = future_annual_cost * level_info["duration"]
            recommended_monthly = self.calculate_monthly_saving_needed(
                total_level_cost, years_until_start, annual_return
            )
            
            shortfall = max(0, recommended_monthly - monthly_saving)
            
            results.append(EducationMonthlyPlan(
                year=2024 + years_until_start,
                child_age=level_info["age_start"],
                education_level=level,
                annual_cost=round(future_annual_cost, 0),
                monthly_cost=round(future_annual_cost / 12, 0),
                recommended_saving=round(recommended_monthly, 0),
                shortfall=round(shortfall, 0)
            ))
        
        return results
    
    def get_required_monthly_saving(
        self,
        current_cost: float,
        child_age: int,
        years: float,
        inflation_rate: float = 0.10,
        annual_return: float = 0.10
    ) -> float:
        """
        Calculate required monthly saving to reach education target.
        
        Parameters
        ----------
        current_cost : float
            Current annual university cost (IDR)
        child_age : int
            Current age of child
        years : float
            Years until college
        inflation_rate : float, default 0.10
            Education inflation rate
        annual_return : float, default 0.10
            Expected annual return rate
            
        Returns
        -------
        float
            Required monthly saving amount (IDR)
        """
        future_cost = self.calculate_future_cost(current_cost, years, inflation_rate)
        total_target = future_cost * 4
        
        required_monthly = self.calculate_monthly_saving_needed(
            total_target, years, annual_return
        )
        
        return round(required_monthly, 0)
    
    def compare_stocks_for_education(
        self,
        stocks: List[str],
        investment_amount: float,
        years: float
    ) -> pd.DataFrame:
        """
        Compare different stocks for education fund performance.
        
        Parameters
        ----------
        stocks : list
            List of stock tickers to compare
        investment_amount : float
            Initial investment amount (IDR)
        years : float
            Investment horizon in years
            
        Returns
        -------
        pd.DataFrame
            Comparison of stock performance
        """
        results = []
        
        for stock in stocks:
            if stock not in self.all_stocks_data:
                continue
            
            df = self.all_stocks_data[stock]
            if 'adjusted_close' in df.columns:
                price_col = 'adjusted_close'
            else:
                price_col = 'close'
            
            # Calculate historical annualized return
            daily_returns = df[price_col].pct_change().dropna()
            annualized_return = daily_returns.mean() * 252 * 100
            
            # Project future value
            future_value = investment_amount * (1 + annualized_return / 100) ** years
            
            # Get dividend yield if available
            current_price = df[price_col].iloc[-1]
            dividend_yield = self.get_dividend_yield(stock, current_price)
            
            results.append({
                'Stock': stock,
                'Annualized_Return_Pct': round(annualized_return, 1),
                'Projected_Value_Rp': round(future_value, 0),
                'Dividend_Yield_Pct': round(dividend_yield, 1),
                'Multiple_of_Initial': round(future_value / investment_amount, 1)
            })
        
        df_result = pd.DataFrame(results)
        df_result = df_result.sort_values('Annualized_Return_Pct', ascending=False)
        
        return df_result


if __name__ == "__main__":
    print("=" * 60)
    print("EDUCATION FUND SIMULATOR MODULE")
    print("=" * 60)
    print("\nThis module provides education fund simulation functionality.")
    print("\nAvailable functions:")
    print("  - EducationSimulator.simulate(): Run education fund simulation")
    print("  - EducationSimulator.simulate_with_breakdown(): Yearly breakdown by education level")
    print("  - EducationSimulator.get_required_monthly_saving(): Calculate required monthly saving")
    print("  - EducationSimulator.compare_stocks_for_education(): Compare stock performance")
    print("  - EducationSimulator.calculate_future_cost(): Project future cost with inflation")
    print("  - EducationSimulator.calculate_monthly_saving_needed(): Calculate required monthly saving")