"""
Wedding Fund Simulator (1-3 years)

Simulates short-term investment outcomes for wedding fund goals.
Strategy: 80 percent low risk + 20 percent medium risk stocks for Conservative profile.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class WeddingResult:
    """Container for wedding simulator results."""
    
    required_monthly: float
    is_sufficient: bool
    probability_success: float
    status: str
    expected_return_percent: float
    recommended_stocks: List[str]
    projected_value: float = 0.0
    target_amount: float = 0.0
    months_to_goal: float = 0.0
    shortfall: float = 0.0
    scenario_analysis: Dict[str, float] = field(default_factory=dict)


@dataclass
class WeddingMonthlyPlan:
    """Container for monthly saving plan."""
    
    month: int
    cumulative_savings: float
    target_progress: float
    projected_value: float
    recommended_saving: float
    on_track: bool


class WeddingSimulator:
    """
    Wedding fund simulator for 1-3 year horizon.
    
    Strategy for each risk profile:
        - Conservative: 80 percent low risk + 20 percent medium risk stocks
        - Moderate: 60 percent low risk + 30 percent medium risk + 10 percent high risk
        - Aggressive: 40 percent low risk + 40 percent medium risk + 20 percent high risk
    
    Recommended stocks adapt to risk profile:
        - Conservative: BBCA, TLKM, ASII, ICBP, INDF
        - Moderate: BBCA, BBRI, BMRI, CPIN, PGAS
        - Aggressive: BBCA, BBRI, ADRO, ITMG, PTBA
    """
    
    # Stock recommendations by risk profile
    STOCK_RECOMMENDATIONS = {
        "Conservative": ["BBCA", "TLKM", "ASII", "ICBP", "INDF"],
        "Moderate": ["BBCA", "BBRI", "BMRI", "CPIN", "PGAS"],
        "Aggressive": ["BBCA", "BBRI", "ADRO", "ITMG", "PTBA"]
    }
    
    # Expected annual returns by risk profile (percentage)
    EXPECTED_RETURNS = {
        "Conservative": 0.08,
        "Moderate": 0.10,
        "Aggressive": 0.12
    }
    
    # Volatility by risk profile (annualized percentage)
    VOLATILITY_BY_PROFILE = {
        "Conservative": 0.12,
        "Moderate": 0.18,
        "Aggressive": 0.25
    }
    
    def __init__(self, bootstrap_simulator=None, all_stocks_data: Dict = None):
        """
        Initialize wedding simulator.
        
        Parameters
        ----------
        bootstrap_simulator : object, optional
            Bootstrap simulator for Monte Carlo projections
        all_stocks_data : dict, optional
            Dictionary of stock dataframes for price lookup
        """
        self.bootstrap_simulator = bootstrap_simulator
        self.all_stocks_data = all_stocks_data or {}
    
    @staticmethod
    def calculate_required_saving(
        target_amount: float,
        years: float,
        annual_return: float
    ) -> float:
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
            Expected annual return rate (e.g., 0.08 for 8 percent)
            
        Returns
        -------
        float
            Required monthly saving (IDR)
        """
        months = years * 12
        monthly_rate = annual_return / 12
        
        if monthly_rate <= 0:
            return target_amount / months
        
        monthly_saving = target_amount * monthly_rate / ((1 + monthly_rate) ** months - 1)
        return monthly_saving
    
    @staticmethod
    def calculate_projected_value(
        monthly_saving: float,
        years: float,
        annual_return: float
    ) -> float:
        """
        Calculate projected future value of monthly savings.
        
        Parameters
        ----------
        monthly_saving : float
            Monthly saving amount (IDR)
        years : float
            Investment horizon in years
        annual_return : float
            Expected annual return rate
            
        Returns
        -------
        float
            Projected future value (IDR)
        """
        months = years * 12
        monthly_rate = annual_return / 12
        
        if monthly_rate <= 0:
            return monthly_saving * months
        
        future_value = monthly_saving * ((1 + monthly_rate) ** months - 1) / monthly_rate
        return future_value
    
    @staticmethod
    def calculate_months_to_goal(
        monthly_saving: float,
        target_amount: float,
        annual_return: float
    ) -> float:
        """
        Calculate months needed to reach target with monthly saving.
        
        Parameters
        ----------
        monthly_saving : float
            Monthly saving amount (IDR)
        target_amount : float
            Target amount to achieve (IDR)
        annual_return : float
            Expected annual return rate
            
        Returns
        -------
        float
            Number of months needed
        """
        monthly_rate = annual_return / 12
        
        if monthly_rate <= 0:
            return target_amount / monthly_saving if monthly_saving > 0 else float('inf')
        
        if monthly_saving <= 0:
            return float('inf')
        
        months = np.log(1 + (target_amount * monthly_rate) / monthly_saving) / np.log(1 + monthly_rate)
        return max(0, months)
    
    def get_scenario_returns(self, risk_profile: str) -> Dict[str, float]:
        """
        Get scenario returns based on risk profile.
        
        Parameters
        ----------
        risk_profile : str
            Risk tolerance ('Conservative', 'Moderate', 'Aggressive')
            
        Returns
        -------
        dict
            Dictionary with pessimistic, moderate, optimistic returns
        """
        base_return = self.EXPECTED_RETURNS.get(risk_profile, 0.10)
        volatility = self.VOLATILITY_BY_PROFILE.get(risk_profile, 0.18)
        
        return {
            "pessimistic": max(0.02, base_return - volatility),
            "moderate": base_return,
            "optimistic": min(0.18, base_return + volatility)
        }
    
    def get_monthly_breakdown(
        self,
        target_amount: float,
        years: float,
        monthly_saving: float,
        annual_return: float = 0.10
    ) -> List[WeddingMonthlyPlan]:
        """
        Get monthly breakdown of savings progress.
        
        Parameters
        ----------
        target_amount : float
            Target wedding fund amount (IDR)
        years : float
            Investment horizon in years
        monthly_saving : float
            Monthly saving amount (IDR)
        annual_return : float, default 0.10
            Expected annual return rate
            
        Returns
        -------
        list
            List of WeddingMonthlyPlan for each month
        """
        results = []
        total_months = int(years * 12)
        monthly_rate = annual_return / 12
        
        cumulative = 0
        target_per_month = target_amount / total_months if total_months > 0 else 0
        
        for month in range(1, total_months + 1):
            # Add monthly saving
            cumulative += monthly_saving
            
            # Apply monthly return
            if monthly_rate > 0:
                cumulative = cumulative * (1 + monthly_rate)
            
            target_progress = target_per_month * month
            on_track = cumulative >= target_progress
            
            # Calculate recommended saving to reach target by this month
            remaining_target = max(0, target_amount - cumulative)
            remaining_months = total_months - month
            if remaining_months > 0:
                recommended = self.calculate_required_saving(
                    remaining_target, remaining_months / 12, annual_return
                )
            else:
                recommended = 0
            
            results.append(WeddingMonthlyPlan(
                month=month,
                cumulative_savings=round(cumulative, 0),
                target_progress=round(target_progress, 0),
                projected_value=round(cumulative, 0),
                recommended_saving=round(recommended, 0),
                on_track=on_track
            ))
        
        return results
    
    def simulate(
        self,
        target_amount: float,
        years: float,
        monthly_saving: float,
        risk_profile: str = "Moderate"
    ) -> WeddingResult:
        """
        Run wedding fund simulation.
        
        Parameters
        ----------
        target_amount : float
            Target wedding fund amount (IDR)
        years : float
            Investment horizon (1-3 years recommended)
        monthly_saving : float
            User's monthly saving capacity (IDR)
        risk_profile : str, default "Moderate"
            Risk tolerance ('Conservative', 'Moderate', 'Aggressive')
            
        Returns
        -------
        WeddingResult
            Container with simulation results including success probability
        """
        expected_return = self.EXPECTED_RETURNS.get(risk_profile, 0.10)
        required = self.calculate_required_saving(target_amount, years, expected_return)
        
        # Calculate projected value
        projected_value = self.calculate_projected_value(monthly_saving, years, expected_return)
        
        # Calculate shortfall
        shortfall = max(0, target_amount - projected_value)
        
        # Calculate months to goal
        months_needed = self.calculate_months_to_goal(monthly_saving, target_amount, expected_return)
        
        # Determine status
        is_sufficient = monthly_saving >= required
        if is_sufficient:
            status = "ON TRACK"
        elif projected_value >= target_amount * 0.8:
            status = "NEAR TARGET"
        elif projected_value >= target_amount * 0.5:
            status = "CAUTION"
        else:
            status = "AT RISK"
        
        # Calculate probability of success using bootstrap (if available)
        probability = 0.0
        if self.bootstrap_simulator is not None:
            initial_investment = monthly_saving * years * 12
            try:
                result = self.bootstrap_simulator.simulate_with_target(
                    initial_investment=initial_investment,
                    target_amount=target_amount,
                    years=years
                )
                probability = result.probability_success
            except Exception:
                # Fallback to simple probability estimate
                if is_sufficient:
                    probability = 80.0
                elif projected_value >= target_amount * 0.9:
                    probability = 70.0
                elif projected_value >= target_amount * 0.7:
                    probability = 50.0
                else:
                    probability = 30.0
        else:
            # Simple probability estimate without bootstrap
            if is_sufficient:
                probability = 80.0
            elif projected_value >= target_amount * 0.9:
                probability = 70.0
            elif projected_value >= target_amount * 0.7:
                probability = 50.0
            else:
                probability = 30.0
        
        # Scenario analysis
        scenario_returns = self.get_scenario_returns(risk_profile)
        scenario_analysis = {}
        for scenario, ret in scenario_returns.items():
            scenario_value = self.calculate_projected_value(monthly_saving, years, ret)
            scenario_analysis[f"{scenario}_value"] = round(scenario_value, 0)
            scenario_analysis[f"{scenario}_coverage_pct"] = round(
                min(100, (scenario_value / target_amount) * 100), 1
            ) if target_amount > 0 else 0
        
        return WeddingResult(
            required_monthly=round(required, 0),
            is_sufficient=is_sufficient,
            probability_success=round(probability, 1),
            status=status,
            expected_return_percent=round(expected_return * 100, 1),
            recommended_stocks=self.STOCK_RECOMMENDATIONS.get(risk_profile, ["BBCA"]),
            projected_value=round(projected_value, 0),
            target_amount=target_amount,
            months_to_goal=round(months_needed, 1),
            shortfall=round(shortfall, 0),
            scenario_analysis=scenario_analysis
        )
    
    def compare_risk_profiles(
        self,
        target_amount: float,
        years: float,
        monthly_saving: float
    ) -> pd.DataFrame:
        """
        Compare outcomes across different risk profiles.
        
        Parameters
        ----------
        target_amount : float
            Target wedding fund amount (IDR)
        years : float
            Investment horizon in years
        monthly_saving : float
            Monthly saving amount (IDR)
            
        Returns
        -------
        pd.DataFrame
            Comparison of results for each risk profile
        """
        results = []
        
        for profile in ["Conservative", "Moderate", "Aggressive"]:
            sim_result = self.simulate(target_amount, years, monthly_saving, risk_profile=profile)
            
            results.append({
                'Risk Profile': profile,
                'Expected Return %': sim_result.expected_return_percent,
                'Required Monthly (Rp)': f"{sim_result.required_monthly:,.0f}",
                'Projected Value (Rp)': f"{sim_result.projected_value:,.0f}",
                'Shortfall (Rp)': f"{sim_result.shortfall:,.0f}",
                'Success Prob %': sim_result.probability_success,
                'Status': sim_result.status,
                'Months to Goal': sim_result.months_to_goal
            })
        
        return pd.DataFrame(results)
    
    def get_recommended_allocation(self, risk_profile: str) -> Dict[str, float]:
        """
        Get recommended allocation percentages for a risk profile.
        
        Parameters
        ----------
        risk_profile : str
            Risk tolerance ('Conservative', 'Moderate', 'Aggressive')
            
        Returns
        -------
        dict
            Allocation percentages for low, medium, high risk
        """
        allocations = {
            "Conservative": {"low": 80, "medium": 20, "high": 0},
            "Moderate": {"low": 60, "medium": 30, "high": 10},
            "Aggressive": {"low": 40, "medium": 40, "high": 20}
        }
        
        return allocations.get(risk_profile, allocations["Moderate"])


if __name__ == "__main__":
    print("=" * 60)
    print("WEDDING FUND SIMULATOR MODULE")
    print("=" * 60)
    print("\nThis module provides wedding fund simulation functionality.")
    print("\nAvailable functions:")
    print("  - WeddingSimulator.simulate(): Run wedding fund simulation")
    print("  - WeddingSimulator.get_monthly_breakdown(): Monthly savings breakdown")
    print("  - WeddingSimulator.compare_risk_profiles(): Compare risk profiles")
    print("  - WeddingSimulator.get_recommended_allocation(): Get asset allocation")
    print("  - WeddingSimulator.calculate_required_saving(): Calculate required monthly saving")
    print("  - WeddingSimulator.calculate_projected_value(): Project future value")
    print("  - WeddingSimulator.calculate_months_to_goal(): Calculate months needed")