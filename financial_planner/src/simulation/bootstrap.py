"""
Weighted Bootstrap Simulation Engine

Implements bootstrap resampling with crisis period weighting for financial projections.
Supports multi-stock portfolio simulation and scenario analysis.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field


# Default constants
DEFAULT_SIMULATIONS = 10000
TRADING_DAYS_PER_YEAR = 252
CRISIS_WEIGHT = 3.0
CRISIS_PERIOD = ("2020-03-01", "2020-06-30")


@dataclass
class BootstrapResult:
    """Container for bootstrap simulation results."""
    
    final_values: np.ndarray
    simulated_returns: np.ndarray
    probability_success: float
    median_value: float
    var_95: float
    expected_shortfall: float
    p5_value: float = 0.0
    p10_value: float = 0.0
    p25_value: float = 0.0
    p75_value: float = 0.0
    p90_value: float = 0.0
    p95_value: float = 0.0


class BootstrapSimulator:
    """
    Weighted bootstrap simulator using historical returns.
    
    This simulator resamples historical daily returns with replacement
    to generate possible future outcomes. Crisis periods can be given
    higher weight to ensure worst-case scenarios are represented.
    """
    
    def __init__(
        self,
        returns: np.ndarray,
        weights: Optional[np.ndarray] = None,
        crisis_weight: float = CRISIS_WEIGHT
    ):
        """
        Initialize bootstrap simulator.
        
        Parameters
        ----------
        returns : np.ndarray
            Array of historical daily returns (%)
        weights : np.ndarray, optional
            Probability weights for each return observation.
            If None, equal weights are used.
        crisis_weight : float, default 3.0
            Weight multiplier for crisis period (used if weights not provided)
        """
        self.crisis_weight = crisis_weight
        
        # Remove NaN values
        valid_mask = ~np.isnan(returns)
        self.returns = returns[valid_mask]
        
        if weights is not None:
            weights = np.array(weights)
            weights = weights[valid_mask]
            self.weights = weights / weights.sum()
        else:
            self.weights = np.ones(len(self.returns)) / len(self.returns)
    
    def simulate(
        self,
        n_days: int,
        n_simulations: int = DEFAULT_SIMULATIONS,
        random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Run bootstrap simulation.
        
        Parameters
        ----------
        n_days : int
            Number of trading days to simulate
        n_simulations : int, default 10000
            Number of simulation paths
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        np.ndarray
            Array of cumulative returns (%) for each simulation path
        """
        if len(self.returns) == 0:
            raise ValueError("No valid returns data available for simulation")
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Weighted sampling with replacement
        sampled_returns = np.random.choice(
            self.returns,
            size=(n_simulations, n_days),
            p=self.weights,
            replace=True
        )
        
        # Calculate cumulative return
        # Total return = (1 + r1/100) * (1 + r2/100) * ... - 1
        cumulative = (1 + sampled_returns / 100).prod(axis=1) - 1
        cumulative = cumulative * 100
        
        return cumulative
    
    def simulate_with_target(
        self,
        initial_investment: float,
        target_amount: float,
        years: float,
        n_simulations: int = DEFAULT_SIMULATIONS,
        random_seed: Optional[int] = None,
        monthly_contribution: float = 0.0
    ) -> BootstrapResult:
        """
        Run simulation and calculate success metrics.
        
        Parameters
        ----------
        initial_investment : float
            Initial investment amount (IDR)
        target_amount : float
            Target amount to achieve (IDR)
        years : float
            Investment horizon in years
        n_simulations : int, default 10000
            Number of simulation paths
        random_seed : int, optional
            Random seed for reproducibility
        monthly_contribution : float, default 0
            Monthly contribution amount (IDR)
            
        Returns
        -------
        BootstrapResult
            Container with simulation results and risk metrics
        """
        n_days = int(years * TRADING_DAYS_PER_YEAR)
        
        # Calculate future value of monthly contributions
        if monthly_contribution > 0:
            # Simulate returns first to get distribution
            simulated_returns = self.simulate(n_days, n_simulations, random_seed)
            
            # Calculate contribution growth for each path
            # For simplicity, use path-specific returns
            contribution_values = np.zeros(n_simulations)
            for i in range(n_simulations):
                # Simulate path for contributions (using same distribution)
                path_returns = np.random.choice(
                    self.returns,
                    size=n_days,
                    p=self.weights,
                    replace=True
                )
                cum_contrib = (1 + path_returns / 100).cumprod()
                # Future value of monthly contributions (simplified)
                monthly_rate = (1 + np.mean(path_returns) / 100) ** (1/12) - 1
                if monthly_rate > 0:
                    months = int(years * 12)
                    contrib_fv = monthly_contribution * ((1 + monthly_rate) ** months - 1) / monthly_rate
                else:
                    contrib_fv = monthly_contribution * months
                contribution_values[i] = contrib_fv
            
            total_final = initial_investment * (1 + simulated_returns / 100) + contribution_values
        else:
            simulated_returns = self.simulate(n_days, n_simulations, random_seed)
            total_final = initial_investment * (1 + simulated_returns / 100)
        
        # Calculate percentiles
        p5 = np.percentile(total_final, 5)
        p10 = np.percentile(total_final, 10)
        p25 = np.percentile(total_final, 25)
        p50 = np.percentile(total_final, 50)
        p75 = np.percentile(total_final, 75)
        p90 = np.percentile(total_final, 90)
        p95 = np.percentile(total_final, 95)
        
        # Calculate key metrics
        probability = (total_final >= target_amount).mean() * 100
        median_value = p50
        var_95 = p5
        expected_shortfall = total_final[total_final <= var_95].mean()
        
        return BootstrapResult(
            final_values=total_final,
            simulated_returns=simulated_returns,
            probability_success=probability,
            median_value=median_value,
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            p5_value=p5,
            p10_value=p10,
            p25_value=p25,
            p75_value=p75,
            p90_value=p90,
            p95_value=p95
        )
    
    def get_confidence_interval(
        self,
        initial_investment: float,
        years: float,
        confidence_level: float = 0.95,
        n_simulations: int = DEFAULT_SIMULATIONS
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for final portfolio value.
        
        Parameters
        ----------
        initial_investment : float
            Initial investment amount (IDR)
        years : float
            Investment horizon in years
        confidence_level : float, default 0.95
            Confidence level (e.g., 0.95 for 95% interval)
        n_simulations : int, default 10000
            Number of simulation paths
            
        Returns
        -------
        tuple
            (lower_bound, upper_bound) for the confidence interval
        """
        n_days = int(years * TRADING_DAYS_PER_YEAR)
        simulated_returns = self.simulate(n_days, n_simulations)
        final_values = initial_investment * (1 + simulated_returns / 100)
        
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100
        
        lower_bound = np.percentile(final_values, lower_percentile)
        upper_bound = np.percentile(final_values, upper_percentile)
        
        return lower_bound, upper_bound


def add_crisis_weights(
    df: pd.DataFrame,
    crisis_weight: float = CRISIS_WEIGHT,
    crisis_start: str = None,
    crisis_end: str = None
) -> pd.DataFrame:
    """
    Add crisis weights to dataframe for weighted bootstrap.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
    crisis_weight : float, default 3.0
        Weight multiplier for crisis period observations
    crisis_start : str, optional
        Start date of crisis period (default: "2020-03-01")
    crisis_end : str, optional
        End date of crisis period (default: "2020-06-30")
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - is_crisis: boolean flag for crisis period
        - bootstrap_weight: normalized sampling weights
    """
    df = df.copy()
    
    if crisis_start is None:
        crisis_start = CRISIS_PERIOD[0]
    if crisis_end is None:
        crisis_end = CRISIS_PERIOD[1]
    
    # Flag crisis period
    df["is_crisis"] = (df.index >= crisis_start) & (df.index <= crisis_end)
    
    # Assign weights: crisis period gets higher weight
    df["bootstrap_weight"] = 1.0
    df.loc[df["is_crisis"], "bootstrap_weight"] = crisis_weight
    
    # Normalize to sum to 1
    df["bootstrap_weight"] = df["bootstrap_weight"] / df["bootstrap_weight"].sum()
    
    return df


def create_bootstrap_simulator_from_df(
    df: pd.DataFrame,
    use_crisis_weight: bool = True,
    crisis_weight: float = CRISIS_WEIGHT
) -> BootstrapSimulator:
    """
    Create a BootstrapSimulator from a DataFrame with daily returns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'daily_return' column and optional 'bootstrap_weight' column
    use_crisis_weight : bool, default True
        Whether to apply crisis weighting
    crisis_weight : float, default 3.0
        Weight multiplier for crisis period (if weights not present)
        
    Returns
    -------
    BootstrapSimulator
        Configured bootstrap simulator
    """
    if 'daily_return' not in df.columns:
        raise ValueError("DataFrame must contain 'daily_return' column")
    
    returns = df['daily_return'].values
    
    if use_crisis_weight and 'bootstrap_weight' in df.columns:
        weights = df['bootstrap_weight'].values
    elif use_crisis_weight:
        df_weighted = add_crisis_weights(df, crisis_weight=crisis_weight)
        weights = df_weighted['bootstrap_weight'].values
    else:
        weights = None
    
    return BootstrapSimulator(returns, weights=weights, crisis_weight=crisis_weight)


def run_scenario_analysis(
    simulator: BootstrapSimulator,
    initial_investment: float,
    years_list: List[float],
    target_multiple: float = 2.0,
    n_simulations: int = DEFAULT_SIMULATIONS
) -> pd.DataFrame:
    """
    Run scenario analysis for different time horizons.
    
    Parameters
    ----------
    simulator : BootstrapSimulator
        Configured bootstrap simulator
    initial_investment : float
        Initial investment amount (IDR)
    years_list : list
        List of investment horizons in years
    target_multiple : float, default 2.0
        Target multiple (e.g., 2.0 means double the investment)
    n_simulations : int, default 10000
        Number of simulation paths per scenario
        
    Returns
    -------
    pd.DataFrame
        Scenario analysis results
    """
    results = []
    
    for years in years_list:
        target_amount = initial_investment * target_multiple
        result = simulator.simulate_with_target(
            initial_investment=initial_investment,
            target_amount=target_amount,
            years=years,
            n_simulations=n_simulations
        )
        
        results.append({
            'Years': years,
            'Target_Amount': target_amount,
            'Median_Value': result.median_value,
            'Success_Probability': round(result.probability_success, 1),
            'VaR_95': result.var_95,
            'Expected_Shortfall': result.expected_shortfall
        })
    
    return pd.DataFrame(results)


def compare_crisis_weighting(
    df: pd.DataFrame,
    initial_investment: float,
    target_amount: float,
    years: float,
    n_simulations: int = DEFAULT_SIMULATIONS
) -> Dict[str, BootstrapResult]:
    """
    Compare simulation results with and without crisis weighting.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'daily_return' column
    initial_investment : float
        Initial investment amount (IDR)
    target_amount : float
        Target amount to achieve (IDR)
    years : float
        Investment horizon in years
    n_simulations : int, default 10000
        Number of simulation paths
        
    Returns
    -------
    dict
        Dictionary with 'with_crisis' and 'without_crisis' results
    """
    # With crisis weighting
    simulator_with = create_bootstrap_simulator_from_df(df, use_crisis_weight=True)
    result_with = simulator_with.simulate_with_target(
        initial_investment=initial_investment,
        target_amount=target_amount,
        years=years,
        n_simulations=n_simulations
    )
    
    # Without crisis weighting
    simulator_without = create_bootstrap_simulator_from_df(df, use_crisis_weight=False)
    result_without = simulator_without.simulate_with_target(
        initial_investment=initial_investment,
        target_amount=target_amount,
        years=years,
        n_simulations=n_simulations
    )
    
    return {
        'with_crisis': result_with,
        'without_crisis': result_without
    }


if __name__ == "__main__":
    print("=" * 60)
    print("BOOTSTRAP SIMULATION MODULE")
    print("=" * 60)
    print("\nThis module provides bootstrap simulation functionality.")
    print("\nAvailable functions:")
    print("  - BootstrapSimulator: Class for weighted bootstrap simulation")
    print("  - add_crisis_weights(): Add crisis period weights to dataframe")
    print("  - create_bootstrap_simulator_from_df(): Create simulator from DataFrame")
    print("  - run_scenario_analysis(): Run scenario analysis for multiple horizons")
    print("  - compare_crisis_weighting(): Compare with/without crisis weighting")