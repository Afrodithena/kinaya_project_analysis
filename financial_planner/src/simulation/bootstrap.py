"""
Weighted Bootstrap Simulation Engine

Implements bootstrap resampling with crisis period weighting for financial projections.
"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass

from src.config import DEFAULT_SIMULATIONS, TRADING_DAYS_PER_YEAR, CRISIS_WEIGHT, CRISIS_PERIOD


@dataclass
class BootstrapResult:
    """Container for bootstrap simulation results."""
    
    final_values: np.ndarray
    simulated_returns: np.ndarray
    probability_success: float
    median_value: float
    var_95: float
    expected_shortfall: float


class BootstrapSimulator:
    """
    Weighted bootstrap simulator using historical returns.
    
    This simulator resamples historical daily returns with replacement
    to generate possible future outcomes. Crisis periods can be given
    higher weight to ensure worst-case scenarios are represented.
    """
    
    def __init__(self, returns: np.ndarray, weights: Optional[np.ndarray] = None):
        """
        Initialize bootstrap simulator.
        
        Parameters
        ----------
        returns : np.ndarray
            Array of historical daily returns (%)
        weights : np.ndarray, optional
            Probability weights for each return observation.
            If None, equal weights are used.
        """
        # Remove NaN values
        valid_mask = ~np.isnan(returns)
        self.returns = returns[valid_mask]
        
        if weights is not None:
            weights = np.array(weights)
            weights = weights[valid_mask]
            self.weights = weights / weights.sum()
        else:
            self.weights = np.ones(len(self.returns)) / len(self.returns)
    
    def simulate(self, n_days: int, n_simulations: int = DEFAULT_SIMULATIONS) -> np.ndarray:
        """
        Run bootstrap simulation.
        
        Parameters
        ----------
        n_days : int
            Number of trading days to simulate
        n_simulations : int
            Number of simulation paths
            
        Returns
        -------
        np.ndarray
            Array of cumulative returns (%) for each simulation path
        """
        if len(self.returns) == 0:
            raise ValueError("No valid returns data available for simulation")
        
        # Weighted sampling with replacement
        sampled_returns = np.random.choice(
            self.returns,
            size=(n_simulations, n_days),
            p=self.weights,
            replace=True
        )
        
        # Calculate cumulative return
        # Total return = (1 + r1) * (1 + r2) * ... - 1
        cumulative = (1 + sampled_returns / 100).prod(axis=1) - 1
        cumulative = cumulative * 100
        
        return cumulative
    
    def simulate_with_target(
        self,
        initial_investment: float,
        target_amount: float,
        years: float,
        n_simulations: int = DEFAULT_SIMULATIONS
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
        n_simulations : int
            Number of simulation paths
            
        Returns
        -------
        BootstrapResult
            Container with simulation results and risk metrics
        """
        n_days = int(years * TRADING_DAYS_PER_YEAR)
        simulated_returns = self.simulate(n_days, n_simulations)
        
        final_values = initial_investment * (1 + simulated_returns / 100)
        
        # Calculate key metrics
        probability = (final_values >= target_amount).mean() * 100
        median_value = np.median(final_values)
        var_95 = np.percentile(final_values, 5)
        expected_shortfall = final_values[final_values <= var_95].mean()
        
        return BootstrapResult(
            final_values=final_values,
            simulated_returns=simulated_returns,
            probability_success=probability,
            median_value=median_value,
            var_95=var_95,
            expected_shortfall=expected_shortfall
        )


def add_crisis_weights(
    df: pd.DataFrame,
    crisis_weight: float = CRISIS_WEIGHT
) -> pd.DataFrame:
    """
    Add crisis weights to dataframe for weighted bootstrap.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
    crisis_weight : float, default CRISIS_WEIGHT (3.0)
        Weight multiplier for crisis period observations
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - is_crisis: boolean flag for crisis period
        - bootstrap_weight: normalized sampling weights
    """
    df = df.copy()
    crisis_start, crisis_end = CRISIS_PERIOD
    
    # Flag crisis period
    df["is_crisis"] = (df.index >= crisis_start) & (df.index <= crisis_end)
    
    # Assign weights: crisis period gets higher weight
    df["bootstrap_weight"] = 1.0
    df.loc[df["is_crisis"], "bootstrap_weight"] = crisis_weight
    
    # Normalize to sum to 1
    df["bootstrap_weight"] = df["bootstrap_weight"] / df["bootstrap_weight"].sum()
    
    return df