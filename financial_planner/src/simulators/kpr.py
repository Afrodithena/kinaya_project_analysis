"""
KPR Readiness Simulator (3-5 years)

Simulates down payment accumulation for house purchase, including hidden costs
(BPHTB, notary fees, bank provisions) and dividend synergy from bank stocks.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class KPRResult:
    """Container for KPR simulator results."""
    
    down_payment: float
    bphtb: float
    notary_fee: float
    bank_provision: float
    total_needed: float
    total_saved: float
    progress_percent: float
    readiness_score: str
    dividend_coverage_months: float
    projected_investment_value: float = 0.0
    additional_needed: float = 0.0
    recommended_monthly_saving: float = 0.0
    best_stock_for_dividend: str = ""
    annual_dividend_income: float = 0.0


@dataclass
class KPRMonthlyPlan:
    """Container for monthly saving plan."""
    
    year: int
    target_savings: float
    current_savings: float
    shortfall: float
    recommended_monthly: float
    dividend_contribution: float


# Default hidden costs (percentages)
DEFAULT_HIDDEN_COSTS = {
    "bphtb": 0.05,       # 5 percent of house price
    "notary": 0.01,      # 1 percent of house price
    "bank_provision": 0.01  # 1 percent of loan amount
}

# Recommended dividend stocks for KPR coverage
RECOMMENDED_DIVIDEND_STOCKS = ["BBRI", "BBCA", "BMRI", "ASII", "PGAS"]


class KPRSimulator:
    """
    KPR readiness simulator for 3-5 year horizon.
    
    Strategy: 50 percent low risk + 40 percent medium risk + 10 percent high risk stocks.
    Hidden costs included:
        - BPHTB: 5 percent of house price
        - Notary fee: 1 percent of house price
        - Bank provision: 1 percent of loan amount
    
    Dividend synergy: Bank stocks (BBCA, BBRI, BMRI) typically yield 3-4 percent,
    which can cover 2-4 months of KPR payments annually.
    """
    
    def __init__(
        self,
        dividend_data: Optional[Dict] = None,
        all_stocks_data: Optional[Dict] = None,
        hidden_costs: Optional[Dict] = None
    ):
        """
        Initialize KPR Simulator.
        
        Parameters
        ----------
        dividend_data : dict, optional
            Dictionary of dividend data for stock yield calculation
        all_stocks_data : dict, optional
            Dictionary of stock dataframes for price lookup
        hidden_costs : dict, optional
            Custom hidden cost percentages (defaults: bphtb 0.05, notary 0.01, bank_provision 0.01)
        """
        self.dividend_data = dividend_data or {}
        self.all_stocks_data = all_stocks_data or {}
        self.hidden_costs = hidden_costs or DEFAULT_HIDDEN_COSTS
    
    @staticmethod
    def calculate_hidden_costs(
        house_price: float,
        dp_percent: float,
        hidden_costs: Dict = None
    ) -> Tuple[float, float, float, float, float]:
        """
        Calculate all costs for KPR down payment.
        
        Parameters
        ----------
        house_price : float
            Total house price (IDR)
        dp_percent : float
            Down payment percentage (5-30 percent)
        hidden_costs : dict, optional
            Custom hidden cost percentages
            
        Returns
        -------
        tuple
            (down_payment, bphtb, notary_fee, bank_provision, total_needed)
        """
        if hidden_costs is None:
            hidden_costs = DEFAULT_HIDDEN_COSTS
        
        down_payment = house_price * dp_percent / 100
        bphtb = house_price * hidden_costs["bphtb"]
        notary_fee = house_price * hidden_costs["notary"]
        bank_provision = (house_price - down_payment) * hidden_costs["bank_provision"]
        total_needed = down_payment + bphtb + notary_fee + bank_provision
        
        return down_payment, bphtb, notary_fee, bank_provision, total_needed
    
    @staticmethod
    def calculate_dividend_coverage(
        investment_amount: float,
        dividend_yield: float,
        monthly_kpr: float
    ) -> float:
        """
        Calculate how many months of KPR payments dividends can cover.
        
        Parameters
        ----------
        investment_amount : float
            Total investment amount (IDR)
        dividend_yield : float
            Average dividend yield of portfolio (e.g., 0.035 for 3.5 percent)
        monthly_kpr : float
            Monthly KPR installment (IDR)
            
        Returns
        -------
        float
            Number of months covered by annual dividends
        """
        if monthly_kpr <= 0:
            return 0.0
        
        yearly_dividend = investment_amount * dividend_yield
        return yearly_dividend / monthly_kpr
    
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
    def project_future_value(
        monthly_saving: float,
        years: float,
        annual_return: float
    ) -> float:
        """
        Project future value of monthly savings.
        
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
    
    def get_dividend_yield(self, stock: str, current_price: float = None) -> float:
        """
        Calculate average dividend yield from historical data.
        
        Parameters
        ----------
        stock : str
            Stock ticker symbol
        current_price : float, optional
            Current stock price (if None, fetches from data)
            
        Returns
        -------
        float
            Average dividend yield (as decimal, e.g., 0.035 for 3.5 percent)
        """
        if stock not in self.dividend_data:
            return 0.0
        
        div_values = list(self.dividend_data[stock].values())
        if not div_values:
            return 0.0
        
        avg_dps = sum(div_values) / len(div_values)
        
        if current_price is None and stock in self.all_stocks_data:
            df = self.all_stocks_data[stock]
            if 'adjusted_close' in df.columns:
                current_price = df['adjusted_close'].iloc[-1]
            else:
                current_price = df['close'].iloc[-1]
        
        if current_price and current_price > 0:
            return (avg_dps / current_price) / 100  # Convert to decimal
        
        return 0.0
    
    def get_best_dividend_stock(self) -> Tuple[str, float]:
        """
        Find the stock with highest dividend yield among recommended stocks.
        
        Returns
        -------
        tuple
            (stock_ticker, dividend_yield_percent)
        """
        best_stock = None
        best_yield = 0.0
        
        for stock in RECOMMENDED_DIVIDEND_STOCKS:
            if stock in self.dividend_data:
                yield_pct = self.get_dividend_yield(stock) * 100
                if yield_pct > best_yield:
                    best_yield = yield_pct
                    best_stock = stock
        
        return best_stock, best_yield
    
    def simulate(
        self,
        house_price: float,
        dp_percent: float,
        monthly_kpr: float,
        years: float,
        monthly_saving: float,
        annual_return: float = 0.10,
        dividend_yield: float = None
    ) -> KPRResult:
        """
        Run KPR readiness simulation.
        
        Parameters
        ----------
        house_price : float
            Total house price (IDR)
        dp_percent : float
            Down payment percentage (5-30 percent)
        monthly_kpr : float
            Target monthly KPR installment (IDR)
        years : float
            Years to save for down payment (3-5)
        monthly_saving : float
            Monthly saving capacity (IDR)
        annual_return : float, default 0.10
            Expected annual return rate (10 percent default)
        dividend_yield : float, optional
            Custom dividend yield (if None, uses best available stock)
            
        Returns
        -------
        KPRResult
            Container with simulation results including readiness score
        """
        # Calculate all costs
        down_payment, bphtb, notary_fee, bank_provision, total_needed = self.calculate_hidden_costs(
            house_price, dp_percent, self.hidden_costs
        )
        
        # Project future value of savings
        projected_investment = self.project_future_value(monthly_saving, years, annual_return)
        
        # Total savings accumulated (simple without return)
        total_saved_simple = monthly_saving * years * 12
        
        # Progress and readiness
        progress = min(100.0, (projected_investment / total_needed) * 100) if total_needed > 0 else 0.0
        readiness = "READY" if projected_investment >= total_needed else "NOT READY"
        
        # Additional needed
        additional_needed = max(0, total_needed - projected_investment)
        
        # Recommended monthly saving to reach target
        recommended_monthly = self.calculate_monthly_saving_needed(total_needed, years, annual_return)
        
        # Dividend coverage
        if dividend_yield is None:
            best_stock, dividend_yield_pct = self.get_best_dividend_stock()
            dividend_yield = dividend_yield_pct / 100
        else:
            best_stock, dividend_yield_pct = self.get_best_dividend_stock()
        
        coverage_months = self.calculate_dividend_coverage(projected_investment, dividend_yield, monthly_kpr)
        annual_dividend = projected_investment * dividend_yield if dividend_yield else 0
        
        return KPRResult(
            down_payment=round(down_payment, 0),
            bphtb=round(bphtb, 0),
            notary_fee=round(notary_fee, 0),
            bank_provision=round(bank_provision, 0),
            total_needed=round(total_needed, 0),
            total_saved=round(total_saved_simple, 0),
            progress_percent=round(progress, 1),
            readiness_score=readiness,
            dividend_coverage_months=round(coverage_months, 1),
            projected_investment_value=round(projected_investment, 0),
            additional_needed=round(additional_needed, 0),
            recommended_monthly_saving=round(recommended_monthly, 0),
            best_stock_for_dividend=best_stock or "",
            annual_dividend_income=round(annual_dividend, 0)
        )
    
    def simulate_with_breakdown(
        self,
        house_price: float,
        dp_percent: float,
        monthly_kpr: float,
        years: float,
        monthly_saving: float,
        annual_return: float = 0.10
    ) -> List[KPRMonthlyPlan]:
        """
        Simulate KPR readiness with yearly breakdown.
        
        Parameters
        ----------
        house_price : float
            Total house price (IDR)
        dp_percent : float
            Down payment percentage
        monthly_kpr : float
            Monthly KPR installment (IDR)
        years : float
            Years to save
        monthly_saving : float
            Monthly saving amount
        annual_return : float, default 0.10
            Expected annual return rate
            
        Returns
        -------
        list
            List of KPRMonthlyPlan for each year
        """
        down_payment, bphtb, notary_fee, bank_provision, total_needed = self.calculate_hidden_costs(
            house_price, dp_percent, self.hidden_costs
        )
        
        results = []
        cumulative_savings = 0
        
        for year in range(1, int(years) + 1):
            # Project savings with returns
            cumulative_savings = self.project_future_value(monthly_saving, year, annual_return)
            
            # Target savings by this year (linear progression)
            target_by_year = total_needed * (year / years)
            
            shortfall = max(0, target_by_year - cumulative_savings)
            
            # Dividend contribution estimate (using 3.5 percent yield on current savings)
            dividend_contribution = cumulative_savings * 0.035
            
            results.append(KPRMonthlyPlan(
                year=2024 + year,
                target_savings=round(target_by_year, 0),
                current_savings=round(cumulative_savings, 0),
                shortfall=round(shortfall, 0),
                recommended_monthly=round(monthly_saving, 0),
                dividend_contribution=round(dividend_contribution, 0)
            ))
        
        return results
    
    def compare_stocks_for_kpr(
        self,
        stocks: List[str],
        investment_amount: float,
        monthly_kpr: float
    ) -> pd.DataFrame:
        """
        Compare different stocks for KPR dividend coverage.
        
        Parameters
        ----------
        stocks : list
            List of stock tickers to compare
        investment_amount : float
            Total investment amount (IDR)
        monthly_kpr : float
            Monthly KPR installment (IDR)
            
        Returns
        -------
        pd.DataFrame
            Comparison of dividend coverage across stocks
        """
        results = []
        
        for stock in stocks:
            if stock not in self.dividend_data:
                continue
            
            dividend_yield = self.get_dividend_yield(stock) * 100
            annual_dividend = investment_amount * (dividend_yield / 100)
            months_covered = annual_dividend / monthly_kpr if monthly_kpr > 0 else 0
            
            # Get current price for share calculation
            current_price = 0
            if stock in self.all_stocks_data:
                df = self.all_stocks_data[stock]
                if 'adjusted_close' in df.columns:
                    current_price = df['adjusted_close'].iloc[-1]
                else:
                    current_price = df['close'].iloc[-1]
            
            shares = int(investment_amount / current_price) if current_price > 0 else 0
            
            results.append({
                'Stock': stock,
                'Dividend_Yield_Pct': round(dividend_yield, 2),
                'Annual_Dividend_Rp': round(annual_dividend, 0),
                'Months_Covered': round(months_covered, 1),
                'Shares_Owned': shares,
                'Current_Price_Rp': round(current_price, 0)
            })
        
        df_result = pd.DataFrame(results)
        df_result = df_result.sort_values('Months_Covered', ascending=False)
        
        return df_result
    
    def get_scenario_analysis(
        self,
        house_price: float,
        dp_percent: float,
        monthly_kpr: float,
        years: float,
        monthly_saving: float
    ) -> Dict[str, Dict]:
        """
        Run scenario analysis with different return assumptions.
        
        Parameters
        ----------
        house_price : float
            Total house price (IDR)
        dp_percent : float
            Down payment percentage
        monthly_kpr : float
            Monthly KPR installment (IDR)
        years : float
            Years to save
        monthly_saving : float
            Monthly saving amount
            
        Returns
        -------
        dict
            Scenario analysis results for pessimistic, moderate, optimistic returns
        """
        scenarios = {
            "pessimistic": 0.06,
            "moderate": 0.10,
            "optimistic": 0.14
        }
        
        results = {}
        
        for scenario_name, annual_return in scenarios.items():
            result = self.simulate(
                house_price=house_price,
                dp_percent=dp_percent,
                monthly_kpr=monthly_kpr,
                years=years,
                monthly_saving=monthly_saving,
                annual_return=annual_return
            )
            
            results[scenario_name] = {
                "projected_value": result.projected_investment_value,
                "progress_percent": result.progress_percent,
                "readiness_score": result.readiness_score,
                "dividend_coverage_months": result.dividend_coverage_months,
                "recommended_monthly": result.recommended_monthly_saving
            }
        
        return results


if __name__ == "__main__":
    print("=" * 60)
    print("KPR READINESS SIMULATOR MODULE")
    print("=" * 60)
    print("\nThis module provides KPR simulation functionality.")
    print("\nAvailable functions:")
    print("  - KPRSimulator.simulate(): Run KPR readiness simulation")
    print("  - KPRSimulator.simulate_with_breakdown(): Yearly breakdown")
    print("  - KPRSimulator.compare_stocks_for_kpr(): Compare stocks for dividend coverage")
    print("  - KPRSimulator.get_scenario_analysis(): Scenario analysis with different returns")
    print("  - KPRSimulator.calculate_hidden_costs(): Calculate down payment with hidden costs")
    print("  - KPRSimulator.calculate_monthly_saving_needed(): Calculate required monthly saving")