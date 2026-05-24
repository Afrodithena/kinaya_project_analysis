"""
Shift Planner - Generates recommendations to transition from current to target portfolio.
Enhanced with transaction cost calculation, tax impact, and rebalancing suggestions.
"""

from typing import List, Dict, Optional, Any
from datetime import date
import pandas as pd
import numpy as np


class ShiftPlanner:
    """
    Generates shift plan from current portfolio to recommended portfolio.
    Calculates what to sell, what to buy, transaction costs, and tax implications.
    Supports gradual rebalancing and threshold-based adjustments.
    """
    
    # Transaction costs (matching PortfolioManager)
    BROKER_FEE_BUY = 0.0015      # 0.15% for buying
    BROKER_FEE_SELL = 0.0025     # 0.25% for selling
    VAT_RATE = 0.11              # 11% VAT on broker fees
    CAPITAL_GAIN_TAX = 0.001     # 0.1% final tax on capital gains
    
    def __init__(self, portfolio_manager, all_stocks_data: dict):
        """
        Initialize shift planner.
        
        Parameters
        ----------
        portfolio_manager : PortfolioManager
            Instance with user's current positions (from portfolio_manager.py)
        all_stocks_data : dict
            Dictionary of stock dataframes with 'close' column
        """
        self.portfolio_manager = portfolio_manager
        self.all_stocks_data = all_stocks_data
    
    def get_current_price(self, stock: str, use_adjusted: bool = True) -> Optional[float]:
        """Get current market price."""
        df = self.all_stocks_data.get(stock)
        if df is None:
            return None
        
        price_col = 'adjusted_close' if use_adjusted and 'adjusted_close' in df.columns else 'close'
        
        if price_col not in df.columns:
            return None
        
        return float(df[price_col].iloc[-1])
    
    def calculate_transaction_cost(self, amount: float, is_buy: bool) -> Dict[str, float]:
        """
        Calculate transaction costs including broker fee and VAT.
        
        Parameters
        ----------
        amount : float
            Transaction amount in IDR
        is_buy : bool
            True for buy, False for sell
        
        Returns
        -------
        dict
            Breakdown of transaction costs
        """
        fee_rate = self.BROKER_FEE_BUY if is_buy else self.BROKER_FEE_SELL
        broker_fee = amount * fee_rate
        vat = broker_fee * self.VAT_RATE
        total_cost = broker_fee + vat
        
        return {
            'broker_fee': round(broker_fee, 0),
            'vat': round(vat, 0),
            'total': round(total_cost, 0)
        }
    
    def calculate_capital_gain_tax(self, gain: float) -> float:
        """Calculate capital gains tax (0.1% of gain)."""
        return max(0, gain * self.CAPITAL_GAIN_TAX)
    
    def get_unrealized_gain_loss(self) -> tuple:
        """Get current portfolio unrealized gain/loss using portfolio manager."""
        return self.portfolio_manager.get_unrealized_gain_loss()
    
    def generate_shift_plan(
        self,
        recommended_stocks: List[str],
        recommended_allocations: Dict[str, float],
        target_portfolio_value: float = None,
        rebalancing_threshold_pct: float = 5.0,
        preserve_cash: bool = True,
        consider_tax: bool = True
    ) -> Dict:
        """
        Generate plan to shift from current to recommended portfolio.
        
        Parameters
        ----------
        recommended_stocks : list
            List of recommended stock tickers
        recommended_allocations : dict
            Allocation percentage for each recommended stock
        target_portfolio_value : float, optional
            Target portfolio value (default: maintain current value)
        rebalancing_threshold_pct : float
            Threshold percentage for rebalancing (default: 5%)
        preserve_cash : bool
            Whether to preserve cash from sales (vs reinvest immediately)
        consider_tax : bool
            Whether to include tax impact in recommendations
        
        Returns
        -------
        dict
            Shift plan with sell/buy recommendations and cost analysis
        """
        current_df, total_cost, total_current = self.get_unrealized_gain_loss()
        
        if target_portfolio_value is None:
            target_portfolio_value = total_current
        
        plan = {
            'generated_date': date.today().strftime('%Y-%m-%d'),
            'current_portfolio': current_df.to_dict('records') if not current_df.empty else [],
            'total_invested': round(total_cost, 0),
            'current_value': round(total_current, 0),
            'target_value': round(target_portfolio_value, 0),
            'total_return_pct': round(((total_current - total_cost) / total_cost * 100), 1) if total_cost > 0 else 0,
            'sell_recommendations': [],
            'buy_recommendations': [],
            'keep_recommendations': [],
            'adjust_recommendations': [],
            'notes': [],
            'estimated_transaction_costs': 0,
            'estimated_tax_impact': 0,
            'net_cash_required': 0,
            'net_cash_from_sales': 0
        }
        
        # No positions case
        if current_df.empty:
            plan['notes'].append("No existing positions found. Start building portfolio with buy recommendations below.")
            
            for stock in recommended_stocks:
                allocation_pct = recommended_allocations.get(stock, 0)
                target_value_stock = target_portfolio_value * (allocation_pct / 100) if allocation_pct > 0 else 0
                current_price = self.get_current_price(stock)
                
                if current_price and current_price > 0 and target_value_stock > 0:
                    shares_to_buy = int(target_value_stock / current_price)
                    buy_cost = shares_to_buy * current_price
                    
                    if shares_to_buy > 0:
                        fees = self.calculate_transaction_cost(buy_cost, is_buy=True)
                        plan['buy_recommendations'].append({
                            'stock': stock,
                            'target_allocation_pct': round(allocation_pct, 1),
                            'estimated_shares': shares_to_buy,
                            'estimated_cost': round(buy_cost, 0),
                            'estimated_fees': round(fees['total'], 0),
                            'total_cost': round(buy_cost + fees['total'], 0),
                            'reason': f"Recommended for your portfolio with {allocation_pct:.0f}% allocation"
                        })
                        plan['estimated_transaction_costs'] += fees['total']
            
            plan['net_cash_required'] = sum([b['total_cost'] for b in plan['buy_recommendations']])
            return plan
        
        # Analysis of current vs target
        current_stocks = set(current_df['Stock'].tolist())
        recommended_stocks_set = set(recommended_stocks)
        
        overlap = current_stocks.intersection(recommended_stocks_set)
        to_sell = current_stocks - recommended_stocks_set
        to_buy = recommended_stocks_set - current_stocks
        
        # Calculate current allocations
        current_allocations = {}
        for _, row in current_df.iterrows():
            current_allocations[row['Stock']] = (row['Current Value'] / total_current) * 100 if total_current > 0 else 0
        
        # Generate sell recommendations (stocks not in target)
        for _, row in current_df.iterrows():
            if row['Stock'] in to_sell:
                sell_fees = self.calculate_transaction_cost(row['Current Value'], is_buy=False)
                capital_gain = row['Unrealized Gain/Loss']
                tax = self.calculate_capital_gain_tax(capital_gain) if consider_tax and capital_gain > 0 else 0
                
                sell_rec = {
                    'stock': row['Stock'],
                    'shares': int(row['Shares']),
                    'current_price': row['Current Price'],
                    'estimated_proceeds': round(row['Current Value'], 0),
                    'estimated_fees': round(sell_fees['total'], 0),
                    'estimated_tax': round(tax, 0),
                    'net_proceeds': round(row['Current Value'] - sell_fees['total'] - tax, 0),
                    'unrealized_return_pct': row['Return %'],
                    'reason': "Not in recommended portfolio for your goal and risk profile"
                }
                plan['sell_recommendations'].append(sell_rec)
                plan['estimated_transaction_costs'] += sell_fees['total']
                plan['estimated_tax_impact'] += tax
        
        # Generate keep recommendations (stocks already in target, allocation within threshold)
        for _, row in current_df.iterrows():
            if row['Stock'] in overlap:
                current_allocation = current_allocations.get(row['Stock'], 0)
                target_allocation = recommended_allocations.get(row['Stock'], 0)
                allocation_diff = current_allocation - target_allocation
                
                if abs(allocation_diff) <= rebalancing_threshold_pct:
                    plan['keep_recommendations'].append({
                        'stock': row['Stock'],
                        'current_allocation_pct': round(current_allocation, 1),
                        'target_allocation_pct': target_allocation,
                        'current_value': round(row['Current Value'], 0),
                        'return_pct': row['Return %'],
                        'note': f"Within {rebalancing_threshold_pct}% of target allocation"
                    })
                else:
                    adjustment = {
                        'stock': row['Stock'],
                        'current_allocation_pct': round(current_allocation, 1),
                        'target_allocation_pct': target_allocation,
                        'current_value': round(row['Current Value'], 0),
                        'allocation_diff_pct': round(allocation_diff, 1),
                        'return_pct': row['Return %']
                    }
                    
                    if allocation_diff > 0:
                        reduction_value = row['Current Value'] * (allocation_diff / 100)
                        shares_to_sell = int(reduction_value / row['Current Price'])
                        
                        if shares_to_sell > 0:
                            sell_fees = self.calculate_transaction_cost(shares_to_sell * row['Current Price'], is_buy=False)
                            adjustment['action'] = 'SELL'
                            adjustment['estimated_shares'] = shares_to_sell
                            adjustment['estimated_proceeds'] = round(shares_to_sell * row['Current Price'], 0)
                            adjustment['estimated_fees'] = round(sell_fees['total'], 0)
                            adjustment['reason'] = f"Overweight by {round(allocation_diff, 1)}%"
                            plan['sell_recommendations'].append(adjustment)
                            plan['estimated_transaction_costs'] += sell_fees['total']
                    else:
                        increase_value = row['Current Value'] * (-allocation_diff / 100)
                        shares_to_buy = int(increase_value / row['Current Price'])
                        
                        if shares_to_buy > 0:
                            buy_fees = self.calculate_transaction_cost(shares_to_buy * row['Current Price'], is_buy=True)
                            adjustment['action'] = 'BUY'
                            adjustment['estimated_shares'] = shares_to_buy
                            adjustment['estimated_cost'] = round(shares_to_buy * row['Current Price'], 0)
                            adjustment['estimated_fees'] = round(buy_fees['total'], 0)
                            adjustment['total_cost'] = round(shares_to_buy * row['Current Price'] + buy_fees['total'], 0)
                            adjustment['reason'] = f"Underweight by {round(-allocation_diff, 1)}%"
                            plan['adjust_recommendations'].append(adjustment)
                            plan['estimated_transaction_costs'] += buy_fees['total']
        
        # Calculate net cash from sales
        net_cash_from_sales = sum([s.get('net_proceeds', s.get('estimated_proceeds', 0)) 
                                   for s in plan['sell_recommendations']])
        plan['net_cash_from_sales'] = round(net_cash_from_sales, 0)
        
        # Generate buy recommendations (stocks not currently held)
        total_alloc = sum(recommended_allocations.values())
        
        if preserve_cash:
            available_for_buy = net_cash_from_sales
        else:
            remaining_value = total_current - sum([s.get('estimated_proceeds', 0) for s in plan['sell_recommendations']])
            available_for_buy = remaining_value + net_cash_from_sales
        
        for stock in to_buy:
            allocation_pct = recommended_allocations.get(stock, 0)
            target_value_stock = (available_for_buy + remaining_value) * (allocation_pct / total_alloc) if total_alloc > 0 else 0
            
            current_price = self.get_current_price(stock)
            if current_price and current_price > 0 and target_value_stock > 0:
                shares_to_buy = int(target_value_stock / current_price)
                buy_cost = shares_to_buy * current_price
                
                if shares_to_buy > 0:
                    fees = self.calculate_transaction_cost(buy_cost, is_buy=True)
                    buy_rec = {
                        'stock': stock,
                        'target_allocation_pct': round(allocation_pct, 1),
                        'estimated_shares': shares_to_buy,
                        'estimated_cost': round(buy_cost, 0),
                        'estimated_fees': round(fees['total'], 0),
                        'total_cost': round(buy_cost + fees['total'], 0),
                        'reason': f"Recommended for your portfolio with {allocation_pct:.0f}% allocation",
                        'current_price': round(current_price, 0)
                    }
                    plan['buy_recommendations'].append(buy_rec)
                    plan['estimated_transaction_costs'] += fees['total']
        
        # Calculate net cash required
        total_buy_cost = sum([b.get('total_cost', b.get('estimated_cost', 0)) 
                              for b in plan['buy_recommendations']])
        plan['net_cash_required'] = round(max(0, total_buy_cost - net_cash_from_sales), 0)
        
        # Add execution notes
        if plan['sell_recommendations']:
            plan['notes'].append(f"Sell {len(plan['sell_recommendations'])} stock(s) to rebalance")
        if plan['buy_recommendations']:
            plan['notes'].append(f"Buy {len(plan['buy_recommendations'])} new stock(s) to achieve target allocation")
        if plan['adjust_recommendations']:
            plan['notes'].append(f"{len(plan['adjust_recommendations'])} position(s) need adjustment")
        if plan['net_cash_required'] > 0:
            plan['notes'].append(f"Additional capital of Rp {plan['net_cash_required']:,.0f} needed for buys")
        elif plan['net_cash_from_sales'] > total_buy_cost:
            plan['notes'].append(f"Excess cash of Rp {plan['net_cash_from_sales'] - total_buy_cost:,.0f} available")
        
        if consider_tax and plan['estimated_tax_impact'] > 0:
            plan['notes'].append(f"Estimated capital gains tax: Rp {plan['estimated_tax_impact']:,.0f}")
        
        plan['estimated_transaction_costs'] = round(plan['estimated_transaction_costs'], 0)
        plan['estimated_tax_impact'] = round(plan['estimated_tax_impact'], 0)
        
        return plan
    
    def generate_gradual_shift_plan(
        self,
        recommended_stocks: List[str],
        recommended_allocations: Dict[str, float],
        months: int = 3,
        monthly_budget: float = None
    ) -> Dict:
        """
        Generate a gradual shift plan spread over multiple months.
        
        Parameters
        ----------
        recommended_stocks : list
            List of recommended stock tickers
        recommended_allocations : dict
            Allocation percentage for each recommended stock
        months : int
            Number of months to spread the shift over
        monthly_budget : float, optional
            Monthly budget for buying (default: calculated from sales)
        
        Returns
        -------
        dict
            Gradual shift plan with monthly recommendations
        """
        full_plan = self.generate_shift_plan(recommended_stocks, recommended_allocations)
        
        total_sell_proceeds = full_plan['net_cash_from_sales']
        total_buy_cost = sum([b['total_cost'] for b in full_plan['buy_recommendations']])
        
        if monthly_budget is None:
            monthly_budget = max(total_sell_proceeds / months, 1_000_000)
        
        gradual_plan = {
            'total_months': months,
            'monthly_budget': round(monthly_budget, 0),
            'total_sell_proceeds': total_sell_proceeds,
            'total_buy_cost': total_buy_cost,
            'monthly_schedule': []
        }
        
        remaining_sell_proceeds = total_sell_proceeds
        remaining_buy_cost = total_buy_cost
        
        for month in range(1, months + 1):
            month_plan = {
                'month': month,
                'sell_this_month': [],
                'buy_this_month': [],
                'cash_flow': 0
            }
            
            if remaining_sell_proceeds > 0:
                sell_this_month = min(remaining_sell_proceeds / (months - month + 1), monthly_budget)
                month_plan['cash_flow'] += sell_this_month
                remaining_sell_proceeds -= sell_this_month
            
            if remaining_buy_cost > 0 and month_plan['cash_flow'] > 0:
                buy_this_month = min(remaining_buy_cost, month_plan['cash_flow'])
                month_plan['cash_flow'] -= buy_this_month
                remaining_buy_cost -= buy_this_month
            
            gradual_plan['monthly_schedule'].append(month_plan)
        
        return gradual_plan
    
    def print_shift_plan(self, plan: Dict, show_details: bool = True) -> None:
        """Print shift plan in a readable, professional format."""
        print("\n" + "=" * 80)
        print(" " * 28 + "PORTFOLIO SHIFT PLAN")
        print("=" * 80)
        
        print(f"\nGenerated: {plan['generated_date']}")
        print(f"Current Portfolio Value: Rp {plan['current_value']:,.0f}")
        print(f"Target Portfolio Value: Rp {plan['target_value']:,.0f}")
        
        if plan['total_return_pct'] != 0:
            status = "Positive" if plan['total_return_pct'] > 0 else "Negative"
            print(f"Current Unrealized Return: {plan['total_return_pct']:.1f}% ({status})")
        
        print(f"\nEstimated Transaction Costs: Rp {plan['estimated_transaction_costs']:,.0f}")
        print(f"Estimated Tax Impact: Rp {plan['estimated_tax_impact']:,.0f}")
        
        if plan['sell_recommendations']:
            print("\n" + "-" * 80)
            print("SELL RECOMMENDATIONS")
            print("-" * 80)
            
            for sell in plan['sell_recommendations']:
                print(f"\n  Stock: {sell['stock']}")
                print(f"     Shares to Sell: {sell['shares']:,} shares")
                print(f"     Current Price: Rp {sell['current_price']:,.0f}")
                print(f"     Estimated Proceeds: Rp {sell['estimated_proceeds']:,.0f}")
                print(f"     Estimated Fees: Rp {sell['estimated_fees']:,.0f}")
                if sell.get('estimated_tax', 0) > 0:
                    print(f"     Estimated Tax: Rp {sell['estimated_tax']:,.0f}")
                print(f"     Net Proceeds: Rp {sell.get('net_proceeds', sell['estimated_proceeds']):,.0f}")
                print(f"     Unrealized Return: {sell['unrealized_return_pct']:.1f}%")
                print(f"     Reason: {sell['reason']}")
        
        if plan['keep_recommendations']:
            print("\n" + "-" * 80)
            print("KEEP RECOMMENDATIONS (Within Target Allocation)")
            print("-" * 80)
            
            for keep in plan['keep_recommendations']:
                print(f"\n  Stock: {keep['stock']}")
                print(f"     Current Allocation: {keep['current_allocation_pct']:.1f}%")
                print(f"     Target Allocation: {keep['target_allocation_pct']:.0f}%")
                print(f"     Current Value: Rp {keep['current_value']:,.0f}")
                print(f"     Return: {keep['return_pct']:.1f}%")
                print(f"     Note: {keep['note']}")
        
        if plan['adjust_recommendations']:
            print("\n" + "-" * 80)
            print("ADJUSTMENT RECOMMENDATIONS (Overweight/Underweight)")
            print("-" * 80)
            
            for adj in plan['adjust_recommendations']:
                print(f"\n  Stock: {adj['stock']} - Action: {adj['action']}")
                print(f"     Current Allocation: {adj['current_allocation_pct']:.1f}%")
                print(f"     Target Allocation: {adj['target_allocation_pct']:.0f}%")
                print(f"     Difference: {adj['allocation_diff_pct']:.1f}%")
                if adj['action'] == 'SELL':
                    print(f"     Shares to Sell: {adj['estimated_shares']:,}")
                    print(f"     Estimated Proceeds: Rp {adj['estimated_proceeds']:,.0f}")
                else:
                    print(f"     Shares to Buy: {adj['estimated_shares']:,}")
                    print(f"     Estimated Cost: Rp {adj['estimated_cost']:,.0f}")
                print(f"     Reason: {adj['reason']}")
        
        if plan['buy_recommendations']:
            print("\n" + "-" * 80)
            print("BUY RECOMMENDATIONS (New Positions)")
            print("-" * 80)
            
            for buy in plan['buy_recommendations']:
                print(f"\n  Stock: {buy['stock']}")
                print(f"     Target Allocation: {buy['target_allocation_pct']:.0f}%")
                print(f"     Current Price: Rp {buy.get('current_price', 0):,.0f}")
                print(f"     Estimated Shares: {buy['estimated_shares']:,}")
                print(f"     Estimated Cost: Rp {buy['estimated_cost']:,.0f}")
                print(f"     Estimated Fees: Rp {buy['estimated_fees']:,.0f}")
                print(f"     Total Cost: Rp {buy['total_cost']:,.0f}")
                print(f"     Reason: {buy['reason']}")
        
        if plan['notes']:
            print("\n" + "-" * 80)
            print("IMPORTANT NOTES")
            print("-" * 80)
            for note in plan['notes']:
                print(f"  - {note}")
        
        print("\n" + "-" * 80)
        print("CASH FLOW SUMMARY")
        print("-" * 80)
        print(f"  Net Cash from Sales: Rp {plan.get('net_cash_from_sales', 0):,.0f}")
        print(f"  Net Cash Required: Rp {plan.get('net_cash_required', 0):,.0f}")
        
        print("\n" + "=" * 80)
        print("EXECUTION GUIDANCE")
        print("=" * 80)
        print("""
1. Sell recommendations: Consider selling gradually if the position has significant unrealized loss
2. Buy recommendations: Use limit orders 1-2 percent below market price for better entry
3. Tax consideration: Capital gains tax on IDX listed stocks is 0.1 percent of proceeds (only on gains)
4. Transaction cost: Broker fees typically range from 0.15 percent to 0.3 percent per transaction
5. Rebalancing: Review your portfolio every 6 months to maintain target allocation
6. Execution window: Complete the shift within 2 to 4 weeks to maintain target allocation accuracy
7. Market timing: Consider spreading large trades over multiple days to minimize price impact
        """)
    
    def print_gradual_plan(self, plan: Dict) -> None:
        """Print gradual shift plan over multiple months."""
        print("\n" + "=" * 80)
        print(" " * 22 + "GRADUAL PORTFOLIO SHIFT PLAN")
        print("=" * 80)
        
        print(f"\nDuration: {plan['total_months']} months")
        print(f"Monthly Budget: Rp {plan['monthly_budget']:,.0f}")
        print(f"Total Sell Proceeds: Rp {plan['total_sell_proceeds']:,.0f}")
        print(f"Total Buy Cost: Rp {plan['total_buy_cost']:,.0f}")
        
        print("\n" + "-" * 80)
        print("MONTHLY EXECUTION SCHEDULE")
        print("-" * 80)
        
        for month_plan in plan['monthly_schedule']:
            month = month_plan['month']
            print(f"\nMonth {month}:")
            print(f"   Available Cash for This Month: Rp {month_plan.get('cash_flow', 0):,.0f}")
            
            if month_plan.get('sell_this_month', 0) > 0:
                print(f"   Sell Target: Rp {month_plan['sell_this_month']:,.0f}")
            if month_plan.get('buy_this_month', 0) > 0:
                print(f"   Buy Target: Rp {month_plan['buy_this_month']:,.0f}")
        
        print("\n" + "=" * 80)
        print("Note: Adjust monthly budget based on your cash flow availability")
        print("=" * 80)


if __name__ == "__main__":
    print("=" * 60)
    print("SHIFT PLANNER - MODULE LOADED")
    print("=" * 60)
    print("\nThis module provides portfolio rebalancing recommendations.")
    print("To use: Import ShiftPlanner and initialize with PortfolioManager instance.")