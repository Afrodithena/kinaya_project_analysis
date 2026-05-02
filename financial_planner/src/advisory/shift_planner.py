"""
Shift Planner - Generates recommendations to transition from current to target portfolio.
"""

from typing import List, Dict
import pandas as pd


class ShiftPlanner:
    """
    Generates shift plan from current portfolio to recommended portfolio.
    Calculates what to sell and what to buy.
    """
    
    def __init__(self, portfolio_manager, all_stocks_data: dict):
        """
        Initialize shift planner.
        
        Parameters
        ----------
        portfolio_manager : PortfolioManager
            Instance with user's current positions
        all_stocks_data : dict
            Dictionary of stock dataframes
        """
        self.portfolio_manager = portfolio_manager
        self.all_stocks_data = all_stocks_data
    
    def get_current_price(self, stock: str) -> float:
        """Get current market price."""
        df = self.all_stocks_data.get(stock)
        return float(df['close'].iloc[-1]) if df is not None else 0
    
    def generate_shift_plan(
        self,
        recommended_stocks: List[str],
        recommended_allocations: Dict[str, float]
    ) -> Dict:
        """
        Generate plan to shift from current to recommended portfolio.
        
        Parameters
        ----------
        recommended_stocks : list
            List of recommended stock tickers
        recommended_allocations : dict
            Allocation percentage for each recommended stock
        
        Returns
        -------
        dict
            Shift plan with sell/buy recommendations
        """
        current_df, total_cost, total_current = self.portfolio_manager.calculate_unrealized_gain_loss()
        
        plan = {
            'current_portfolio': current_df.to_dict('records') if not current_df.empty else [],
            'total_invested': total_cost,
            'current_value': total_current,
            'total_return_pct': ((total_current - total_cost) / total_cost * 100) if total_cost > 0 else 0,
            'sell_recommendations': [],
            'buy_recommendations': [],
            'keep_recommendations': [],
            'notes': []
        }
        
        # If no positions, only show buy recommendations
        if current_df.empty:
            target_value = 100_000_000  # Default target value
            for stock in recommended_stocks:
                allocation_pct = recommended_allocations.get(stock, 0)
                target_value_stock = target_value * (allocation_pct / 100)
                current_price = self.get_current_price(stock)
                shares_to_buy = int(target_value_stock / current_price) if current_price > 0 else 0
                
                if shares_to_buy > 0:
                    plan['buy_recommendations'].append({
                        'stock': stock,
                        'target_allocation_pct': allocation_pct,
                        'estimated_shares': shares_to_buy,
                        'estimated_cost': shares_to_buy * current_price,
                        'reason': f"Recommended for your portfolio with {allocation_pct:.0f}% allocation"
                    })
            
            plan['notes'].append("No existing positions found. Start building portfolio with buy recommendations above.")
            return plan
        
        # Overlap analysis
        current_stocks = set(current_df['Stock'].tolist())
        recommended_stocks_set = set(recommended_stocks)
        
        overlap = current_stocks.intersection(recommended_stocks_set)
        to_sell = current_stocks - recommended_stocks_set
        to_buy = recommended_stocks_set - current_stocks
        
        # Generate sell recommendations
        for _, row in current_df.iterrows():
            if row['Stock'] in to_sell:
                plan['sell_recommendations'].append({
                    'stock': row['Stock'],
                    'shares': int(row['Shares']),
                    'current_price': row['Current Price'],
                    'estimated_proceeds': row['Current Value'],
                    'unrealized_return': row['Return %'],
                    'reason': f"Not in recommended portfolio for your goal and risk profile"
                })
            elif row['Stock'] in overlap:
                plan['keep_recommendations'].append({
                    'stock': row['Stock'],
                    'current_allocation': (row['Current Value'] / total_current) * 100 if total_current > 0 else 0,
                    'note': f"Already in your portfolio. Maintain position."
                })
        
        # Generate buy recommendations
        target_portfolio_value = total_current  # Maintain same capital
        total_allocated = sum([recommended_allocations.get(s, 0) for s in recommended_stocks])
        
        for stock in to_buy:
            allocation_pct = recommended_allocations.get(stock, 0)
            target_value = target_portfolio_value * (allocation_pct / total_allocated) if total_allocated > 0 else 0
            current_price = self.get_current_price(stock)
            shares_to_buy = int(target_value / current_price) if current_price > 0 else 0
            
            if shares_to_buy > 0:
                plan['buy_recommendations'].append({
                    'stock': stock,
                    'target_allocation_pct': round(allocation_pct, 1),
                    'estimated_shares': shares_to_buy,
                    'estimated_cost': shares_to_buy * current_price,
                    'reason': f"Recommended for your portfolio with {allocation_pct:.0f}% allocation"
                })
        
        return plan
    
    def print_shift_plan(self, plan: Dict) -> None:
        """Print shift plan in a readable format."""
        print("\n" + "=" * 70)
        print("PORTFOLIO SHIFT PLAN")
        print("=" * 70)
        
        print(f"\nCurrent Portfolio Summary:")
        print(f"  Total Invested: Rp {plan['total_invested']:,.0f}")
        print(f"  Current Market Value: Rp {plan['current_value']:,.0f}")
        print(f"  Unrealized Return: {plan['total_return_pct']:.1f}%")
        
        if plan['sell_recommendations']:
            print("\n" + "-" * 50)
            print("SELL RECOMMENDATIONS")
            print("-" * 50)
            for sell in plan['sell_recommendations']:
                print(f"\n  Stock: {sell['stock']}")
                print(f"    Current Price: Rp {sell['current_price']:,.0f}")
                print(f"    Shares to Sell: {sell['shares']:,}")
                print(f"    Estimated Proceeds: Rp {sell['estimated_proceeds']:,.0f}")
                print(f"    Unrealized Return: {sell['unrealized_return']:.1f}%")
                print(f"    Reason: {sell['reason']}")
        
        if plan['keep_recommendations']:
            print("\n" + "-" * 50)
            print("KEEP RECOMMENDATIONS")
            print("-" * 50)
            for keep in plan['keep_recommendations']:
                print(f"\n  Stock: {keep['stock']}")
                print(f"    Current Allocation: {keep['current_allocation']:.1f}%")
                print(f"    Note: {keep['note']}")
        
        if plan['buy_recommendations']:
            print("\n" + "-" * 50)
            print("BUY RECOMMENDATIONS")
            print("-" * 50)
            for buy in plan['buy_recommendations']:
                print(f"\n  Stock: {buy['stock']}")
                print(f"    Target Allocation: {buy['target_allocation_pct']:.0f}%")
                print(f"    Estimated Shares: {buy['estimated_shares']:,}")
                print(f"    Estimated Cost: Rp {buy['estimated_cost']:,.0f}")
                print(f"    Reason: {buy['reason']}")
        
        if plan['notes']:
            print("\n" + "-" * 50)
            print("ADDITIONAL NOTES")
            print("-" * 50)
            for note in plan['notes']:
                print(f"  {note}")
        
        print("\n" + "=" * 70)
        print("EXECUTION GUIDANCE")
        print("=" * 70)
        print("""
1. For sell recommendations: Consider selling gradually if the position has significant unrealized loss
2. For buy recommendations: Use limit orders to avoid paying excessive premiums
3. Tax Consideration: Capital gains tax on IDX listed stocks is 0.1% of proceeds
4. Transaction Cost: Broker fees typically range from 0.1% to 0.3% per transaction
5. Rebalancing: Review your portfolio every 6 months to maintain target allocation
""")