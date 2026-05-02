"""
Portfolio Manager - Tracks user's stock positions and calculates unrealized gain/loss.
"""

from datetime import date
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np


class PortfolioManager:
    """
    Manages user's existing stock positions.
    Tracks purchase date, price, lot size, and calculates unrealized gain/loss.
    """
    
    def __init__(self, all_stocks_data: dict, user_positions: list = None):
        """
        Initialize PortfolioManager.
        
        Parameters
        ----------
        all_stocks_data : dict
            Dictionary of stock dataframes
        user_positions : list, optional
            Existing user positions (default: empty list)
        """
        self.all_stocks_data = all_stocks_data
        self.user_positions = user_positions if user_positions is not None else []
    
    def get_price_on_date(self, stock: str, target_date: date) -> Optional[float]:
        """Get historical closing price on a specific trading day."""
        df = self.all_stocks_data.get(stock)
        if df is None or 'close' not in df.columns:
            return None
        
        available_dates = df[df.index <= pd.to_datetime(target_date)].index
        if len(available_dates) == 0:
            return None
        
        closest_date = available_dates[-1]
        return float(df.loc[closest_date, 'close'])
    
    def add_position(self, stock: str, purchase_date: date, purchase_price: float, lot_size: int) -> bool:
        """
        Add a stock position. 1 lot = 100 shares.
        
        Returns
        -------
        bool
            True if position was added successfully
        """
        if purchase_price == 0:
            purchase_price = self.get_price_on_date(stock, purchase_date)
            if purchase_price is None:
                return False
        
        shares = lot_size * 100
        position = {
            'stock': stock,
            'purchase_date': purchase_date.strftime('%Y-%m-%d'),
            'purchase_price': purchase_price,
            'lot_size': lot_size,
            'shares': shares,
            'cost_basis': purchase_price * shares
        }
        self.user_positions.append(position)
        return True
    
    def get_current_price(self, stock: str) -> Optional[float]:
        """Get current market price."""
        df = self.all_stocks_data.get(stock)
        if df is None or 'close' not in df.columns:
            return None
        return float(df['close'].iloc[-1])
    
    def get_unrealized_gain_loss(self) -> Tuple[pd.DataFrame, float, float]:
        """Calculate unrealized capital gain/loss for all positions."""
        results = []
        total_cost = 0
        total_current = 0
        
        for pos in self.user_positions:
            current_price = self.get_current_price(pos['stock'])
            if current_price is None:
                continue
            
            current_value = current_price * pos['shares']
            gain_loss = current_value - pos['cost_basis']
            gain_loss_pct = (gain_loss / pos['cost_basis']) * 100 if pos['cost_basis'] > 0 else 0
            
            results.append({
                'Stock': pos['stock'],
                'Purchase Date': pos['purchase_date'],
                'Purchase Price': pos['purchase_price'],
                'Current Price': current_price,
                'Lot Size': pos['lot_size'],
                'Shares': pos['shares'],
                'Cost Basis': pos['cost_basis'],
                'Current Value': current_value,
                'Unrealized Gain/Loss': gain_loss,
                'Return %': round(gain_loss_pct, 1)
            })
            
            total_cost += pos['cost_basis']
            total_current += current_value
        
        df = pd.DataFrame(results)
        return df, total_cost, total_current
    
    def clear_positions(self):
        """Clear all user positions."""
        self.user_positions = []
    
    def generate_shift_plan(self, recommended_stocks: List[str], recommended_allocations: Dict[str, float]) -> Dict:
        """Generate plan to shift from current to recommended portfolio."""
        current_df, total_cost, total_current = self.get_unrealized_gain_loss()
        plan = {
            'sell_recommendations': [],
            'buy_recommendations': [],
            'notes': []
        }
        
        if current_df.empty:
            plan['notes'].append("No existing positions recorded.")
            return plan
        
        current_set = set(current_df['Stock'].tolist())
        recommended_set = set(recommended_stocks)
        to_sell = current_set - recommended_set
        to_buy = recommended_set - current_set
        overlap = current_set.intersection(recommended_set)
        
        for _, row in current_df.iterrows():
            if row['Stock'] in to_sell:
                plan['sell_recommendations'].append({
                    'stock': row['Stock'],
                    'shares': int(row['Shares']),
                    'current_price': row['Current Price'],
                    'estimated_proceeds': row['Current Value'],
                    'unrealized_return': row['Return %']
                })
            elif row['Stock'] in overlap:
                plan['notes'].append(f"{row['Stock']} is already in your portfolio and recommended. Maintain position.")
        
        target_value = total_current
        total_alloc = sum(recommended_allocations.values())
        
        for stock in to_buy:
            alloc_pct = recommended_allocations.get(stock, 0)
            target_stock_value = target_value * (alloc_pct / total_alloc) if total_alloc > 0 else 0
            current_price = self.get_current_price(stock)
            if current_price and current_price > 0:
                shares_to_buy = int(target_stock_value / current_price)
                if shares_to_buy > 0:
                    plan['buy_recommendations'].append({
                        'stock': stock,
                        'target_allocation_pct': round(alloc_pct, 1),
                        'estimated_shares': shares_to_buy,
                        'estimated_cost': shares_to_buy * current_price
                    })
        
        return plan