"""
Portfolio Manager - Tracks user's stock positions, calculates unrealized gain/loss,
and provides rebalancing recommendations based on target allocations.
Enhanced with dividend tracking, tax estimation, and performance metrics.
"""

from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np


class PortfolioManager:
    """
    Manages user's existing stock positions.
    Tracks purchase date, price, lot size, and calculates unrealized gain/loss.
    Provides rebalancing recommendations and performance analytics.
    """
    
    # Transaction costs (Indonesian stock exchange standard)
    BROKER_FEE_BUY = 0.0015      # 0.15% for buying
    BROKER_FEE_SELL = 0.0025     # 0.25% for selling (including exchange fee)
    VAT_RATE = 0.11              # 11% VAT on broker fees
    CAPITAL_GAIN_TAX = 0.001     # 0.1% final tax on capital gains
    DIVIDEND_TAX = 0.10          # 10% final withholding tax
    
    def __init__(self, all_stocks_data: dict, user_positions: list = None, dividend_data: dict = None):
        """
        Initialize PortfolioManager.
        
        Parameters
        ----------
        all_stocks_data : dict
            Dictionary of stock dataframes with 'close' and 'adjusted_close' columns
        user_positions : list, optional
            Existing user positions (default: empty list)
        dividend_data : dict, optional
            Historical dividend data for dividend tracking
        """
        self.all_stocks_data = all_stocks_data
        self.dividend_data = dividend_data or {}
        self.user_positions = user_positions if user_positions is not None else []
        self.transaction_history = []
    
    def get_price_on_date(self, stock: str, target_date: date, use_adjusted: bool = True) -> Optional[float]:
        """Get historical closing price on a specific trading day."""
        df = self.all_stocks_data.get(stock)
        if df is None:
            return None
        
        price_col = 'adjusted_close' if use_adjusted and 'adjusted_close' in df.columns else 'close'
        
        if price_col not in df.columns:
            return None
        
        available_dates = df[df.index <= pd.to_datetime(target_date)].index
        if len(available_dates) == 0:
            return None
        
        closest_date = available_dates[-1]
        return float(df.loc[closest_date, price_col])
    
    def get_current_price(self, stock: str, use_adjusted: bool = True) -> Optional[float]:
        """Get current market price."""
        df = self.all_stocks_data.get(stock)
        if df is None:
            return None
        
        price_col = 'adjusted_close' if use_adjusted and 'adjusted_close' in df.columns else 'close'
        
        if price_col not in df.columns:
            return None
        
        return float(df[price_col].iloc[-1])
    
    def get_historical_prices(self, stock: str, start_date: date, end_date: date) -> pd.Series:
        """Get historical prices for a date range."""
        df = self.all_stocks_data.get(stock)
        if df is None:
            return pd.Series()
        
        price_col = 'adjusted_close' if 'adjusted_close' in df.columns else 'close'
        mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
        return df.loc[mask, price_col]
    
    def calculate_transaction_cost(self, amount: float, is_buy: bool) -> Dict[str, float]:
        """
        Calculate transaction costs including broker fee and VAT.
        
        Parameters
        ----------
        amount : float
            Transaction amount in IDR
        is_buy : bool
            True for buy, False for sell (sell has higher fee)
        
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
    
    def add_position(self, stock: str, purchase_date: date, purchase_price: float = 0, 
                     lot_size: int = 1, include_fees: bool = True) -> bool:
        """
        Add a stock position. 1 lot = 100 shares.
        
        Parameters
        ----------
        stock : str
            Stock ticker symbol
        purchase_date : date
            Date of purchase
        purchase_price : float
            Price per share (if 0, auto-fetch from historical data)
        lot_size : int
            Number of lots (1 lot = 100 shares)
        include_fees : bool
            Whether to include transaction fees in cost basis
        
        Returns
        -------
        bool
            True if position was added successfully
        """
        if purchase_price == 0:
            purchase_price = self.get_price_on_date(stock, purchase_date)
            if purchase_price is None:
                print(f"Could not fetch price for {stock} on {purchase_date}")
                return False
        
        shares = lot_size * 100
        gross_cost = purchase_price * shares
        
        if include_fees:
            fees = self.calculate_transaction_cost(gross_cost, is_buy=True)
            cost_basis = gross_cost + fees['total']
        else:
            cost_basis = gross_cost
            fees = {'broker_fee': 0, 'vat': 0, 'total': 0}
        
        position = {
            'stock': stock,
            'purchase_date': purchase_date.strftime('%Y-%m-%d'),
            'purchase_price': purchase_price,
            'lot_size': lot_size,
            'shares': shares,
            'cost_basis': cost_basis,
            'gross_cost': gross_cost,
            'transaction_fees': fees['total']
        }
        
        # Check if position already exists (average down)
        existing_idx = None
        for i, pos in enumerate(self.user_positions):
            if pos['stock'] == stock:
                existing_idx = i
                break
        
        if existing_idx is not None:
            # Average the cost basis
            old_pos = self.user_positions[existing_idx]
            old_shares = old_pos['shares']
            old_cost = old_pos['cost_basis']
            new_shares = old_shares + shares
            new_cost = old_cost + cost_basis
            
            self.user_positions[existing_idx]['shares'] = new_shares
            self.user_positions[existing_idx]['cost_basis'] = new_cost
            self.user_positions[existing_idx]['lot_size'] = old_pos['lot_size'] + lot_size
            self.user_positions[existing_idx]['average_price'] = new_cost / new_shares
        else:
            position['average_price'] = cost_basis / shares
            self.user_positions.append(position)
        
        # Record transaction
        self.transaction_history.append({
            'date': purchase_date.strftime('%Y-%m-%d'),
            'stock': stock,
            'type': 'BUY',
            'shares': shares,
            'price': purchase_price,
            'amount': gross_cost,
            'fees': fees['total']
        })
        
        return True
    
    def remove_position(self, stock: str, shares_to_sell: int = None, sell_date: date = None) -> bool:
        """
        Remove or reduce a stock position.
        
        Parameters
        ----------
        stock : str
            Stock ticker symbol
        shares_to_sell : int, optional
            Number of shares to sell (None = sell all)
        sell_date : date, optional
            Date of sale (default: today)
        
        Returns
        -------
        bool
            True if position was removed/reduced successfully
        """
        sell_date = sell_date or date.today()
        
        for i, pos in enumerate(self.user_positions):
            if pos['stock'] == stock:
                current_price = self.get_price_on_date(stock, sell_date) or self.get_current_price(stock)
                if current_price is None:
                    print(f"Could not fetch price for {stock}")
                    return False
                
                shares_owned = pos['shares']
                sell_shares = shares_to_sell if shares_to_sell is not None else shares_owned
                
                if sell_shares > shares_owned:
                    print(f"Cannot sell {sell_shares} shares. Only {shares_owned} owned.")
                    return False
                
                # Calculate proceeds and gain
                gross_proceeds = current_price * sell_shares
                fees = self.calculate_transaction_cost(gross_proceeds, is_buy=False)
                
                cost_basis_portion = pos['cost_basis'] * (sell_shares / shares_owned)
                capital_gain = gross_proceeds - cost_basis_portion
                capital_gain_tax = max(0, capital_gain * self.CAPITAL_GAIN_TAX)
                
                net_proceeds = gross_proceeds - fees['total'] - capital_gain_tax
                
                # Update or remove position
                if sell_shares == shares_owned:
                    removed = self.user_positions.pop(i)
                else:
                    pos['shares'] -= sell_shares
                    pos['cost_basis'] -= cost_basis_portion
                    pos['lot_size'] = int(pos['shares'] / 100)
                
                # Record transaction
                self.transaction_history.append({
                    'date': sell_date.strftime('%Y-%m-%d'),
                    'stock': stock,
                    'type': 'SELL',
                    'shares': sell_shares,
                    'price': current_price,
                    'amount': gross_proceeds,
                    'fees': fees['total'],
                    'capital_gain_tax': capital_gain_tax,
                    'net_proceeds': net_proceeds,
                    'realized_gain': capital_gain
                })
                
                return True
        
        print(f"Position {stock} not found")
        return False
    
    def get_dividend_income(self, year: int = None) -> Dict[str, Any]:
        """
        Calculate dividend income received from positions.
        
        Parameters
        ----------
        year : int, optional
            Specific year to calculate (None = all time)
        
        Returns
        -------
        dict
            Dividend income breakdown
        """
        total_dividends = 0
        dividends_by_stock = {}
        
        for pos in self.user_positions:
            stock = pos['stock']
            if stock not in self.dividend_data:
                continue
            
            shares = pos['shares']
            stock_dividends = 0
            
            for yr, dps in self.dividend_data[stock].items():
                if year is None or yr == year:
                    div_amount = dps * shares
                    stock_dividends += div_amount
                    total_dividends += div_amount
            
            if stock_dividends > 0:
                dividends_by_stock[stock] = stock_dividends
        
        # Apply dividend tax
        dividend_tax = total_dividends * self.DIVIDEND_TAX
        net_dividends = total_dividends - dividend_tax
        
        return {
            'gross_dividends': total_dividends,
            'dividend_tax': dividend_tax,
            'net_dividends': net_dividends,
            'by_stock': dividends_by_stock
        }
    
    def get_unrealized_gain_loss(self, use_adjusted: bool = True) -> Tuple[pd.DataFrame, float, float, float]:
        """
        Calculate unrealized capital gain/loss for all positions.
        
        Returns
        -------
        tuple
            (DataFrame with position details, total cost, total current value, total unrealized gain/loss)
        """
        results = []
        total_cost = 0
        total_current = 0
        total_unrealized = 0
        
        for pos in self.user_positions:
            current_price = self.get_current_price(pos['stock'], use_adjusted=use_adjusted)
            if current_price is None:
                continue
            
            current_value = current_price * pos['shares']
            gain_loss = current_value - pos['cost_basis']
            gain_loss_pct = (gain_loss / pos['cost_basis']) * 100 if pos['cost_basis'] > 0 else 0
            
            # Calculate if position would be profitable to sell after fees
            sell_fees = self.calculate_transaction_cost(current_value, is_buy=False)
            after_fees_value = current_value - sell_fees['total']
            capital_gain_tax = max(0, gain_loss * self.CAPITAL_GAIN_TAX)
            net_realizable = after_fees_value - capital_gain_tax
            
            results.append({
                'Stock': pos['stock'],
                'Purchase Date': pos['purchase_date'],
                'Purchase Price': round(pos['purchase_price'], 0),
                'Current Price': round(current_price, 0),
                'Lot Size': pos['lot_size'],
                'Shares': pos['shares'],
                'Cost Basis': round(pos['cost_basis'], 0),
                'Current Value': round(current_value, 0),
                'Unrealized Gain/Loss': round(gain_loss, 0),
                'Return Percent': round(gain_loss_pct, 1),
                'Net Realizable Value': round(net_realizable, 0)
            })
            
            total_cost += pos['cost_basis']
            total_current += current_value
            total_unrealized += gain_loss
        
        df = pd.DataFrame(results)
        return df, total_cost, total_current, total_unrealized
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary including dividends and returns."""
        df, total_cost, total_current, total_unrealized = self.get_unrealized_gain_loss()
        dividend_info = self.get_dividend_income()
        
        total_return_pct = (total_unrealized / total_cost) * 100 if total_cost > 0 else 0
        total_return_with_dividends = total_unrealized + dividend_info['net_dividends']
        total_return_with_dividends_pct = (total_return_with_dividends / total_cost) * 100 if total_cost > 0 else 0
        
        weighted_return = 0
        if not df.empty:
            weighted_return = (df['Unrealized Gain/Loss'].sum() / df['Cost Basis'].sum()) * 100 if df['Cost Basis'].sum() > 0 else 0
        
        return {
            'total_invested': round(total_cost, 0),
            'current_value': round(total_current, 0),
            'unrealized_gain_loss': round(total_unrealized, 0),
            'unrealized_return_pct': round(total_return_pct, 1),
            'weighted_avg_return_pct': round(weighted_return, 1),
            'total_dividends_gross': round(dividend_info['gross_dividends'], 0),
            'total_dividends_net': round(dividend_info['net_dividends'], 0),
            'total_return_with_dividends': round(total_return_with_dividends, 0),
            'total_return_with_dividends_pct': round(total_return_with_dividends_pct, 1),
            'num_positions': len(self.user_positions),
            'num_stocks': df['Stock'].nunique() if not df.empty else 0
        }
    
    def get_performance_by_stock(self) -> pd.DataFrame:
        """Get performance metrics for each stock in portfolio."""
        df, _, _, _ = self.get_unrealized_gain_loss()
        
        if df.empty:
            return pd.DataFrame()
        
        # Add additional metrics
        df['Cost per Share'] = df['Cost Basis'] / df['Shares']
        df['Gain per Share'] = df['Current Price'] - df['Cost per Share']
        df['Days Held'] = (pd.to_datetime(date.today()) - pd.to_datetime(df['Purchase Date'])).dt.days
        
        # Sort by return percentage (best performers first)
        df = df.sort_values('Return Percent', ascending=False)
        
        return df[['Stock', 'Shares', 'Cost per Share', 'Current Price', 
                   'Gain per Share', 'Return Percent', 'Days Held', 'Net Realizable Value']]
    
    def generate_rebalancing_plan(self, target_stocks: List[str], 
                                   target_allocations: Dict[str, float],
                                   preserve_cash: bool = True) -> Dict[str, Any]:
        """
        Generate plan to rebalance from current to target portfolio.
        
        Parameters
        ----------
        target_stocks : List[str]
            List of target stock tickers
        target_allocations : Dict[str, float]
            Target allocation percentages (sum to 100)
        preserve_cash : bool
            Whether to preserve cash from sales (vs reinvest immediately)
        
        Returns
        -------
        dict
            Rebalancing plan with buy/sell recommendations
        """
        current_df, total_cost, total_current, _ = self.get_unrealized_gain_loss()
        
        plan = {
            'current_portfolio_value': round(total_current, 0),
            'sell_recommendations': [],
            'buy_recommendations': [],
            'hold_recommendations': [],
            'notes': [],
            'estimated_transaction_costs': 0,
            'estimated_tax_impact': 0
        }
        
        if current_df.empty:
            plan['notes'].append("No existing positions. Consider building from scratch.")
            return plan
        
        current_stocks = set(current_df['Stock'].tolist())
        target_stocks_set = set(target_stocks)
        
        # Stocks to sell (in current but not in target)
        to_sell = current_stocks - target_stocks_set
        
        for _, row in current_df.iterrows():
            if row['Stock'] in to_sell:
                sell_fees = self.calculate_transaction_cost(row['Current Value'], is_buy=False)
                capital_gain = row['Unrealized Gain/Loss']
                tax = max(0, capital_gain * self.CAPITAL_GAIN_TAX)
                
                plan['sell_recommendations'].append({
                    'stock': row['Stock'],
                    'shares': int(row['Shares']),
                    'current_price': row['Current Price'],
                    'estimated_proceeds': round(row['Current Value'], 0),
                    'estimated_fees': round(sell_fees['total'], 0),
                    'estimated_tax': round(tax, 0),
                    'net_proceeds': round(row['Current Value'] - sell_fees['total'] - tax, 0),
                    'unrealized_return_pct': row['Return Percent']
                })
                
                plan['estimated_transaction_costs'] += sell_fees['total']
                plan['estimated_tax_impact'] += tax
        
        # Stocks to hold (in both current and target)
        to_hold = current_stocks.intersection(target_stocks_set)
        
        for _, row in current_df.iterrows():
            if row['Stock'] in to_hold:
                current_allocation = (row['Current Value'] / total_current) * 100 if total_current > 0 else 0
                target_allocation = target_allocations.get(row['Stock'], 0)
                
                plan['hold_recommendations'].append({
                    'stock': row['Stock'],
                    'current_allocation_pct': round(current_allocation, 1),
                    'target_allocation_pct': target_allocation,
                    'current_value': round(row['Current Value'], 0),
                    'return_pct': row['Return Percent']
                })
        
        # Calculate net cash from sales
        net_sale_proceeds = sum([rec['net_proceeds'] for rec in plan['sell_recommendations']])
        
        # Stocks to buy (in target but not in current)
        to_buy = target_stocks_set - current_stocks
        
        total_alloc = sum(target_allocations.values())
        remaining_value = total_current - sum([rec['estimated_proceeds'] for rec in plan['sell_recommendations']])
        
        if preserve_cash:
            available_for_buy = net_sale_proceeds
        else:
            available_for_buy = remaining_value + net_sale_proceeds
        
        for stock in to_buy:
            alloc_pct = target_allocations.get(stock, 0)
            target_value = (available_for_buy + remaining_value) * (alloc_pct / total_alloc) if total_alloc > 0 else 0
            
            current_price = self.get_current_price(stock)
            if current_price and current_price > 0 and target_value > 0:
                shares_to_buy = int(target_value / current_price)
                buy_cost = shares_to_buy * current_price
                buy_fees = self.calculate_transaction_cost(buy_cost, is_buy=True)
                total_cost_buy = buy_cost + buy_fees['total']
                
                if shares_to_buy > 0:
                    plan['buy_recommendations'].append({
                        'stock': stock,
                        'target_allocation_pct': round(alloc_pct, 1),
                        'estimated_shares': shares_to_buy,
                        'estimated_cost': round(buy_cost, 0),
                        'estimated_fees': round(buy_fees['total'], 0),
                        'total_cost': round(total_cost_buy, 0)
                    })
        
        # Adjust holdings that need rebalancing
        for _, row in current_df.iterrows():
            if row['Stock'] in to_hold:
                current_allocation = (row['Current Value'] / total_current) * 100 if total_current > 0 else 0
                target_allocation = target_allocations.get(row['Stock'], 0)
                
                diff = current_allocation - target_allocation
                
                if diff > 5:
                    reduction_value = row['Current Value'] * (diff / 100)
                    shares_to_sell = int(reduction_value / row['Current Price'])
                    if shares_to_sell > 0:
                        sell_fees = self.calculate_transaction_cost(shares_to_sell * row['Current Price'], is_buy=False)
                        plan['sell_recommendations'].append({
                            'stock': row['Stock'],
                            'shares': shares_to_sell,
                            'current_price': row['Current Price'],
                            'estimated_proceeds': round(shares_to_sell * row['Current Price'], 0),
                            'estimated_fees': round(sell_fees['total'], 0),
                            'estimated_tax': 0,
                            'net_proceeds': round(shares_to_sell * row['Current Price'] - sell_fees['total'], 0),
                            'unrealized_return_pct': row['Return Percent'],
                            'reason': f'Overweight by {round(diff, 1)}%'
                        })
                
                elif diff < -5:
                    increase_value = row['Current Value'] * (-diff / 100)
                    shares_to_buy = int(increase_value / row['Current Price'])
                    if shares_to_buy > 0:
                        buy_fees = self.calculate_transaction_cost(shares_to_buy * row['Current Price'], is_buy=True)
                        plan['buy_recommendations'].append({
                            'stock': row['Stock'],
                            'target_allocation_pct': target_allocation,
                            'estimated_shares': shares_to_buy,
                            'estimated_cost': round(shares_to_buy * row['Current Price'], 0),
                            'estimated_fees': round(buy_fees['total'], 0),
                            'total_cost': round(shares_to_buy * row['Current Price'] + buy_fees['total'], 0),
                            'reason': f'Underweight by {round(-diff, 1)}%'
                        })
        
        plan['estimated_transaction_costs'] = round(plan['estimated_transaction_costs'], 0)
        plan['estimated_tax_impact'] = round(plan['estimated_tax_impact'], 0)
        plan['net_cash_from_sales'] = round(net_sale_proceeds, 0)
        
        return plan
    
    def clear_positions(self):
        """Clear all user positions."""
        self.user_positions = []
        self.transaction_history = []
    
    def get_transaction_history(self) -> pd.DataFrame:
        """Get transaction history as DataFrame."""
        if not self.transaction_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.transaction_history)
        return df.sort_values('date', ascending=False)
    
    def export_portfolio(self, filepath: str) -> bool:
        """
        Export portfolio to CSV file.
        
        Parameters
        ----------
        filepath : str
            Path to save the CSV file
        
        Returns
        -------
        bool
            True if export successful
        """
        try:
            df, _, _, _ = self.get_unrealized_gain_loss()
            if not df.empty:
                df.to_csv(filepath, index=False)
                return True
        except Exception as e:
            print(f"Export failed: {e}")
        return False
    
    def calculate_portfolio_risk_metrics(self) -> Dict[str, Any]:
        """
        Calculate risk metrics for the portfolio using historical volatility.
        
        Returns
        -------
        dict
            Portfolio risk metrics including weighted volatility and VaR
        """
        df, total_cost, total_current, _ = self.get_unrealized_gain_loss()
        
        if df.empty:
            return {'error': 'No positions in portfolio'}
        
        # Get volatility for each stock
        stock_volatilities = {}
        for _, row in df.iterrows():
            stock = row['Stock']
            stock_df = self.all_stocks_data.get(stock)
            if stock_df is not None and 'daily_return' in stock_df.columns:
                daily_vol = stock_df['daily_return'].std()
                annual_vol = daily_vol * np.sqrt(252)
                stock_volatilities[stock] = annual_vol
        
        # Calculate weighted portfolio volatility
        weights = df['Current Value'] / total_current
        weighted_vol = sum([weights.iloc[i] * stock_volatilities.get(row['Stock'], 0) 
                           for i, row in df.iterrows() if row['Stock'] in stock_volatilities])
        
        # Simple VaR approximation
        portfolio_value = total_current
        daily_var_95 = portfolio_value * (weighted_vol / 100) * 1.645 / np.sqrt(252)
        
        return {
            'portfolio_value': round(portfolio_value, 0),
            'weighted_volatility_annual': round(weighted_vol, 1),
            'daily_var_95': round(daily_var_95, 0),
            'stocks_analyzed': len(stock_volatilities),
            'risk_level': 'High' if weighted_vol > 30 else 'Medium' if weighted_vol > 18 else 'Low'
        }


if __name__ == "__main__":
    print("=" * 60)
    print("PORTFOLIO MANAGER MODULE LOADED")
    print("=" * 60)
    print("\nThis module provides portfolio tracking and rebalancing functionality.")
    print("To use: Import PortfolioManager and initialize with stock data.")