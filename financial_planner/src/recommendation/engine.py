"""
Recommendation Engine - Data-Driven Stock Selection
Based on historical performance: return, volatility, drawdown, and goal alignment
Enhanced with dividend yield, crisis resilience, and Sharpe ratio metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def safe_normalize(series: pd.Series) -> pd.Series:
    """
    Safe min-max normalization that handles constant series.
    Returns 0.5 for constant series to avoid division by zero.
    """
    if series.max() == series.min():
        return pd.Series(0.5, index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def classify_risk(volatility: float) -> str:
    """
    Classify stock risk based on annualized volatility (in percentage).
    
    Criteria:
        Low: < 18%
        Medium: 18% - 30%
        High: > 30%
    """
    if volatility < 18.0:
        return "Low Risk"
    elif volatility < 30.0:
        return "Medium Risk"
    else:
        return "High Risk"


def compute_dividend_yield(stock: str, dividend_data: dict, current_price: float) -> float:
    """
    Calculate average dividend yield from historical data.
    
    Parameters
    ----------
    stock : str
        Stock ticker symbol
    dividend_data : dict
        Dictionary of dividend data
    current_price : float
        Current stock price
    
    Returns
    -------
    float
        Average dividend yield percentage
    """
    if stock not in dividend_data:
        return 0.0
    
    div_values = list(dividend_data[stock].values())
    if not div_values:
        return 0.0
    
    avg_dps = sum(div_values) / len(div_values)
    return (avg_dps / current_price) * 100 if current_price > 0 else 0


def compute_crisis_resilience(df: pd.DataFrame, price_col: str = 'close') -> float:
    """
    Calculate resilience score based on COVID-19 recovery.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price column
    price_col : str, default 'close'
        Name of the price column
    
    Returns
    -------
    float
        Resilience score (0-100, higher is better)
    """
    crisis_start = '2020-03-01'
    crisis_end = '2020-05-31'
    
    crisis_df = df.loc[crisis_start:crisis_end]
    if len(crisis_df) == 0:
        return 50.0
    
    pre_crisis = df.loc[:crisis_start].tail(21)
    pre_price = pre_crisis[price_col].max() if len(pre_crisis) > 0 else df[price_col].iloc[0]
    trough_price = crisis_df[price_col].min()
    
    drop_pct = ((trough_price - pre_price) / pre_price) * 100 if pre_price > 0 else 0
    
    # Resilience score: 100 = no drop, 0 = dropped 50% or more
    resilience = max(0, min(100, 100 + drop_pct * 2))
    return round(resilience, 1)


def compute_stock_features(
    df: pd.DataFrame,
    stock: str = None,
    dividend_data: dict = None
) -> Dict[str, float]:
    """
    Extract key features from a stock's historical data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' or 'adjusted_close' column
    stock : str, optional
        Stock ticker symbol for dividend lookup
    dividend_data : dict, optional
        Dictionary of dividend data
    
    Returns
    -------
    dict
        Contains annual_return (%), volatility (%), max_drawdown (%),
        sharpe_ratio, dividend_yield (%), crisis_resilience
    """
    df = df.copy()
    
    # Use adjusted_close if available
    if 'adjusted_close' in df.columns:
        price_col = 'adjusted_close'
    else:
        price_col = 'close'
    
    # Daily returns
    df["daily_ret"] = df[price_col].pct_change()
    
    # Annualized metrics (in percentage)
    annual_return = df["daily_ret"].mean() * 252 * 100
    volatility = df["daily_ret"].std() * np.sqrt(252) * 100
    
    # Maximum drawdown
    cummax = df[price_col].cummax()
    drawdown = (df[price_col] - cummax) / cummax * 100
    max_drawdown = drawdown.min()
    
    # Sharpe Ratio (risk-free rate = 5%)
    risk_free_rate = 5.0
    sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Dividend yield
    dividend_yield = 0.0
    if stock and dividend_data:
        current_price = df[price_col].iloc[-1]
        dividend_yield = compute_dividend_yield(stock, dividend_data, current_price)
    
    # Crisis resilience
    crisis_resilience = compute_crisis_resilience(df, price_col=price_col)
    
    return {
        "annual_return": round(annual_return, 2),
        "volatility": round(volatility, 2),
        "max_drawdown": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe_ratio, 3),
        "dividend_yield": round(dividend_yield, 2),
        "crisis_resilience": crisis_resilience
    }


def build_stock_features_dataframe(
    all_stocks_data: Dict[str, pd.DataFrame],
    dividend_data: dict = None
) -> pd.DataFrame:
    """
    Build a dataframe with features for all stocks.
    
    Parameters
    ----------
    all_stocks_data : dict
        Dictionary of stock dataframes
    dividend_data : dict, optional
        Dictionary of dividend data
    
    Returns
    -------
    pd.DataFrame
        DataFrame with features for each stock
    """
    features_list = []
    
    for stock, df in all_stocks_data.items():
        # Skip non-dataframe entries (metadata dictionaries)
        if not isinstance(df, pd.DataFrame):
            continue
        
        # Skip if no price column
        if "close" not in df.columns and "adjusted_close" not in df.columns:
            continue
        
        try:
            features = compute_stock_features(df, stock=stock, dividend_data=dividend_data)
            features["stock"] = stock
            features_list.append(features)
        except Exception as e:
            print(f"Warning: Could not process {stock}: {e}")
            continue
    
    df_features = pd.DataFrame(features_list)
    df_features = df_features.set_index("stock")
    
    return df_features


def normalize_features(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize features to 0-1 scale.
    For volatility and drawdown, lower values are better (inverted normalization).
    """
    df_norm = df_features.copy()
    
    df_norm["return_norm"] = safe_normalize(df_features["annual_return"])
    df_norm["vol_norm"] = 1 - safe_normalize(df_features["volatility"])
    df_norm["drawdown_norm"] = 1 - safe_normalize(df_features["max_drawdown"])
    df_norm["sharpe_norm"] = safe_normalize(df_features["sharpe_ratio"])
    df_norm["dividend_norm"] = safe_normalize(df_features["dividend_yield"])
    df_norm["resilience_norm"] = safe_normalize(df_features["crisis_resilience"])
    
    return df_norm


def get_goal_weights(goal: str) -> Dict[str, float]:
    """
    Return weights based on investment goal.
    
    Wedding (1-3 years): prioritize stability
    KPR (3-5 years): balance between return and stability
    Education (10+ years): prioritize growth
    """
    if "Wedding" in goal:
        return {
            "return": 0.25,
            "volatility": 0.35,
            "drawdown": 0.25,
            "sharpe": 0.08,
            "dividend": 0.04,
            "resilience": 0.03
        }
    elif "KPR" in goal:
        return {
            "return": 0.30,
            "volatility": 0.25,
            "drawdown": 0.15,
            "sharpe": 0.10,
            "dividend": 0.15,
            "resilience": 0.05
        }
    else:  # Education
        return {
            "return": 0.40,
            "volatility": 0.15,
            "drawdown": 0.10,
            "sharpe": 0.15,
            "dividend": 0.05,
            "resilience": 0.15
        }


def adjust_weights_for_risk(
    weights: Dict[str, float],
    risk_profile: str
) -> Dict[str, float]:
    """Adjust weights based on user's risk tolerance."""
    adjusted = weights.copy()
    
    if risk_profile == "Conservative":
        adjusted["volatility"] += 0.10
        adjusted["drawdown"] += 0.05
        adjusted["return"] -= 0.10
        adjusted["sharpe"] -= 0.05
    elif risk_profile == "Aggressive":
        adjusted["return"] += 0.10
        adjusted["volatility"] -= 0.05
        adjusted["sharpe"] += 0.05
        adjusted["dividend"] -= 0.05
    
    # Re-normalize to sum to 1
    total = sum(adjusted.values())
    if total > 0:
        return {k: v / total for k, v in adjusted.items()}
    return weights


def score_stocks(
    df_norm: pd.DataFrame,
    weights: Dict[str, float]
) -> pd.DataFrame:
    """Calculate final score for each stock based on weights."""
    df_score = df_norm.copy()
    
    df_score["final_score"] = (
        weights.get("return", 0) * df_score["return_norm"] +
        weights.get("volatility", 0) * df_score["vol_norm"] +
        weights.get("drawdown", 0) * df_score["drawdown_norm"] +
        weights.get("sharpe", 0) * df_score["sharpe_norm"] +
        weights.get("dividend", 0) * df_score["dividend_norm"] +
        weights.get("resilience", 0) * df_score["resilience_norm"]
    )
    
    return df_score.sort_values("final_score", ascending=False)


def get_smooth_allocations(scores: pd.Series, top_n: int = 5) -> Dict[str, float]:
    """
    Convert scores to allocations using softmax for smoother distribution.
    """
    top_scores = scores.head(top_n)
    
    std_val = top_scores.std()
    if std_val == 0:
        exp_scores = np.ones(len(top_scores))
    else:
        exp_scores = np.exp(top_scores.values / std_val)
    
    softmax_alloc = exp_scores / exp_scores.sum()
    
    allocations = {stock: round(pct * 100, 1) for stock, pct in zip(top_scores.index, softmax_alloc)}
    
    # Adjust rounding to sum to 100
    total = sum(allocations.values())
    if total != 100 and len(allocations) > 0:
        first_stock = list(allocations.keys())[0]
        allocations[first_stock] += round(100 - total, 1)
    
    return allocations


def get_strategy_explanation(goal: str, risk_profile: str) -> str:
    """Generate strategy explanation based on goal and risk profile."""
    if "Wedding" in goal:
        base_strategy = "80 percent low risk plus 20 percent medium risk stocks"
        focus = "capital protection for short-term fixed-date goal"
    elif "KPR" in goal:
        base_strategy = "60 percent low risk plus 30 percent medium risk plus 10 percent high risk"
        focus = "balanced return with dividend synergy for KPR coverage"
    else:
        base_strategy = "40 percent low risk plus 35 percent medium risk plus 25 percent high risk"
        focus = "long-term growth with consumer stocks as inflation hedge"
    
    risk_note = ""
    if risk_profile == "Conservative":
        risk_note = " with extra emphasis on stability and capital preservation"
    elif risk_profile == "Aggressive":
        risk_note = " with extra emphasis on growth and return potential"
    
    return f"Strategy: {base_strategy}. Focus: {focus}{risk_note}."


def generate_recommendation(
    goal: str,
    risk_profile: str,
    df_features: pd.DataFrame,
    df_norm: pd.DataFrame,
    top_n: int = 5
) -> Dict:
    """
    Generate stock recommendation based on goal and risk profile.
    
    Parameters
    ----------
    goal : str
        Investment goal (Wedding, KPR, Education)
    risk_profile : str
        Risk tolerance (Conservative, Moderate, Aggressive)
    df_features : pd.DataFrame
        Raw feature values for all stocks
    df_norm : pd.DataFrame
        Normalized feature values for all stocks
    top_n : int, default 5
        Number of stocks to recommend
    
    Returns
    -------
    dict
        Contains recommended stocks, allocations, scores, and strategy explanation
    """
    # Get base weights and adjust for risk
    weights = get_goal_weights(goal)
    weights = adjust_weights_for_risk(weights, risk_profile)
    
    # Score all stocks
    df_scored = score_stocks(df_norm, weights)
    top_stocks = df_scored.head(top_n)
    
    # Get smooth allocations
    allocations = get_smooth_allocations(top_stocks["final_score"], top_n)
    
    # Add risk levels and details for each recommended stock
    risk_levels = {}
    stock_details = {}
    for stock in top_stocks.index:
        vol = df_features.loc[stock, "volatility"]
        risk_levels[stock] = classify_risk(vol)
        stock_details[stock] = {
            "annual_return": df_features.loc[stock, "annual_return"],
            "volatility": df_features.loc[stock, "volatility"],
            "max_drawdown": df_features.loc[stock, "max_drawdown"],
            "sharpe_ratio": df_features.loc[stock, "sharpe_ratio"],
            "dividend_yield": df_features.loc[stock, "dividend_yield"],
            "crisis_resilience": df_features.loc[stock, "crisis_resilience"]
        }
    
    strategy_text = get_strategy_explanation(goal, risk_profile)
    
    # Generate explanation based on goal
    if "Wedding" in goal:
        explanation = f"For your wedding goal with {risk_profile.lower()} risk profile, these stocks prioritize stability and capital protection. Selected from {len(df_features)} LQ45 stocks based on 2019-2025 historical data, with extra weight on low volatility and low drawdown."
        allocation_guide = "Prioritize low volatility and low drawdown stocks. Avoid high risk stocks entirely for this time horizon."
    elif "KPR" in goal:
        explanation = f"For your KPR goal with {risk_profile.lower()} risk profile, these stocks offer balanced return with moderate risk. High dividend stocks can supplement monthly KPR payments."
        allocation_guide = "Include high dividend bank stocks (BBCA, BBRI, BMRI) to supplement KPR payments. Rebalance annually as down payment deadline approaches."
    else:
        explanation = f"Your education goal with {risk_profile.lower()} risk profile prioritizes long-term growth. Consumer stocks serve as natural inflation hedge for education costs rising 10-15 percent annually."
        allocation_guide = "Prioritize stocks with high Sharpe ratio and crisis resilience. Long time horizon allows higher allocation to growth stocks."
    
    return {
        "recommended_stocks": top_stocks.index.tolist(),
        "allocations": allocations,
        "scores": top_stocks["final_score"].to_dict(),
        "risk_levels": risk_levels,
        "stock_details": stock_details,
        "strategy": strategy_text,
        "weights_used": weights,
        "explanation": explanation,
        "allocation_guide": allocation_guide,
        "goal": goal,
        "risk_profile": risk_profile
    }


def generate_recommendations_batch(
    df_features: pd.DataFrame,
    df_norm: pd.DataFrame,
    goals: List[str],
    risk_profiles: List[str],
    top_n: int = 5
) -> Dict[str, Dict]:
    """
    Generate recommendations for multiple goal-risk combinations.
    
    Parameters
    ----------
    df_features : pd.DataFrame
        Raw feature values for all stocks
    df_norm : pd.DataFrame
        Normalized feature values for all stocks
    goals : list
        List of goal strings
    risk_profiles : list
        List of risk profile strings
    top_n : int, default 5
        Number of stocks to recommend per combination
    
    Returns
    -------
    dict
        Nested dictionary with recommendations for each combination
    """
    results = {}
    
    for goal in goals:
        results[goal] = {}
        for risk in risk_profiles:
            key = f"{goal}_{risk}"
            results[goal][risk] = generate_recommendation(
                goal, risk, df_features, df_norm, top_n=top_n
            )
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("RECOMMENDATION ENGINE MODULE")
    print("=" * 60)
    print("\nThis module provides stock recommendation functionality.")
    print("\nAvailable functions:")
    print("  - build_stock_features_dataframe(): Build features for all stocks")
    print("  - normalize_features(): Normalize features to 0-1 scale")
    print("  - generate_recommendation(): Generate stock recommendations")
    print("  - generate_recommendations_batch(): Batch recommendations for multiple scenarios")
    print("  - get_goal_weights(): Get weights for each goal type")
    print("  - classify_risk(): Classify risk level from volatility")